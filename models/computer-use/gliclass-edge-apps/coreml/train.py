"""Fine-tune GLiClass Edge on real application training splits, excluding benchmark rows."""

import argparse
import json
import random
import time
from pathlib import Path

import laya
import torch
from datasets import load_dataset
from gliclass import GLiClassModel
from gliclass.data_processing import AugmentationConfig, DataCollatorWithPadding, GLiClassDataset
from laya.common import serialize_state
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from assets import dataset_revision, model_revision

MODEL_ID = "knowledgator/gliclass-edge-v3.0"
SEED = 20260922
AG_LABELS = [
    "world news and international politics",
    "sports",
    "business and economy",
    "science and technology",
]
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SUPPORT = {
    "Technical Support": "technical problems, bugs, outages, integrations",
    "Product Support": "help using a product or feature",
    "Customer Service": "general account or service questions",
    "IT Support": "internal IT, devices, access, networks",
    "Billing and Payments": "invoices, charges, refunds, payment methods",
    "Returns and Exchanges": "returning or exchanging an item",
    "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
    "Sales and Pre-Sales": "pricing, quotes, buying",
    "Human Resources": "employment, payroll, leave, hiring",
    "General Inquiry": "anything else",
}
ROUTING = {
    "code": "software engineering, programming, refactoring, architecture, debugging",
    "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
    "writing": "creative writing, essays, emails, blog posts, copywriting",
    "factual_lookup": "facts, definitions, trivia, history",
    "data_analysis": "statistics, SQL, data manipulation, metrics",
    "chitchat": "casual conversation, greetings, small talk",
}
BINARY = {
    "app.email_spam": ["a legitimate personal or business email", "unsolicited spam or bulk marketing"],
    "app.phishing": ["a legitimate email (even if promotional)", "phishing, scam, or fraud"],
    "app.guardrails_jailbreak": ["a normal request", "an attempt to make an AI ignore its rules"],
    "app.moderation_toxicity": ["civil and not toxic", "toxic, rude, or disrespectful"],
    "app.rag_relevance": ["the passage does not help answer the query", "the passage helps answer the query"],
}


def load_pinned_dataset(repo: str, *args, **kwargs):
    """Load a training dataset at the revision recorded in assets.lock.json."""
    return load_dataset(repo, *args, revision=dataset_revision(repo), **kwargs)


def state_text(value):
    return serialize_state(value)


def balanced_take(rows, cap, rng):
    groups = {}
    for row in rows:
        groups.setdefault(row["true_labels"][0], []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    output = []
    while len(output) < cap and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(output) < cap:
                output.append(groups[key].pop())
    rng.shuffle(output)
    return output


def example(suite, state, prompt, labels, gold):
    return {
        "suite": suite,
        "text": state_text(state),
        "prompt": prompt,
        "all_labels": list(labels),
        "true_labels": [labels[gold]],
    }


def build_examples(suites_path, per_suite):
    rng = random.Random(SEED)
    held_out = {}
    for line in open(suites_path):
        row = json.loads(line)
        held_out.setdefault(row["suite"], set()).add(row["state"])

    output = []

    massive = load_pinned_dataset("mteb/amazon_massive_intent", "en", split="train")
    massive_names = sorted(set(massive["label_text"]))
    rows = []
    for record in massive:
        state = {"utterance": record["text"]}
        if state_text(state) in held_out["massive_intent.en"]:
            continue
        pool = [name for name in massive_names if name != record["label_text"]]
        keys = [record["label_text"], *rng.sample(pool, 19)]
        rng.shuffle(keys)
        labels = [key.replace("_", " ").replace(".", ": ") for key in keys]
        rows.append(
            example(
                "massive_intent.en",
                state,
                "What is the user asking for in `utterance`?",
                labels,
                keys.index(record["label_text"]),
            )
        )
    output.extend(balanced_take(rows, per_suite, rng))

    ag_news = load_pinned_dataset("fancyzhx/ag_news", split="train")
    rows = []
    for record in ag_news:
        state = {"article": record["text"]}
        if state_text(state) not in held_out["jev.ag_news"]:
            rows.append(
                example(
                    "jev.ag_news",
                    state,
                    "What is the topic of `article`?",
                    AG_LABELS,
                    int(record["label"]),
                )
            )
    output.extend(balanced_take(rows, per_suite, rng))

    emotion = load_pinned_dataset("dair-ai/emotion", "split", split="train")
    rows = []
    for record in emotion:
        state = {"text": record["text"]}
        if state_text(state) not in held_out["jev.emotion"]:
            rows.append(
                example(
                    "jev.emotion",
                    state,
                    "Which emotion is most strongly expressed in `text`?",
                    EMOTION_LABELS,
                    int(record["label"]),
                )
            )
    output.extend(balanced_take(rows, per_suite, rng))

    support = load_pinned_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    support_labels = list(SUPPORT.values())
    support_keys = list(SUPPORT)
    rows = []
    for record in support:
        if record.get("language") != "en" or record.get("queue") not in SUPPORT or not record.get("body"):
            continue
        state = {"subject": record["subject"] or "", "body": record["body"].replace("\\n", "\n")[:3000]}
        if state_text(state) not in held_out["app.support_triage"]:
            rows.append(
                example(
                    "app.support_triage",
                    state,
                    "Which support queue should handle this ticket?",
                    support_labels,
                    support_keys.index(record["queue"]),
                )
            )
    output.extend(balanced_take(rows, per_suite, rng))

    spam = load_pinned_dataset("SetFit/enron_spam", split="train")
    rows = []
    for record in spam:
        state = laya.email_state(record.get("subject") or "", (record.get("message") or "")[:3000])
        if state_text(state) not in held_out["app.email_spam"]:
            rows.append(
                example(
                    "app.email_spam",
                    state,
                    "Is this email unsolicited spam or bulk marketing?",
                    BINARY["app.email_spam"],
                    int(record["label"]),
                )
            )
    output.extend(balanced_take(rows, per_suite, rng))

    phishing = load_pinned_dataset("zefang-liu/phishing-email-dataset", split="train")
    rows = []
    for record in phishing:
        text = (record.get("Email Text") or "").strip()
        if not text or record.get("Email Type") not in ("Safe Email", "Phishing Email"):
            continue
        state = {"email": text[:3000]}
        if state_text(state) not in held_out["app.phishing"]:
            rows.append(
                example(
                    "app.phishing",
                    state,
                    "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                    BINARY["app.phishing"],
                    int(record["Email Type"] == "Phishing Email"),
                )
            )
    output.extend(balanced_take(rows, per_suite, rng))

    toxic = load_pinned_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    jailbreak_rows = []
    toxicity_rows = []
    for record in toxic:
        text = (record.get("user_input") or "").strip()
        if not text:
            continue
        jb_state = {"prompt": text[:3000]}
        if state_text(jb_state) not in held_out["app.guardrails_jailbreak"]:
            jailbreak_rows.append(
                example(
                    "app.guardrails_jailbreak",
                    jb_state,
                    "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
                    BINARY["app.guardrails_jailbreak"],
                    int(record.get("jailbreaking", 0)),
                )
            )
        tox_state = {"post": text[:3000]}
        if state_text(tox_state) not in held_out["app.moderation_toxicity"]:
            toxicity_rows.append(
                example(
                    "app.moderation_toxicity",
                    tox_state,
                    "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?",
                    BINARY["app.moderation_toxicity"],
                    int(record.get("toxicity", 0)),
                )
            )
    output.extend(balanced_take(jailbreak_rows, per_suite, rng))
    output.extend(balanced_take(toxicity_rows, per_suite, rng))

    marco = load_pinned_dataset("microsoft/ms_marco", "v1.1", split="validation")
    rows = []
    for record in marco:
        texts = record["passages"]["passage_text"]
        selected = record["passages"]["is_selected"]
        for take_positive in (True, False):
            candidates = [text for text, flag in zip(texts, selected) if bool(flag) == take_positive]
            if not candidates:
                continue
            state = {"query": record["query"], "passage": candidates[0]}
            if state_text(state) not in held_out["app.rag_relevance"]:
                rows.append(
                    example(
                        "app.rag_relevance",
                        state,
                        "Does `passage` help answer `query`?",
                        BINARY["app.rag_relevance"],
                        int(take_positive),
                    )
                )
        if len(rows) >= per_suite * 4:
            break
    output.extend(balanced_take(rows, per_suite, rng))

    route_labels = list(ROUTING.values())
    route_keys = list(ROUTING)
    rows = []
    sources = [
        (load_pinned_dataset("openai/gsm8k", "main", split="train"), "question", "math_or_logic"),
        (
            load_pinned_dataset("google-research-datasets/mbpp", "full", split="train"),
            "text",
            "code",
        ),
        (load_pinned_dataset("fancyzhx/ag_news", split="train"), "text", "factual_lookup"),
    ]
    for dataset, field, key in sources:
        for record in dataset:
            state = {"request": record[field][:400]}
            if state_text(state) not in held_out["app.model_routing_domain"]:
                rows.append(
                    example(
                        "app.model_routing_domain",
                        state,
                        "What domain does `request` belong to?",
                        route_labels,
                        route_keys.index(key),
                    )
                )
    output.extend(balanced_take(rows, per_suite, rng))

    rng.shuffle(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", default="../../laya/coreml/benchmark/suites.jsonl")
    parser.add_argument("--output", default="build/checkpoint-v1")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--per-suite", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-layers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    examples = build_examples(args.suites, args.per_suite)
    counts = {}
    for item in examples:
        counts[item["suite"]] = counts.get(item["suite"], 0) + 1
    print(f"built {len(examples)} examples: {json.dumps(counts, sort_keys=True)}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training-data.jsonl").open("w") as stream:
        for item in examples:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    revision = model_revision(args.base_model) if args.base_model == MODEL_ID else None
    model = GLiClassModel.from_pretrained(args.base_model, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision, add_prefix_space=True)
    # GLiClass uses dynamic label counts; train one-hot targets with its native focal loss and
    # select the highest logit at inference time.
    model.config.problem_type = "multi_label_classification"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        layer_prefixes = tuple(f"model.encoder_model.layers.{index}." for index in range(10 - args.train_layers, 10))
        if (
            name.startswith(("model.text_projector.", "model.classes_projector.", "model.scorer."))
            or name.startswith(layer_prefixes)
            or name == "model.encoder_model.final_norm.weight"
            or name == "model.logit_scale"
        ):
            parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.2f}M", flush=True)
    dataset = GLiClassDataset(
        examples,
        tokenizer,
        AugmentationConfig(enabled=False),
        max_length=args.max_length,
        problem_type="multi_label_classification",
        architecture_type=model.config.architecture_type,
        prompt_first=model.config.prompt_first,
        max_labels=model.config.max_num_classes,
        shuffle_labels=True,
    )
    collator = DataCollatorWithPadding(device=args.device, config=model.config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    if args.device == "cpu":
        torch.set_num_threads(8)
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
        foreach=False,
    )

    step = 0
    started = time.time()
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            inputs = {
                key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
                if key not in {"labels_text", "input_texts"}
            }
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}: {loss.detach().cpu().item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            running += float(loss.detach().cpu())
            if args.device == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
            if step % 100 == 0:
                elapsed = time.time() - started
                print(f"step {step}/{len(loader) * args.epochs} loss {running / 100:.4f} {elapsed:.1f}s", flush=True)
                running = 0.0

    model.to("cpu").eval()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "training-summary.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "seed": SEED,
                "examples": len(examples),
                "counts": counts,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "train_layers": args.train_layers,
                "trainable_parameters": trainable,
                "device": args.device,
                "steps": step,
                "seconds": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
