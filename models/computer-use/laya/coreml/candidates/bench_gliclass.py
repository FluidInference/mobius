"""Score a GLiClass checkpoint on laya's published application suites."""

import argparse
import json
import statistics
import time
import warnings

import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


ORDER = [
    "jev.ag_news",
    "jev.emotion",
    "massive_intent.en",
    "app.support_triage",
    "app.email_spam",
    "app.phishing",
    "app.guardrails_jailbreak",
    "app.moderation_toxicity",
    "app.rag_relevance",
    "app.model_routing_domain",
]
LAYA = {
    "jev.ag_news": 0.935,
    "jev.emotion": 0.537,
    "massive_intent.en": 0.657,
    "app.support_triage": 0.542,
    "app.email_spam": 0.9925,
    "app.phishing": 0.993,
    "app.guardrails_jailbreak": 0.805,
    "app.moderation_toxicity": 0.535,
    "app.rag_relevance": 0.672,
    "app.model_routing_domain": 0.441,
}
BINARY = {
    "app.email_spam": [("a legitimate personal or business email", 0), ("unsolicited spam or bulk marketing", 1)],
    "app.guardrails_jailbreak": [("a normal request", 0), ("an attempt to make an AI ignore its rules", 1)],
    "app.moderation_toxicity": [("civil and not toxic", 0), ("toxic, rude, or disrespectful", 1)],
    "app.rag_relevance": [("the passage does not help answer the query", 0), ("the passage helps answer the query", 1)],
}


def labels_for(row, label_format):
    if not row["options"]:
        return BINARY[row["suite"]]
    labels = []
    for index, (key, description) in enumerate(row["options"]):
        if label_format == "key":
            label = key
        elif label_format == "full":
            label = f"{key}: {description}" if description else key
        else:
            label = description or key
        labels.append((label, index))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="knowledgator/gliclass-edge-v3.0")
    parser.add_argument(
        "--suites",
        default="/Users/hanweng/Documents/mobius-laya/models/computer-use/laya/coreml/benchmark/suites.jsonl",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=10, help="Rows per suite; zero runs all rows")
    parser.add_argument("--label-format", choices=["description", "key", "full"], default="description")
    parser.add_argument("--instruction-in-text", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    started = time.time()
    model = GLiClassModel.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    pipeline = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        max_length=args.max_length,
        classification_type="single-label",
        device=args.device,
        progress_bar=False,
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"loaded {args.model} ({params / 1e6:.1f}M) in {time.time() - started:.1f}s", flush=True)

    rows = [json.loads(line) for line in open(args.suites)]
    if args.limit:
        counts = {}
        selected = []
        for row in rows:
            counts.setdefault(row["suite"], 0)
            if counts[row["suite"]] < args.limit:
                selected.append(row)
                counts[row["suite"]] += 1
        rows = selected

    per_suite = {}
    times = []
    for index, row in enumerate(rows):
        choices = labels_for(row, args.label_format)
        labels = [label for label, _ in choices]
        label_to_gold = {label: gold for label, gold in choices}
        text = row["state"]
        prompt = row["instructions"]
        if args.instruction_in_text:
            text = f"Question: {prompt}\nState: {text}"
            prompt = None
        if args.device == "mps":
            torch.mps.synchronize()
        tick = time.perf_counter()
        prediction = pipeline(text, labels, threshold=0.0, prompt=prompt)[0][0]["label"]
        if args.device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - tick) * 1000)
        result = per_suite.setdefault(row["suite"], {"n": 0, "ok": 0})
        result["n"] += 1
        result["ok"] += int(label_to_gold[prediction] == row["gold"])
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)

    accuracies = {key: value["ok"] / value["n"] for key, value in per_suite.items()}
    macro = statistics.mean(accuracies.values())
    micro = sum(value["ok"] for value in per_suite.values()) / sum(value["n"] for value in per_suite.values())
    times.sort()
    output = {
        "model": args.model,
        "params_m": params / 1e6,
        "rows": len(rows),
        "label_format": args.label_format,
        "instruction_in_text": args.instruction_in_text,
        "max_length": args.max_length,
        "macro": macro,
        "micro": micro,
        "ms_p50": statistics.median(times),
        "ms_p95": times[int(0.95 * len(times)) - 1],
        "suites": {key: {**value, "accuracy": accuracies[key]} for key, value in per_suite.items()},
    }
    for key in ORDER:
        print(f"{key:28s} {accuracies[key]:.3f}  laya {LAYA[key]:.3f}")
    print(f"macro {macro:.3f}  micro {micro:.3f}  p50 {output['ms_p50']:.1f} ms")
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()
