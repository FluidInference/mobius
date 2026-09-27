"""Score LFM2.5-350M constrained candidate likelihoods on laya's application suites."""

import argparse
import copy
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "notnotsamuel/LFM2.5-350M-RLCD"
REVISION = "deb589d803d141cabd158ef55f6617b128529f36"
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


def fork_cache(cache, count):
    cloned = copy.deepcopy(cache)
    device = next(layer.keys.device for layer in cache.layers if hasattr(layer, "keys"))
    cloned.reorder_cache(torch.zeros(count, dtype=torch.long, device=device))
    return cloned


class Engine:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(device).eval()
        self.model.requires_grad_(False)

    def tensor(self, tokens):
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def choose(self, context, instruction, options):
        definitions = "; ".join(f"{key}: {description}" for key, description in options)
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": f"{instruction} Allowed answers: {definitions}",
                    "enum": [key for key, _ in options],
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the decision question about the supplied state. Return only a JSON "
                    "object matching this schema and use an exact allowed value.\n"
                    + json.dumps(schema, ensure_ascii=False)
                ),
            },
            {"role": "user", "content": context},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "{\n"
        prefix = self.encode(prompt)
        cache = self.model(self.tensor([prefix]), use_cache=True, logits_to_keep=1).past_key_values
        suffix = self.encode('  "answer": ')
        branches = []
        values = []
        for key, _ in options:
            value = self.encode(json.dumps(key, ensure_ascii=False) + "\n")
            branches.append(suffix + value)
            values.append(value)
        width = max(map(len, branches))
        ids = self.tensor([branch + [self.tokenizer.pad_token_id] * (width - len(branch)) for branch in branches])
        mask = self.tensor(
            [[1] * (len(prefix) + len(branch)) + [0] * (width - len(branch)) for branch in branches]
        )
        out = self.model(
            ids,
            past_key_values=fork_cache(cache, len(branches)),
            attention_mask=mask,
            use_cache=False,
        )
        scores = []
        for row, value in enumerate(values):
            logp = out.logits[row, len(suffix) - 1 : len(suffix) + len(value) - 1].float().log_softmax(-1)
            scores.append(logp.gather(1, self.tensor(value)[:, None]).sum().item())
        return max(range(len(scores)), key=scores.__getitem__)


def options_for(row):
    if row["options"]:
        return row["options"]
    return [["false", "No"], ["true", "Yes"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites",
        default="/Users/hanweng/Documents/mobius-laya/models/computer-use/laya/coreml/benchmark/suites.jsonl",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=10, help="Rows per suite; zero runs all rows")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

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

    started = time.time()
    engine = Engine(args.device)
    print(f"loaded {MODEL_ID} in {time.time() - started:.1f}s; evaluating {len(rows)} rows", flush=True)
    per_suite = {}
    times = []
    for index, row in enumerate(rows):
        engine.sync()
        tick = time.perf_counter()
        prediction = engine.choose(row["state"], row["instructions"], options_for(row))
        engine.sync()
        times.append((time.perf_counter() - tick) * 1000)
        result = per_suite.setdefault(row["suite"], {"n": 0, "ok": 0})
        result["n"] += 1
        result["ok"] += int(prediction == row["gold"])
        if (index + 1) % 20 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)

    accuracies = {key: value["ok"] / value["n"] for key, value in per_suite.items()}
    macro = statistics.mean(accuracies.values())
    micro = sum(value["ok"] for value in per_suite.values()) / sum(value["n"] for value in per_suite.values())
    times.sort()
    output = {
        "model": MODEL_ID,
        "revision": REVISION,
        "rows": len(rows),
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
