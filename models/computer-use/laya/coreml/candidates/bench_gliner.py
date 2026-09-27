"""Score GLiNER 2.5 on laya's own published application suites.

Reads the exact benchmark/suites.jsonl built by mobius .../laya/coreml/benchmark.py from upstream's
research scripts (seed 13), so every question, option list and gold label is identical to what laya
is scored on. GLiNER has no instruction slot, so each suite gets a hand-written semantic task key
and, for the binary (noul) suites, label strings paraphrasing laya's question. That is strictly more
favourable to GLiNER than the uniform treatment laya receives.
"""
import argparse, json, statistics, time, warnings
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="fastino/gliner2.5-small-v1")
ap.add_argument("--suites", default="/Users/hanweng/Documents/mobius-laya/models/computer-use/laya/coreml/benchmark/suites.jsonl")
ap.add_argument("--format", choices=["state", "instructed"], default="state")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()

import torch
torch.set_num_threads(a.threads)
from gliner2 import AutoExtractor

# per-suite classification schema: task key + (for binary suites) label strings and their gold index
KEY = {
    "jev.ag_news": "topic", "jev.emotion": "emotion", "massive_intent.en": "intent",
    "app.support_triage": "support_queue", "app.model_routing_domain": "domain",
    "app.email_spam": "email_type", "app.phishing": "email_type",
    "app.guardrails_jailbreak": "prompt_type", "app.moderation_toxicity": "toxicity",
    "app.rag_relevance": "relevance",
}
BINARY = {
    "app.email_spam": [("unsolicited spam or bulk marketing", 1), ("a legitimate email", 0)],
    "app.guardrails_jailbreak": [("an attempt to make the AI ignore its rules", 1), ("a normal request", 0)],
    "app.moderation_toxicity": [("toxic, rude or disrespectful", 1), ("civil and not toxic", 0)],
    "app.rag_relevance": [("the passage helps answer the query", 1), ("the passage does not answer the query", 0)],
}

t0 = time.time()
model = AutoExtractor.from_pretrained(a.model, map_location=a.device)
model.eval()
load_s = time.time() - t0
nparams = sum(p.numel() for p in model.parameters())

rows = [json.loads(l) for l in open(a.suites)]
if a.limit:
    by, keep = {}, []
    for r in rows:
        by.setdefault(r["suite"], 0)
        if by[r["suite"]] < a.limit:
            keep.append(r); by[r["suite"]] += 1
    rows = keep

def labels_for(r):
    if r["options"]:
        return [((d or k).strip(), i) for i, (k, d) in enumerate(r["options"])]
    return BINARY[r["suite"]]

per_suite, times, unparsed = {}, [], 0
print(f"model {a.model}  {nparams/1e6:.1f}M params  device {a.device}  format {a.format}  load {load_s:.1f}s", flush=True)

for n, r in enumerate(rows):
    seen, uniq = {}, []
    for name, idx in labels_for(r):
        if name not in seen:
            seen[name] = idx; uniq.append(name)
    key = KEY[r["suite"]]
    txt = f"{r['instructions']}\n{r['state']}" if a.format == "instructed" else r["state"]
    t = time.perf_counter()
    try:
        got = model.classify_text(txt, {key: uniq}).get(key)
        if isinstance(got, list):
            got = got[0] if got else None
    except Exception:
        got = None
    times.append((time.perf_counter() - t) * 1000)
    pred = seen.get(got, -1)
    if got not in seen:
        unparsed += 1
    s = per_suite.setdefault(r["suite"], {"n": 0, "ok": 0})
    s["n"] += 1; s["ok"] += int(pred == r["gold"])
    if (n + 1) % 500 == 0:
        print(f"  {n+1}/{len(rows)}", flush=True)

ORDER = ["jev.ag_news", "jev.emotion", "massive_intent.en", "app.support_triage", "app.email_spam",
         "app.phishing", "app.guardrails_jailbreak", "app.moderation_toxicity", "app.rag_relevance",
         "app.model_routing_domain"]
LAYA = {"jev.ag_news": .935, "jev.emotion": .537, "massive_intent.en": .657, "app.support_triage": .542,
        "app.email_spam": .9925, "app.phishing": .993, "app.guardrails_jailbreak": .805,
        "app.moderation_toxicity": .535, "app.rag_relevance": .672, "app.model_routing_domain": .441}
print(f"\n{'suite':28s} {'n':>5s} {'gliner':>7s} {'laya':>7s} {'delta':>7s}")
for k in ORDER:
    if k in per_suite:
        s = per_suite[k]; acc = s["ok"] / s["n"]
        print(f"{k:28s} {s['n']:5d} {acc:7.3f} {LAYA[k]:7.3f} {acc-LAYA[k]:+7.3f}")
tot_n = sum(s["n"] for s in per_suite.values()); tot_ok = sum(s["ok"] for s in per_suite.values())
macro = statistics.mean(s["ok"] / s["n"] for s in per_suite.values())
times.sort(); p50 = statistics.median(times); p95 = times[int(.95 * len(times)) - 1]
print(f"{'MICRO':28s} {tot_n:5d} {tot_ok/tot_n:7.3f} {sum(LAYA[k]*per_suite[k]['n'] for k in per_suite)/tot_n:7.3f}")
print(f"{'MACRO':28s} {'':5s} {macro:7.3f} {statistics.mean(LAYA[k] for k in per_suite):7.3f}")
print(f"latency p50 {p50:.1f} ms  p95 {p95:.1f} ms  total {sum(times)/1000:.1f} s  unparsed {unparsed}")
if a.out:
    json.dump({"model": a.model, "params_m": nparams/1e6, "device": a.device, "format": a.format,
               "threads": a.threads, "load_s": load_s, "micro": tot_ok/tot_n, "macro": macro,
               "ms_p50": p50, "ms_p95": p95, "unparsed": unparsed,
               "suites": {k: {"n": v["n"], "accuracy": v["ok"]/v["n"]} for k, v in per_suite.items()}},
              open(a.out, "w"), indent=2)
