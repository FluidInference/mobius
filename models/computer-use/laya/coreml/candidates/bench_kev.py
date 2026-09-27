"""Score a Kev checkpoint on laya's published application suites.

Kev speaks the same typed-decision contract as laya (a shared state plus typed choice / noul / score
questions, pointer head over option markers, one forward pass), so each suite row maps onto a System
One request with no prompt translation: option keys, descriptions, order, instructions and the gold
label all pass through unchanged.

Scoring goes through kev.serve.Server, which is how Kev is actually served: it allows
INFER_MAX_STATE = 8192 state tokens. kev.predictors.LocalPredictor instead enforces the 384-token
*training* limit and hard-fails ~20% of the two email suites, which would not be a like-for-like
comparison against laya (laya truncates to its own max_len rather than refusing). The prefix cache
is disabled so latency reflects a cold state every time, as it would here: every state is unique.

Shards so a long run fits inside one foreground call: --start / --end are row indices.
"""
import argparse, json, os, statistics, time, warnings
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
ap.add_argument("--suites", default="/Users/hanweng/Documents/mobius-laya/models/computer-use/laya/coreml/benchmark/suites.jsonl")
ap.add_argument("--device", default="mps")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=0)      # 0 = to the end
ap.add_argument("--out", default="")
a = ap.parse_args()

os.environ["KEV_PREFIX_CACHE"] = "0"               # must be set before kev.serve is imported

from kev.api import question_keys
from kev.checkpoint import LoadOptions
from kev.data import materialize
from kev.predictors import LocalPredictor
from kev.serve import Server, PREFIX_CACHE_SIZE
assert PREFIX_CACHE_SIZE == 0, "prefix cache must be off for honest latency"

t0 = time.time()
pred = LocalPredictor(a.run, a.device, LoadOptions())
server = Server(None, pred.tok, pred.model, a.device)
load_s = time.time() - t0
print(f"run {a.run}  device {a.device}  load {load_s:.1f}s  temperature {pred.temperature}", flush=True)

rows = [json.loads(l) for l in open(a.suites)]
end = a.end or len(rows)
rows = rows[a.start:end]
print(f"rows [{a.start}:{end}) = {len(rows)}", flush=True)

def to_request(r):
    """laya suite row -> Kev labelled request. Keys, descriptions and order are laya's, unchanged."""
    state = json.loads(r["state"])
    if r["type"] == "choice":
        q = {"type": "choice", "instructions": r["instructions"],
             "criteria": {k: d for k, d in r["options"]},
             "label": r["options"][r["gold"]][0]}
    else:
        q = {"type": "noul", "instructions": r["instructions"],
             "criteria": ({k: d for k, d in r["options"]} if r["options"] else None),
             "label": bool(r["gold"])}
    q["src"] = r["suite"]
    return {"state": state, "questions": {"q": q}}

out_rows, times, errs, details = [], [], 0, []
for n, r in enumerate(rows):
    try:
        req = to_request(r)
        ps, meta = server.probs(materialize(req))
        keys = question_keys(req["questions"]["q"]["type"], req["questions"]["q"].get("criteria"))
        p = ps[0]
        pred_idx = max(range(len(p)), key=lambda i: p[i])
        times.append(meta["latency_ms"])
    except Exception as e:
        errs += 1
        if len(details) < 10:
            details.append({"suite": r["suite"], "error": repr(e)[:160]})
        pred_idx = -1
    out_rows.append({"suite": r["suite"], "ok": int(pred_idx == r["gold"])})
    if (n + 1) % 250 == 0:
        print(f"  {n+1}/{len(rows)}  {statistics.median(times):.0f} ms median", flush=True)

per_suite = {}
for o in out_rows:
    s = per_suite.setdefault(o["suite"], {"n": 0, "ok": 0}); s["n"] += 1; s["ok"] += o["ok"]
times.sort()
res = {"run": a.run, "device": a.device, "start": a.start, "end": end, "load_s": load_s,
       "errors": errs, "error_details": details,
       "ms_p50": statistics.median(times) if times else None,
       "ms_p95": times[int(.95*len(times))-1] if times else None,
       "times_ms": times,
       "suites": {k: v for k, v in per_suite.items()}}
for k, v in per_suite.items():
    print(f"{k:28s} {v['n']:5d} {v['ok']/v['n']:7.3f}")
print(f"errors {errs}  p50 {res['ms_p50']} ms")
for d in details:
    print("  ", d)
if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
