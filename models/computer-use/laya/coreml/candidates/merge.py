import json, statistics, sys, glob
LAYA = {"jev.ag_news": .9350, "jev.emotion": .5375, "massive_intent.en": .6567, "app.support_triage": .5425,
        "app.email_spam": .9925, "app.phishing": .9925, "app.guardrails_jailbreak": .8075,
        "app.moderation_toxicity": .5350, "app.rag_relevance": .6725, "app.model_routing_domain": .4411}
ORDER = ["jev.ag_news","jev.emotion","massive_intent.en","app.support_triage","app.email_spam",
         "app.phishing","app.guardrails_jailbreak","app.moderation_toxicity","app.rag_relevance",
         "app.model_routing_domain"]
per, times, errs = {}, [], 0
files = sorted(glob.glob(sys.argv[1]))
for f in files:
    d = json.load(open(f))
    errs += d["errors"]; times += d["times_ms"]
    for k, v in d["suites"].items():
        s = per.setdefault(k, {"n":0,"ok":0}); s["n"] += v["n"]; s["ok"] += v["ok"]
print(f"merged {len(files)} shards")
print(f"{'suite':28s} {'n':>5s} {'model':>7s} {'laya':>7s} {'delta':>7s}")
for k in ORDER:
    s = per[k]; acc = s["ok"]/s["n"]
    print(f"{k:28s} {s['n']:5d} {acc:7.3f} {LAYA[k]:7.3f} {acc-LAYA[k]:+7.3f}")
n = sum(s["n"] for s in per.values()); ok = sum(s["ok"] for s in per.values())
macro = statistics.mean(s["ok"]/s["n"] for s in per.values())
times.sort()
print(f"{'MICRO':28s} {n:5d} {ok/n:7.3f} {sum(LAYA[k]*per[k]['n'] for k in per)/n:7.3f}")
print(f"{'MACRO':28s} {'':5s} {macro:7.3f} {statistics.mean(LAYA.values()):7.3f}")
print(f"latency p50 {statistics.median(times):.1f} ms  p95 {times[int(.95*len(times))-1]:.1f} ms  errors {errs}")
