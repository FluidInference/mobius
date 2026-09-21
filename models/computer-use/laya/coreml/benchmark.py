"""Build laya's published application suites, score them with the PyTorch reference, and export them.

The suite builders are transcribed from the upstream research scripts
(NandhaKishorM/laya `research/scripts/bench_apps.py` and `bench_local.py`, Apache-2.0) so the
questions, sampling (seed 13, 400 cases per task, 300 MASSIVE cases with 20 options) and gold
labels are identical to the numbers laya publishes in BENCHMARKS.md. Suites whose dataset cannot be
downloaded are skipped, as upstream does. banking77 (77 options) is excluded because the Core ML
buckets have 32 option slots; upstream reports 0.425 for both checkpoints on it.

Outputs:
  benchmark/suites.jsonl            one question per line with the exact serialized state
  reports/benchmark-reference.json  PyTorch FP32 CPU accuracy per suite + per-row argmax/probabilities
"""

from __future__ import annotations

import argparse
import json
import random
import time

import laya
import numpy as np
import torch
from laya.common import QTYPES, build_sequence, collate_items, render_options, serialize_state, temp_bucket

from assets import ROOT, checkpoint_dir, load_lock, verify_assets
from preprocessing import to_internal

SEED = 13
MASSIVE_OPTIONS = 20
SUITE_DIR = ROOT / "benchmark"

UPSTREAM_PUBLISHED = {
    # laya BENCHMARKS.md, laya-multilingual column (Tesla T4, PyTorch)
    "jev.ag_news": 0.930,
    "jev.emotion": 0.530,
    "massive_intent.en": 0.657,
    "app.support_triage": 0.522,
    "app.email_spam": 0.993,
    "app.phishing": 0.993,
    "app.guardrails_jailbreak": 0.755,
    "app.moderation_toxicity": 0.525,
    "app.rag_relevance": 0.657,
    "app.model_routing_domain": 0.123,
}


def register(suites: dict, name: str, cases, gold, **meta):
    suites[name] = {"cases": cases, "gold": gold, "meta": meta}
    print(f"   {name:26s} {len(cases):4d} cases  {meta.get('note', '')}", flush=True)


def build(n: int) -> dict:
    from datasets import load_dataset

    rng = random.Random(SEED)
    suites: dict = {}

    try:
        d = load_dataset("mteb/amazon_massive_intent", "en", split="test")
        labels = sorted(set(d["label_text"]))
        mrng = random.Random(SEED)
        cases, gold = [], []
        for r in list(d)[:300]:
            pool = [x for x in labels if x != r["label_text"]]
            keys = [r["label_text"]] + mrng.sample(pool, min(MASSIVE_OPTIONS - 1, len(pool)))
            mrng.shuffle(keys)
            cases.append(
                (
                    {"utterance": r["text"]},
                    {
                        "intent": {
                            "type": "choice",
                            "instructions": "What is the user asking for in `utterance`?",
                            "criteria": {k: k.replace("_", " ").replace(".", ": ") for k in keys},
                        }
                    },
                )
            )
            gold.append(keys.index(r["label_text"]))
        register(suites, "massive_intent.en", cases, gold, note="20 options, upstream 0.657")
    except Exception as e:  # noqa: BLE001
        print("   FAIL massive", str(e)[:80])

    try:
        d = load_dataset("fancyzhx/ag_news", split="test")
        crit = {
            "world": "world news and international politics",
            "sports": "sports",
            "business": "business and economy",
            "sci_tech": "science and technology",
        }
        keys = list(crit)
        cases, gold = [], []
        for r in list(d)[:n]:
            cases.append(
                (
                    {"article": r["text"]},
                    {
                        "topic": {
                            "type": "choice",
                            "instructions": "What is the topic of `article`?",
                            "criteria": dict(crit),
                        }
                    },
                )
            )
            gold.append(keys.index(keys[int(r["label"])]))
        register(suites, "jev.ag_news", cases, gold, note="Jev 0.910, upstream 0.930")
    except Exception as e:  # noqa: BLE001
        print("   FAIL ag_news", str(e)[:80])

    try:
        d = load_dataset("dair-ai/emotion", "split", split="test")
        names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        cases, gold = [], []
        for r in list(d)[:n]:
            cases.append(
                (
                    {"text": r["text"]},
                    {
                        "emotion": {
                            "type": "choice",
                            "instructions": "Which emotion is most strongly expressed in `text`?",
                            "criteria": {x: None for x in names},
                        }
                    },
                )
            )
            gold.append(int(r["label"]))
        register(suites, "jev.emotion", cases, gold, note="Jev 0.480, upstream 0.530")
    except Exception as e:  # noqa: BLE001
        print("   FAIL emotion", str(e)[:80])

    try:
        d = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
        queues = {
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
        keys = list(queues)
        cases, gold = [], []
        for r in d:
            if r.get("language") != "en" or r.get("queue") not in queues or not r.get("body"):
                continue
            cases.append(
                (
                    {"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]},
                    {
                        "queue": {
                            "type": "choice",
                            "instructions": "Which support queue should handle this ticket?",
                            "criteria": dict(queues),
                        }
                    },
                )
            )
            gold.append(keys.index(r["queue"]))
            if len(cases) >= n:
                break
        register(suites, "app.support_triage", cases, gold, note="10-way, upstream 0.522")
    except Exception as e:  # noqa: BLE001
        print("   FAIL support_triage", str(e)[:80])

    try:
        d = load_dataset("SetFit/enron_spam", split="test")
        cases, gold = [], []
        for r in list(d)[:n]:
            st = laya.email_state(r.get("subject") or "", (r.get("message") or "")[:3000])
            cases.append(
                (st, {"is_spam": {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"}})
            )
            gold.append(int(r["label"]))
        register(suites, "app.email_spam", cases, gold, note="enron, upstream 0.993")
    except Exception as e:  # noqa: BLE001
        print("   FAIL email_spam", str(e)[:80])

    try:
        d = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        rows = [
            r
            for r in list(d)[:6000]
            if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")
        ]
        rng.shuffle(rows)
        cases, gold = [], []
        for r in rows[:n]:
            cases.append(
                (
                    {"email": r["Email Text"][:3000]},
                    {
                        "is_phishing": {
                            "type": "noul",
                            "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
                            "criteria": {
                                "true": "phishing, scam, or fraud",
                                "false": "a legitimate email (even if promotional)",
                            },
                        }
                    },
                )
            )
            gold.append(int(r["Email Type"] == "Phishing Email"))
        register(suites, "app.phishing", cases, gold, note="upstream 0.993")
    except Exception as e:  # noqa: BLE001
        print("   FAIL phishing", str(e)[:80])

    try:
        d = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        rows = [r for r in d if (r.get("user_input") or "").strip()]
        jb = [r for r in rows if int(r.get("jailbreaking", 0)) == 1][: n // 2]
        nj = [r for r in rows if int(r.get("jailbreaking", 0)) == 0][: n - len(jb)]
        mix = jb + nj
        rng.shuffle(mix)
        cases, gold = [], []
        for r in mix:
            cases.append(
                (
                    {"prompt": r["user_input"][:3000]},
                    {
                        "jailbreak": {
                            "type": "noul",
                            "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
                        }
                    },
                )
            )
            gold.append(int(r["jailbreaking"]))
        register(suites, "app.guardrails_jailbreak", cases, gold, note="held out, upstream 0.755")

        tox = [r for r in rows if int(r.get("toxicity", 0)) == 1][: n // 2]
        ntox = [r for r in rows if int(r.get("toxicity", 0)) == 0][: n - len(tox)]
        mix2 = tox + ntox
        rng.shuffle(mix2)
        cases, gold = [], []
        for r in mix2:
            cases.append(
                (
                    {"post": r["user_input"][:3000]},
                    {
                        "toxic": {
                            "type": "noul",
                            "instructions": "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?",
                        }
                    },
                )
            )
            gold.append(int(r["toxicity"]))
        register(suites, "app.moderation_toxicity", cases, gold, note="held out, upstream 0.525")
    except Exception as e:  # noqa: BLE001
        print("   FAIL toxic-chat", str(e)[:80])

    try:
        d = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
        cases, gold = [], []
        for r in d:
            texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
            pos = [t for t, s in zip(texts, sel) if s == 1]
            neg = [t for t, s in zip(texts, sel) if s == 0]
            if not pos or not neg:
                continue
            take_pos = len(cases) % 2 == 0
            p = rng.choice(pos if take_pos else neg)
            cases.append(
                (
                    {"query": r["query"], "passage": p},
                    {"relevant": {"type": "noul", "instructions": "Does `passage` help answer `query`?"}},
                )
            )
            gold.append(1 if take_pos else 0)
            if len(cases) >= n:
                break
        register(suites, "app.rag_relevance", cases, gold, note="MS MARCO, upstream 0.657")
    except Exception as e:  # noqa: BLE001
        print("   FAIL rag", str(e)[:80])

    try:
        dom = {
            "code": "software engineering, programming, refactoring, architecture, debugging",
            "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
            "writing": "creative writing, essays, emails, blog posts, copywriting",
            "factual_lookup": "facts, definitions, trivia, history",
            "data_analysis": "statistics, SQL, data manipulation, metrics",
            "chitchat": "casual conversation, greetings, small talk",
        }
        keys = list(dom)
        pool = []
        g = load_dataset("openai/gsm8k", "main", split="test")
        pool += [(r["question"], "math_or_logic") for r in list(g)[: n // 3]]
        m = load_dataset("google-research-datasets/mbpp", "full", split="test")
        pool += [(r["text"], "code") for r in list(m)[: n // 3]]
        t = load_dataset("fancyzhx/ag_news", split="test")
        pool += [(r["text"][:400], "factual_lookup") for r in list(t)[: n // 3]]
        rng.shuffle(pool)
        cases, gold = [], []
        for text, label in pool[:n]:
            cases.append(
                (
                    {"request": text},
                    {
                        "domain": {
                            "type": "choice",
                            "instructions": "What domain does `request` belong to?",
                            "criteria": dict(dom),
                        }
                    },
                )
            )
            gold.append(keys.index(label))
        register(suites, "app.model_routing_domain", cases, gold, note="held out, upstream 0.123")
    except Exception as e:  # noqa: BLE001
        print("   FAIL routing", str(e)[:80])
    return suites


def export_rows(suites: dict) -> list[dict]:
    rows = []
    for name, suite in suites.items():
        for index, ((state, questions), gold) in enumerate(zip(suite["cases"], suite["gold"])):
            ((qid, qdef),) = questions.items()
            criteria = qdef.get("criteria")
            if qdef["type"] == "choice":
                options = [[k, v] for k, v in criteria.items()]
            elif qdef["type"] == "score":
                options = [[c, None] for c in criteria]
            else:
                options = [[k, criteria[k]] for k in ("false", "true")] if criteria else []
            rows.append(
                {
                    "suite": name,
                    "index": index,
                    "state": serialize_state(state),
                    "type": qdef["type"],
                    "instructions": qdef["instructions"],
                    "options": options,
                    "gold": gold,
                }
            )
    return rows


def reference(agent, rows: list[dict], max_len: int) -> tuple[dict, list[dict]]:
    """Score every row with the reference model at `max_len`; returns per-suite metrics and per-row outputs."""
    per_row = []
    started = time.perf_counter()
    for row in rows:
        qdef = {"type": row["type"], "instructions": row["instructions"]}
        if row["type"] == "choice":
            qdef["criteria"] = {k: v for k, v in row["options"]}
        elif row["type"] == "noul" and row["options"]:
            qdef["criteria"] = {k: v for k, v in row["options"]}
        q = to_internal(qdef)
        ids, markers = build_sequence(agent.tok, row["state"], q, max_len, agent.cfg["head_max_len"])
        if len(markers) != len(render_options(q)):
            per_row.append({"suite": row["suite"], "index": row["index"], "dropped": True})
            continue
        b = collate_items([[{"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]}]], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, _ = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        k = len(markers)
        scale = agent.temperature_by_options.get(temp_bucket(QTYPES[q["t"]], k), agent.temperature[QTYPES[q["t"]]])
        z = logits[0, :k].float().numpy() / max(1e-3, float(scale))
        p = np.exp(z - z.max())
        p /= p.sum()
        per_row.append(
            {
                "suite": row["suite"],
                "index": row["index"],
                "tokens": len(ids),
                "argmax": int(p.argmax()),
                "probabilities": [round(float(x), 6) for x in p],
                "correct": int(p.argmax()) == row["gold"],
            }
        )
    seconds = time.perf_counter() - started
    metrics = {}
    for name in sorted({r["suite"] for r in rows}):
        scored = [r for r in per_row if r["suite"] == name and not r.get("dropped")]
        metrics[name] = {
            "n": len(scored),
            "dropped": sum(1 for r in per_row if r["suite"] == name and r.get("dropped")),
            "accuracy": round(sum(r["correct"] for r in scored) / max(1, len(scored)), 4),
            "max_tokens": max((r["tokens"] for r in scored), default=0),
            "mean_tokens": round(float(np.mean([r["tokens"] for r in scored])), 1) if scored else 0,
            "upstream_published": UPSTREAM_PUBLISHED.get(name),
        }
    return {
        "seconds": round(seconds, 1),
        "ms_per_question": round(1000 * seconds / max(1, len(rows)), 2),
        "suites": metrics,
    }, per_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--max-len", type=int, default=1024, help="reference max_len (upstream default 1024)")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    verify_assets("multilingual")
    SUITE_DIR.mkdir(exist_ok=True)
    print(f"=== building suites (n={args.n}, seed={SEED}) ===", flush=True)
    rows = export_rows(build(args.n))
    with (SUITE_DIR / "suites.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions to {SUITE_DIR / 'suites.jsonl'}", flush=True)

    agent = laya.load(str(checkpoint_dir("multilingual")), device="cpu")
    agent.model.eval()
    summary, per_row = reference(agent, rows, args.max_len)
    for name, m in summary["suites"].items():
        published = m["upstream_published"]
        delta = f"  upstream {published:.3f} -> {m['accuracy'] - published:+.3f}" if published else ""
        print(
            f"   {name:26s} n={m['n']:4d} acc {m['accuracy']:.3f}  max tokens {m['max_tokens']:4d}{delta}", flush=True
        )
    report = {
        "source_repo": load_lock()["repo"],
        "source_revision": load_lock()["revision"],
        "variant": "multilingual",
        "device": "cpu",
        "torch": torch.__version__,
        "threads": 4,
        "seed": SEED,
        "n_per_task": args.n,
        "max_len": args.max_len,
        **summary,
    }
    (ROOT / "reports" / "benchmark-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    with (SUITE_DIR / "reference-rows.jsonl").open("w") as stream:
        for row in per_row:
            stream.write(json.dumps(row) + "\n")
    print(f"reference: {summary['ms_per_question']} ms/question on CPU; wrote reports/benchmark-reference.json")


if __name__ == "__main__":
    main()
