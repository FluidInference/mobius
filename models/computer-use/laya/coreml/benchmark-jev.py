"""Reproduce the five-task "Laya vs Jev, measured" table (brainfunctioncollapse.com/laya) on device.

The post gives task names, 100 labelled examples per task, laya 0.3.4 English checkpoint on an M1 Max
GPU, and Jev's accuracy, but not its datasets, sampling or question wording. This script uses the
obvious public dataset for each task, the first 100 rows of its test split (train for SMS spam, which
has no test split), and laya's own question wording where one exists. Jev is a closed API and is not
measured here; its column is copied from the post. Both laya checkpoints are scored with the
unmodified PyTorch runtime so the English numbers can be compared with the post and the
multilingual rows give the reference for the Core ML buckets (`FluidUseLaya benchmark`).

Outputs:
  benchmark/jev-suites.jsonl           the 500 questions with serialized states and gold labels
  benchmark/jev-reference-rows.jsonl   multilingual PyTorch answers per row
  reports/benchmark-jev-reference.json accuracy per task for both checkpoints
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import laya
import numpy as np
import torch
from laya.common import QTYPES, build_sequence, collate_items, render_options, serialize_state, temp_bucket

from assets import ROOT, checkpoint_dir, verify_assets
from preprocessing import to_internal

N = 100
SUITE_DIR = ROOT / "benchmark"
POST = {  # brainfunctioncollapse.com/laya, "Laya vs Jev, measured", 2026-09-20
    "news_topic": {"laya_english_m1max": 0.93, "jev": 0.92},
    "sms_spam": {"laya_english_m1max": 0.96, "jev": 0.96},
    "emotion": {"laya_english_m1max": 0.45, "jev": 0.53},
    "review_stars": {"laya_english_m1max": 0.35, "jev": 0.70},
    "prompt_injection": {"laya_english_m1max": 0.65, "jev": 0.71},
    "all": {"laya_english_m1max": 0.668, "jev": 0.764},
}


def build() -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []

    def add(suite, index, state, qdef, gold):
        criteria = qdef.get("criteria")
        if qdef["type"] == "choice":
            options = [[k, v] for k, v in criteria.items()]
        elif qdef["type"] == "score":
            options = [[c, None] for c in criteria]
        else:
            options = [[k, criteria[k]] for k in ("false", "true")] if criteria else []
        rows.append(
            {
                "suite": suite,
                "index": index,
                "state": serialize_state(state),
                "type": qdef["type"],
                "instructions": qdef["instructions"],
                "options": options,
                "gold": gold,
            }
        )

    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {
        "world": "world news and international politics",
        "sports": "sports",
        "business": "business and economy",
        "sci_tech": "science and technology",
    }
    for i, r in enumerate(list(d)[:N]):
        add(
            "news_topic",
            i,
            {"article": r["text"]},
            {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": dict(crit)},
            int(r["label"]),
        )

    d = load_dataset("ucirvine/sms_spam", split="train")
    for i, r in enumerate(list(d)[:N]):
        add(
            "sms_spam",
            i,
            {"message": r["sms"]},
            {"type": "noul", "instructions": "Is this message unsolicited spam or bulk marketing?"},
            int(r["label"]),
        )

    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for i, r in enumerate(list(d)[:N]):
        add(
            "emotion",
            i,
            {"text": r["text"]},
            {
                "type": "choice",
                "instructions": "Which emotion is most strongly expressed in `text`?",
                "criteria": {n: None for n in names},
            },
            int(r["label"]),
        )

    d = load_dataset("Yelp/yelp_review_full", split="test")
    stars = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for i, r in enumerate(list(d)[:N]):
        add(
            "review_stars",
            i,
            {"review": r["text"][:3000]},
            {"type": "score", "instructions": "How many stars did the reviewer give in `review`?", "criteria": stars},
            int(r["label"]),
        )

    d = load_dataset("deepset/prompt-injections", split="test")
    for i, r in enumerate(list(d)[:N]):
        add(
            "prompt_injection",
            i,
            {"prompt": r["text"]},
            {
                "type": "noul",
                "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
            },
            int(r["label"]),
        )
    return rows


def score(agent, rows: list[dict], max_len: int) -> tuple[dict, list[dict], float]:
    per_row = []
    started = time.perf_counter()
    for row in rows:
        qdef = {"type": row["type"], "instructions": row["instructions"]}
        if row["type"] == "choice" or (row["type"] == "noul" and row["options"]):
            qdef["criteria"] = {k: v for k, v in row["options"]}
        elif row["type"] == "score":
            qdef["criteria"] = [k for k, _ in row["options"]]
        q = to_internal(qdef)
        ids, markers = build_sequence(agent.tok, row["state"], q, max_len, agent.cfg["head_max_len"])
        if len(markers) != len(render_options(q)):
            per_row.append({"suite": row["suite"], "index": row["index"], "dropped": True})
            continue
        b = collate_items([[{"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]}]], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, _ = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
        k = len(markers)
        qt = QTYPES[q["t"]]
        scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
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
            "accuracy": round(sum(r["correct"] for r in scored) / max(1, len(scored)), 4),
            "max_tokens": max((r["tokens"] for r in scored), default=0),
        }
    scored = [r for r in per_row if not r.get("dropped")]
    metrics["all"] = {"n": len(scored), "accuracy": round(sum(r["correct"] for r in scored) / max(1, len(scored)), 4)}
    return metrics, per_row, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english-dir", type=Path, default=ROOT / "artifacts" / "english")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    verify_assets("multilingual")
    SUITE_DIR.mkdir(exist_ok=True)
    rows = build()
    with (SUITE_DIR / "jev-suites.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"built {len(rows)} questions", flush=True)

    report = {"protocol": __doc__.strip(), "n_per_task": N, "post": POST, "checkpoints": {}}
    for name, path, max_len in (
        ("multilingual", checkpoint_dir("multilingual"), 1024),
        ("english", args.english_dir, 512),
    ):
        if not (path / "model.safetensors").exists():
            print(f"skip {name}: no checkpoint at {path}")
            continue
        agent = laya.load(str(path), device="cpu")
        agent.model.eval()
        metrics, per_row, seconds = score(agent, rows, max_len)
        report["checkpoints"][name] = {
            "max_len": max_len,
            "seconds": round(seconds, 1),
            "ms_per_question": round(1000 * seconds / len(rows), 2),
            **metrics,
        }
        for task, m in metrics.items():
            post = POST[task]
            print(
                f"  {name:12s} {task:18s} acc {m['accuracy']:.3f}   post laya {post['laya_english_m1max']:.3f}  jev {post['jev']:.3f}",
                flush=True,
            )
        if name == "multilingual":
            with (SUITE_DIR / "jev-reference-rows.jsonl").open("w") as stream:
                for row in per_row:
                    stream.write(json.dumps(row) + "\n")
        del agent
    (ROOT / "reports" / "benchmark-jev-reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote reports/benchmark-jev-reference.json")


if __name__ == "__main__":
    main()
