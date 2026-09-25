"""Core ML Kev row vs Kev's own DecisionModel (fp32, the published path) on real development records.

    cd <kev repo> && PYTHONPATH=<this dir> uv run --with coremltools==9.0 python <this dir>/parity_coreml.py \
        --merged <this dir>/build/merged --package <this dir>/build/L512_K16/KevRow_fp16.mlpackage
"""
import argparse
import json
import random
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.model import rows_of
from kev.suite import load_split
from kev_export import load_kev_row, row_inputs

UNITS = {"all": ct.ComputeUnit.ALL, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu": ct.ComputeUnit.CPU_ONLY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-options", type=int, default=16)
    ap.add_argument("--per-suite", type=int, default=30)
    ap.add_argument("--units", choices=list(UNITS), default="all")
    ap.add_argument("--suites", nargs="+", default=["evals/v7/decision-v7", "evals/documents-v1"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(8)
    tok, model = Checkpoint(args.run).load("cpu", LoadOptions(backend="torch", dtype=torch.float32, merge=True))
    model.eval()
    _, cfg, meta, embed = load_kev_row(args.merged, args.length, args.max_options)
    coreml = ct.models.MLModel(str(args.package), compute_units=UNITS[args.units])
    names = ["hidden", "cos", "sin", "decide_onehot", "option_onehot", "option_mask"]
    rows_out, skipped, latencies = [], 0, []
    for suite in args.suites:
        records = load_split(suite, "development")
        random.Random(0).shuffle(records)
        used = 0
        for raw in records:
            if used >= args.per_suite:
                break
            rec = materialize(raw)
            enc = model.encode(tok, rec)
            state_ids, state_pos, rows = rows_of(enc)
            if any(len(state_ids) + len(r["ids"]) > args.length or len(r["opts"]) > args.max_options for r in rows):
                skipped += 1
                continue
            with torch.no_grad():
                reference = model.probs(enc)
            for q, r, ref in zip(rec["questions"], rows, reference):
                inputs = row_inputs(cfg, embed, state_ids + r["ids"], state_pos + r["pos"], len(state_ids) + r["decide"],
                                    [len(state_ids) + o for o in r["opts"]], args.length, args.max_options, meta["pad_id"])
                feed = {n: t.numpy().astype(np.float32) for n, t in zip(names, inputs)}
                start = time.perf_counter()
                got = np.asarray(coreml.predict(feed)["probabilities"])[: len(r["opts"])]
                latencies.append((time.perf_counter() - start) * 1000)
                ref = np.asarray(ref, dtype=np.float32)
                rows_out.append({"suite": suite, "max_dp": float(np.abs(ref - got).max()), "flip": int(ref.argmax() != got.argmax()),
                                 "ref_correct": int(ref.argmax() == q["label"]), "coreml_correct": int(got.argmax() == q["label"])})
            used += 1
            if used % 10 == 0:
                print(f"{suite}: {used} records, worst |dp| {max(x['max_dp'] for x in rows_out):.4f}, "
                      f"flips {sum(x['flip'] for x in rows_out)}", flush=True)
    by_suite = {}
    for suite in args.suites:
        rs = [x for x in rows_out if x["suite"] == suite]
        by_suite[suite] = {"questions": len(rs), "max_dp": max(x["max_dp"] for x in rs), "flips": sum(x["flip"] for x in rs),
                           "reference_accuracy": sum(x["ref_correct"] for x in rs) / len(rs),
                           "coreml_accuracy": sum(x["coreml_correct"] for x in rs) / len(rs)}
    latencies.sort()
    summary = {"package": str(args.package), "units": args.units, "records_skipped_over_bucket": skipped,
               "questions": len(rows_out), "max_dp": max(x["max_dp"] for x in rows_out),
               "mean_dp": float(np.mean([x["max_dp"] for x in rows_out])), "argmax_flips": sum(x["flip"] for x in rows_out),
               "p50_ms_per_row": latencies[len(latencies) // 2], "by_suite": by_suite}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows_out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
