"""Our export wrapper (PyTorch) vs Kev's own DecisionModel (fp32, the published path) on real development records.

    cd <kev repo> && PYTHONPATH=<this dir> uv run python <this dir>/parity_torch.py --merged <this dir>/build/merged
"""
import argparse
import json
import random
from pathlib import Path

import torch

from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.model import rows_of
from kev.suite import load_split
from kev_export import load_kev_row, row_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-options", type=int, default=16)
    ap.add_argument("--per-suite", type=int, default=40)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(8)
    tok, model = Checkpoint(args.run).load("cpu", LoadOptions(backend="torch", dtype=torch.float32, merge=True))
    model.eval()
    row, cfg, meta, embed = load_kev_row(args.merged, args.length, args.max_options)
    results, skipped = [], 0
    for suite in ("evals/v7/decision-v7", "evals/documents-v1"):
        records = load_split(suite, "development")
        random.Random(0).shuffle(records)
        used = 0
        for rec in records:
            if used >= args.per_suite:
                break
            enc = model.encode(tok, materialize(rec))
            state_ids, state_pos, rows = rows_of(enc)
            if any(len(state_ids) + len(r["ids"]) > args.length or len(r["opts"]) > args.max_options for r in rows):
                skipped += 1
                continue
            with torch.no_grad():
                reference = model.probs(enc)
                ours = []
                for r in rows:
                    inputs = row_inputs(cfg, embed, state_ids + r["ids"], state_pos + r["pos"], len(state_ids) + r["decide"],
                                        [len(state_ids) + o for o in r["opts"]], args.length, args.max_options, meta["pad_id"])
                    ours.append(row(*inputs)[1][: len(r["opts"])])
            for ref, got in zip(reference, ours):
                ref = torch.as_tensor(ref, dtype=torch.float32)
                results.append({"suite": suite, "max_dp": float((ref - got).abs().max()), "flip": int(ref.argmax() != got.argmax())})
            used += 1
            print(f"{suite} record {used}: questions {len(rows)}, worst |dp| so far {max(r['max_dp'] for r in results):.2e}, "
                  f"flips {sum(r['flip'] for r in results)}", flush=True)
    summary = {"questions": len(results), "records_skipped_over_bucket": skipped,
               "max_dp": max(r["max_dp"] for r in results), "mean_dp": sum(r["max_dp"] for r in results) / len(results),
               "argmax_flips": sum(r["flip"] for r in results)}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
