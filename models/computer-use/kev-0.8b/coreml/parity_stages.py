"""Two-stage (state pass + batched question pass) vs the single-row KevRow, both PyTorch fp32, on real records."""
import argparse
import json
import random
from pathlib import Path

import torch

from kev.data import materialize
from kev.model import encode, load_tokenizer, rows_of
from kev.suite import load_split
from kev_export import load_kev_row, row_inputs
from kev_stages import load_stages, stage_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--state-len", type=int, default=512)
    ap.add_argument("--question-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-options", type=int, default=16)
    ap.add_argument("--per-suite", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(8)
    tok = load_tokenizer("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    row, cfg, meta, embed = load_kev_row(args.merged, 1024, 80)
    first, second, _, _, _ = load_stages(args.merged, args.state_len, args.question_len, args.batch, args.max_options)
    worst, flips, n = 0.0, 0, 0
    for suite in ("evals/documents-v1", "evals/v7/decision-v7", "evals/v4/transfer-v4"):
        records = load_split(suite, "development"); random.Random(3).shuffle(records); used = 0
        for raw in records:
            if used >= args.per_suite: break
            enc = encode(tok, materialize(raw))
            state_ids, state_pos, rows = rows_of(enc)
            if len(state_ids) > args.state_len or any(len(r["ids"]) > args.question_len or len(r["opts"]) > args.max_options for r in rows):
                continue
            with torch.no_grad():
                reference = []
                for r in rows:
                    inputs = row_inputs(cfg, embed, state_ids + r["ids"], state_pos + r["pos"], len(state_ids) + r["decide"],
                                        [len(state_ids) + o for o in r["opts"]], 1024, 80, meta["pad_id"])
                    reference.append(row(*inputs)[1][: len(r["opts"])])
                state_in, question_in = stage_inputs(cfg, embed, state_ids, [(r["ids"], r["decide"], r["opts"]) for r in rows],
                                                     args.state_len, args.question_len, args.batch, args.max_options, meta["pad_id"])
                keys, values, states, tails = first(*state_in)
                hidden, q_cos, q_sin, valid, decide, options, mask = question_in
                probs = second(hidden, q_cos, q_sin, keys, values, valid, states, tails, decide, options, mask)[1]
            for b, ref in enumerate(reference):
                got = probs[b, : len(ref)]
                worst = max(worst, float((got - ref).abs().max())); flips += int(got.argmax() != ref.argmax()); n += 1
            used += 1
            print(f"{suite} record {used}: state {len(state_ids)} tokens, {len(rows)} questions, worst |dp| {worst:.2e}", flush=True)
    print(json.dumps({"questions": n, "max_dp": worst, "flips": flips}))


if __name__ == "__main__":
    main()
