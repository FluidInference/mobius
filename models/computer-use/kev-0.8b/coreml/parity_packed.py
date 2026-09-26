"""Fused pass vs the single-row KevRow (and vs the two-stage packed pass), all PyTorch fp32, on real records."""
import argparse
import json
import random
from pathlib import Path

import torch

from kev.data import materialize
from kev.model import encode, load_tokenizer, rows_of
from kev.suite import load_split
from kev_export import load_kev_row, row_inputs
from kev_stages import fused_inputs, pack_groups, load_fused, load_packed, load_stages, packed_inputs, stage_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--state-len", type=int, default=512)
    ap.add_argument("--packed-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-options", type=int, default=16)
    ap.add_argument("--per-suite", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=1, help="pack each question this many times (segment isolation)")
    ap.add_argument("--block-recursion", action="store_true", help="GatedDeltaNet.unit_lower_inverse instead")
    args = ap.parse_args()
    import kev_stages
    kev_stages.MASKED_INVERSE = not args.block_recursion
    torch.set_num_threads(8)
    tok = load_tokenizer("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    row, cfg, meta, embed = load_kev_row(args.merged, 1024, 80)
    first, _, _, _, _ = load_stages(args.merged, args.state_len, 32, 2, args.max_options)
    # the two-stage packed pass keeps a power-of-two chunk; the fused pass takes any packed length
    power_of_two = args.packed_len & (args.packed_len - 1) == 0
    packed = load_packed(args.merged, args.state_len, args.packed_len, args.batch, args.max_options) if power_of_two else None
    fused = load_fused(args.merged, args.state_len, args.packed_len, args.batch, args.max_options)
    worst, flips, n, worst_fused = 0.0, 0, 0, 0.0
    for suite in ("evals/documents-v1", "evals/v7/decision-v7", "evals/v4/transfer-v4"):
        records = load_split(suite, "development"); random.Random(3).shuffle(records); used = 0
        for raw in records:
            if used >= args.per_suite: break
            enc = encode(tok, materialize(raw))
            state_ids, state_pos, rows = rows_of(enc)
            rows = rows * args.repeat
            if len(state_ids) > args.state_len or any(len(r["opts"]) > args.max_options for r in rows):
                continue
            lane = min(128, args.packed_len)
            if any(len(r["ids"]) > lane for r in rows):
                continue
            groups = [[rows[i] for i in g] for g in pack_groups([len(r["ids"]) for r in rows], args.packed_len, args.batch, lane)]
            with torch.no_grad():
                reference = []
                for r in rows:
                    inputs = row_inputs(cfg, embed, state_ids + r["ids"], state_pos + r["pos"], len(state_ids) + r["decide"],
                                        [len(state_ids) + o for o in r["opts"]], 1024, 80, meta["pad_id"])
                    reference.append(row(*inputs)[1][: len(r["opts"])])
                state_in, _ = stage_inputs(cfg, embed, state_ids, [], args.state_len, 32, 2, args.max_options, meta["pad_id"])
                keys, values, states, tails = first(*state_in)
                got = []
                for group in groups:
                    fused_probs = fused(*fused_inputs(cfg, embed, state_ids, [(r["ids"], r["decide"], r["opts"]) for r in group],
                                                      args.state_len, args.packed_len, args.batch, args.max_options,
                                                      meta["pad_id"]))[1]
                    got += [fused_probs[b, : len(r["opts"])] for b, r in enumerate(group)]
                    if packed is None:
                        continue
                    hidden, cos, sin, segment, keep, lag_tail, decide, options, mask = packed_inputs(
                        cfg, embed, len(state_ids), [(r["ids"], r["decide"], r["opts"]) for r in group],
                        args.packed_len, args.batch, args.max_options, meta["pad_id"], lane=lane)
                    probs = packed(hidden, cos, sin, keys, values, state_in[3], states, tails, segment, keep, lag_tail,
                                   decide, options, mask)[1]
                    worst_fused = max(worst_fused, float((fused_probs - probs).abs().max()))
            for g, ref in zip(got, reference):
                worst = max(worst, float((g - ref).abs().max())); flips += int(g.argmax() != ref.argmax()); n += 1
            used += 1
            print(f"{suite} record {used}: state {len(state_ids)}, {len(rows)} questions in {len(groups)} packs, "
                  f"worst |dp| {worst:.2e}, fused vs packed {worst_fused:.2e}", flush=True)
    print(json.dumps({"questions": n, "max_dp": worst, "flips": flips, "fused_vs_packed": worst_fused}))


if __name__ == "__main__":
    main()
