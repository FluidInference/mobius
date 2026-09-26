"""Convert Kev's two-stage export (state pass + batched question pass) to Core ML.

Outputs build/stages/S<state>/:
  StatePass.mlpackage                      hidden [1,S,D], cos/sin [S,R], valid [S], tail_onehot [k-1,S]
                                           -> keys, values [A,kv,S,d], states [Ld,H,Dk,Dv], tails [Ld,conv,k-1]
  QuestionPass_Q<q>_B<b>_K<k>.mlpackage    hidden [B,Q,D], cos/sin [Q,R], keys, values, cache_valid [S], states, tails,
                                           decide_onehot [B,Q], option_onehot [B,K,Q], option_mask [B,K]
                                           -> logits [B,K] (temperature applied), probabilities [B,K]
  packed_C<c>_P<p>_B<b>_K<k>.mlpackage     the questions packed end to end in P tokens: hidden [1,P,D], cos/sin [P,R],
                                           keys, values, cache_valid, states, tails, segment [P,P], lag_keep [k-1,P],
                                           lag_tail [k-1,P,k-1], decide_onehot [B,P], option_onehot [B,K,P], option_mask
  fused_S<s>_P<p>_B<b>_K<k>.mlpackage      state + packed questions in one call: hidden [1,S+P,D], cos/sin [S+P,R],
                                           valid [S], tail_onehot [k-1,S], segment, lag_keep, lag_tail, readouts
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from kev_stages import fused_inputs, load_fused, load_packed, load_stages, packed_inputs, stage_inputs


def _types(names, shapes, dtype):
    return [ct.TensorType(name=n, shape=sh, dtype=dtype) for n, sh in zip(names, shapes)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, default=Path("build/merged"))
    ap.add_argument("--stage", choices=["state", "question", "packed", "fused"], required=True)
    ap.add_argument("--length", type=int, required=True, help="state bucket (state) or cache bucket (question)")
    ap.add_argument("--question-len", type=int, default=64, help="per-question length (question) or packed length (packed)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-options", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--lane", type=int, default=128, help="fused: packed questions never cross a lane of this size")
    ap.add_argument("--build", type=Path, default=Path("build/stages/functions"))
    args = ap.parse_args()
    S, Q, B, K = args.length, args.question_len, args.batch, args.max_options
    chunk = min(args.chunk_size, S)
    # only the question stage uses `second`; its per-row chunk must be a power of two, packed lengths need not
    first, second, cfg, meta, embed = load_stages(args.merged, S, Q if args.stage == "question" else 32, B, K, chunk)
    special = meta["special_tokens"]
    state_ids = [special["state"]] + list(range(100, 100 + min(40, S - 1)))
    branch = [special["q"], 300, special["opt"], 400, special["close_opt"], special["opt"], 500, special["close_opt"], special["decide"]]
    state_in, question_in = stage_inputs(cfg, embed, state_ids, [(branch, len(branch) - 1, [4, 7])], S, Q, B, K, meta["pad_id"])
    args.build.mkdir(parents=True, exist_ok=True)
    A = sum(1 for t in cfg.layer_types if t != "linear_attention")
    Ld = cfg.num_layers - A
    conv_dim = 2 * cfg.lin_k_heads * cfg.lin_k_dim + cfg.lin_v_heads * cfg.lin_v_dim
    kernel = cfg.conv_kernel - 1
    half = np.float16  # fp16 I/O: the program computes in fp16 anyway, this only halves the copies
    start = time.time()
    if args.stage == "state":
        with torch.no_grad():
            traced = torch.jit.trace(first, state_in, check_trace=False)
        model = ct.convert(
            traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY,
            inputs=_types(["hidden", "cos", "sin", "valid", "tail_onehot"],
                          [(1, S, cfg.hidden_size), (S, cfg.rotary_dim), (S, cfg.rotary_dim), (S,), (kernel, S)], half),
            outputs=[ct.TensorType(name=n, dtype=half) for n in ("keys", "values", "states", "tails")])
        name = f"state_S{S}"
    elif args.stage == "fused":
        fused = load_fused(args.merged, S, Q, B, K, chunk, lane=args.lane)
        example = fused_inputs(cfg, embed, state_ids, [(branch, len(branch) - 1, [4, 7])] * 2, S, Q, B, K, meta["pad_id"],
                               lane=args.lane)
        with torch.no_grad():
            traced = torch.jit.trace(fused, example, check_trace=False)
        model = ct.convert(
            traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY,
            inputs=_types(
                ["hidden", "cos", "sin", "valid", "tail_onehot", "segment", "lag_keep", "lag_tail", "decide_onehot",
                 "option_onehot", "option_mask"],
                [(1, S + Q, cfg.hidden_size), (S + Q, cfg.rotary_dim), (S + Q, cfg.rotary_dim), (S,), (kernel, S), (Q, Q),
                 (kernel, Q), (kernel, Q, kernel), (B, Q), (B, K, Q), (B, K)], half),
            outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)])
        name = f"fused_S{S}_P{Q}_B{B}_K{K}"
    elif args.stage == "packed":
        packed = load_packed(args.merged, S, Q, B, K)
        with torch.no_grad():
            keys, values, states, tails = first(*state_in)
            packed_in = packed_inputs(cfg, embed, len(state_ids), [(branch, len(branch) - 1, [4, 7])] * 2, Q, B, K,
                                      meta["pad_id"])
            hidden, p_cos, p_sin, segment, keep, lag_tail, decide, options, mask = packed_in
            traced = torch.jit.trace(packed, (hidden, p_cos, p_sin, keys, values, state_in[3], states, tails, segment, keep,
                                              lag_tail, decide, options, mask), check_trace=False)
        model = ct.convert(
            traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY,
            inputs=_types(
                ["hidden", "cos", "sin", "keys", "values", "cache_valid", "states", "tails", "segment", "lag_keep",
                 "lag_tail", "decide_onehot", "option_onehot", "option_mask"],
                [(1, Q, cfg.hidden_size), (Q, cfg.rotary_dim), (Q, cfg.rotary_dim), (A, cfg.num_kv_heads, S, cfg.head_dim),
                 (A, cfg.num_kv_heads, S, cfg.head_dim), (S,), (Ld, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim),
                 (Ld, conv_dim, kernel), (Q, Q), (kernel, Q), (kernel, Q, kernel), (B, Q), (B, K, Q), (B, K)], half),
            outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)])
        name = f"packed_C{S}_P{Q}_B{B}_K{K}"
    else:
        with torch.no_grad():
            keys, values, states, tails = first(*state_in)
            hidden, q_cos, q_sin, valid, decide, options, mask = question_in
            traced = torch.jit.trace(second, (hidden, q_cos, q_sin, keys, values, valid, states, tails, decide, options, mask),
                                     check_trace=False)
        model = ct.convert(
            traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY,
            inputs=_types(
                ["hidden", "cos", "sin", "keys", "values", "cache_valid", "states", "tails", "decide_onehot",
                 "option_onehot", "option_mask"],
                [(B, Q, cfg.hidden_size), (Q, cfg.rotary_dim), (Q, cfg.rotary_dim), (A, cfg.num_kv_heads, S, cfg.head_dim),
                 (A, cfg.num_kv_heads, S, cfg.head_dim), (S,), (Ld, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim),
                 (Ld, conv_dim, kernel), (B, Q), (B, K, Q), (B, K)], half),
            outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)])
        name = f"question_C{S}_Q{Q}_B{B}_K{K}"
    model.author = "Jared Palmer (Kev, Apache-2.0); Fluid Inference (Core ML conversion)"
    model.save(str(args.build / f"{name}.mlpackage"))
    config = {"chunk_size": chunk, "lane": args.lane, "hidden_size": cfg.hidden_size, "rotary_dim": cfg.rotary_dim, "rope_theta": cfg.rope_theta,
              "conv_kernel": cfg.conv_kernel, "attention_layers": A, "delta_layers": Ld, "kv_heads": cfg.num_kv_heads,
              "head_dim": cfg.head_dim, "delta_heads": cfg.lin_v_heads, "delta_key_dim": cfg.lin_k_dim,
              "delta_value_dim": cfg.lin_v_dim, "conv_dim": conv_dim, "vocab_size": int(embed.shape[0]),
              "pad_id": meta["pad_id"], "special_tokens": special, "temperature": meta["temperature"], "source_run": meta["run"]}
    (args.build / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{name}: {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
