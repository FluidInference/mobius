"""Selective int8 weight-quantization of flowlm_step (Kyutai recipe).

Mirrors kyutai-labs/pocket-tts#147's `apply_dynamic_int8` with
`RECOMMENDED_CONFIG = {"attention", "ffn"}` but in the CoreML domain via
`coremltools.optimize.coreml.linear_quantize_weights` with an explicit
op-name allowlist.

Quantized (W8A16, per-channel symmetric):
  - transformer.layers[i].self_attn.in_proj  → const `attn{i}_in_proj_weight`
  - transformer.layers[i].self_attn.out_proj → const `attn{i}_out_proj_weight`
  - transformer.layers[i].linear1            → const `linear{i}_1_weight`
  - transformer.layers[i].linear2            → const `linear{i}_2_weight`

Untouched (fp32 / fp16):
  - input_linear, out_norm, out_eos (output head — quantizing this is what
    breaks autoregressive EOS gating, per our bisection)
  - All LayerNorm scales/biases
  - All attention bookkeeping (RoPE freqs, masks)

Input :  fp32 flowlm_step.mlpackage  (root-level HF or local convert output)
Output:  flowlm_step_int8_selective.mlpackage with the matching linear ops
         W8A16 and everything else fp16-cast.

Usage (from mobius/models/tts/pocket_tts/):
    uv run python coreml/convert_models/convert/quantize_flowlm_step_selective_int8.py \
        --input  /tmp/flowlm-int8-work/flowlm_step.mlpackage \
        --output /tmp/flowlm-int8-work/flowlm_step_int8_selective.mlpackage
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    OptimizationConfig,
    get_weights_metadata,
    linear_quantize_weights,
)


# Match both fp32 (`_weight`) and fp16-cast (`_weight_to_fp16`) variants.
ATTENTION_RE = re.compile(r"^attn\d+_(in_proj|out_proj)_weight(_to_fp16)?$")
FFN_RE       = re.compile(r"^linear\d+_[12]_weight(_to_fp16)?$")
# Output head: must NOT be quantized — bisection (Kyutai PR#147 + our flow_decoder
# parity) shows quantizing this kills autoregressive EOS gating.
EOS_HEAD_RE  = re.compile(r"^out_eos(_weight|_bias)?(_to_fp16)?$")


def _natural_op_key(s: str) -> tuple[int, str]:
    """Sort `linear_3` before `linear_10` while tolerating `_to_fp16` suffixes."""
    m = re.search(r"linear_(\d+)", s)
    return (int(m.group(1)) if m else 1_000_000, s)


def collect_target_op_names(
    mlmodel: ct.models.MLModel,
) -> tuple[list[str], list[str], list[str]]:
    """Walk weights metadata and split linear ops into:
      - target: Kyutai recipe (attention + FFN linears)
      - exclude: output head (out_eos) — kept fp32
      - shared_extras: non-Kyutai linears that share a bias const with a target,
                       so MUST be quantized too to avoid CoreML config-conflict
                       (input_linear is the canonical case)

    Returns (target_op_names_quantize, op_names_keep_fp32, all_quantize_op_names_for_reporting).
    """
    md = get_weights_metadata(mlmodel, weight_threshold=0)

    # Step 1: classify weight consts and collect their consuming linear ops.
    kyutai_linears: set[str] = set()
    eos_linears: set[str] = set()
    other_linears: set[str] = set()
    matched_consts: list[str] = []

    for const_name, info in md.items():
        is_kyutai = bool(ATTENTION_RE.match(const_name) or FFN_RE.match(const_name))
        is_eos = bool(EOS_HEAD_RE.match(const_name))
        for child in info.child_ops:
            if child.op_type != "linear":
                continue
            if is_kyutai:
                kyutai_linears.add(child.name)
            elif is_eos:
                eos_linears.add(child.name)
            else:
                # Could be input_linear, or a bias shared with target ops.
                # We'll resolve below.
                other_linears.add(child.name)
        if is_kyutai:
            matched_consts.append(const_name)

    # Step 2: find biases shared between Kyutai targets and non-target linears.
    # Any non-target linear sharing a bias with a Kyutai target must be quantized
    # too (else CoreML's per-const config-resolution raises a conflict).
    shared_extras: set[str] = set()
    for const_name, info in md.items():
        consumers = {c.name for c in info.child_ops if c.op_type == "linear"}
        if not consumers:
            continue
        if consumers & kyutai_linears and consumers - kyutai_linears:
            shared_extras |= (consumers - kyutai_linears - eos_linears)

    quantize_ops = sorted(kyutai_linears | shared_extras,
                          key=_natural_op_key)
    keep_fp32_ops = sorted(eos_linears,
                           key=_natural_op_key)
    return quantize_ops, keep_fp32_ops, sorted(matched_consts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="fp32 flowlm_step.mlpackage to quantize")
    ap.add_argument("--output", required=True, type=Path,
                    help="Destination .mlpackage path")
    ap.add_argument("--mode", default="linear_symmetric",
                    choices=["linear", "linear_symmetric"],
                    help="Per-channel quantization mode (default: linear_symmetric)")
    ap.add_argument("--weight-threshold", type=int, default=2048,
                    help="Skip ops whose weight tensor has < this many elements (default: 2048)")
    ap.add_argument("--list-only", action="store_true",
                    help="Print matched linear ops + their weight const names and exit (no quant)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    print(f"Loading {args.input}...", flush=True)
    mlmodel = ct.models.MLModel(str(args.input), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)

    print("Walking weight metadata for target linear ops...", flush=True)
    quantize_ops, keep_fp32_ops, matched_consts = collect_target_op_names(mlmodel)

    print(f"  Kyutai weight consts matched: {len(matched_consts)}")
    for c in matched_consts:
        print(f"    + {c}")
    print(f"  Linear ops to quantize ({len(quantize_ops)}):")
    for n in quantize_ops:
        print(f"    Q {n}")
    print(f"  Linear ops kept fp32 ({len(keep_fp32_ops)}):")
    for n in keep_fp32_ops:
        print(f"    . {n}")

    if args.list_only:
        return 0

    op_config = OpLinearQuantizerConfig(
        mode=args.mode,
        dtype="int8",
        weight_threshold=args.weight_threshold,
        granularity="per_channel",
    )

    # Strategy: quantize all linears EXCEPT the explicit fp32 set (out_eos head).
    # `op_name_configs[name] = None` opts a specific op out of the type-level config.
    # This makes every linear op resolve to the same config (op_config or None),
    # avoiding the per-const child-op config conflict that occurs when biases are
    # deduplicated across mixed-config linears.
    op_name_configs = {n: op_config for n in quantize_ops}
    op_name_configs.update({n: None for n in keep_fp32_ops})

    config = OptimizationConfig(
        global_config=None,
        op_type_configs={"linear": op_config},
        op_name_configs=op_name_configs,
    )

    print(f"Applying selective W8A16 quantization to {len(quantize_ops)} linear ops "
          f"({len(keep_fp32_ops)} kept fp32)...", flush=True)
    quantized = linear_quantize_weights(mlmodel, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {args.output}...", flush=True)
    quantized.save(str(args.output))

    in_size  = sum(p.stat().st_size for p in args.input.rglob("*") if p.is_file())
    out_size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    print(f"\nSizes: input={in_size/1024/1024:.1f} MB  output={out_size/1024/1024:.1f} MB  "
          f"(saved {(in_size-out_size)/1024/1024:.1f} MB, "
          f"{100*(1 - out_size/in_size):.1f}% reduction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
