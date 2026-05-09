"""Apply post-training linear int8 weight quantization to an existing
``decoder_step.mlpackage``.

Unlike ``convert_nanocodec.py``'s ``--palettize`` (8-bit kmeans), this uses
``cto.linear_quantize_weights`` so that weight matmuls can dispatch to the
ANE's int8 fast path on iOS 17+ / macOS 14+ devices, instead of dequantizing
back to fp16 at runtime. The trade-off is potential ANE rejection on a few
ops which would force CPU+GPU fallback (cf. the stateful variant's 2.2x
regression for that exact reason).

This script does NOT re-trace from PyTorch. It loads the already-converted
``mlpackage`` and rewrites the weight tensors in place. That keeps the
turnaround tight (no ``nemo_toolkit`` install required) and lets us iterate
on the quant config without redoing ``ct.convert()``.

------------------------------------------------------------------------------
RESULT (2026-05, M2, macOS 26.5, coremltools 8.x):

  Per-step latency (coreml-cli synthetic 1-step bench):
      fp16             : 16.05 ms (all)   ANE 97.3 %
      int8 per-tensor  : 12.56 ms (all)   ANE 97.3 %   -21.7 %
      int8 per-channel : 14.62 ms (all)   ANE 97.3 %    -8.9 %
      (per-channel pays a runtime scale-broadcast vs per-tensor; both stay
      fully on ANE because constexpr_affine_dequantize is a constant-prep
      op, not in the forward path.)

  End-to-end streaming synth (Magpie CLI):

  Short input — 3-word "Hello from Magpie." (1-2 chunks):
      fp16             : TTFA 1.482 s  total 2.00 s  audio  2.59 s  EOS@29 ✓
      int8 per-tensor  : TTFA 6.837 s  total 15.13 s audio 24.04 s  EOS✗  (runaway)
      int8 per-channel : TTFA 0.873 s  total 1.46 s  audio  2.96 s  EOS@36 ✓

  Long input — 5-clause sentence (5 chunks):
      fp16             : TTFA 2.785 s  total 14.96 s audio 25.17 s  all chunks EOS ✓
      int8 per-channel : TTFA 2.121 s  total 18.74 s audio 37.34 s  CHUNK 4 RUNAWAY
                         (chunks 0-3 EOS clean, chunk 4 hits maxSteps=500 with
                         ~12 s of garbage tail audio)
      int8 per-channel + skip-head (linear_60_cast_fp16 fp16):
                       : TTFA 2.726 s  total 20.23 s audio 38.50 s  CHUNK 4 STILL
                         RUNAWAY. Skipping the LM head alone did NOT fix the
                         long-context EOS drift, indicating the drift is
                         accumulated in the int8 body's KV/representations
                         (prefill + 4 chunks worth of state) and an fp16 head
                         can't recover the distribution. Mlpackage kept under
                         build/decoder_step_int8_pc_skiphead.mlpackage.

  Diagnosis:
    - Per-tensor int8 drifts the EOS logit enough that no chunk terminates
      after the prefill-anchored chunk 0 → unconditional runaway.
    - Per-channel preserves dynamic range across the LM head's 16192 output
      codes and recovers EOS on most chunks, but the final / longest chunk
      can still drift on inputs ≥ ~5 chunks → intermittent runaway.
    - The TTFA win on short inputs is real (-41 % vs fp16) and the per-step
      latency is genuine (-9 %). What still drifts is the EOS calibration
      on the longest-context chunks.

  Verdict:
    - Not shippable as a default — long inputs produce garbage tails.
    - Possibly shippable as opt-in for short / always-streaming workloads
      where every chunk is bounded.
    - True fix likely requires op-name skip-list to keep the LM head
      (`logits_proj` / final projection) and / or final attention block
      in fp16 — same approach NVIDIA uses for INT8 transformer LMs.

  Next levers (not yet tried, gated on user approval):
    1. Op-name skip-list: leave LM head + last LN in fp16 via
       cto.OptimizationConfig op_name_configs / op_type_configs.
    2. Group-wise / blockwise int4 with palette to recover EOS while keeping
       most of the weight savings.
    3. KV cache quant (separate code path — only weights are touched here).

  Until one of those works, prefer the fp16 graph + chunker tweaks for TTFA.
  The int8 per-channel mlpackage / mlmodelc are kept under
  ``build/decoder_step_int8_pc.mlpackage`` and
  ``compiled/build/decoder_step_int8_pc.mlmodelc`` for the op-allowlist
  follow-up; fp16 cache restored at
  ``~/.cache/fluidaudio/Models/magpie-tts/decoder_step.mlmodelc``.
------------------------------------------------------------------------------

Usage:
    uv run python quantize_decoder_step_int8.py \
        --input  build/upstream/decoder_step.mlpackage \
        --output build/decoder_step_int8.mlpackage
"""

import argparse
import os
import time

import coremltools as ct
import coremltools.optimize.coreml as cto


def quantize(
    input_path: str,
    output_path: str,
    weight_threshold: int = 512,
    granularity: str = "per_channel",
    skip_ops: tuple = ("linear_60_cast_fp16",),
) -> str:
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input mlpackage not found: {input_path}")

    print(f"Loading {input_path} ...")
    t0 = time.time()
    mlmodel = ct.models.MLModel(input_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f"  Loaded in {time.time() - t0:.2f}s")

    global_cfg = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity=granularity,
        weight_threshold=weight_threshold,
    )

    # Per-op overrides: pass `None` to leave the op's weights untouched (fp16).
    # This is the textbook fix for INT8 transformer LMs — the output projection
    # ("LM head") is the most calibration-sensitive op because EOS sits in the
    # tail of a 16192-way softmax and small per-output bias drift kills it.
    op_name_configs = {name: None for name in skip_ops}

    config = cto.OptimizationConfig(
        global_config=global_cfg,
        op_name_configs=op_name_configs,
    )

    skip_list = ", ".join(skip_ops) if skip_ops else "(none)"
    print(
        f"Quantizing weights (linear_symmetric int8, granularity={granularity}, "
        f"weight_threshold={weight_threshold}, skip_ops=[{skip_list}]) ..."
    )
    t0 = time.time()
    qmodel = cto.linear_quantize_weights(mlmodel, config)
    print(f"  Quantized in {time.time() - t0:.2f}s")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    qmodel.save(output_path)
    print(f"Saved to {output_path}")

    in_size = _dir_bytes(input_path)
    out_size = _dir_bytes(output_path)
    print(
        f"Size: {in_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB "
        f"({out_size / in_size:.2%} of original)"
    )
    return output_path


def _dir_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="build/upstream/decoder_step.mlpackage",
        help="Path to existing decoder_step.mlpackage (fp16, from HF).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="build/decoder_step_int8.mlpackage",
        help="Output path for the quantized mlpackage.",
    )
    parser.add_argument(
        "--weight-threshold",
        type=int,
        default=512,
        help=(
            "Skip quantizing weight tensors with fewer than this many "
            "elements (default 512, mirrors coremltools recommendation "
            "to leave tiny biases / norms in fp16)."
        ),
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="per_channel",
        choices=["per_tensor", "per_channel"],
        help=(
            "Quant scale granularity. ``per_tensor`` uses one scale for the "
            "whole weight (smallest, fastest, most lossy); ``per_channel`` "
            "(default) uses one scale per output channel which preserves the "
            "LM head's dynamic range and keeps EOS prediction intact."
        ),
    )
    parser.add_argument(
        "--skip-ops",
        type=str,
        default="linear_60_cast_fp16",
        help=(
            "Comma-separated list of MIL op names to leave in fp16 "
            "(default ``linear_60_cast_fp16`` = the final LM head / "
            "``final_proj`` linear that produces the 16192 codebook logits). "
            "Pass an empty string to quantize the whole graph."
        ),
    )
    args = parser.parse_args()
    skip_ops = tuple(s for s in args.skip_ops.split(",") if s.strip())
    quantize(
        args.input,
        args.output,
        args.weight_threshold,
        args.granularity,
        skip_ops,
    )
