#!/usr/bin/env python3
"""Bit-exact alternative to fuse_decoder_joint_decision.py: a CoreML
*pipeline* of the shipped decoder + joint_decision specs.

A MIL-level fusion (fuse_decoder_joint_decision.py) elides the fp32 model
boundary; the runtime then schedules the joint GEMMs differently and the fp16
logits move by up to ~3.5 (4e-3 relative). That flips argmax on near-tie
frames (~1% of decode steps on LibriSpeech), changing emission timing and
occasionally tokens. A pipeline keeps the two original compiled programs
byte-identical — outputs are bit-exact vs the two-model reference — while
still collapsing the host side to ONE MLModel.prediction per RNNT step
(458 -> 229 dispatches/utt).

The decoder's `decoder` output is renamed to `decoder_step` so it feeds the
joint's input; `h_out`/`c_out` are surfaced as pipeline outputs alongside the
joint's `token_id`/`token_prob`/top-k.

Usage:
    python pipeline_decoder_joint_decision.py \
        --decoder /path/to/decoder.mlpackage \
        --joint /path/to/joint_decision.mlpackage \
        --output-dir /tmp/eou_fused
"""
from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct


def build(decoder_path: Path, joint_path: Path, output_dir: Path) -> Path:
    decoder = ct.models.MLModel(
        str(decoder_path), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True
    )
    # decoder output "decoder" -> joint input "decoder_step"
    dec_spec = decoder.get_spec()
    ct.utils.rename_feature(dec_spec, "decoder", "decoder_step", rename_inputs=False)
    decoder = ct.models.MLModel(
        dec_spec,
        weights_dir=decoder.weights_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    joint = ct.models.MLModel(
        str(joint_path), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True
    )

    # make_pipeline: "decoder_step" feeds the joint; the decoder's unconsumed
    # h_out/c_out and the joint's outputs all surface as pipeline outputs.
    model = ct.utils.make_pipeline(decoder, joint, compute_units=ct.ComputeUnit.CPU_ONLY)
    model.short_description = (
        "Parakeet EOU decoder + joint_decision pipeline (1 dispatch per RNNT "
        "step, bit-exact vs the two-model reference)."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "decoder_joint_decision_pipeline.mlpackage"
    model.save(str(out_path))
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", type=Path, required=True)
    ap.add_argument("--joint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    build(args.decoder, args.joint, args.output_dir)


if __name__ == "__main__":
    main()
