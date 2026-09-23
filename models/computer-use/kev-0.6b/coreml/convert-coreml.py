"""Convert pinned Kev 0.6B weights to a fixed-shape FP16 Core ML program."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from assets import ROOT, load_model
from export_model import KevExport
from preprocessing import Shape, prepare_inputs


def fixture_request() -> dict:
    return {
        "state": "The piece leaves one hole beneath it and creates a small bump on top.",
        "questions": {
            "placement": {
                "type": "choice",
                "instructions": "Classify the placement.",
                "criteria": {
                    "clean": "No buried holes and a flat surface",
                    "risky": "Creates a cavity or an awkward surface",
                },
                "label": "risky",
                "src": "coreml-fixture",
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)

    lock, tokenizer, decision_model = load_model()
    decision_model.eval()
    shape = Shape(args.length, args.max_options)
    wrapper = KevExport(decision_model, shape.length, shape.max_options).eval()
    arrays, encoded = prepare_inputs(decision_model, tokenizer, fixture_request(), shape)
    example = tuple(torch.from_numpy(value) for value in arrays.values())
    with torch.no_grad():
        reference_logits = decision_model.forward(encoded)[0]
        export_logits, export_probabilities = wrapper(*example)
    option_count = len(encoded["opt_idx"][0])
    max_logit_error = float((reference_logits - export_logits[0, :option_count]).abs().max())
    same_argmax = int(reference_logits.argmax()) == int(export_logits[0, :option_count].argmax())
    print(f"PyTorch wrapper parity: argmax={same_argmax} max_logit_error={max_logit_error:.8f}", flush=True)
    if not same_argmax or max_logit_error > 1e-4:
        raise RuntimeError("explicit export wrapper does not match upstream Kev")
    if args.parity_only:
        return

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    started = time.perf_counter()
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, shape.length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, shape.length), dtype=np.int32),
            ct.TensorType(name="decide_map", shape=(1, 1, shape.length), dtype=np.float32),
            ct.TensorType(name="option_map", shape=(1, shape.max_options, shape.length), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="probabilities", dtype=np.float32),
        ],
    )
    converted.short_description = (
        f"Kev 0.6B typed decision scoring, {shape.length} tokens, {shape.max_options} option slots"
    )
    converted.author = "Jared Palmer (Kev); Qwen team (base); Fluid Inference (Core ML conversion)"
    converted.license = "Apache-2.0"
    converted.version = "1"
    converted.user_defined_metadata.update(
        {
            "checkpoint_repo": lock["checkpoint"]["repo"],
            "checkpoint_revision": lock["checkpoint"]["revision"],
            "base_repo": lock["base"]["repo"],
            "base_revision": lock["base"]["revision"],
            "length": str(shape.length),
            "max_options": str(shape.max_options),
            "sequence_format": "<state> state <q> instructions (<opt> option </opt>)* <decide>",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"kev_0_6b_fp16_L{shape.length}_options{shape.max_options}.mlpackage"
    converted.save(str(path))
    manifest = {
        "model": path.name,
        "precision": "float16",
        "minimum_target": "iOS17/macOS14",
        "length": shape.length,
        "max_options": shape.max_options,
        "parameters": sum(parameter.numel() for parameter in decision_model.parameters()),
        "export_seconds": round(time.perf_counter() - started, 1),
        "pytorch_wrapper_max_logit_error": max_logit_error,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "sources": lock,
    }
    (args.output_dir / f"{path.stem}.conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {path} ({manifest['parameters']:,} parameters, {manifest['export_seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
