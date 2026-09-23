"""Export Verdict's trained encoder/head to a fixed-shape Core ML program.

The graph returns raw logits. The host applies assets/calibrator.json using
native_reference.decode and retains the trained abstention candidate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer

from assets import LOCK, ROOT, sha256, verify_assets
from export_model import GLiClassExport
from native_reference import build_request
from preprocessing import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    if args.max_candidates != 25:
        raise ValueError("released Verdict head has 25 candidate slots")

    source = verify_assets()
    torch.set_num_threads(4)
    model = GLiClassModel.from_pretrained(source).eval()
    tokenizer = AutoTokenizer.from_pretrained(source)
    example = build_request(
        "I lost my wallet and need to stop my card.",
        {
            "type": "choice",
            "question": "What is the request?",
            "options": [
                {"id": "lost", "description": "a lost card"},
                {"id": "pin", "description": "a PIN reset"},
            ],
        },
    )
    arrays = prepare(tokenizer, model.config.class_token_index, example.text, args.length, args.max_candidates)
    if int(np.sum(arrays["class_marker_map"])) != len(example.labels):
        raise ValueError("candidate marker count differs from native request")
    tensors = tuple(torch.from_numpy(value) for value in arrays.values())
    export = GLiClassExport(model, args.length, args.max_candidates).eval()
    with torch.inference_mode():
        reference = model(
            input_ids=tensors[0].long(), attention_mask=tensors[1].long(), max_num_classes=len(example.labels)
        ).logits
        explicit = export(*tensors)[0][:, : len(example.labels)]
    wrapper_error = float((reference - explicit).abs().max())
    if wrapper_error > 1e-4:
        raise RuntimeError(f"wrapper/native logit mismatch: {wrapper_error}")

    traced = torch.jit.trace(export, tensors)
    started = time.perf_counter()
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, args.length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, args.length), dtype=np.int32),
            ct.TensorType(name="class_marker_map", shape=(1, args.max_candidates, args.length), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)],
    )
    converted.short_description = "Verdict decision logits; use abstention and per-K calibration in host"
    converted.author = "Heman10x-NGU (Verdict weights); Fluid Inference (Core ML conversion)"
    converted.license = "Apache-2.0"
    converted.user_defined_metadata.update(
        {
            "source_repo": LOCK["source_repo"],
            "source_revision": LOCK["source_revision"],
            "source_code_revision_audited": LOCK["source_code_revision_audited"],
            "calibrator_sha256": LOCK["files"]["calibrator.json"],
            "candidate_capacity_including_abstention": str(args.max_candidates),
            "length": str(args.length),
            "minimum_target": "iOS17/macOS14",
            "output_contract": "raw_logits; calibrate in host",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = args.output_dir / f"verdict_fp16_L{args.length}_candidates25.mlpackage"
    converted.save(str(package))
    report = {
        "package": package.name,
        "source_repo": LOCK["source_repo"],
        "source_revision": LOCK["source_revision"],
        "source_weight_sha256": LOCK["files"]["model.safetensors"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "wrapper_max_logit_error": wrapper_error,
        "conversion_seconds": time.perf_counter() - started,
        "package_bytes": sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
        "package_files_sha256": {str(p.relative_to(package)): sha256(p) for p in package.rglob("*") if p.is_file()},
    }
    (args.output_dir / f"{package.stem}.conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
