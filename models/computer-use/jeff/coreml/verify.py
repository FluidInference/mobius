"""Compare the exported Jeff FP16 Core ML package with its trained native model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from jeff.backends.torch_backend import TorchBackend

from export import FIXTURES, REVISION, SOURCE, make_batch, model_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--package", type=Path, help="explicit package path for an optimized variant")
    parser.add_argument("--units", choices=("cpu", "cpu-gpu", "cpu-ne", "all"), default="cpu")
    parser.add_argument("--report", type=Path, help="JSON report path for an optimized variant")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = snapshot_download(SOURCE, revision=REVISION, local_files_only=True)
    backend = TorchBackend(checkpoint, device="cpu", dtype="float32", attn_kernel="eager", batch_size=1)
    package = args.package or Path(f"build/JeffDecision-L128-{args.precision.upper()}.mlpackage")
    units = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }
    model = ct.models.MLModel(str(package), compute_units=units[args.units])
    output_name = model.get_spec().description.output[0].name
    results = []
    for name, text, group in FIXTURES:
        batch = make_batch(backend, text, group)
        native = backend.model.model(**batch, include_media=False).cat_logits.detach().float().numpy()[0]
        tensor_inputs = model_inputs(batch, backend.model.config)
        inputs = {
            key: tensor.numpy()
            for key, tensor in zip(
                ("input_ids", "attention_mask", "parent_position", "category_positions"),
                tensor_inputs,
            )
        }
        start = time.perf_counter()
        converted = model.predict(inputs)[output_name][0, :len(group.labels)].astype(np.float32)
        elapsed_ms = (time.perf_counter() - start) * 1000
        error = float(np.max(np.abs(native - converted))) if np.isfinite(converted).all() else float("inf")
        agreement = int(np.argmax(native) == np.argmax(converted))
        results.append({
            "name": name,
            "labels": list(group.labels),
            "native_logits": native.tolist(),
            "coreml_logits": converted.tolist(),
            "max_logit_error": error,
            "top_label_agreement": bool(agreement),
            "coreml_wall_ms": elapsed_ms,
        })
        print(
            f"{name}: logits={converted.tolist()}, error={error:.6f}, "
            f"top_label={bool(agreement)}, wall_ms={elapsed_ms:.1f}",
            flush=True,
        )
    report = args.report or Path(f"build/coreml-parity-{args.precision}.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + "\n")
    if not all(row["top_label_agreement"] for row in results):
        raise AssertionError("Core ML changed the chosen label on a parity fixture")
    if max(row["max_logit_error"] for row in results) > 0.25:
        raise AssertionError("Core ML logit error exceeds the 0.25 tolerance")


if __name__ == "__main__":
    main()
