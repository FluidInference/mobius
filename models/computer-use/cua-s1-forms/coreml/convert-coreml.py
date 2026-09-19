"""Convert the pinned real CUA-S1-FORMS checkpoint using an existing demo input."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from ane_gather import rewrite_byte_gathers
from assets import LOCK_PATH, ROOT, load_demo, load_reference, sha256, verify_assets
from export_model import ExportScorer
from preprocessing import InputLimits, prepare_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--optimization", choices=["baseline", "ane-gather"], default="baseline")
    args = parser.parse_args()
    torch.set_num_threads(2)
    # Prevent PyTorch's fused inference-only encoder from hiding the traceable ops.
    torch.backends.mha.set_fastpath_enabled(False)
    lock = verify_assets()
    reference, _, config = load_reference()
    limits = InputLimits(config["context_tokens"], config["option_tokens"], args.max_options)
    rows = load_demo()
    row = rows[0]
    arrays = prepare_inputs(row["context"], row["options"], limits)
    example = tuple(torch.from_numpy(value) for value in arrays.values())
    wrapper = ExportScorer(reference, optimization=args.optimization).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    started = time.perf_counter()
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType(name=name, shape=value.shape, dtype=np.int32) for name, value in arrays.items()],
        outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)],
    )
    if args.optimization == "ane-gather":
        converted = rewrite_byte_gathers(converted)
    converted.short_description = "CUA-S1-FORMS: one-pass scoring of document entities and form actions"
    converted.author = "Cua AI (weights); Fluid Inference (Core ML conversion)"
    converted.license = "MIT; see pinned model card and vendor/CUA-LICENSE"
    converted.version = "1"
    converted.user_defined_metadata.update(
        {
            "model_revision": lock["model_revision"],
            "source_revision": lock["source_revision"],
            "input_limits": json.dumps(asdict(limits), sort_keys=True),
            "encoding": "UTF-8 bytes truncated to limits; byte+1; zero padding",
            "optimization": args.optimization,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"cua_s1_forms_fp16_options{args.max_options}.mlpackage"
    converted.save(str(path))
    manifest = {
        "model": path.name,
        "precision": "float16",
        "optimization": args.optimization,
        "minimum_target": "iOS17/macOS14",
        "limits": asdict(limits),
        "model_config": config,
        "parameters": sum(parameter.numel() for parameter in reference.parameters()),
        "assets_lock_sha256": sha256(LOCK_PATH),
        "model_revision": lock["model_revision"],
        "source_revision": lock["source_revision"],
        "trace_row": 0,
        "trace_dataset_revision": lock["dataset_revision"],
        "export_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "package_files": {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()},
    }
    (args.output_dir / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {path} ({manifest['parameters']:,} parameters)", flush=True)


if __name__ == "__main__":
    main()
