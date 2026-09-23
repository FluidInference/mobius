"""Convert the pinned laya checkpoint to a fixed-shape FP16 Core ML program."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import coremltools as ct
import laya
import numpy as np
import torch

from assets import LOCK_PATH, ROOT, checkpoint_dir, load_lock, sha256, verify_assets
from export_model import LayaExport
from preprocessing import Shape, package_name, prepare_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="multilingual")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    entry = verify_assets(args.variant)
    lock = load_lock()
    agent = laya.load(str(checkpoint_dir(args.variant)), device="cpu")
    agent.model.eval()
    shape = Shape(args.length, args.max_options)
    export = LayaExport(agent.model, shape.length, shape.max_options).eval()
    arrays, _ = prepare_inputs(
        agent.tok,
        "The piece leaves one hole under it and makes a small bump on top.",
        {"type": "noul", "instructions": "Is this placement clean?"},
        shape,
        agent.cfg["head_max_len"],
    )
    example = tuple(torch.from_numpy(value) for value in arrays.values())
    with torch.no_grad():
        traced = torch.jit.trace(export, example)
    started = time.perf_counter()
    inputs = [
        ct.TensorType(name="input_ids", shape=(1, shape.length), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=(1, shape.length), dtype=np.int32),
        ct.TensorType(name="marker_map", shape=(1, shape.max_options, shape.length), dtype=np.float32),
        ct.TensorType(name="question_type", shape=(1, 3), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
        ct.TensorType(name="probabilities", dtype=np.float32),
        ct.TensorType(name="action_probabilities", dtype=np.float32),
    ]
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=inputs,
        outputs=outputs,
    )
    converted.short_description = (
        f"laya {args.variant}: typed decision scoring (choice/score/noul), {shape.length} tokens, "
        f"{shape.max_options} option slots"
    )
    converted.author = "Convai Innovations (weights); Fluid Inference (Core ML conversion)"
    converted.license = "Apache-2.0"
    converted.version = "1"
    converted.user_defined_metadata.update(
        {
            "source_repo": lock["repo"],
            "source_revision": lock["revision"],
            "source_subfolder": entry["subfolder"],
            "encoder": entry["encoder"],
            "length": str(shape.length),
            "max_options": str(shape.max_options),
            "head_max_len": str(agent.cfg["head_max_len"]),
            "temperature": json.dumps(agent.cfg.get("temperature", [1.0, 1.0, 1.0])),
            "temperature_by_options": json.dumps(agent.cfg.get("temperature_by_options", {})),
            "sequence_format": "[CLS] <type> question: instructions [SEP] ([MASK] option)* [SEP] state [SEP]",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{package_name(args.variant, shape.length, shape.max_options)}.mlpackage"
    converted.save(str(path))
    manifest = {
        "model": path.name,
        "variant": args.variant,
        "precision": "float16",
        "minimum_target": "iOS17/macOS14",
        "length": shape.length,
        "max_options": shape.max_options,
        "head_max_len": agent.cfg["head_max_len"],
        "parameters": sum(p.numel() for p in agent.model.parameters()),
        "assets_lock_sha256": sha256(LOCK_PATH),
        "source_repo": lock["repo"],
        "source_revision": lock["revision"],
        "export_seconds": round(time.perf_counter() - started, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "package_files": {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob("*")) if p.is_file()},
    }
    (args.output_dir / f"{path.stem}.conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {path} ({manifest['parameters']:,} parameters, {manifest['export_seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
