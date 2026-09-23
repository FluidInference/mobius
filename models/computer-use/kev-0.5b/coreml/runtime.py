"""Serve unlabelled Kev 0.5B typed decisions from the Core ML artifact."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from kev.api import output_tokens, to_answers
from transformers import AutoTokenizer

from preprocessing import Shape, prepare_runtime_inputs

COMPUTE_UNITS = {
    "all": ct.ComputeUnit.ALL,
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
}
PACKAGE = "kev_0_5b_fp16_L128_options32.mlpackage"


def stage_symlinked_package(package: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Give Core ML real package files when a Hub snapshot contains blob symlinks."""
    if not package.is_symlink() and not any(path.is_symlink() for path in package.rglob("*")):
        return package, None
    temporary = tempfile.TemporaryDirectory(prefix="kev-coreml-")
    staged = Path(temporary.name) / package.name
    shutil.copytree(package, staged, symlinks=False)
    return staged, temporary


class KevCoreML:
    """Reusable serving session; initialization and request latency stay separate."""

    def __init__(self, model_dir: str | Path, *, units: str = "all", package: str | Path | None = None):
        root = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        self.shape = Shape()
        model_path = Path(package) if package is not None else root / PACKAGE
        model_path, self._staged_package = stage_symlinked_package(model_path)
        self.model = ct.models.MLModel(str(model_path), compute_units=COMPUTE_UNITS[units])

    def predict(self, request: dict) -> dict:
        arrays, encoded, metadata, parsed = prepare_runtime_inputs(self.tokenizer, request, self.shape)
        result = np.asarray(self.model.predict(arrays)["probabilities"], dtype=np.float64)
        count = len(metadata[0]["keys"])
        if result.shape != (1, self.shape.max_options) or not np.isfinite(result).all():
            raise ValueError("Invalid Core ML probabilities")
        selected = result[0, :count].tolist()
        keys = metadata[0]["keys"]
        answers = to_answers([selected], metadata)
        return {
            "model": parsed.model,
            "answers": answers,
            "usage": {
                "input_tokens": len(encoded["ids"]),
                "output_tokens": output_tokens(self.tokenizer, answers),
            },
            "option_keys": keys,
            "probabilities": selected,
            "chosen": keys[int(np.argmax(selected))],
        }


def predict(model_dir: str | Path, request: dict, *, units: str = "all", package: str | Path | None = None) -> dict:
    """One-shot compatibility wrapper; reuse `KevCoreML` for repeated calls."""
    return KevCoreML(model_dir, units=units, package=package).predict(request)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--units", choices=COMPUTE_UNITS, default="all")
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request_json.read_text())
    print(json.dumps(predict(args.model_dir, request, units=args.units, package=args.package), indent=2))


if __name__ == "__main__":
    main()
