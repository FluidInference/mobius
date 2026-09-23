"""Run the published Kev 0.6B Core ML artifact without native model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from kev.api import output_tokens, to_answers
from transformers import AutoTokenizer

from preprocessing import Shape, prepare_runtime_inputs

PACKAGES = {
    "fp16": "kev_0_6b_fp16_L128_options32.mlpackage",
    "w8": "kev_0_6b_w8_L128_options32.mlpackage",
}
COMPUTE_UNITS = {
    "all": ct.ComputeUnit.ALL,
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
}


def predict(
    model_dir: Path, request: dict, *, precision: str = "fp16", units: str = "all",
    package: Path | None = None,
) -> dict:
    """Return one typed System One answer from the published package and tokenizer."""
    root = Path(model_dir)
    training = json.loads((root / "config" / "training_config.json").read_text())
    if training["args"]["option_isolation"] not in (0, False):
        raise ValueError("This Core ML export requires Kev option_isolation=False")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    arrays, encoded, metadata, parsed = prepare_runtime_inputs(tokenizer, request, Shape())
    model_path = Path(package) if package is not None else root / PACKAGES[precision]
    model = ct.models.MLModel(str(model_path), compute_units=COMPUTE_UNITS[units])
    result = model.predict(arrays)
    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    count = len(metadata[0]["keys"])
    if probabilities.shape != (1, Shape().max_options) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid Core ML probabilities")
    selected = probabilities[0, :count].tolist()
    answers = to_answers([selected], metadata)
    return {
        "model": parsed.model,
        "answers": answers,
        "usage": {
            "input_tokens": len(encoded["ids"]),
            "output_tokens": output_tokens(tokenizer, answers),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--precision", choices=PACKAGES, default="fp16")
    parser.add_argument("--units", choices=COMPUTE_UNITS, default="all")
    parser.add_argument("--package", type=Path, help="override package path while keeping the published tokenizer")
    args = parser.parse_args()
    request = json.loads(args.request_json.read_text())
    print(json.dumps(predict(args.model_dir, request, precision=args.precision, units=args.units,
                             package=args.package), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
