"""Run a GLiNER2 Core ML classifier without loading the original model weights."""
import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np

from preprocessing import load_processor, prepare_with_processor


def classify(
    model_dir: str,
    text: str,
    task: str,
    labels: list[str],
    length: int = 128,
    max_options: int = 8,
    *,
    package: str | None = None,
    units: str = "all",
):
    model_dir = Path(model_dir)
    default_package = f"gliner2_small_classification_fp16_L{length}_K{max_options}.mlpackage"
    package_path = Path(package or default_package)
    if not package_path.is_absolute():
        package_path = model_dir / package_path
    compute_units = {"all": ct.ComputeUnit.ALL, "cpu-ane": ct.ComputeUnit.CPU_AND_NE}[units]
    processor = load_processor(str(model_dir / "tokenizer"))
    arrays = prepare_with_processor(processor, text, task, labels, length, max_options)
    model = ct.models.MLModel(str(package_path), compute_units=compute_units)
    scores = np.asarray(model.predict(arrays)["probabilities"])[0, : len(labels)]
    return {"label": labels[int(scores.argmax())], "confidence": float(scores.max()),
            "probabilities": {label: float(score) for label, score in zip(labels, scores)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--task", default="decision")
    parser.add_argument("--labels", required=True, help="JSON list of label strings")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--package", help="Package name inside model-dir, or absolute package path")
    parser.add_argument("--units", choices=["all", "cpu-ane"], default="all")
    args = parser.parse_args()
    print(json.dumps(classify(args.model_dir, args.text, args.task, json.loads(args.labels),
                              args.length, args.max_options, package=args.package, units=args.units), indent=2))


if __name__ == "__main__":
    main()
