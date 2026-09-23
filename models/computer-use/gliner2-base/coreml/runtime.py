"""Run a GLiNER2 Core ML classifier without loading the original model weights."""
import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np

from preprocessing import load_processor, prepare_with_processor


def classify(model_dir: str, text: str, task: str, labels: list[str], length: int = 128, max_options: int = 8,
             precision: str = "fp16"):
    model_dir = Path(model_dir)
    if precision not in ("fp16", "embedding_w8"):
        raise ValueError(f"unsupported classification precision: {precision}")
    package = model_dir / f"gliner2_base_classification_{precision}_L{length}_K{max_options}.mlpackage"
    processor = load_processor(str(model_dir / "tokenizer"))
    arrays = prepare_with_processor(processor, text, task, labels, length, max_options)
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
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
    parser.add_argument("--precision", choices=["fp16", "embedding_w8"], default="fp16")
    args = parser.parse_args()
    print(json.dumps(classify(args.model_dir, args.text, args.task, json.loads(args.labels),
                              args.length, args.max_options, args.precision), indent=2))


if __name__ == "__main__":
    main()
