"""Standalone Core ML RLCD inference with the pinned tokenizer and schema contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

from preprocessing import Shape, batch_arrays, prepare_candidates, select_values


class RLCDCoreML:
    def __init__(self, package: Path, tokenizer_dir: Path, shape: Shape = Shape()):
        self.shape = shape
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)

    def constrained(self, context: str, schema: dict) -> dict:
        candidates = prepare_candidates(self.tokenizer, context, schema, self.shape)
        scores = []
        model_calls = 0
        for offset in range(0, len(candidates), self.shape.candidates):
            group = candidates[offset : offset + self.shape.candidates]
            arrays = batch_arrays(self.tokenizer, group, self.shape)
            output = self.model.predict(arrays)
            scores.extend(np.asarray(output["scores"]).reshape(-1).tolist()[: len(group)])
            model_calls += 1

        selected = select_values(candidates, scores)
        telemetry: dict[str, list[dict]] = {}
        for candidate, score in zip(candidates, scores):
            telemetry.setdefault(candidate.field, []).append({"value": candidate.value, "log_likelihood": score})
        return {
            "text": json.dumps(selected, ensure_ascii=False, allow_nan=False),
            "scores": telemetry,
            "branches": len(candidates),
            "model_calls": model_calls,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("."))
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-value-tokens", type=int, default=16)
    parser.add_argument("--context", required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text())
    runtime = RLCDCoreML(args.package, args.tokenizer_dir, Shape(args.length, args.batch, args.max_value_tokens))
    print(json.dumps(runtime.constrained(args.context, schema), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
