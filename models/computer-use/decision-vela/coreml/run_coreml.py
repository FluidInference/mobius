"""Serve the pinned System One contract from the published Core ML packages."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from packing import MarkerCollator
from transformers import AutoTokenizer
from typed_coreml import KINDS, marker_tensor, padded_batch
from upstream_system_one import _answer, system_one_records


class CoreMLSystemOne:
    def __init__(self, repo_root: Path, embedding_w8_kinds: tuple[str, ...] = ()):
        self.root = repo_root.resolve()
        if not set(embedding_w8_kinds).issubset(KINDS):
            raise ValueError(f"unknown compressed decision path: {embedding_w8_kinds}")
        self.embedding_w8_kinds = frozenset(embedding_w8_kinds)
        lock = json.loads((self.root / "conversion" / "assets.lock.json").read_text())
        self.model_name = {
            "decision-kai": "Decision-1.0-Kai",
            "decision-lex": "Decision-1.0-Lex",
        }[lock["engine"]]
        tokenizer = AutoTokenizer.from_pretrained(
            self.root / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        # The public upstream API enforces complete input within 1,024 tokens.
        self.collator = MarkerCollator(tokenizer, max_length=1024, state_truncation="error")
        self.models = {}
        self.model_inputs = {}

    def _model(self, kind: str):
        import coremltools as ct

        if kind not in self.models:
            # Holding two typed MLModel proxies crashes Core ML on this macOS runtime.
            self.models.clear()
            self.model_inputs.clear()
            gc.collect()
            suffix = "-embedding-w8" if kind in self.embedding_w8_kinds else ""
            path = self.root / "coreml" / (kind + suffix + ".mlpackage")
            if not path.is_dir():
                raise FileNotFoundError(f"selected Core ML package is missing: {path}")
            self.models[kind] = ct.models.MLModel(
                os.path.relpath(path, Path.cwd()), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
        return self.models[kind]

    def predict_row(self, record: dict) -> dict:
        kind = record["question"]["type"].lower()
        metadata = json.loads((self.root / "reports" / (kind + ".json")).read_text())
        shape = metadata["shape"]
        if shape["batch"] != 1:
            raise ValueError("This package expects one question per model call")
        batch, encoded = padded_batch(
            self.collator, [record], shape["tokens"], shape["candidates"]
        )
        model = self._model(kind)
        inputs = {
            "input_ids": batch["input_ids"].to(torch.int32).numpy(),
            "attention_mask": batch["attention_mask"].to(torch.int32).numpy(),
        }
        if kind not in self.model_inputs:
            self.model_inputs[kind] = {item.name for item in model.get_spec().description.input}
        input_names = self.model_inputs[kind]
        if "marker_map" in input_names:
            inputs["marker_map"] = marker_tensor(batch, marker_map=True).numpy()
        elif "marker_positions" in input_names:
            inputs["marker_positions"] = marker_tensor(batch, marker_map=False).numpy()
        else:
            raise ValueError("Core ML package lacks a supported marker input")
        result = model.predict(inputs)
        count = len(encoded[0]["positions"])
        logits = torch.from_numpy(result["logits"][0, :count]).float()
        probabilities = logits.softmax(-1)
        row = encoded[0]
        output = {
            key: row[key]
            for key in (
                "id", "question_id", "candidate_ids", "state_tokens_original",
                "state_tokens_kept", "input_tokens"
            )
        }
        output.update(
            type=kind.capitalize(),
            logits=logits.tolist(),
            probabilities=probabilities.tolist(),
            confidence=float(probabilities.max()),
        )
        if kind == "noul":
            output["probability"] = float(probabilities[1])
        else:
            output["choice_id"] = row["candidate_ids"][int(probabilities.argmax())]
            if kind == "score":
                output["score"] = float(
                    (probabilities * torch.arange(count, dtype=probabilities.dtype)).sum()
                )
                output["expected_value"] = float(
                    (probabilities * batch["values"][0, :count]).sum()
                )
        return output

    def evaluate(self, request: dict) -> dict:
        if request.get("model") != self.model_name:
            raise ValueError("Request model does not match the published checkpoint")
        rows = system_one_records(request)
        answers = {}
        input_tokens = 0
        for row in rows:
            prediction = self.predict_row(row)
            answers[row["question"]["id"]] = _answer(row, prediction)
            input_tokens += prediction["input_tokens"]
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--request-json", required=True, type=Path)
    parser.add_argument("--embedding-w8-kinds", nargs="+", choices=KINDS, default=(),
                        help="use optional embedding-W8 packages for the listed typed paths")
    args = parser.parse_args()
    runtime = CoreMLSystemOne(args.repo_root, tuple(args.embedding_w8_kinds))
    result = runtime.evaluate(json.loads(args.request_json.read_text()))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
