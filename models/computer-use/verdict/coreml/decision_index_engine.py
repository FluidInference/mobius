"""Decision Index adapter for the current pinned Verdict checkpoint.

This is a newly frozen renderer. The historical tracker adapter/checkpoint pair
has not been authenticated; any score produced here is a new reproduction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import coremltools as ct
from decision_index.engines import Engine, NativeAbstention, Unsupported
from transformers import AutoTokenizer

from assets import LOCK, ROOT, verify_assets
from native_reference import build_request, decode, load_calibrator
from preprocessing import prepare


def as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def adapt_question(question: dict) -> dict:
    kind = question["type"]
    instructions = as_text(question.get("instructions", ""))
    if kind == "choice":
        criteria = question["criteria"]
        if not isinstance(criteria, dict):
            raise Unsupported("choice criteria must be an ordered key/description mapping")
        options = [
            {"id": key, "description": as_text(description) if description is not None else key}
            for key, description in criteria.items()
        ]
        return {"type": "choice", "question": instructions, "options": options}
    if kind == "noul":
        return {"type": "noul", "proposition": instructions}
    if kind == "score":
        criteria = question["criteria"]
        if not isinstance(criteria, list):
            raise Unsupported("score criteria must be an ordered list")
        return {
            "type": "score",
            "question": instructions,
            "levels": [
                {"id": str(index), "description": as_text(label), "value": index}
                for index, label in enumerate(criteria)
            ],
        }
    raise Unsupported(f"unsupported Verdict question type: {kind}")


class VerdictCoreMLEngine(Engine):
    name = "verdict-current-coreml"
    latency = "End-to-end in-process request time; excludes model loading."

    def __init__(self, packages: str = str(ROOT / "build"), lengths: str = "128,512", **options):
        super().__init__(**options)
        source = verify_assets(required=("config.json", "tokenizer.json", "tokenizer_config.json", "calibrator.json"))
        self.class_token_index = int(json.loads((source / "config.json").read_text())["class_token_index"])
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.calibrator = load_calibrator(source)
        self.lengths = tuple(sorted({int(length) for length in lengths.split(",")}))
        base = Path(packages)
        self.models = {
            length: ct.models.MLModel(
                str(base / f"verdict_fp16_L{length}_candidates25.mlpackage"), compute_units=ct.ComputeUnit.ALL
            )
            for length in self.lengths
        }
        self.provenance = {
            "source_repo": LOCK["source_repo"],
            "source_revision": LOCK["source_revision"],
            "historical_tracker_checkpoint_verified": False,
            "rendering": "Native labels, abstention and calibration; newly frozen Decision Index mapping",
            "max_tokens": max(self.lengths),
            "max_substantive_options": 24,
        }

    def __call__(self, state, questions):
        state_text = as_text(state)
        answers = {}
        raw = {}
        started = time.perf_counter()
        for question_id, question in questions.items():
            request = build_request(state_text, adapt_question(question))
            count = len(self.tokenizer(request.text, truncation=False)["input_ids"])
            available = [length for length in self.lengths if length >= count]
            if not available:
                raise Unsupported(f"Verdict input needs {count} tokens; largest exported bucket is {max(self.lengths)}")
            length = min(available)
            inputs = prepare(self.tokenizer, self.class_token_index, request.text, length, 25)
            output = self.models[length].predict(inputs)
            prediction = decode(output["logits"], request, self.calibrator)
            raw[question_id] = {"tokens": count, "bucket": length, **prediction}
            if prediction["is_abstention"]:
                raise NativeAbstention(raw)
            if question["type"] == "choice":
                substantive = {key: prediction["probabilities"][key] for key in question["criteria"]}
                mass = sum(substantive.values())
                answers[question_id] = {
                    "type": "choice",
                    "choice": prediction["choice"],
                    "probabilities": {key: value / mass for key, value in substantive.items()},
                }
            elif question["type"] == "noul":
                answers[question_id] = {"type": "noul", "noul": prediction["noul"]}
            else:
                answers[question_id] = {"type": "score", "score": prediction["score"]}
        response = {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": sum(x["tokens"] for x in raw.values())},
        }
        raw["elapsed_ms"] = (time.perf_counter() - started) * 1000
        return response, raw
