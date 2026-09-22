"""Decision Index engine for the application-tuned GLiClass checkpoint."""

from __future__ import annotations

import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from decision_index.engines import Engine, Unsupported
from gliclass import GLiClassModel
from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
from laya.common import serialize_state
from transformers import AutoTokenizer

from verify import make_arrays, select_bucket

ROOT = Path(__file__).resolve().parent


def option_labels(question: dict) -> tuple[list[str], list[str]]:
    """Map one Decision Index question to stable model labels and response keys."""
    if question["type"] == "choice":
        keys = list(question["criteria"])
        labels = []
        for key, description in question["criteria"].items():
            labels.append(f"{key}: {description}" if description is not None else key)
        return keys, labels
    if question["type"] == "noul":
        return ["false", "true"], ["false", "true"]
    raise Unsupported(f"unsupported question type {question['type']}")


def instruction_text(question: dict) -> str | None:
    """Render structured instructions without dropping benchmark content."""
    instructions = question.get("instructions")
    if instructions is None or isinstance(instructions, str):
        return instructions
    return serialize_state(instructions)


class GLiClassEdgeAppsEngine(Engine):
    """Run the tuned GLiClass model without truncation or option filtering."""

    name = "gliclass-edge-apps-v2"
    latency = "CPU-synchronized in-process request time including fixed rendering and tokenization; excludes loading."

    def __init__(
        self,
        model: str = str(ROOT / "build/checkpoint-v2"),
        device: str = "cpu",
        max_tokens: int = 512,
        max_options: int = 25,
        threads: int = 8,
        **options,
    ) -> None:
        super().__init__(**options)
        self.device = torch.device(device)
        self.max_tokens = int(max_tokens)
        self.max_options = int(max_options)
        self.model_id = model
        self.model = GLiClassModel.from_pretrained(model).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
        self.formatter = UniEncoderZeroShotClassificationPipeline(
            self.model,
            self.tokenizer,
            max_classes=self.max_options,
            max_length=self.max_tokens,
            classification_type="single-label",
            device=self.device,
            progress_bar=False,
        )
        if self.device.type == "cpu":
            torch.set_num_threads(int(threads))
        self.provenance = {
            "kind": "gliclass",
            "checkpoint": model,
            "device": str(self.device),
            "context_limit_tokens": self.max_tokens,
            "option_limit": self.max_options,
            "policy": (
                "Each question is rendered with the model's fixed uni-encoder helper. The complete serialized state, "
                "instruction, option keys, and option descriptions are tokenized without truncation. Requests beyond "
                "the declared context or option capacity are unsupported; every option is retained."
            ),
        }

    def runtime(self) -> dict:
        import gliclass
        import transformers

        return {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gliclass": getattr(gliclass, "__version__", "0.1.20"),
            "device": str(self.device),
        }

    def synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()

    def __call__(self, state, questions) -> tuple[dict, dict]:
        if any(len(option_labels(question)[0]) > self.max_options for question in questions.values()):
            raise Unsupported(f"request exceeds the {self.max_options}-option limit")

        state_text = serialize_state(state)
        answers = {}
        raw = {}
        for question_key, question in questions.items():
            keys, labels = option_labels(question)
            rendered = self.formatter.prepare_input(state_text, labels, prompt=instruction_text(question))
            encoded = self.tokenizer(rendered, truncation=False, return_tensors="pt")
            token_count = int(encoded["input_ids"].shape[1])
            if token_count > self.max_tokens:
                raise Unsupported(f"prompt has {token_count} tokens; context limit is {self.max_tokens}")

            model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
            started = time.perf_counter()
            with torch.inference_mode():
                logits = self.model(**model_inputs, max_num_classes=len(labels)).logits[0].float()
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            elapsed_ms = (time.perf_counter() - started) * 1000
            distribution = dict(zip(keys, probabilities))
            choice = keys[max(range(len(keys)), key=probabilities.__getitem__)]
            if question["type"] == "choice":
                answers[question_key] = {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": distribution,
                }
            else:
                answers[question_key] = {"type": "noul", "noul": distribution["true"]}
            raw[question_key] = {
                "tokens": token_count,
                "labels": len(labels),
                "inference_ms": elapsed_ms,
            }
        response = {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": sum(item["tokens"] for item in raw.values())},
        }
        return response, raw


class GLiClassEdgeAppsCoreMLEngine(GLiClassEdgeAppsEngine):
    """Run the same deploy limits and the converted Core ML buckets directly."""

    name = "gliclass-edge-apps-v2-coreml"
    latency = "Core ML in-process request time including fixed rendering and tokenization; excludes loading."

    def __init__(
        self,
        packages: str = str(ROOT / "build/coreml"),
        lengths: str = "128,256,512",
        **options,
    ) -> None:
        super().__init__(device="cpu", **options)
        self.lengths = sorted({int(value) for value in lengths.split(",")})
        if self.lengths[-1] != self.max_tokens:
            raise ValueError("largest Core ML bucket must equal max_tokens")
        package_dir = Path(packages)
        self.packages = {
            length: ct.models.MLModel(
                str(package_dir / f"gliclass_edge_apps_fp16_L{length}_options{self.max_options}.mlpackage"),
                compute_units=ct.ComputeUnit.ALL,
            )
            for length in self.lengths
        }
        self.provenance = {
            **self.provenance,
            "kind": "coreml",
            "packages": str(package_dir),
            "buckets": self.lengths,
        }

    def runtime(self) -> dict:
        return {
            **super().runtime(),
            "coremltools": ct.__version__,
            "device": "Core ML ALL",
        }

    def __call__(self, state, questions) -> tuple[dict, dict]:
        if any(len(option_labels(question)[0]) > self.max_options for question in questions.values()):
            raise Unsupported(f"request exceeds the {self.max_options}-option limit")

        state_text = serialize_state(state)
        answers = {}
        raw = {}
        for question_key, question in questions.items():
            keys, labels = option_labels(question)
            rendered = self.formatter.prepare_input(state_text, labels, prompt=instruction_text(question))
            token_count = len(self.tokenizer(rendered, truncation=False)["input_ids"])
            if token_count > self.max_tokens:
                raise Unsupported(f"prompt has {token_count} tokens; context limit is {self.max_tokens}")

            length = select_bucket(token_count, self.lengths)
            arrays = make_arrays(self.tokenizer, self.model, rendered, len(labels), length, self.max_options)
            started = time.perf_counter()
            output = self.packages[length].predict(arrays)
            logits = np.asarray(output["logits"], dtype=np.float64)[0, : len(labels)]
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            probabilities = probabilities.tolist()
            elapsed_ms = (time.perf_counter() - started) * 1000
            distribution = dict(zip(keys, probabilities))
            choice = keys[max(range(len(keys)), key=probabilities.__getitem__)]
            if question["type"] == "choice":
                answers[question_key] = {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": distribution,
                }
            else:
                answers[question_key] = {"type": "noul", "noul": distribution["true"]}
            raw[question_key] = {
                "tokens": token_count,
                "labels": len(labels),
                "bucket": length,
                "inference_ms": elapsed_ms,
            }
        response = {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": sum(item["tokens"] for item in raw.values())},
        }
        return response, raw
