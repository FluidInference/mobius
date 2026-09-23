"""Export GLiClass variants that remove CPU-side preprocessing from the encoder graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliclass import GLiClassModel
from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
from transformers import AutoTokenizer

from export_model import GLiClassExport


class FloatMaskExport(GLiClassExport):
    """Keep token lookup in Core ML but accept a ready-to-add floating-point pad bias."""

    def forward(self, input_ids, attention_bias, class_marker_map):
        full = self.full_mask + attention_bias
        band = self.band_mask + attention_bias
        hidden = self.embeddings(input_ids=input_ids.long())
        return self._encode(hidden, full, band, class_marker_map)

    def _encode(self, hidden, full, band, class_marker_map):
        for layer, layer_type in zip(self.layers, self.layer_types):
            hidden = layer(
                hidden,
                attention_mask=full if layer_type == "full_attention" else band,
                position_embeddings=(
                    getattr(self, f"cos_{layer_type}"),
                    getattr(self, f"sin_{layer_type}"),
                ),
            )
        hidden = self.final_norm(hidden)
        classes = torch.matmul(class_marker_map, hidden)
        text = self.text_projector(hidden[:, 0])
        classes = self.classes_projector(classes)
        logits = self.scorer(text, classes)
        option_mask = class_marker_map.sum(-1)
        logits = logits * option_mask + (1.0 - option_mask) * -1e4
        return logits, torch.softmax(logits, dim=-1)


class PreembeddedExport(FloatMaskExport):
    """Accept gathered token vectors so the encoder graph contains only floating-point work."""

    def __init__(self, model, length, max_options):
        super().__init__(model, length, max_options)
        self.embedding_norm = self.embeddings.norm
        self.embeddings = None

    def forward(self, input_embeddings, attention_bias, class_marker_map):
        hidden = self.embedding_norm(input_embeddings)
        full = self.full_mask + attention_bias
        band = self.band_mask + attention_bias
        return self._encode(hidden, full, band, class_marker_map)


class PreembeddedIO16Export(PreembeddedExport):
    """Use FP16 feature I/O while preserving the source module's export-friendly dtypes."""

    def forward(self, input_embeddings, attention_bias, class_marker_map):
        logits, probabilities = super().forward(
            input_embeddings.float(), attention_bias.float(), class_marker_map.float()
        )
        return logits.half(), probabilities.half()


def package_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def prepare_fixture(tokenizer, model, length: int, max_options: int):
    labels = [
        "world news and international politics",
        "sports",
        "business and economy",
        "science and technology",
    ]
    formatter = UniEncoderZeroShotClassificationPipeline(
        model,
        tokenizer,
        max_classes=max_options,
        max_length=length,
        classification_type="single-label",
        device="cpu",
        progress_bar=False,
    )
    rendered = formatter.prepare_input(
        '{"article":"Apple introduced a smaller processor for laptop computers."}',
        labels,
        prompt="What is the topic of `article`?",
    )
    encoded = tokenizer(
        rendered,
        truncation=True,
        max_length=length,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int32)
    attention_mask = encoded["attention_mask"].astype(np.int32)
    positions = np.flatnonzero(input_ids[0] == model.config.class_token_index)
    marker_map = np.zeros((1, max_options, length), dtype=np.float32)
    for index, position in enumerate(positions):
        marker_map[0, index, position] = 1.0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "class_marker_map": marker_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="build/checkpoint-v2")
    parser.add_argument("--output", default="build/ane")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=25)
    args = parser.parse_args()

    model = GLiClassModel.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    arrays = prepare_fixture(tokenizer, model, args.length, args.max_options)
    ids = torch.from_numpy(arrays["input_ids"])
    marker_map = torch.from_numpy(arrays["class_marker_map"])
    attention_bias = (1.0 - torch.from_numpy(arrays["attention_mask"]).float()).view(1, 1, 1, args.length) * -1e4
    with torch.no_grad():
        reference = model(
            input_ids=ids.long(),
            attention_mask=torch.from_numpy(arrays["attention_mask"]).long(),
            max_num_classes=4,
        ).logits
        token_embeddings = model.model.encoder_model.embeddings.tok_embeddings(ids.long())

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    variants = [
        (
            "float_mask",
            FloatMaskExport(model, args.length, args.max_options).eval(),
            (ids, attention_bias, marker_map),
            [
                ct.TensorType(name="input_ids", shape=(1, args.length), dtype=np.int32),
                ct.TensorType(name="attention_bias", shape=(1, 1, 1, args.length), dtype=np.float32),
                ct.TensorType(
                    name="class_marker_map",
                    shape=(1, args.max_options, args.length),
                    dtype=np.float32,
                ),
            ],
            np.float32,
        ),
        (
            "preembedded",
            PreembeddedExport(model, args.length, args.max_options).eval(),
            (token_embeddings, attention_bias, marker_map),
            [
                ct.TensorType(
                    name="input_embeddings",
                    shape=(1, args.length, model.config.hidden_size),
                    dtype=np.float32,
                ),
                ct.TensorType(name="attention_bias", shape=(1, 1, 1, args.length), dtype=np.float32),
                ct.TensorType(
                    name="class_marker_map",
                    shape=(1, args.max_options, args.length),
                    dtype=np.float32,
                ),
            ],
            np.float32,
        ),
        (
            "preembedded_io16",
            PreembeddedIO16Export(model, args.length, args.max_options).eval(),
            (token_embeddings.half(), attention_bias.half(), marker_map.half()),
            [
                ct.TensorType(
                    name="input_embeddings",
                    shape=(1, args.length, model.config.hidden_size),
                    dtype=np.float16,
                ),
                ct.TensorType(name="attention_bias", shape=(1, 1, 1, args.length), dtype=np.float16),
                ct.TensorType(
                    name="class_marker_map",
                    shape=(1, args.max_options, args.length),
                    dtype=np.float16,
                ),
            ],
            np.float16,
        ),
    ]
    for name, export, tensors, inputs, output_dtype in variants:
        with torch.no_grad():
            explicit = export(*tensors)[0][:, :4]
        wrapper_error = float((reference - explicit).abs().max())
        traced = torch.jit.trace(export, tensors)
        converted = ct.convert(
            traced,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            inputs=inputs,
            outputs=[
                ct.TensorType(name="logits", dtype=output_dtype),
                ct.TensorType(name="probabilities", dtype=output_dtype),
            ],
        )
        package = output / f"gliclass_edge_apps_{name}_fp16_L{args.length}_options{args.max_options}.mlpackage"
        converted.save(str(package))
        coreml = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
        prediction_inputs = {
            inputs[0].name: tensors[0].numpy(),
            "attention_bias": tensors[1].numpy(),
            "class_marker_map": tensors[2].numpy(),
        }
        predicted = np.asarray(coreml.predict(prediction_inputs)["logits"])[0, :4]
        coreml_error = float(np.max(np.abs(predicted - reference.numpy())))
        results[name] = {
            "package": str(package),
            "package_bytes": package_bytes(package),
            "wrapper_max_logit_error": wrapper_error,
            "coreml_max_logit_error": coreml_error,
            "argmax_match": int(predicted.argmax()) == int(reference.numpy().argmax()),
        }
    report = output / "conversion.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
