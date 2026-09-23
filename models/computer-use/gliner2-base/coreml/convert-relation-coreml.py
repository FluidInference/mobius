"""Export the trained GLiNER2.5 base sparse relation scorer to Core ML."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2 import AutoExtractor, Schema
from gliner2.models.base import QueryLayout
from gliner2.training.trainer import ExtractorCollator
from huggingface_hub import snapshot_download

from extraction_export import ExtractionRelationExport

MODEL_ID = "fastino/gliner2.5-base-v1"
MODEL_REVISION = "1a8bc24e00dc7300b9017c81d63e3dcdabb26596"
INPUT_NAMES = (
    "text_states",
    "text_length",
    "relation_states",
    "batch_index",
    "relation_index",
    "head_start",
    "head_end",
    "tail_start",
    "tail_end",
    "pair_mask",
)


def pad(value, size: int, fill=0):
    if value.shape[0] > size:
        raise ValueError(f"Relation fixture exceeds capacity {size}")
    output = value.new_full((size, *value.shape[1:]), fill)
    output[: value.shape[0]] = value
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/extraction")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--max-relations", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=[
            "config.json",
            "encoder_config/*",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
    native = AutoExtractor.from_pretrained(source, map_location="cpu").eval()
    text = "Alice founded Acme in Toronto."
    schema = Schema().relations(["founded"])
    batch = ExtractorCollator(native.processor, is_training=False, max_len=None, architecture="boundary")(
        [(text, schema.build())]
    )
    with torch.no_grad():
        core = native._encode_core(batch)
        output = native.boundary_head(core["text_states"], core["text_mask"], core["query_states"], core["query_mask"])
        sample = native._single_sample_candidates(output.candidates, 0)
        relation_specs = core["rel_specs"][0]
        pairs = native.relation_pair_generator.generate_batched(
            sample, [QueryLayout(queries=())], [[entry["spec"] for entry in relation_specs]], compact=False
        )
        relation_states = torch.stack([entry["query_state"] for entry in relation_specs]).unsqueeze(0)
        native_scores = native.relation_scorer(core["text_states"], relation_states, sample, pairs)
    pair_cap = args.max_relations * native.boundary_settings.relation_pair_cap
    text_states = torch.zeros(1, args.max_words, core["text_states"].shape[-1])
    text_states[:, : core["text_states"].shape[1]] = core["text_states"]
    relation_padded = torch.zeros(1, args.max_relations, relation_states.shape[-1])
    relation_padded[:, : relation_states.shape[1]] = relation_states
    arguments = (
        text_states,
        torch.tensor([core["text_states"].shape[1]], dtype=torch.int32),
        relation_padded,
        pad(pairs.batch_index.int(), pair_cap),
        pad(pairs.relation_index.int(), pair_cap),
        pad(pairs.head_start.int(), pair_cap),
        pad(pairs.head_end.int(), pair_cap),
        pad(pairs.tail_start.int(), pair_cap),
        pad(pairs.tail_end.int(), pair_cap),
        pad(pairs.pair_mask.float(), pair_cap),
    )
    wrapper = ExtractionRelationExport(native).eval()
    with torch.no_grad():
        reference = wrapper(*arguments)
        wrapper_error = float((reference[: len(pairs)] - native_scores).abs().max())
        traced = torch.jit.trace(wrapper, arguments, check_trace=False)
    if wrapper_error > 1e-4:
        raise RuntimeError(f"Relation wrapper differs from native: {wrapper_error}")
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name=name,
                shape=tuple(value.shape),
                dtype=np.float32 if name in ("text_states", "relation_states", "pair_mask") else np.int32,
            )
            for name, value in zip(INPUT_NAMES, arguments)
        ],
        outputs=[ct.TensorType(name="relation_logits", dtype=np.float32)],
    )
    converted.short_description = "GLiNER2.5 base trained sparse relation scoring head"
    converted.author = "Fastino (original); Fluid Inference (Core ML conversion)"
    converted.license = "Apache-2.0"
    converted.user_defined_metadata.update(
        {
            "source_model": MODEL_ID,
            "source_revision": MODEL_REVISION,
            "stage": "trained relation scorer",
            "word_capacity": str(args.max_words),
            "relation_capacity": str(args.max_relations),
            "pair_capacity": str(pair_cap),
        }
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.precision}_W{args.max_words}_R{args.max_relations}_P{pair_cap}"
    package = out / f"gliner2_base_relation_{suffix}.mlpackage"
    converted.save(str(package))
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    prediction = model.predict(
        {
            name: value.numpy().astype(
                np.float32 if name in ("text_states", "relation_states", "pair_mask") else np.int32
            )
            for name, value in zip(INPUT_NAMES, arguments)
        }
    )["relation_logits"]
    runtime_error = float(np.max(np.abs(prediction[: len(pairs)] - reference.numpy()[: len(pairs)])))
    if not np.isfinite(runtime_error):
        raise RuntimeError("Relation scorer produced non-finite values")
    report = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "precision": args.precision,
        "fixture": text,
        "valid_pairs": int(pairs.pair_mask.sum()),
        "wrapper_max_absolute_error": wrapper_error,
        "coreml_max_absolute_error": runtime_error,
        "package": str(package),
        "package_bytes": sum(file.stat().st_size for file in package.rglob("*") if file.is_file()),
        "coremltools": ct.__version__,
        "torch": torch.__version__,
    }
    (out / f"relation-{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
