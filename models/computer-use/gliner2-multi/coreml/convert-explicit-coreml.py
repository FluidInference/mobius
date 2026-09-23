"""Export GLiNER2.5 multilingual trained explicit-span scorer for attributes and enums."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2 import AutoExtractor, Schema
from huggingface_hub import snapshot_download

from extraction_export import ExtractionExplicitSpanExport, ExtractionFeaturesExport, coreml_trace_patches
from preprocessing import prepare_extraction

MODEL_ID = "fastino/gliner2.5-multi-v1"
MODEL_REVISION = "a221b77a8baf4a613b8f8652661d41fa10a5641e"
INPUT_NAMES = (
    "text_states",
    "text_mask",
    "query_states",
    "query_mask",
    "boundary_states",
    "start_logits",
    "end_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "span_indices",
    "span_mask",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/extraction")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--max-spans", type=int, default=64)
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
    text = "Alice founded Acme in Toronto in 2020."
    schema = Schema().entities(["person", "organization", "location"])
    arrays, batch = prepare_extraction(native.processor, text, schema, args.length, args.max_words, args.max_queries)
    tensors = tuple(torch.from_numpy(value) for value in arrays.values())
    with torch.no_grad(), coreml_trace_patches():
        features = ExtractionFeaturesExport(native).eval()(*tensors)
    with torch.no_grad():
        core = native._encode_core(batch)
        candidates = native.boundary_head(
            core["text_states"], core["text_mask"], core["query_states"], core["query_mask"]
        ).candidates
    query_count = core["query_states"].shape[1]
    indices = torch.zeros(1, args.max_queries, args.max_spans, 2, dtype=torch.int32)
    mask = torch.zeros(1, args.max_queries, args.max_spans, dtype=torch.float32)
    count = min(args.max_spans, candidates.indices.shape[2])
    indices[:, :query_count, :count] = candidates.indices[:, :query_count, :count].int()
    mask[:, :query_count, :count] = candidates.valid_mask[:, :query_count, :count].float()
    wrapper = ExtractionExplicitSpanExport(native).eval()
    arguments = (
        features[0],
        tensors[3],
        features[1],
        tensors[5],
        features[2],
        features[4],
        features[5],
        features[6],
        features[7],
        indices,
        mask,
    )
    with torch.no_grad():
        reference = wrapper(*arguments)
        native_reference = native.boundary_head.score_explicit_spans(
            core["text_states"],
            core["text_mask"],
            core["query_states"],
            core["query_mask"],
            indices[:, :query_count].long(),
            mask[:, :query_count].bool(),
        )
        wrapper_error = float(
            (reference[:, :query_count][mask[:, :query_count].bool()] - native_reference[mask[:, :query_count].bool()])
            .abs()
            .max()
        )
        traced = torch.jit.trace(wrapper, arguments, check_trace=False)
    if wrapper_error > 1e-4:
        raise RuntimeError(f"Explicit span wrapper differs from native: {wrapper_error}")
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.int32 if name == "span_indices" else np.float32)
            for name, value in zip(INPUT_NAMES, arguments)
        ],
        outputs=[ct.TensorType(name="span_logits", dtype=np.float32)],
    )
    converted.short_description = "GLiNER2.5 multilingual trained explicit-span extraction scorer"
    converted.author = "Fastino (original); Fluid Inference (Core ML conversion)"
    converted.license = "Apache-2.0"
    converted.user_defined_metadata.update(
        {
            "source_model": MODEL_ID,
            "source_revision": MODEL_REVISION,
            "stage": "trained explicit-span proposal and reranker",
            "word_capacity": str(args.max_words),
            "query_capacity": str(args.max_queries),
            "span_capacity": str(args.max_spans),
        }
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.precision}_W{args.max_words}_Q{args.max_queries}_S{args.max_spans}"
    package = out / f"gliner2_multi_explicit_{suffix}.mlpackage"
    converted.save(str(package))
    runtime = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    predicted = runtime.predict(
        {
            name: value.detach().numpy().astype(np.int32 if name == "span_indices" else np.float32)
            for name, value in zip(INPUT_NAMES, arguments)
        }
    )["span_logits"]
    runtime_error = float(
        np.max(
            np.abs(
                predicted[:, :query_count][mask[:, :query_count].bool().numpy()]
                - reference.numpy()[:, :query_count][mask[:, :query_count].bool().numpy()]
            )
        )
    )
    if not np.isfinite(runtime_error):
        raise RuntimeError("Explicit-span scorer produced non-finite logits")
    report = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "precision": args.precision,
        "fixture": text,
        "valid_spans": int(mask.sum()),
        "wrapper_max_absolute_error": wrapper_error,
        "coreml_max_absolute_error": runtime_error,
        "package": str(package),
        "package_bytes": sum(file.stat().st_size for file in package.rglob("*") if file.is_file()),
        "coremltools": ct.__version__,
        "torch": torch.__version__,
    }
    (out / f"explicit-{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
