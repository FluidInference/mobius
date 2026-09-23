"""Export the trained GLiNER2.5 base boundary extraction stages to Core ML."""

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2 import AutoExtractor, Schema
from huggingface_hub import snapshot_download

from convert_extraction_names import FEATURE_NAMES, SCORE_INPUT_NAMES
from extraction_export import ExtractionFeaturesExport, ExtractionScoreExport, coreml_trace_patches
from extraction_pool import select_candidates
from preprocessing import prepare_extraction

MODEL_ID = "fastino/gliner2.5-base-v1"
MODEL_REVISION = "1a8bc24e00dc7300b9017c81d63e3dcdabb26596"
FIXTURE_TEXT = "Alice founded Acme in Toronto in 2020."
FIXTURE_SCHEMA = Schema().entities(["person", "organization", "location"])


def package_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/extraction")
    parser.add_argument("--length", type=int, default=128, help="Subword capacity")
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
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
    arrays, batch = prepare_extraction(
        native.processor, FIXTURE_TEXT, FIXTURE_SCHEMA, args.length, args.max_words, args.max_queries
    )
    tensors = tuple(torch.from_numpy(value) for value in arrays.values())
    features_wrapper = ExtractionFeaturesExport(native).eval()
    with torch.no_grad(), coreml_trace_patches():
        features_reference = features_wrapper(*tensors)
        traced_features = torch.jit.trace(features_wrapper, tensors, check_trace=False)
    with torch.no_grad():
        native_core = native._encode_core(batch)
        valid_words = native_core["text_states"].shape[1]
        valid_queries = native_core["query_states"].shape[1]
        routing_error = max(
            float((features_reference[0][:, :valid_words] - native_core["text_states"]).abs().max()),
            float((features_reference[1][:, :valid_queries] - native_core["query_states"]).abs().max()),
        )
    if routing_error > 1e-4:
        raise RuntimeError(f"Traced routing differs from native: {routing_error}")

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    input_names = tuple(arrays)
    features_model = ct.convert(
        traced_features,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType(name=name, shape=arrays[name].shape, dtype=arrays[name].dtype) for name in input_names],
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in FEATURE_NAMES],
    )
    features_model.short_description = "GLiNER2.5 base trained boundary extraction features"
    features_model.author = "Fastino (original); Fluid Inference (Core ML conversion)"
    features_model.license = "Apache-2.0"
    features_model.user_defined_metadata.update(
        {
            "source_model": MODEL_ID,
            "source_revision": MODEL_REVISION,
            "stage": "extraction features and trained boundary heads",
            "subword_capacity": str(args.length),
            "word_capacity": str(args.max_words),
            "query_capacity": str(args.max_queries),
        }
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source) / "config.json", out / "config.json")
    tokenizer_dir = out / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(Path(source) / name, tokenizer_dir / name)
    suffix = f"{args.precision}_L{args.length}_W{args.max_words}_Q{args.max_queries}"
    features_path = out / f"gliner2_base_extraction_features_{suffix}.mlpackage"
    if features_path.exists():
        shutil.rmtree(features_path)
    features_model.save(str(features_path))
    print(f"Saved {features_path}", flush=True)

    head = native.boundary_head
    pooled = select_candidates(
        features_reference[8],
        features_reference[9],
        features_reference[3].bool(),
        tensors[5].bool(),
        features_reference[4],
        features_reference[5],
        boundary_top_k=head.shared_pool_builder.pool_boundary_top_k,
        pool_size=head.shared_pool_builder.pool_size,
        min_pool_per_query=head.shared_pool_builder.min_pool_per_query,
    )
    score_tensors = (
        features_reference[0],
        tensors[3],
        features_reference[1],
        tensors[5],
        features_reference[2],
        features_reference[4],
        features_reference[5],
        features_reference[6],
        features_reference[7],
        pooled.indices.int(),
        pooled.mask.float(),
        pooled.compat_logits,
    )
    scorer_wrapper = ExtractionScoreExport(native).eval()
    with torch.no_grad():
        scores_reference = scorer_wrapper(*score_tensors)
        traced_scores = torch.jit.trace(scorer_wrapper, score_tensors, check_trace=False)
    scorer_model = ct.convert(
        traced_scores,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name=name, shape=tuple(value.shape), dtype=np.int32 if name == "candidate_indices" else np.float32
            )
            for name, value in zip(SCORE_INPUT_NAMES, score_tensors)
        ],
        outputs=[
            ct.TensorType(name="pair_logits", dtype=np.float32),
            ct.TensorType(name="candidate_states", dtype=np.float32),
        ],
    )
    scorer_model.short_description = "GLiNER2.5 base trained shared-pool extraction scorer"
    scorer_model.author = "Fastino (original); Fluid Inference (Core ML conversion)"
    scorer_model.license = "Apache-2.0"
    scorer_model.user_defined_metadata.update(
        {
            "source_model": MODEL_ID,
            "source_revision": MODEL_REVISION,
            "stage": "trained extraction candidate scorer",
            "candidate_capacity": str(head.shared_pool_builder.pool_size),
        }
    )
    scorer_path = out / f"gliner2_base_extraction_scorer_{suffix}.mlpackage"
    if scorer_path.exists():
        shutil.rmtree(scorer_path)
    scorer_model.save(str(scorer_path))
    print(f"Saved {scorer_path}", flush=True)

    # The first runtime check uses the same selected real fixture as the trace.
    runtime_features = ct.models.MLModel(str(features_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    predicted_features = runtime_features.predict(arrays)
    errors = {
        name: float(np.max(np.abs(np.asarray(predicted_features[name]) - reference.detach().numpy())))
        for name, reference in zip(FEATURE_NAMES, features_reference)
    }
    if any(not np.isfinite(value) for value in errors.values()):
        raise RuntimeError("Extraction features contain non-finite values")
    runtime_scorer = ct.models.MLModel(str(scorer_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    score_arrays = {
        name: value.detach().numpy().astype(np.int32 if name == "candidate_indices" else np.float32)
        for name, value in zip(SCORE_INPUT_NAMES, score_tensors)
    }
    predicted_scores = runtime_scorer.predict(score_arrays)
    errors["pair_logits"] = float(np.max(np.abs(predicted_scores["pair_logits"] - scores_reference[0].numpy())))
    errors["candidate_states"] = float(
        np.max(np.abs(predicted_scores["candidate_states"] - scores_reference[1].numpy()))
    )
    if any(not np.isfinite(value) for value in errors.values()):
        raise RuntimeError("Extraction scorer contains non-finite values")
    report = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "precision": args.precision,
        "fixture": FIXTURE_TEXT,
        "shape": {
            "subwords": args.length,
            "words": args.max_words,
            "queries": args.max_queries,
            "candidates": head.shared_pool_builder.pool_size,
        },
        "routing_max_absolute_error": routing_error,
        "runtime_max_absolute_errors": errors,
        "packages": {
            "features": {"path": str(features_path), "bytes": package_bytes(features_path)},
            "scorer": {"path": str(scorer_path), "bytes": package_bytes(scorer_path)},
        },
        "coremltools": ct.__version__,
        "torch": torch.__version__,
    }
    (out / f"conversion-{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
