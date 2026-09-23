"""Small real-fixture native/Core ML parity check for entity extraction."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2 import AutoExtractor, Schema
from gliner2.models.boundary.model import _group_scored_candidates
from gliner2.models.outputs import CandidateTensorBatch
from huggingface_hub import snapshot_download

from convert_extraction_names import FEATURE_NAMES, SCORE_INPUT_NAMES
from extraction_pool import select_candidates
from preprocessing import prepare_extraction

MODEL_ID = "fastino/gliner2.5-base-v1"
MODEL_REVISION = "1a8bc24e00dc7300b9017c81d63e3dcdabb26596"
FIXTURES = [
    ("Alice founded Acme in Toronto in 2020.", ["person", "organization", "location"]),
    ("Apple acquired Beats for three billion dollars.", ["company", "product", "money"]),
    ("Marie Curie was born in Warsaw and worked in Paris.", ["person", "city"]),
]


def coreml_entity_result(native, features_model, scorer_model, text, schema, length, words, queries):
    arrays, batch = prepare_extraction(native.processor, text, schema, length, words, queries)
    features = features_model.predict(arrays)
    tensor = {name: torch.from_numpy(np.asarray(features[name]).copy()) for name in FEATURE_NAMES}
    text_mask = torch.from_numpy(arrays["text_mask"]).bool()
    query_mask = torch.from_numpy(arrays["query_mask"]).bool()
    head = native.boundary_head
    pool = select_candidates(
        tensor["pool_start_projection"],
        tensor["pool_end_projection"],
        tensor["boundary_mask"].bool(),
        query_mask,
        tensor["start_logits"],
        tensor["end_logits"],
        boundary_top_k=head.shared_pool_builder.pool_boundary_top_k,
        pool_size=head.shared_pool_builder.pool_size,
        min_pool_per_query=head.shared_pool_builder.min_pool_per_query,
    )
    score_values = (
        tensor["text_states"],
        text_mask.float(),
        tensor["query_states"],
        query_mask.float(),
        tensor["boundary_states"],
        tensor["start_logits"],
        tensor["end_logits"],
        tensor["inside_prefix"],
        tensor["inside_prefix_mean"],
        pool.indices.int(),
        pool.mask.float(),
        pool.compat_logits,
    )
    score_arrays = {
        name: value.numpy().astype(np.int32 if name == "candidate_indices" else np.float32)
        for name, value in zip(SCORE_INPUT_NAMES, score_values)
    }
    scores = scorer_model.predict(score_arrays)
    candidate_batch = CandidateTensorBatch(
        indices=pool.indices.unsqueeze(1).expand(1, queries, -1, 2),
        proposal_logits=pool.proposal_logits.unsqueeze(1).expand(1, queries, -1),
        pair_logits=torch.from_numpy(np.asarray(scores["pair_logits"]).copy()),
        valid_mask=pool.mask.unsqueeze(1).expand(1, queries, -1),
        query_mask=query_mask,
    )
    _, metadata = native._build_schema_dicts_and_metadata([schema])
    with torch.no_grad():
        core = native._encode_core(batch)
        native_output = native._extract_from_batch(batch, 0.5, metadata, True, True)[0]
    probs = torch.sigmoid(candidate_batch.pair_logits / native.boundary_settings.pair_temperature)
    grouped = _group_scored_candidates(
        candidate_batch,
        threshold=0.5,
        probabilities=probs,
        count_log_rates=tensor["count_log_rates"],
        adaptive_threshold=native.boundary_settings.adaptive_threshold,
    )
    extracted = native._decode_entities(
        0,
        core,
        core["ext_specs"][0],
        grouped[0],
        metadata[0],
        torch.sigmoid(tensor["null_logits"])[0],
        "flat",
        core["word_offsets"][0],
        batch.start_mappings[0],
        batch.end_mappings[0],
        text,
        len(batch.start_mappings[0]),
        True,
        True,
    )
    coreml_output = {"entities": [extracted]} if extracted else {}
    return native_output, coreml_output, pool, candidate_batch


def stripped_spans(result):
    output = {}
    for name, entities in result.get("entities", [{}])[0].items():
        output[name] = [(item["text"], item["start"], item["end"]) for item in entities]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    native = AutoExtractor.from_pretrained(str(source), map_location="cpu").eval()
    suffix = f"{args.precision}_L{args.length}_W{args.max_words}_Q{args.max_queries}"
    folder = Path(args.model_dir)
    features = ct.models.MLModel(
        str(folder / f"gliner2_base_extraction_features_{suffix}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    scorer = ct.models.MLModel(
        str(folder / f"gliner2_base_extraction_scorer_{suffix}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
    )
    cases = []
    for text, labels in FIXTURES:
        reference, actual, _, _ = coreml_entity_result(
            native,
            features,
            scorer,
            text,
            Schema().entities(labels),
            args.length,
            args.max_words,
            args.max_queries,
        )
        expected_spans = stripped_spans(reference)
        actual_spans = stripped_spans(actual)
        cases.append(
            {
                "text": text,
                "native_spans": expected_spans,
                "coreml_spans": actual_spans,
                "span_match": expected_spans == actual_spans,
            }
        )
    report = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "precision": args.precision,
        "selected_manifest": "three fixed real-text entity fixtures",
        "cases": cases,
        "span_matches": sum(case["span_match"] for case in cases),
    }
    path = folder / f"verify-entities-{suffix}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["span_matches"] != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
