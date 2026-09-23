"""Real-checkpoint parity for trained extraction graph wrappers."""

from pathlib import Path

import torch
from gliner2 import AutoExtractor, Schema
from gliner2.training.trainer import ExtractorCollator

from extraction_export import ExtractionFeaturesExport, ExtractionScoreExport, coreml_trace_patches
from extraction_pool import select_candidates
from preprocessing import prepare_extraction

SOURCE = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1/snapshots/1a8bc24e00dc7300b9017c81d63e3dcdabb26596"
)


def test_export_wrappers_match_native_entity_scores():
    torch.set_num_threads(4)
    native = AutoExtractor.from_pretrained(str(SOURCE), map_location="cpu").eval()
    text = "Alice founded Acme in Toronto in 2020."
    schema = Schema().entities(["person", "organization", "location"])
    batch = ExtractorCollator(native.processor, is_training=False, max_len=128, architecture=native.architecture)(
        [(text, schema.build())]
    )
    arrays, _ = prepare_extraction(native.processor, text, schema, 128, 64, 8)
    arguments = tuple(torch.from_numpy(value) for value in arrays.values())
    with torch.no_grad():
        core = native._encode_core(batch)
        expected = native.boundary_head(
            core["text_states"], core["text_mask"], core["query_states"], core["query_mask"]
        )
        with coreml_trace_patches():
            features = ExtractionFeaturesExport(native).eval()(*arguments)
        assert torch.allclose(features[0][:, : core["text_states"].shape[1]], core["text_states"], atol=1e-5)
        assert torch.allclose(features[1][:, : core["query_states"].shape[1]], core["query_states"], atol=1e-5)
        pool = select_candidates(
            features[8],
            features[9],
            features[3].bool(),
            arguments[5].bool(),
            features[4],
            features[5],
            boundary_top_k=native.boundary_head.shared_pool_builder.pool_boundary_top_k,
            pool_size=native.boundary_head.shared_pool_builder.pool_size,
            min_pool_per_query=native.boundary_head.shared_pool_builder.min_pool_per_query,
        )
        assert torch.equal(
            pool.indices.unsqueeze(1).expand_as(expected.candidates.indices), expected.candidates.indices
        )
        scores, candidate_states = ExtractionScoreExport(native).eval()(
            features[0],
            arguments[3],
            features[1],
            arguments[5],
            features[2],
            features[4],
            features[5],
            features[6],
            features[7],
            pool.indices.int(),
            pool.mask.float(),
            pool.compat_logits,
        )
        assert torch.allclose(scores[:, : core["query_states"].shape[1]], expected.candidates.pair_logits, atol=1e-4)
        assert torch.allclose(candidate_states.unsqueeze(1), expected.candidates.candidate_states[:, :1], atol=1e-4)
