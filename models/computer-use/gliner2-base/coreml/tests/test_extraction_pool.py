"""Real-checkpoint parity for the weight-free candidate selection stage."""

from pathlib import Path

import torch
from gliner2 import AutoExtractor, Schema
from gliner2.training.trainer import ExtractorCollator

from extraction_pool import select_candidates

SOURCE = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1/snapshots/1a8bc24e00dc7300b9017c81d63e3dcdabb26596"
)


def test_candidate_pool_matches_native_with_real_weights():
    torch.set_num_threads(4)
    native = AutoExtractor.from_pretrained(str(SOURCE), map_location="cpu").eval()
    fixtures = [
        ("Alice founded Acme in Toronto in 2020.", ["person", "organization", "location"]),
        ("Apple acquired Beats for three billion dollars.", ["company", "product", "money"]),
    ]
    for text, labels in fixtures:
        schema = Schema().entities(labels)
        batch = ExtractorCollator(native.processor, is_training=False, max_len=128, architecture=native.architecture)(
            [(text, schema.build())]
        )
        with torch.no_grad():
            core = native._encode_core(batch)
            head = native.boundary_head
            encoded = head.boundary_encoder(core["text_states"], core["text_mask"])
            marginals = head.boundary_query_head(
                encoded.states,
                encoded.mask,
                core["text_states"],
                core["text_mask"],
                core["query_states"],
                core["query_mask"],
            )
            expected = head.shared_pool_builder(
                encoded.states,
                encoded.mask,
                core["query_mask"],
                marginals.start_logits,
                marginals.end_logits,
            )
            actual = select_candidates(
                head.shared_pool_builder.start_projection(encoded.states),
                head.shared_pool_builder.end_projection(encoded.states),
                encoded.mask,
                core["query_mask"],
                marginals.start_logits,
                marginals.end_logits,
                boundary_top_k=head.shared_pool_builder.pool_boundary_top_k,
                pool_size=head.shared_pool_builder.pool_size,
                min_pool_per_query=head.shared_pool_builder.min_pool_per_query,
            )
        assert torch.equal(actual.indices, expected.indices)
        assert torch.equal(actual.mask, expected.mask)
        assert torch.allclose(actual.compat_logits, expected.compat_logits, atol=1e-5)
        assert torch.allclose(actual.proposal_logits, expected.proposal_logits, atol=1e-5)
