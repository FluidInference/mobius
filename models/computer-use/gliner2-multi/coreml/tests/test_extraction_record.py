"""Real-checkpoint parity of all three native record formation modes."""

from pathlib import Path

import torch
from gliner2 import AutoExtractor, Schema
from gliner2.training.trainer import ExtractorCollator

from extraction_export import ExtractionRecordAnchorlessExport, ExtractionRecordAssignmentExport

SOURCE = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/snapshots/a221b77a8baf4a613b8f8652661d41fa10a5641e"
)


def test_record_heads_match_native_natural_latent_and_anchorless():
    torch.set_num_threads(4)
    native = AutoExtractor.from_pretrained(str(SOURCE), map_location="cpu").eval()
    text = "Alice works at Acme. Bob works at Beta."
    for mode in ("natural", "latent", "anchorless"):
        schema = Schema()
        builder = schema.structure("employment", mode=mode, anchor="person" if mode == "natural" else None)
        builder.field("person", dtype="str")
        builder.field("company", dtype="str")
        batch = ExtractorCollator(native.processor, is_training=False, max_len=None, architecture="boundary")(
            [(text, schema.build())]
        )
        with torch.no_grad():
            core = native._encode_core(batch)
            candidates = native.boundary_head(
                core["text_states"], core["text_mask"], core["query_states"], core["query_mask"]
            ).candidates
            spec = next(iter(batch.record_specs[0].values()))
            group = native.record_decoder.forward_group(spec, core["query_states"][0], candidates, 0)
            field_states = []
            for query_id in group.field_query_ids:
                keep = candidates.valid_mask[0, query_id]
                field_states.append(candidates.candidate_states[0, query_id][keep])
            field_queries = core["query_states"][0][group.field_query_ids]
            max_candidates = max(states.shape[0] for states in field_states)
            padded_fields = torch.stack(
                [
                    torch.nn.functional.pad(states, (0, 0, 0, max_candidates - states.shape[0]))
                    for states in field_states
                ]
            )
            if mode == "natural":
                anchor_index = group.field_query_ids.index(spec.anchor_query_id)
                instances = field_states[anchor_index]
            elif mode == "latent":
                instances = torch.cat(field_states, 0)
            else:
                instances = native.record_decoder._anchorless_states(field_states)
                context = torch.cat(field_states, 0)
                predicted = ExtractionRecordAnchorlessExport(native).eval()(context, torch.ones(context.shape[0]))
                assert torch.allclose(predicted, instances, atol=1e-5)
            assignment, object_scores, latent_scores = ExtractionRecordAssignmentExport(native).eval()(
                instances, field_queries, padded_fields
            )
            for field_index, expected in enumerate(group.assign_logits):
                assert torch.allclose(assignment[:, field_index, : expected.shape[1]], expected, atol=1e-5)
            if mode == "anchorless":
                assert torch.allclose(object_scores, group.object_logits, atol=1e-5)
            if mode == "latent":
                assert torch.allclose(latent_scores, group.object_logits, atol=1e-5)
