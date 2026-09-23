"""GLiNER2.5 base extraction runtime using only Core ML trained weights."""

import json
from contextvars import ContextVar
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.engine import BoundaryExtractor
from gliner2.models.boundary.records import RecordGroupOutput
from gliner2.models.boundary.relations import (
    RelationProposalSettings,
    RelationTypeSpec,
    TypedRelationPairGenerator,
)
from gliner2.models.outputs import CandidateTensorBatch, ExtractorOutput

from convert_extraction_names import FEATURE_NAMES, SCORE_INPUT_NAMES
from extraction_pool import select_candidates
from preprocessing import load_processor, prepare_extraction

MODEL_PREFIX = "gliner2_base"


def to_tensor(value):
    return torch.from_numpy(np.asarray(value).copy())


def padded(value, size):
    if value.shape[0] > size:
        raise ValueError(f"Request exceeds Core ML bucket capacity {size}")
    result = value.new_zeros((size, *value.shape[1:]))
    result[: value.shape[0]] = value
    return result


class CoreMLBoundaryHead(torch.nn.Module):
    """Provide native decoder tensors from the converted extraction stages."""

    def __init__(self, context: ContextVar, explicit_model, max_queries: int, max_spans: int):
        super().__init__()
        self.context = context
        self.explicit_model = explicit_model
        self.max_queries = max_queries
        self.max_spans = max_spans

    def forward(self, text, text_mask, query, query_mask, return_candidates=True):
        state = self.context.get()
        features = state["features"]
        full = state["candidates"]
        count = query.shape[1]
        selected = CandidateTensorBatch(
            indices=full.indices[:, :count],
            proposal_logits=full.proposal_logits[:, :count],
            pair_logits=full.pair_logits[:, :count],
            valid_mask=full.valid_mask[:, :count],
            query_mask=full.query_mask[:, :count],
            candidate_states=full.candidate_states[:, :count],
        )
        return ExtractorOutput(
            candidates=selected if return_candidates else None,
            start_logits=features["start_logits"][:, :count],
            end_logits=features["end_logits"][:, :count],
            null_logits=features["null_logits"][:, :count],
            count_log_rates=features["count_log_rates"][:, :count],
            batch_size=1,
        )

    def score_explicit_spans(self, text, text_mask, query, query_mask, indices, valid_mask=None):
        state = self.context.get()
        features = state["features"]
        full_queries = features["query_states"][0]
        active_queries = int(state["arrays"]["query_mask"].sum())
        query_count = query.shape[1]
        span_count = indices.shape[2]
        if query_count > self.max_queries or span_count > self.max_spans:
            raise ValueError("Explicit-span request exceeds Core ML bucket capacity")
        selected = []
        for row in query[0]:
            equal = torch.isclose(full_queries[:active_queries], row, atol=1e-6, rtol=0).all(-1)
            matches = equal.nonzero(as_tuple=False).flatten()
            if matches.numel() != 1:
                raise ValueError("Explicit-span query cannot be mapped to the encoded schema")
            selected.append(int(matches[0]))
        selection = torch.tensor(selected, dtype=torch.long)
        query_states = padded(query[0], self.max_queries).unsqueeze(0)
        query_mask_padded = padded(query_mask[0].float(), self.max_queries).unsqueeze(0)

        def selected_feature(name):
            source = features[name][0].index_select(0, selection)
            return padded(source, self.max_queries).unsqueeze(0)

        span_indices = torch.zeros(1, self.max_queries, self.max_spans, 2, dtype=torch.int32)
        span_mask = torch.zeros(1, self.max_queries, self.max_spans, dtype=torch.float32)
        span_indices[:, :query_count, :span_count] = indices.int()
        span_mask[:, :query_count, :span_count] = valid_mask.float() if valid_mask is not None else 1.0
        values = (
            text,
            text_mask.float(),
            query_states,
            query_mask_padded,
            features["boundary_states"],
            selected_feature("start_logits"),
            selected_feature("end_logits"),
            selected_feature("inside_prefix"),
            selected_feature("inside_prefix_mean"),
            span_indices,
            span_mask,
        )
        names = (
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
        output = self.explicit_model.predict(
            {
                name: value.numpy().astype(np.int32 if name == "span_indices" else np.float32)
                for name, value in zip(names, values)
            }
        )["span_logits"]
        return to_tensor(output)[:, :query_count, :span_count]


class CoreMLRelationScorer(torch.nn.Module):
    """Call the trained Core ML relation graph after native pair selection."""

    def __init__(self, context: ContextVar, model, max_words: int, max_relations: int, pair_cap: int):
        super().__init__()
        self.context = context
        self.model = model
        self.max_words = max_words
        self.max_relations = max_relations
        self.pair_cap = pair_cap

    def forward(self, text, relation, candidates, pairs):
        count = len(pairs)
        if text.shape[1] != self.max_words or relation.shape[1] > self.max_relations or count > self.pair_cap:
            raise ValueError("Relation request exceeds Core ML bucket capacity")
        relation_states = torch.zeros(1, self.max_relations, relation.shape[-1], dtype=relation.dtype)
        relation_states[:, : relation.shape[1]] = relation
        state = self.context.get()
        text_length = int(state["arrays"]["text_mask"].sum())
        fields = (
            text,
            torch.tensor([text_length], dtype=torch.int32),
            relation_states,
            padded(pairs.batch_index.int(), self.pair_cap),
            padded(pairs.relation_index.int(), self.pair_cap),
            padded(pairs.head_start.int(), self.pair_cap),
            padded(pairs.head_end.int(), self.pair_cap),
            padded(pairs.tail_start.int(), self.pair_cap),
            padded(pairs.tail_end.int(), self.pair_cap),
            padded(pairs.pair_mask.float(), self.pair_cap),
        )
        names = (
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
        output = self.model.predict(
            {
                name: value.numpy().astype(
                    np.float32 if name in ("text_states", "relation_states", "pair_mask") else np.int32
                )
                for name, value in zip(names, fields)
            }
        )["relation_logits"]
        return to_tensor(output)[:count]


class CoreMLRecordHead(torch.nn.Module):
    """Keep GLiNER2's instance ordering while running learned layers in Core ML."""

    def __init__(self, assignment_model, anchorless_model, max_fields=8, max_candidates=192, max_instances=1536):
        super().__init__()
        self.assignment_model = assignment_model
        self.anchorless_model = anchorless_model
        self.max_fields = max_fields
        self.max_candidates = max_candidates
        self.max_instances = max_instances

    def forward_group(self, spec, query_states, candidates, sample_index):
        field_specs = list(spec.fields)
        field_query_ids = [field.query_id for field in field_specs]
        if len(field_specs) > self.max_fields:
            raise ValueError("Record has more fields than the Core ML bucket")
        query_count = query_states.shape[0]
        if any(query_id < 0 or query_id >= query_count for query_id in field_query_ids):
            raise ValueError("Record field query is outside the encoded schema")
        field_states, field_spans, field_logits, field_masks = [], [], [], []
        for query_id in field_query_ids:
            mask = candidates.valid_mask[sample_index, query_id]
            if int(mask.sum()) > self.max_candidates:
                raise ValueError("Record candidate count exceeds Core ML bucket")
            field_states.append(candidates.candidate_states[sample_index, query_id][mask])
            field_spans.append(candidates.indices[sample_index, query_id][mask])
            field_logits.append(candidates.pair_logits[sample_index, query_id][mask])
            field_masks.append(torch.ones(int(mask.sum()), dtype=torch.bool))

        instance_seed = []
        instance_spans = []
        if spec.mode == "natural":
            anchor = field_query_ids.index(spec.anchor_query_id)
            instances = field_states[anchor]
            for index, span in enumerate(field_spans[anchor]):
                instance_seed.append((anchor, index))
                instance_spans.append((int(span[0]), int(span[1])))
        elif spec.mode == "latent":
            instances = (
                torch.cat(field_states, 0) if field_states else query_states.new_zeros((0, query_states.shape[-1]))
            )
            for field_index, spans in enumerate(field_spans):
                for index, span in enumerate(spans):
                    instance_seed.append((field_index, index))
                    instance_spans.append((int(span[0]), int(span[1])))
        else:
            context_states = (
                torch.cat(field_states, 0) if field_states else query_states.new_zeros((0, query_states.shape[-1]))
            )
            context_capacity = self.max_fields * self.max_candidates
            context_mask = torch.zeros(context_capacity, dtype=torch.float32)
            context_mask[: context_states.shape[0]] = 1.0
            output = self.anchorless_model.predict(
                {
                    "context_states": padded(context_states, context_capacity).numpy().astype(np.float32),
                    "context_mask": context_mask.numpy(),
                }
            )["instance_states"]
            instances = to_tensor(output)
            instance_seed = [None] * instances.shape[0]
            instance_spans = [None] * instances.shape[0]

        count = instances.shape[0]
        if count > self.max_instances:
            raise ValueError("Record instance count exceeds Core ML bucket")
        hidden = query_states.shape[-1]
        candidate_states = torch.zeros(self.max_fields, self.max_candidates, hidden)
        for index, states in enumerate(field_states):
            candidate_states[index, : states.shape[0]] = states
        fields = (
            padded(instances, self.max_instances),
            padded(query_states[field_query_ids], self.max_fields),
            candidate_states,
        )
        predicted = self.assignment_model.predict(
            {
                name: value.numpy().astype(np.float32)
                for name, value in zip(("instance_states", "field_queries", "field_candidate_states"), fields)
            }
        )
        assignment = to_tensor(predicted["assignment_logits"])
        assignment_by_field = [
            assignment[:count, field_index, : 1 + states.shape[0]] for field_index, states in enumerate(field_states)
        ]
        if spec.mode == "natural":
            object_logits = field_logits[field_query_ids.index(spec.anchor_query_id)]
        elif spec.mode == "latent":
            object_logits = to_tensor(predicted["latent_seed_logits"])[:count]
        else:
            object_logits = to_tensor(predicted["object_logits"])[:count]
        return RecordGroupOutput(
            spec=spec,
            object_logits=object_logits,
            assign_logits=assignment_by_field,
            field_query_ids=field_query_ids,
            field_specs=field_specs,
            field_spans=field_spans,
            field_cand_mask=field_masks,
            field_cand_logits=field_logits,
            instance_seed=instance_seed,
            instance_spans=instance_spans,
        )


class CoreMLBoundaryExtractor(BoundaryExtractor):
    """Native GLiNER2 schema/decoder with all trained extraction heads in Core ML."""

    def __init__(
        self,
        model_dir: str,
        precision: str = "fp32",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        feature_package: str | None = None,
    ):
        if precision not in ("fp16", "fp32"):
            raise ValueError("precision must be fp16 or fp32")
        torch.nn.Module.__init__(self)
        folder = Path(model_dir)
        config = json.loads((folder / "config.json").read_text())
        self.boundary_settings = BoundaryHeadSettings(**config["boundary_head"])
        self.processor = load_processor(str(folder / "tokenizer"))
        self.enable_records = self.boundary_settings.enable_records
        self.enable_relations = self.boundary_settings.enable_relations
        self.strict_extraction = True
        self.length, self.max_words, self.max_queries = 128, 64, 8
        self._context = ContextVar("gliner2_coreml_extraction_context")
        suffix = f"{precision}_L128_W64_Q8"
        feature_path = (
            Path(feature_package) if feature_package else Path(f"{MODEL_PREFIX}_extraction_features_{suffix}.mlpackage")
        )
        if not feature_path.is_absolute():
            feature_path = folder / feature_path
        self.features_model = ct.models.MLModel(str(feature_path), compute_units=compute_units)
        self.scorer_model = ct.models.MLModel(
            str(folder / f"{MODEL_PREFIX}_extraction_scorer_{suffix}.mlpackage"), compute_units=compute_units
        )
        explicit = ct.models.MLModel(
            str(folder / f"{MODEL_PREFIX}_explicit_{precision}_W64_Q8_S64.mlpackage"), compute_units=compute_units
        )
        relation = ct.models.MLModel(
            str(folder / f"{MODEL_PREFIX}_relation_{precision}_W64_R4_P256.mlpackage"), compute_units=compute_units
        )
        assignment = ct.models.MLModel(
            str(folder / f"{MODEL_PREFIX}_record_assignment_{precision}_F8_C192_I1536.mlpackage"),
            compute_units=compute_units,
        )
        anchorless = ct.models.MLModel(
            str(folder / f"{MODEL_PREFIX}_record_anchorless_{precision}_F8_C192_I1536.mlpackage"),
            compute_units=compute_units,
        )
        self.boundary_head = CoreMLBoundaryHead(self._context, explicit, self.max_queries, 64)
        self.relation_scorer = CoreMLRelationScorer(self._context, relation, self.max_words, 4, 256)
        self.record_decoder = CoreMLRecordHead(assignment, anchorless)
        self.relation_pair_generator = TypedRelationPairGenerator(
            RelationProposalSettings(
                heads_per_relation=self.boundary_settings.relation_heads_per_type,
                tails_per_relation=self.boundary_settings.relation_tails_per_type,
                pair_cap=self.boundary_settings.relation_pair_cap,
                argument_threshold=self.boundary_settings.relation_argument_proposal_threshold,
            )
        )

    def _encode_core(self, batch):
        features = self._context.get()["features"]
        query_states = features["query_states"]
        text_states = features["text_states"]
        query_mask = to_tensor(self._context.get()["arrays"]["query_mask"]).bool()
        query_count = int(query_mask.sum())
        query_states = query_states[:, :query_count]
        query_mask = query_mask[:, :query_count]
        text_mask = to_tensor(self._context.get()["arrays"]["text_mask"]).bool()
        ext_specs, cls_specs, rel_specs, word_offsets = [], [], [], []
        for sample_index in range(len(batch)):
            specs = [
                {
                    "group_index": item.task_index,
                    "field_index": item.role_index,
                    "task_type": item.task_type,
                    "task_name": item.task_name,
                    "field_name": item.role_name,
                }
                for item in batch.query_layouts[sample_index].queries
            ]
            ext_specs.append(specs)
            classifications = []
            choice_offset = 0
            for group_index in range(batch.schema_counts[sample_index]):
                if batch.task_types[sample_index][group_index] != "classifications":
                    continue
                count = max(len(batch.schema_special_indices[sample_index][group_index]) - 1, 0)
                schema_tokens = batch.schema_tokens_list[sample_index][group_index]
                if count:
                    classifications.append(
                        {
                            "group_index": group_index,
                            "task_name": schema_tokens[2],
                            "schema_tokens": schema_tokens,
                            "group_embs": features["classification_logits"][
                                sample_index, choice_offset : choice_offset + count
                            ],
                        }
                    )
                choice_offset += count
            cls_specs.append(classifications)
            word_offsets.append(
                max(int(batch.text_word_counts[sample_index]) - len(batch.start_mappings[sample_index]), 0)
            )
            groups = {}
            for query_id, spec in enumerate(specs):
                groups.setdefault(spec["group_index"], []).append(query_id)
            relations = []
            for group_index, role_ids in groups.items():
                if batch.task_types[sample_index][group_index] != "relations" or len(role_ids) < 2:
                    continue
                head_id, tail_id = role_ids[:2]
                role_states = query_states[sample_index, [head_id, tail_id]]
                state = (
                    torch.cat((role_states[0], role_states[1]), -1)
                    if self.boundary_settings.directional_relation_states
                    else role_states.mean(0)
                )
                relations.append(
                    {
                        "group_index": group_index,
                        "relation_type": specs[head_id]["task_name"],
                        "spec": RelationTypeSpec(
                            specs[head_id]["task_name"], head_query_ids=(head_id,), tail_query_ids=(tail_id,)
                        ),
                        "query_state": state,
                    }
                )
            rel_specs.append(relations)
        return {
            "text_states": text_states,
            "text_mask": text_mask,
            "text_lengths": text_mask.sum(-1).long(),
            "query_states": query_states,
            "query_mask": query_mask,
            "ext_specs": ext_specs,
            "cls_specs": cls_specs,
            "rel_specs": rel_specs,
            "word_offsets": word_offsets,
        }

    def _extract_classification_result(self, results, schema_name, schema, embs, schema_tokens, temperature=1.0):
        cls_config = self._resolve_classification_config(schema_tokens[2], schema.get("classifications", []))
        if cls_config is None:
            return
        if temperature <= 0:
            raise ValueError("Classification temperature must be positive")
        logits = embs / temperature
        activation = cls_config.get("class_act", "auto")
        multi_label = cls_config.get("multi_label", False)
        if activation == "sigmoid" or (activation != "softmax" and multi_label):
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)
        labels = cls_config["labels"]
        if multi_label:
            threshold = cls_config.get("cls_threshold", 0.5)
            chosen = [
                (labels[index], float(probabilities[index]))
                for index in range(len(labels))
                if float(probabilities[index]) >= threshold
            ]
            if not chosen:
                best = int(probabilities.argmax())
                chosen = [(labels[best], float(probabilities[best]))]
            results[cls_config["task"]] = chosen
            return
        best = int(probabilities.argmax())
        results[cls_config["task"]] = (labels[best], float(probabilities[best]))

    def extract(
        self,
        text: str,
        schema,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        overlap_policy=None,
    ):
        schema_dicts, metadata_list = self._build_schema_dicts_and_metadata([schema])
        if overlap_policy is not None:
            metadata_list[0]["_overlap_policy"] = self._resolved_overlap_policy(overlap_policy)
        arrays, batch = prepare_extraction(
            self.processor, text, schema_dicts[0], self.length, self.max_words, self.max_queries
        )
        predicted = self.features_model.predict(arrays)
        features = {name: to_tensor(predicted[name]) for name in FEATURE_NAMES}
        query_mask = to_tensor(arrays["query_mask"]).bool()
        candidates = None
        if bool(query_mask.any()):
            head = self.boundary_settings
            pool = select_candidates(
                features["pool_start_projection"],
                features["pool_end_projection"],
                features["boundary_mask"].bool(),
                query_mask,
                features["start_logits"],
                features["end_logits"],
                boundary_top_k=head.pool_boundary_top_k,
                pool_size=head.pool_size,
                min_pool_per_query=head.min_pool_per_query,
            )
            values = (
                features["text_states"],
                to_tensor(arrays["text_mask"]),
                features["query_states"],
                to_tensor(arrays["query_mask"]),
                features["boundary_states"],
                features["start_logits"],
                features["end_logits"],
                features["inside_prefix"],
                features["inside_prefix_mean"],
                pool.indices.int(),
                pool.mask.float(),
                pool.compat_logits,
            )
            scored = self.scorer_model.predict(
                {
                    name: value.numpy().astype(np.int32 if name == "candidate_indices" else np.float32)
                    for name, value in zip(SCORE_INPUT_NAMES, values)
                }
            )
            candidates = CandidateTensorBatch(
                indices=pool.indices.unsqueeze(1).expand(1, self.max_queries, -1, 2),
                proposal_logits=pool.proposal_logits.unsqueeze(1).expand(1, self.max_queries, -1),
                pair_logits=to_tensor(scored["pair_logits"]),
                valid_mask=pool.mask.unsqueeze(1).expand(1, self.max_queries, -1),
                query_mask=query_mask,
                candidate_states=to_tensor(scored["candidate_states"]).unsqueeze(1).expand(1, self.max_queries, -1, -1),
            )
        token = self._context.set({"arrays": arrays, "features": features, "candidates": candidates})
        try:
            raw = self._extract_from_batch(batch, threshold, metadata_list, include_confidence, include_spans)[0]
            if format_results:
                return self.format_results(
                    raw,
                    include_confidence,
                    metadata_list[0].get("relation_order", []),
                    metadata_list[0].get("classification_tasks", []),
                )
            return raw
        finally:
            self._context.reset(token)
