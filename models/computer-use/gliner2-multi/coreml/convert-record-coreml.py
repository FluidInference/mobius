"""Export GLiNER2.5 multilingual trained record assignment and anchorless heads."""

import argparse
import json
import shutil
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2 import AutoExtractor, Schema
from gliner2.training.trainer import ExtractorCollator
from huggingface_hub import snapshot_download

from extraction_export import ExtractionRecordAnchorlessExport, ExtractionRecordAssignmentExport

MODEL_ID = "fastino/gliner2.5-multi-v1"
MODEL_REVISION = "a221b77a8baf4a613b8f8652661d41fa10a5641e"
FIXTURE_TEXT = "Alice works at Acme. Bob works at Beta."


def record_fixture(native, mode):
    schema = Schema()
    builder = schema.structure("employment", mode=mode, anchor="person" if mode == "natural" else None)
    builder.field("person", dtype="str")
    builder.field("company", dtype="str")
    batch = ExtractorCollator(native.processor, is_training=False, max_len=None, architecture="boundary")(
        [(FIXTURE_TEXT, schema.build())]
    )
    with torch.no_grad():
        core = native._encode_core(batch)
        candidates = native.boundary_head(
            core["text_states"], core["text_mask"], core["query_states"], core["query_mask"]
        ).candidates
        spec = next(iter(batch.record_specs[0].values()))
        group = native.record_decoder.forward_group(spec, core["query_states"][0], candidates, 0)
        field_states = [
            candidates.candidate_states[0, query_id][candidates.valid_mask[0, query_id]]
            for query_id in group.field_query_ids
        ]
        queries = core["query_states"][0][group.field_query_ids]
    return group, spec, field_states, queries


def pad_first(value, size: int):
    if value.shape[0] > size:
        raise ValueError(f"Real record fixture exceeds bucket capacity {size}")
    result = value.new_zeros((size, *value.shape[1:]))
    result[: value.shape[0]] = value
    return result


def package_bytes(path):
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/extraction")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--max-fields", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=192)
    parser.add_argument("--max-instances", type=int, default=1536)
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
    head = native.record_decoder
    if args.max_instances < args.max_fields * args.max_candidates:
        raise ValueError("Instance bucket must hold all latent field candidates")

    group, spec, field_states, queries = record_fixture(native, "natural")
    anchor_index = group.field_query_ids.index(spec.anchor_query_id)
    instances = field_states[anchor_index]
    hidden = instances.shape[-1]
    field_candidates = torch.zeros(args.max_fields, args.max_candidates, hidden)
    for field_index, states in enumerate(field_states):
        if states.shape[0] > args.max_candidates:
            raise ValueError("Record candidate count exceeds bucket")
        field_candidates[field_index, : states.shape[0]] = states
    assignment_args = (pad_first(instances, args.max_instances), pad_first(queries, args.max_fields), field_candidates)
    assignment_wrapper = ExtractionRecordAssignmentExport(native).eval()
    with torch.no_grad():
        assignment_reference = assignment_wrapper(*assignment_args)
        for field_index, expected in enumerate(group.assign_logits):
            actual = assignment_reference[0][: instances.shape[0], field_index, : expected.shape[1]]
            if not torch.allclose(actual, expected, atol=1e-4):
                raise RuntimeError("Record assignment wrapper differs from native")
        assignment_trace = torch.jit.trace(assignment_wrapper, assignment_args, check_trace=False)

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    assignment_model = ct.convert(
        assignment_trace,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name=name, shape=tuple(value.shape), dtype=np.float32)
            for name, value in zip(("instance_states", "field_queries", "field_candidate_states"), assignment_args)
        ],
        outputs=[
            ct.TensorType(name="assignment_logits", dtype=np.float32),
            ct.TensorType(name="object_logits", dtype=np.float32),
            ct.TensorType(name="latent_seed_logits", dtype=np.float32),
        ],
    )
    assignment_model.short_description = "GLiNER2.5 multilingual trained record assignment and object heads"
    assignment_model.author = "Fastino (original); Fluid Inference (Core ML conversion)"
    assignment_model.license = "Apache-2.0"
    assignment_model.user_defined_metadata.update(
        {
            "source_model": MODEL_ID,
            "source_revision": MODEL_REVISION,
            "stage": "trained record assignment, object and latent seed heads",
            "field_capacity": str(args.max_fields),
            "candidate_capacity": str(args.max_candidates),
            "instance_capacity": str(args.max_instances),
        }
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.precision}_F{args.max_fields}_C{args.max_candidates}_I{args.max_instances}"
    assignment_path = out / f"gliner2_multi_record_assignment_{suffix}.mlpackage"
    if assignment_path.exists():
        shutil.rmtree(assignment_path)
    assignment_model.save(str(assignment_path))
    assignment_runtime = ct.models.MLModel(str(assignment_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    assignment_prediction = assignment_runtime.predict(
        {
            name: value.numpy().astype(np.float32)
            for name, value in zip(("instance_states", "field_queries", "field_candidate_states"), assignment_args)
        }
    )
    assignment_errors = {
        name: float(np.max(np.abs(assignment_prediction[name] - expected.numpy())))
        for name, expected in zip(("assignment_logits", "object_logits", "latent_seed_logits"), assignment_reference)
    }

    _, _, anchorless_field_states, _ = record_fixture(native, "anchorless")
    context = torch.cat(anchorless_field_states, 0)
    context_size = args.max_fields * args.max_candidates
    context_states = pad_first(context, context_size)
    context_mask = torch.zeros(context_size, dtype=torch.float32)
    context_mask[: context.shape[0]] = 1.0
    anchorless_wrapper = ExtractionRecordAnchorlessExport(native).eval()
    with torch.no_grad():
        anchorless_reference = anchorless_wrapper(context_states, context_mask)
        native_states = head._anchorless_states(anchorless_field_states)
        wrapper_error = float((anchorless_reference - native_states).abs().max())
        if wrapper_error > 1e-4:
            raise RuntimeError(f"Anchorless wrapper differs from native: {wrapper_error}")
        anchorless_trace = torch.jit.trace(anchorless_wrapper, (context_states, context_mask), check_trace=False)
    anchorless_model = ct.convert(
        anchorless_trace,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="context_states", shape=tuple(context_states.shape), dtype=np.float32),
            ct.TensorType(name="context_mask", shape=tuple(context_mask.shape), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="instance_states", dtype=np.float32)],
    )
    anchorless_model.short_description = "GLiNER2.5 multilingual trained anchorless record instance head"
    anchorless_model.author = "Fastino (original); Fluid Inference (Core ML conversion)"
    anchorless_model.license = "Apache-2.0"
    anchorless_model.user_defined_metadata.update(
        {
            "source_model": MODEL_ID,
            "source_revision": MODEL_REVISION,
            "stage": "trained anchorless record instance head",
            "context_capacity": str(context_size),
        }
    )
    anchorless_path = out / f"gliner2_multi_record_anchorless_{suffix}.mlpackage"
    if anchorless_path.exists():
        shutil.rmtree(anchorless_path)
    anchorless_model.save(str(anchorless_path))
    anchorless_runtime = ct.models.MLModel(str(anchorless_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    anchorless_prediction = anchorless_runtime.predict(
        {
            "context_states": context_states.numpy().astype(np.float32),
            "context_mask": context_mask.numpy().astype(np.float32),
        }
    )["instance_states"]
    anchorless_error = float(np.max(np.abs(anchorless_prediction - anchorless_reference.numpy())))
    if not all(np.isfinite(value) for value in (*assignment_errors.values(), anchorless_error)):
        raise RuntimeError("Record head produced non-finite values")
    report = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "precision": args.precision,
        "fixture": FIXTURE_TEXT,
        "shape": {"fields": args.max_fields, "candidates": args.max_candidates, "instances": args.max_instances},
        "assignment_max_absolute_errors": assignment_errors,
        "anchorless_wrapper_max_absolute_error": wrapper_error,
        "anchorless_coreml_max_absolute_error": anchorless_error,
        "packages": {
            "assignment": {"path": str(assignment_path), "bytes": package_bytes(assignment_path)},
            "anchorless": {"path": str(anchorless_path), "bytes": package_bytes(anchorless_path)},
        },
        "coremltools": ct.__version__,
        "torch": torch.__version__,
    }
    (out / f"record-{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
