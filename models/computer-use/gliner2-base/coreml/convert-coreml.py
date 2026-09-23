"""Convert and verify the pinned GLiNER2.5-base classification decision path."""
import argparse
import json
import math
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download
from transformers.models.deberta_v2 import modeling_deberta_v2

from export_model import GLiNER2ClassificationExport, coreml_safe_attention_forward
from preprocessing import native_batch, prepare_classification

MODEL_ID = "fastino/gliner2.5-base-v1"
MODEL_REVISION = "1a8bc24e00dc7300b9017c81d63e3dcdabb26596"
EXAMPLES = [
    ("The rocket launched successfully.", "topic", ["science", "sports", "politics"]),
    ("The team won the football championship.", "topic", ["science", "sports", "politics"]),
    ("The budget was approved by parliament.", "topic", ["science", "sports", "politics"]),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=[
            "config.json", "encoder_config/*", "model.safetensors", "tokenizer.json", "tokenizer_config.json"
        ],
    )
    native = AutoExtractor.from_pretrained(source, map_location="cpu").eval()
    wrapper = GLiNER2ClassificationExport(native).eval()
    text, task, labels = EXAMPLES[0]
    arrays = prepare_classification(native, text, task, labels, args.length, args.max_options)
    tensors = tuple(torch.from_numpy(value) for value in arrays.values())
    with torch.no_grad():
        batch = native_batch(native, text, task, labels, args.length)
        core = native._encode_core(batch)
        expected = native.classifier(core["cls_specs"][0][0]["choice_states"]).squeeze(-1)
        actual = wrapper(*tensors)[0][0, : len(labels)]
        wrapper_error = float((expected - actual).abs().max())
    if wrapper_error > 1e-4:
        raise RuntimeError(f"Wrapper/native logit mismatch: {wrapper_error}")
    # The upstream scale is a constant for a fixed DeBERTa attention head width.
    # Its traced int32 sqrt is rejected by Core ML; freeze the identical float32
    # value while tracing, and restore the upstream implementation immediately.
    original_scale = modeling_deberta_v2.scaled_size_sqrt
    original_rpos = modeling_deberta_v2.build_rpos
    original_attention = modeling_deberta_v2.DisentangledSelfAttention.forward

    def static_scale(query_layer, scale_factor):
        value = math.sqrt(float(query_layer.shape[-1] * scale_factor))
        return torch.tensor(value, dtype=torch.float32, device=query_layer.device)

    modeling_deberta_v2.scaled_size_sqrt = static_scale
    # The encoder only uses self-attention: query and key sequence lengths are
    # identical, so the scripted build_rpos returns relative_pos unchanged.
    # Freeze that branch to avoid a Core ML conditional with mismatched ranks.
    modeling_deberta_v2.build_rpos = lambda query, key, relative_pos, buckets, max_pos: relative_pos
    modeling_deberta_v2.DisentangledSelfAttention.forward = coreml_safe_attention_forward
    try:
        with torch.no_grad():
            frozen = wrapper(*tensors)[0][0, : len(labels)]
            frozen_error = float((expected - frozen).abs().max())
            if frozen_error > 1e-4:
                raise RuntimeError(f"Frozen attention scale changed native logits: {frozen_error}")
            traced = torch.jit.trace(wrapper, tensors, check_trace=False)
    finally:
        modeling_deberta_v2.scaled_size_sqrt = original_scale
        modeling_deberta_v2.build_rpos = original_rpos
        modeling_deberta_v2.DisentangledSelfAttention.forward = original_attention
    converted = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, args.length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, args.length), dtype=np.int32),
            ct.TensorType(name="marker_indices", shape=(1, args.max_options), dtype=np.int32),
            ct.TensorType(name="marker_mask", shape=(1, args.max_options), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)],
    )
    converted.short_description = "GLiNER2.5-base native schema classification path"
    converted.author = "Fastino (original); Fluid Inference (Core ML conversion)"
    converted.license = "Apache-2.0"
    converted.user_defined_metadata.update({
        "source_model": MODEL_ID, "source_revision": MODEL_REVISION,
        "scope": "classification only; entity/relation/record extraction heads not exported",
        "length": str(args.length), "max_options": str(args.max_options),
    })
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    package = out / f"gliner2_base_classification_{args.precision}_L{args.length}_K{args.max_options}.mlpackage"
    converted.save(str(package))
    runtime = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
    cases = []
    for text, task, labels in EXAMPLES:
        arrays = prepare_classification(native, text, task, labels, args.length, args.max_options)
        native_output = native.classify_text(text, {task: labels}, include_confidence=True, max_len=args.length)[task]
        prediction = runtime.predict(arrays)
        scores = np.asarray(prediction["probabilities"])[0, : len(labels)]
        choice = labels[int(scores.argmax())]
        if choice != native_output["label"]:
            raise RuntimeError(f"Core ML/native choice mismatch: {choice} != {native_output['label']}")
        cases.append({
            "text": text, "native_label": native_output["label"], "coreml_label": choice,
            "native_confidence": native_output["confidence"], "coreml_confidence": float(scores.max()),
            "absolute_confidence_error": abs(float(scores.max()) - native_output["confidence"]),
        })
    report = {
        "source_model": MODEL_ID, "source_revision": MODEL_REVISION, "package": str(package),
        "package_bytes": sum(f.stat().st_size for f in package.rglob("*") if f.is_file()),
        "native_total_parameters": sum(p.numel() for p in native.parameters()),
        "exported_parameters": sum(p.numel() for p in wrapper.parameters()),
        "wrapper_max_logit_error": wrapper_error, "coremltools": ct.__version__,
        "torch": torch.__version__, "cases": cases,
    }
    (out / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
