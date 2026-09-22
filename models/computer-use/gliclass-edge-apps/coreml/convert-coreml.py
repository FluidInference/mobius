"""Convert and parity-check the tuned GLiClass Edge model at a fixed shape."""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from gliclass import GLiClassModel
from gliclass.pipeline import UniEncoderZeroShotClassificationPipeline
from transformers import AutoTokenizer

from export_model import GLiClassExport


def prepare(tokenizer, model, text, labels, prompt, length, max_options):
    formatter = UniEncoderZeroShotClassificationPipeline(
        model,
        tokenizer,
        max_classes=max_options,
        max_length=length,
        classification_type="single-label",
        device="cpu",
        progress_bar=False,
    )
    rendered = formatter.prepare_input(text, labels, prompt=prompt)
    encoded = tokenizer(
        rendered,
        truncation=True,
        max_length=length,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = encoded["input_ids"].astype(np.int32)
    attention_mask = encoded["attention_mask"].astype(np.int32)
    positions = np.flatnonzero(input_ids[0] == model.config.class_token_index)
    if len(positions) != len(labels):
        raise ValueError(f"labels do not fit L{length}: expected {len(labels)} markers, found {len(positions)}")
    marker_map = np.zeros((1, max_options, length), dtype=np.float32)
    for index, position in enumerate(positions):
        marker_map[0, index, position] = 1.0
    return {"input_ids": input_ids, "attention_mask": attention_mask, "class_marker_map": marker_map}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="build/checkpoint-v2")
    parser.add_argument("--output", default="build/coreml")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=25)
    parser.add_argument("--target", choices=["iOS17", "iOS18"], default="iOS17")
    args = parser.parse_args()

    torch.set_num_threads(4)
    model = GLiClassModel.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    export = GLiClassExport(model, args.length, args.max_options).eval()
    labels = [
        "world news and international politics",
        "sports",
        "business and economy",
        "science and technology",
    ]
    arrays = prepare(
        tokenizer,
        model,
        '{"article":"Apple introduced a smaller processor for laptop computers."}',
        labels,
        "What is the topic of `article`?",
        args.length,
        args.max_options,
    )
    tensors = tuple(torch.from_numpy(value) for value in arrays.values())
    with torch.no_grad():
        reference = model(
            input_ids=tensors[0].long(),
            attention_mask=tensors[1].long(),
            max_num_classes=len(labels),
        ).logits
        explicit = export(*tensors)[0][:, : len(labels)]
    max_logit_error = float((reference - explicit).abs().max())
    if max_logit_error > 1e-4:
        raise RuntimeError(f"explicit wrapper parity failed: {max_logit_error}")
    print(f"PyTorch wrapper max logit error {max_logit_error:.8f}", flush=True)

    with torch.no_grad():
        traced = torch.jit.trace(export, tensors)
    started = time.perf_counter()
    deployment_target = {"iOS17": ct.target.iOS17, "iOS18": ct.target.iOS18}[args.target]
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=deployment_target,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, args.length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, args.length), dtype=np.int32),
            ct.TensorType(
                name="class_marker_map",
                shape=(1, args.max_options, args.length),
                dtype=np.float32,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="probabilities", dtype=np.float32),
        ],
    )
    converted.short_description = "Application-tuned GLiClass Edge: dynamic single-label decisions"
    converted.author = "Knowledgator (base); Fluid Inference (application tuning and Core ML conversion)"
    converted.license = "Apache-2.0"
    converted.user_defined_metadata.update(
        {
            "base_model": "knowledgator/gliclass-edge-v3.0",
            "length": str(args.length),
            "max_options": str(args.max_options),
            "minimum_deployment_target": args.target,
            "sequence_format": "labels + SEP + instruction + serialized state",
        }
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    package = output / f"gliclass_edge_apps_fp16_L{args.length}_options{args.max_options}.mlpackage"
    converted.save(str(package))

    coreml = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
    predicted = coreml.predict(arrays)
    coreml_logits = np.asarray(predicted["logits"])[:, : len(labels)]
    coreml_argmax = int(coreml_logits.argmax())
    reference_argmax = int(reference.numpy().argmax())
    coreml_error = float(np.max(np.abs(coreml_logits - reference.numpy())))
    if coreml_argmax != reference_argmax:
        raise RuntimeError(f"Core ML argmax mismatch: {coreml_argmax} != {reference_argmax}")
    summary = {
        "package": str(package),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "package_bytes": sum(path.stat().st_size for path in package.rglob("*") if path.is_file()),
        "length": args.length,
        "max_options": args.max_options,
        "minimum_deployment_target": args.target,
        "pytorch_wrapper_max_logit_error": max_logit_error,
        "coreml_max_logit_error": coreml_error,
        "argmax": coreml_argmax,
        "conversion_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "coremltools": ct.__version__,
    }
    (output / f"conversion-L{args.length}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
