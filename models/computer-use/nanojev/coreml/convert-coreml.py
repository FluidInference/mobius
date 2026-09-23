"""Export NanoJev's trained candidate encoder and set-dependent decision head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from assets import LOCK, ROOT, load_model
from export_model import NanoEncoder, NanoHead
from fixtures import fixture, requests
from preprocessing import prepare_request


def native_parity(root, tokenizer, model, encoder, head, length, candidates):
    max_error = 0.0
    for request in requests():
        inputs, candidate_mask, example = prepare_request(root, tokenizer, request, length, candidates)
        tensors = tuple(torch.from_numpy(value) for value in inputs.values())
        valid = torch.from_numpy(candidate_mask)
        typ = example["type"]
        set_flag = torch.tensor([[typ == "choice"]], dtype=torch.float32)
        bool_flag = torch.tensor([[typ == "boolean"]], dtype=torch.float32)
        with torch.no_grad():
            expected = model([example], tokenizer.pad_token_id)[0][0, : len(example["candidate_ids"])]
            embeddings = encoder(*tensors)
            actual = head(embeddings, valid, set_flag, bool_flag)[0][0, : len(example["candidate_ids"])]
        error = float(torch.max(torch.abs(expected - actual)))
        max_error = max(max_error, error)
        same = int(torch.argmax(expected)) == int(torch.argmax(actual))
        if not same or error > 1e-3:
            raise RuntimeError(f"NanoJev {typ} parity failed: argmax={same}, max logit error={error:.6f}")
        print(f"NanoJev {typ} parity: argmax={same}, max logit error={error:.8f}", flush=True)
    inputs, candidate_mask, _ = prepare_request(root, tokenizer, fixture(), length, candidates)
    return inputs, candidate_mask, max_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    if args.candidates < 2:
        raise ValueError("candidate bucket must be at least 2 for Boolean output")
    torch.set_num_threads(2)
    root, tokenizer, model = load_model()
    encoder = NanoEncoder(model, args.length, args.candidates).eval()
    head = NanoHead(model, args.candidates).eval()
    inputs, candidate_mask, error = native_parity(
        root, tokenizer, model, encoder, head, args.length, args.candidates
    )
    if args.parity_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    encoder_trace = torch.jit.trace(encoder, tuple(torch.from_numpy(value) for value in inputs.values()))
    if args.trace_only:
        for index, node in enumerate(encoder_trace.inlined_graph.nodes()):
            if node.kind() == "aten::Int":
                print(f"trace node {index}: {node} scope={node.scopeName()}", flush=True)
        return
    encoder_coreml = ct.convert(
        encoder_trace,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(args.candidates, args.length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(args.candidates, args.length), dtype=np.int32),
            ct.TensorType(name="eos_map", shape=(args.candidates, 1, args.length), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="embeddings", dtype=np.float32)],
    )
    encoder_coreml.short_description = "NanoJev trained Qwen3 candidate encoder"
    encoder_coreml.author = "NanoJev contributors; Qwen team; Fluid Inference (Core ML conversion)"
    encoder_path = args.output_dir / f"nanojev_encoder_fp16_L{args.length}_K{args.candidates}.mlpackage"
    encoder_coreml.save(str(encoder_path))

    embedding_example = torch.zeros(1, args.candidates, model.backbone.config.hidden_size)
    head_trace = torch.jit.trace(
        head,
        (embedding_example, torch.from_numpy(candidate_mask), torch.ones(1, 1), torch.zeros(1, 1)),
    )
    head_coreml = ct.convert(
        head_trace,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(
                name="embeddings", shape=(1, args.candidates, model.backbone.config.hidden_size), dtype=np.float32
            ),
            ct.TensorType(name="candidate_mask", shape=(1, args.candidates), dtype=np.float32),
            ct.TensorType(name="use_set_head", shape=(1, 1), dtype=np.float32),
            ct.TensorType(name="is_boolean", shape=(1, 1), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)],
    )
    head_coreml.short_description = "NanoJev trained scalar and set-dependent decision heads"
    head_coreml.author = encoder_coreml.author
    head_path = args.output_dir / f"nanojev_heads_fp16_K{args.candidates}.mlpackage"
    head_coreml.save(str(head_path))
    manifest = {
        "encoder": encoder_path.name,
        "head": head_path.name,
        "length": args.length,
        "candidates": args.candidates,
        "source": LOCK,
        "native_max_logit_error": error,
        "export_seconds": round(time.perf_counter() - started, 1),
        "minimum_target": "iOS17/macOS14",
        "redistribution": "source code only until trained weight license is clarified",
    }
    (args.output_dir / f"nanojev_L{args.length}_K{args.candidates}.conversion.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved {encoder_path} and {head_path}", flush=True)


if __name__ == "__main__":
    main()
