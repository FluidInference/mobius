"""Export pinned RLCD full-candidate likelihood scoring to fixed-shape Core ML."""

from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as functional
from transformers.activations import ACT2FN
from transformers.models.lfm2 import modeling_lfm2

from decision import SOURCE_REVISION, CandidateScorer, load_native
from fixtures import CONTEXT, SCHEMA
from preprocessing import Shape, batch_arrays, prepare_candidates
from release_integrity import package_file_hashes


@contextmanager
def fixed_trace_convolution(model):
    """Freeze only the known convolution dimensions during static-shape tracing.

    Transformers 5.16 computes conv padding from a traced tensor shape. Torch
    2.7 forwards that shape as an empty padding list to conv1d. The checkpoint
    fixes kernel width and hidden channels, so explicit integers reproduce its
    original functional convolution without changing any trained weights.
    """
    original = modeling_lfm2.causal_conv1d_fn
    padding = model.config.conv_L_cache - 1
    groups = model.config.hidden_size

    def static_convolution(hidden_states, weight, bias=None, activation=None, **kwargs):
        sequence_length = hidden_states.shape[-1]
        output = functional.conv1d(
            hidden_states.to(weight.dtype),
            weight=weight.unsqueeze(1),
            bias=bias,
            padding=padding,
            groups=groups,
        )[:, :, :sequence_length]
        if activation is not None:
            output = ACT2FN[activation](output)
        return output.to(hidden_states.dtype)

    modeling_lfm2.causal_conv1d_fn = static_convolution
    try:
        yield
    finally:
        modeling_lfm2.causal_conv1d_fn = original


@contextmanager
def fixed_trace_attention(batch: int, length: int):
    """Use equivalent fixed dimensions in the eager LFM attention graph.

    The upstream module obtains reshape dimensions from traced tensor shapes.
    Core ML Tools 9 cannot lower the resulting non-scalar ``aten::Int`` nodes.
    All dimensions below are already fixed by this conversion bucket.
    """
    original = modeling_lfm2.Lfm2Attention.forward

    def static_attention(module, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        if past_key_values is not None:
            raise ValueError("Core ML export does not use an attention cache")
        heads = module.config.num_attention_heads
        key_heads = module.config.num_key_value_heads
        head_dim = module.head_dim
        repetitions = module.num_key_value_groups

        query = module.q_layernorm(module.q_proj(hidden_states).reshape(batch, length, heads, head_dim)).transpose(1, 2)
        key = module.k_layernorm(module.k_proj(hidden_states).reshape(batch, length, key_heads, head_dim)).transpose(
            1, 2
        )
        value = module.v_proj(hidden_states).reshape(batch, length, key_heads, head_dim).transpose(1, 2)
        cosine, sine = position_embeddings
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        half = head_dim // 2
        query = query * cosine + torch.cat((-query[..., half:], query[..., :half]), dim=-1) * sine
        key = key * cosine + torch.cat((-key[..., half:], key[..., :half]), dim=-1) * sine
        if repetitions != 1:
            key = key.unsqueeze(2).expand(batch, key_heads, repetitions, length, head_dim)
            key = key.reshape(batch, heads, length, head_dim)
            value = value.unsqueeze(2).expand(batch, key_heads, repetitions, length, head_dim)
            value = value.reshape(batch, heads, length, head_dim)

        weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        probabilities = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, length, heads * head_dim)
        return module.out_proj(output.contiguous()), probabilities

    modeling_lfm2.Lfm2Attention.forward = static_attention
    try:
        yield
    finally:
        modeling_lfm2.Lfm2Attention.forward = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-value-tokens", type=int, default=16)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--stable-logprob", action="store_true", help="Try gathered logit minus logsumexp")
    parser.add_argument("--fp32-logsumexp", action="store_true", help="Keep logsumexp in FP32 for the stable variant")
    args = parser.parse_args()
    if args.fp32_logsumexp and (not args.stable_logprob or args.precision != "fp16"):
        parser.error("--fp32-logsumexp requires --stable-logprob and --precision fp16")
    torch.set_num_threads(2)

    shape = Shape(args.length, args.batch, args.max_value_tokens)
    tokenizer, model = load_native("cpu")
    candidates = prepare_candidates(tokenizer, CONTEXT, SCHEMA, shape)
    arrays = batch_arrays(tokenizer, candidates, shape)
    example = tuple(torch.from_numpy(array) for array in arrays.values())
    wrapper = CandidateScorer(model, shape.length, stable_logprob=args.stable_logprob).eval()
    with torch.inference_mode():
        reference = wrapper(*example)
        with fixed_trace_convolution(model), fixed_trace_attention(shape.candidates, shape.length):
            patched = wrapper(*example)
            patch_error = float((patched - reference).abs().max())
            if patch_error > 1e-5:
                raise RuntimeError(f"static trace operators differ from native: {patch_error}")
            traced = torch.jit.trace(wrapper, example)
        trace_output = traced(*example)
    trace_error = float((trace_output - reference).abs().max())
    if trace_error > 1e-4:
        raise RuntimeError(f"trace changes native logits: {trace_error}")
    print(f"PyTorch trace parity: {trace_error:.8f}", flush=True)
    if args.trace_only:
        nodes = list(traced.inlined_graph.nodes())
        first = next(index for index, node in enumerate(nodes) if node.kind() == "aten::Int")
        for node in nodes[first - 15 : first + 15]:
            print(node, flush=True)
        return

    start = time.perf_counter()
    compute_precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    if args.fp32_logsumexp:
        compute_precision = ct.transform.FP16ComputePrecision(
            op_selector=lambda op: op.op_type != "reduce_log_sum_exp"
        )
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(shape.candidates, shape.length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(shape.candidates, shape.length), dtype=np.int32),
            ct.TensorType(name="value_positions", shape=(shape.candidates, shape.max_value_tokens), dtype=np.int32),
            ct.TensorType(name="value_targets", shape=(shape.candidates, shape.max_value_tokens), dtype=np.int32),
            ct.TensorType(name="value_mask", shape=(shape.candidates, shape.max_value_tokens), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="scores", dtype=np.float32)],
    )
    converted.short_description = "LFM2.5-350M-RLCD full candidate log-likelihood scoring, no mutable cache"
    converted.author = "Liquid AI; RLCD implementation by notnotsamuel; Core ML conversion by Fluid Inference"
    converted.license = "LFM Open License v1.0"
    converted.user_defined_metadata.update(
        {
            "checkpoint_repo": "notnotsamuel/LFM2.5-350M-RLCD",
            "checkpoint_revision": "deb589d803d141cabd158ef55f6617b128529f36",
            "length": str(shape.length),
            "candidate_batch": str(shape.candidates),
            "max_value_tokens": str(shape.max_value_tokens),
            "scoring": "sum of full candidate value token log likelihoods",
            "cache": "none; shared prompt recomputed for each candidate",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant = "_stable_fp32lse" if args.fp32_logsumexp else "_stable" if args.stable_logprob else ""
    path = args.output_dir / (
        f"lfm350_rlcd_{args.precision}{variant}_L{shape.length}_B{shape.candidates}_V{shape.max_value_tokens}.mlpackage"
    )
    converted.save(str(path))
    manifest = {
        "model": path.name,
        "source_revision": SOURCE_REVISION,
        "precision": args.precision,
        "logprob_mode": "target_minus_logsumexp" if args.stable_logprob else "log_softmax_gather",
        "fp32_logsumexp": args.fp32_logsumexp,
        "length": shape.length,
        "candidate_batch": shape.candidates,
        "max_value_tokens": shape.max_value_tokens,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trace_error": trace_error,
        "conversion_seconds": round(time.perf_counter() - start, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "package_files_sha256": package_file_hashes(path),
    }
    (args.output_dir / f"{path.stem}.conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
