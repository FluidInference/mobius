"""Convert standalone Qwen3-0.6B to a fused stateful CoreML decoder (lmHead baked in).

Adapted from models/stt/qwen3-asr-0.6b/coreml/convert_decoder_fused.py — the same
proven Qwen3 decoder graph, but pointed at the standalone LLM (Qwen/Qwen3-0.6B) loaded
via AutoModelForCausalLM instead of the ASR "thinker.*" weight nesting.

Purpose: test the "LLM-on-ANE" thesis (see knowledge/coreml/ane-cpu-scheduled-matmul.md).
The output is a stateful decode graph (KV cache as MLState) producing logits [1, 1, VOCAB]
for the last position — this is the *decode* path, whose in-graph cache mutation is exactly
what the paper predicts the ANE rejects, so it is expected to land on CPU/GPU. Prefill is
naturally stateless; profile placement with `coreml-cli` after conversion.

Usage:
    uv run convert-coreml.py --output-dir ./build
    uv run convert-coreml.py --model-id Qwen/Qwen3-0.6B --max-seq-len 512
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Qwen3-0.6B architecture constants (identical to Qwen3-ASR-0.6B's text decoder)
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
VOCAB_SIZE = 151_936
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS  # 2


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class FusedStatefulQwen3Decoder(nn.Module):
    """28 Qwen3 decoder layers + final RMSNorm + lm_head, with stateful KV cache.

    Outputs logits for the last query position only (prefill projects one row).
    """

    def __init__(self, layers, final_norm, lm_head, max_seq_len: int = 512):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

        for i in range(NUM_LAYERS):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )

    def forward(self, hidden_states, position_cos, position_sin, attention_mask):
        q_len = hidden_states.shape[1]
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - q_len

        cos = position_cos.unsqueeze(1)
        sin = position_sin.unsqueeze(1)

        for i in range(NUM_LAYERS):
            layer = self.layers[i]
            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attn = layer.self_attn
            q = attn.q_proj(hidden_states)
            k = attn.k_proj(hidden_states)
            v = attn.v_proj(hidden_states)

            q = q.view(1, q_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            # Qwen3 has per-head q/k RMSNorm (Qwen2 does not)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            k_cache[:, :, past_kv_len:end_step, :] = k.half()
            v_cache[:, :, past_kv_len:end_step, :] = v.half()

            k_full = k_cache[:, :, :end_step, :].float()
            v_full = v_cache[:, :, :end_step, :].float()

            k_full = repeat_kv(k_full, GQA_REPEAT)
            v_full = repeat_kv(v_full, GQA_REPEAT)

            attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_full)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(1, q_len, NUM_Q_HEADS * HEAD_DIM)
            hidden_states = attn.o_proj(attn_output)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp = layer.mlp
            gate = mlp.gate_proj(hidden_states)
            up = mlp.up_proj(hidden_states)
            hidden_states = mlp.down_proj(F.silu(gate) * up)
            hidden_states = residual + hidden_states

        last_hidden = hidden_states[:, -1:, :]
        last_hidden = self.final_norm(last_hidden)
        logits = self.lm_head(last_hidden)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Convert standalone Qwen3-0.6B to stateful CoreML decoder")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--output-dir", default="./build")
    parser.add_argument("--compute-units", default="ALL",
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    args = parser.parse_args()

    MAX_SEQ_LEN = args.max_seq_len
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    print(f"Loading {args.model_id} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    base = model.model
    layers = base.layers
    final_norm = base.norm
    lm_head = model.lm_head  # tied to embed_tokens for Qwen3-0.6B

    # Sanity-check architecture matches our constants
    a0 = layers[0].self_attn
    assert len(layers) == NUM_LAYERS, f"{len(layers)} != {NUM_LAYERS}"
    assert a0.q_proj.out_features == NUM_Q_HEADS * HEAD_DIM
    assert a0.k_proj.out_features == NUM_KV_HEADS * HEAD_DIM
    assert lm_head.weight.shape == (VOCAB_SIZE, HIDDEN_SIZE)
    print(f"  layers={len(layers)} q_norm={hasattr(a0, 'q_norm')} "
          f"q_proj={a0.q_proj.in_features}->{a0.q_proj.out_features}")

    decoder = FusedStatefulQwen3Decoder(layers, final_norm, lm_head, max_seq_len=MAX_SEQ_LEN)
    decoder.eval()

    # Trace in decode shape (Q=1)
    hidden = torch.randn(1, 1, HIDDEN_SIZE)
    cos_in = torch.randn(1, 1, HEAD_DIM)
    sin_in = torch.randn(1, 1, HEAD_DIM)
    mask = torch.zeros(1, 1, 1, 5)
    print("Tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, (hidden, cos_in, sin_in, mask))
    traced.eval()

    import coremltools as ct
    print(f"coremltools {ct.__version__} — converting ...")

    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float32),
    ]
    outputs = [ct.TensorType("logits", dtype=np.float32)]
    states = []
    for i in range(NUM_LAYERS):
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float16),
            name=f"k_cache_{i}"))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float16),
            name=f"v_cache_{i}"))

    cu = getattr(ct.ComputeUnit, args.compute_units)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=cu,
    )
    print(f"  converted in {time.time() - t0:.1f}s")

    out_path = output_dir / "qwen3_0_6b_decoder_stateful.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
