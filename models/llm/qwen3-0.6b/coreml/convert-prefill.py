"""Convert Qwen3-0.6B as a STATELESS prefill graph — the positive LLM-on-ANE test.

The decode graph (convert-coreml.py) mutates a KV cache in-graph and is ANE-rejected
(ANECCompile -14), matching Surgical Inference §6.3. Prefill has no cache to persist: it
computes K/V for the whole prompt, uses them immediately, and discards them. That makes it
a stateless Dense-Static graph — the case the paper predicts the ANE *accepts*.

Fixed sequence length (static shape, no RangeDim) so the ANE compiler has everything at
compile time. Input is embeddings + RoPE tables + causal mask (host-computed, same split as
the runtime); output is last-position logits [1, 1, VOCAB].

Usage:
    uv run convert-prefill.py --seq-len 128 --output-dir ./build
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
VOCAB_SIZE = 151_936
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(h, n_rep):
    if n_rep == 1:
        return h
    b, kv, s, d = h.shape
    return h[:, :, None, :, :].expand(b, kv, n_rep, s, d).reshape(b, kv * n_rep, s, d)


class StatelessPrefillQwen3(nn.Module):
    """28 Qwen3 layers + final norm + lm_head, no KV cache. Last-position logits only."""

    def __init__(self, layers, final_norm, lm_head):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, hidden_states, position_cos, position_sin, attention_mask):
        q_len = hidden_states.shape[1]
        cos = position_cos.unsqueeze(1)
        sin = position_sin.unsqueeze(1)

        for i in range(NUM_LAYERS):
            layer = self.layers[i]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attn = layer.self_attn
            q = attn.q_proj(hidden_states).view(1, q_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
            k = attn.k_proj(hidden_states).view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = attn.v_proj(hidden_states).view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            k = repeat_kv(k, GQA_REPEAT)
            v = repeat_kv(v, GQA_REPEAT)

            attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

            attn_output = attn_output.transpose(1, 2).contiguous().view(1, q_len, NUM_Q_HEADS * HEAD_DIM)
            hidden_states = attn.o_proj(attn_output)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp = layer.mlp
            hidden_states = mlp.down_proj(F.silu(mlp.gate_proj(hidden_states)) * mlp.up_proj(hidden_states))
            hidden_states = residual + hidden_states

        last_hidden = self.final_norm(hidden_states[:, -1:, :])
        return self.lm_head(last_hidden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output-dir", default="./build")
    args = parser.parse_args()

    S = args.seq_len
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    print(f"Loading {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
    model.eval()
    base = model.model
    decoder = StatelessPrefillQwen3(base.layers, base.norm, model.lm_head)
    decoder.eval()

    hidden = torch.randn(1, S, HIDDEN_SIZE)
    cos_in = torch.randn(1, S, HEAD_DIM)
    sin_in = torch.randn(1, S, HEAD_DIM)
    mask = torch.triu(torch.full((1, 1, S, S), -1e4), diagonal=1)

    print(f"Tracing (static seq_len={S}) ...")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, (hidden, cos_in, sin_in, mask))
    traced.eval()

    import coremltools as ct
    print(f"coremltools {ct.__version__} — converting stateless prefill (fixed shapes) ...")
    inputs = [
        ct.TensorType("hidden_states", shape=(1, S, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, S, HEAD_DIM), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, S, HEAD_DIM), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, S, S), dtype=np.float32),
    ]
    outputs = [ct.TensorType("logits", dtype=np.float32)]

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  converted in {time.time() - t0:.1f}s")

    out_path = output_dir / f"qwen3_0_6b_prefill_s{S}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
