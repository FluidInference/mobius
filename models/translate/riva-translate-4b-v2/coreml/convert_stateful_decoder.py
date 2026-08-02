"""Convert Riva-Translate-4B-Instruct-v2 decoder to a stateful CoreML model.

Probe conversion for on-device feasibility. Architecture is standard Mistral
(MistralForCausalLM, 34 layers, hidden 3072, 32Q/8KV heads, head_dim 128,
GQA x4, vocab 131072, tied embeddings). Pattern follows
models/stt/qwen3-asr-0.6b/coreml/convert_stateful_decoder.py, with two
memory-driven differences for a 4.2B model on a 24GB host:

  - the model is loaded and traced in fp16 end to end (no fp32 master copy)
  - embed_tokens and lm_head stay OUT of the decoder graph; embedding lookup
    happens host-side and lm_head converts as a separate model

Usage:
    uv run convert_stateful_decoder.py --output-dir ./out
    uv run convert_stateful_decoder.py --max-seq-len 1024 --skip-lm-head
"""

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.4",
#     "transformers>=4.48",
#     "coremltools>=8.0",
#     "numpy<2",
#     "safetensors",
#     "huggingface_hub",
# ]
# ///

import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Riva-Translate-4B-Instruct-v2 architecture constants (from config.json)
NUM_LAYERS = 34
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 8640
VOCAB_SIZE = 131_072
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS  # 4


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for grouped query attention: [B, H_kv, S, D] -> [B, H_q, S, D]."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class StatefulMistralDecoder(nn.Module):
    """Mistral decoder stack with stateful KV cache for CoreML export.

    Runs entirely in fp16. Final RMSNorm is NOT applied here — it lives in
    the lm_head model.
    """

    def __init__(self, layers: nn.ModuleList, max_seq_len: int):
        super().__init__()
        self.layers = layers
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            hidden_states: [1, Q, 3072] fp16 — input embeddings
            position_cos:  [1, Q, 128]  fp16 — RoPE cosines for query positions
            position_sin:  [1, Q, 128]  fp16 — RoPE sines
            attention_mask: [1, 1, Q, end_step] fp16 (0=attend, -inf-ish=ignore)
        Returns:
            [1, Q, 3072] fp16 — decoder output (pre final norm)
        """
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
            q = attn.q_proj(hidden_states)  # [1, Q, 32*128=4096]
            k = attn.k_proj(hidden_states)  # [1, Q, 8*128=1024]
            v = attn.v_proj(hidden_states)  # [1, Q, 8*128=1024]

            q = q.view(1, q_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            k_cache[:, :, past_kv_len:end_step, :] = k
            v_cache[:, :, past_kv_len:end_step, :] = v

            k_full = k_cache[:, :, :end_step, :]
            v_full = v_cache[:, :, :end_step, :]

            k_full = repeat_kv(k_full, GQA_REPEAT)  # [1, 32, end_step, 128]
            v_full = repeat_kv(v_full, GQA_REPEAT)

            attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_full)  # [1, 32, Q, 128]

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

        return hidden_states


class LmHead(nn.Module):
    """Final RMSNorm + tied-embedding lm_head projection."""

    def __init__(self, norm: nn.Module, weight: torch.Tensor):
        super().__init__()
        self.norm = norm
        self.proj = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden_states))


def main():
    parser = argparse.ArgumentParser(description="Convert Riva-Translate-4B decoder to stateful CoreML")
    parser.add_argument("--model-id", default="nvidia/Riva-Translate-4B-Instruct-v2")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length for KV cache")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    args = parser.parse_args()

    MAX_SEQ_LEN = args.max_seq_len
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import coremltools as ct

    print(f"torch {torch.__version__}, coremltools {ct.__version__}")

    # ---- Step 1: Load model in fp16 (no fp32 master copy — 24GB host) ----
    print(f"Loading {args.model_id} in fp16...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    layers = model.model.layers
    assert len(layers) == NUM_LAYERS
    attn0 = layers[0].self_attn
    assert attn0.q_proj.out_features == NUM_Q_HEADS * HEAD_DIM
    assert attn0.k_proj.out_features == NUM_KV_HEADS * HEAD_DIM
    assert not hasattr(attn0, "q_norm"), "unexpected QK norms for Mistral"

    # ---- Step 2: lm_head model (norm + tied embedding projection) ----
    if not args.skip_lm_head:
        print("\nConverting lm_head (norm + 3072x131072 projection)...")
        lm_head = LmHead(model.model.norm, model.model.embed_tokens.weight.detach())
        lm_head.eval().half()
        ex = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch.float16)
        with torch.no_grad():
            traced_head = torch.jit.trace(lm_head, ex)
        head_ml = ct.convert(
            traced_head,
            inputs=[ct.TensorType("hidden_states", shape=(1, 1, HIDDEN_SIZE), dtype=np.float16)],
            outputs=[ct.TensorType("logits", dtype=np.float16)],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        head_path = output_dir / "riva4b_lm_head.mlpackage"
        head_ml.save(str(head_path))
        print(f"Saved {head_path}")
        del lm_head, traced_head, head_ml
        gc.collect()

    if args.skip_decoder:
        return

    # ---- Step 3: Stateful decoder ----
    # Free everything except the layer stack before tracing.
    embed_path = output_dir / "embed_tokens_fp16.npy"
    if not embed_path.exists():
        np.save(embed_path, model.model.embed_tokens.weight.detach().numpy())
        print(f"Saved host-side embedding table to {embed_path}")
    model.model.embed_tokens = None
    model.lm_head = None
    gc.collect()

    print(f"\nCreating stateful decoder (max_seq_len={MAX_SEQ_LEN})...")
    stateful_model = StatefulMistralDecoder(layers, max_seq_len=MAX_SEQ_LEN)
    stateful_model.eval()

    trace_q, trace_end = 1, 5
    hidden = torch.randn(1, trace_q, HIDDEN_SIZE, dtype=torch.float16)
    cos_in = torch.randn(1, trace_q, HEAD_DIM, dtype=torch.float16)
    sin_in = torch.randn(1, trace_q, HEAD_DIM, dtype=torch.float16)
    mask = torch.zeros(1, 1, trace_q, trace_end, dtype=torch.float16)

    print("Tracing (fp16, CPU)...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(stateful_model, (hidden, cos_in, sin_in, mask))
    traced.eval()
    print(f"Trace complete in {time.time() - t0:.1f}s")

    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)

    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, HIDDEN_SIZE), dtype=np.float16),
        ct.TensorType("position_cos", shape=(1, query_length, HEAD_DIM), dtype=np.float16),
        ct.TensorType("position_sin", shape=(1, query_length, HEAD_DIM), dtype=np.float16),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float16),
    ]
    outputs = [ct.TensorType("output_hidden", dtype=np.float16)]

    states = []
    for i in range(NUM_LAYERS):
        for kv in ("k", "v"):
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float16
                    ),
                    name=f"{kv}_cache_{i}",
                )
            )

    print("Converting decoder to CoreML (this is the slow, memory-heavy step)...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    print(f"CoreML conversion complete in {time.time() - t0:.1f}s")

    out_path = output_dir / "riva4b_decoder_stateful.mlpackage"
    mlmodel.save(str(out_path))
    print(f"Saved {out_path}")

    # ---- Step 4: smoke test ----
    print("\nSmoke test (decode step Q=1)...")
    state = mlmodel.make_state()
    out = mlmodel.predict(
        {
            "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float16),
            "position_cos": np.random.randn(1, 1, HEAD_DIM).astype(np.float16),
            "position_sin": np.random.randn(1, 1, HEAD_DIM).astype(np.float16),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float16),
        },
        state=state,
    )
    arr = out["output_hidden"]
    print(f"  output shape {arr.shape}, range [{np.min(arr):.3f}, {np.max(arr):.3f}]")
    print("Done.")


if __name__ == "__main__":
    main()
