"""Convert Qwen3-ASR decoder to a fused stateful CoreML model with lmHead baked in.

Same as convert_stateful_decoder.py but fuses the final RMSNorm + lm_head linear
projection into the decoder. This eliminates a separate CoreML prediction call for
lmHead (~8.5ms/tok overhead).

The output is logits [1, 1, VOCAB_SIZE] instead of hidden states [1, Q, HIDDEN_SIZE].
Only the last position is projected to logits (saves compute during prefill).

Usage:
    uv run convert_decoder_fused.py --output-dir /path/to/output
    uv run convert_decoder_fused.py --model-id Qwen/Qwen3-ASR-0.6B --max-seq-len 512
"""

# /// script
# requires-python = ">=3.10"
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
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Qwen3-ASR-0.6B architecture constants
NUM_LAYERS = 28
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
VOCAB_SIZE = 151_936
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS  # 2


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE rotation using concatenated-halves layout."""
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


class FusedStatefulQwen3Decoder(nn.Module):
    """Qwen3-ASR decoder with fused lmHead and stateful KV cache for CoreML export.

    This wraps the 28 transformer decoder layers, final RMSNorm, and lm_head
    linear projection. Outputs logits for the last query position only.

    Compared to the non-fused version, this eliminates a separate CoreML
    prediction call for the lm_head model (~8.5ms/tok savings).
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        final_norm: nn.Module,
        lm_head: nn.Linear,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

        # Register 56 state buffers (28 layers x K + V)
        # CoreML states MUST be fp16 — we cast to/from fp32 during computation
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
        """Run decoder layers with in-place KV cache updates, then project to logits.

        Args:
            hidden_states: [1, Q, 1024] — input embeddings (Q tokens)
            position_cos:  [1, Q, 128]  — RoPE cosines for query positions
            position_sin:  [1, Q, 128]  — RoPE sines for query positions
            attention_mask: [1, 1, Q, end_step] — causal mask (0=attend, -1e9=ignore)
                end_step = past_kv_len + Q (total valid sequence length)

        Returns:
            logits: [1, 1, 151936] — vocab logits for the last query position
        """
        q_len = hidden_states.shape[1]
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - q_len

        # Expand RoPE for multi-head broadcasting: [1, Q, 128] -> [1, 1, Q, 128]
        cos = position_cos.unsqueeze(1)
        sin = position_sin.unsqueeze(1)

        for i in range(NUM_LAYERS):
            layer = self.layers[i]
            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")

            # --- Pre-attention LayerNorm ---
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            # --- Q, K, V projections ---
            attn = layer.self_attn
            q = attn.q_proj(hidden_states)  # [1, Q, 16*128=2048]
            k = attn.k_proj(hidden_states)  # [1, Q, 8*128=1024]
            v = attn.v_proj(hidden_states)  # [1, Q, 8*128=1024]

            # Reshape to multi-head: [1, Q, H*D] -> [1, H, Q, D]
            q = q.view(1, q_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            # QK norms (Qwen3 has these; Qwen2 doesn't)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            # --- Apply RoPE ---
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

            # --- In-place KV cache update (CoreML detects as state mutation) ---
            # Cast fp32 -> fp16 for storage (CoreML states must be fp16)
            k_cache[:, :, past_kv_len:end_step, :] = k.half()
            v_cache[:, :, past_kv_len:end_step, :] = v.half()

            # Read valid cache entries and cast back to fp32 for attention math
            k_full = k_cache[:, :, :end_step, :].float()  # [1, 8, end_step, 128]
            v_full = v_cache[:, :, :end_step, :].float()  # [1, 8, end_step, 128]

            # GQA: expand KV heads (8 -> 16)
            k_full = repeat_kv(k_full, GQA_REPEAT)  # [1, 16, end_step, 128]
            v_full = repeat_kv(v_full, GQA_REPEAT)  # [1, 16, end_step, 128]

            # --- Scaled dot-product attention (manual for coremltools) ---
            attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v_full)  # [1, 16, Q, 128]

            # Reshape: [1, 16, Q, 128] -> [1, Q, 2048] -> o_proj -> [1, Q, 1024]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(1, q_len, NUM_Q_HEADS * HEAD_DIM)
            hidden_states = attn.o_proj(attn_output)

            # Residual connection
            hidden_states = residual + hidden_states

            # --- Post-attention LayerNorm + MLP (SwiGLU) ---
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            mlp = layer.mlp
            gate = mlp.gate_proj(hidden_states)  # [1, Q, 3072]
            up = mlp.up_proj(hidden_states)  # [1, Q, 3072]
            hidden_states = mlp.down_proj(F.silu(gate) * up)  # [1, Q, 1024]

            # Residual connection
            hidden_states = residual + hidden_states

        # --- Fused lmHead: slice last position, apply RMSNorm + linear ---
        # Only project the last position to logits (saves compute during prefill)
        last_hidden = hidden_states[:, -1:, :]  # [1, 1, 1024]
        last_hidden = self.final_norm(last_hidden)  # [1, 1, 1024]
        logits = self.lm_head(last_hidden)  # [1, 1, 151936]
        return logits


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-ASR decoder (fused with lmHead) to stateful CoreML")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B", help="HuggingFace model ID")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length for KV cache")
    parser.add_argument("--output-dir", default=".", help="Output directory for .mlpackage")
    parser.add_argument("--skip-validation", action="store_true", help="Skip PyTorch validation")
    args = parser.parse_args()

    MAX_SEQ_LEN = args.max_seq_len
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load model weights from HuggingFace ----
    print(f"Loading model: {args.model_id}")
    t0 = time.time()

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3Model

    # Create Qwen3 text decoder config (matches Qwen3-ASR-0.6B's text_config)
    text_config = Qwen3Config(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=65_536,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        hidden_act="silu",
        attention_bias=False,
        tie_word_embeddings=True,
    )

    # Create empty Qwen3 text model and load weights
    text_model = Qwen3Model(text_config)
    text_model.eval()

    # Download safetensors and remap keys: thinker.model.X -> X
    st_path = hf_hub_download(args.model_id, "model.safetensors")
    all_weights = load_file(st_path)

    decoder_weights = {}
    for k, v in all_weights.items():
        if k.startswith("thinker.model."):
            new_key = k[len("thinker.model."):]  # layers.0.xxx, norm.weight, embed_tokens.weight
            decoder_weights[new_key] = v.float()  # ensure float32

    missing, unexpected = text_model.load_state_dict(decoder_weights, strict=False)
    print(f"Weights loaded in {time.time() - t0:.1f}s")
    print(f"  Loaded: {len(decoder_weights)} tensors")
    if missing:
        print(f"  Missing (expected): {len(missing)} — {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)} — {unexpected[:5]}...")

    # ---- Step 2: Extract decoder layers + final norm ----
    layers = text_model.layers
    final_norm = text_model.norm  # Final RMSNorm before lm_head
    print(f"\nFound {len(layers)} decoder layers")
    print(f"  Final norm: {type(final_norm).__name__} (weight shape: {final_norm.weight.shape})")

    # Verify architecture
    layer0 = layers[0]
    attn0 = layer0.self_attn
    q_out = attn0.q_proj.out_features
    k_out = attn0.k_proj.out_features
    has_qk_norm = hasattr(attn0, "q_norm")

    print(f"  q_proj: {attn0.q_proj.in_features} -> {q_out} (expect {NUM_Q_HEADS * HEAD_DIM})")
    print(f"  k_proj: {attn0.k_proj.in_features} -> {k_out} (expect {NUM_KV_HEADS * HEAD_DIM})")
    print(f"  QK norms: {has_qk_norm}")
    print(f"  MLP gate_proj: {layer0.mlp.gate_proj.in_features} -> {layer0.mlp.gate_proj.out_features}")

    assert len(layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, got {len(layers)}"
    assert q_out == NUM_Q_HEADS * HEAD_DIM, f"Q projection mismatch: {q_out} != {NUM_Q_HEADS * HEAD_DIM}"
    assert k_out == NUM_KV_HEADS * HEAD_DIM, f"K projection mismatch: {k_out} != {NUM_KV_HEADS * HEAD_DIM}"

    # ---- Step 3: Create lm_head linear layer ----
    # Qwen3-ASR uses tie_word_embeddings=True, so lm_head.weight == embed_tokens.weight
    # Check if thinker.lm_head.weight exists separately, otherwise use embed_tokens
    lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
    if "thinker.lm_head.weight" in all_weights:
        lm_head_weight = all_weights["thinker.lm_head.weight"].float()
        print(f"\n  lm_head: loaded from thinker.lm_head.weight ({lm_head_weight.shape})")
    else:
        # Tied weights — use embed_tokens.weight
        lm_head_weight = text_model.embed_tokens.weight.data.float()
        print(f"\n  lm_head: tied to embed_tokens.weight ({lm_head_weight.shape})")
    lm_head.weight = nn.Parameter(lm_head_weight)
    lm_head.eval()

    assert lm_head.weight.shape == (VOCAB_SIZE, HIDDEN_SIZE), \
        f"lm_head weight shape mismatch: {lm_head.weight.shape} != ({VOCAB_SIZE}, {HIDDEN_SIZE})"

    del all_weights, decoder_weights  # Free memory

    # ---- Step 4: Create fused stateful wrapper ----
    print(f"\nCreating fused stateful decoder (max_seq_len={MAX_SEQ_LEN})...")
    stateful_model = FusedStatefulQwen3Decoder(
        layers, final_norm, lm_head, max_seq_len=MAX_SEQ_LEN
    )
    stateful_model.eval()

    # ---- Step 5: Trace the model ----
    # Trace with small representative shapes (decode mode: Q=1, end_step=5)
    trace_q = 1
    trace_end = 5

    hidden = torch.randn(1, trace_q, HIDDEN_SIZE)
    cos_in = torch.randn(1, trace_q, HEAD_DIM)
    sin_in = torch.randn(1, trace_q, HEAD_DIM)
    mask = torch.zeros(1, 1, trace_q, trace_end)

    print("Tracing model...")
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(stateful_model, (hidden, cos_in, sin_in, mask))
    traced.eval()
    print(f"Trace complete in {time.time() - t0:.1f}s")

    # ---- Step 6: Validate traced model ----
    if not args.skip_validation:
        print("\nValidating traced vs eager...")
        # Create a fresh instance to avoid state contamination
        stateful_ref = FusedStatefulQwen3Decoder(
            layers, final_norm, lm_head, max_seq_len=MAX_SEQ_LEN
        )
        stateful_ref.eval()

        test_hidden = torch.randn(1, 1, HIDDEN_SIZE)
        test_cos = torch.randn(1, 1, HEAD_DIM)
        test_sin = torch.randn(1, 1, HEAD_DIM)
        test_mask = torch.zeros(1, 1, 1, 3)

        with torch.no_grad():
            ref_out = stateful_ref(test_hidden, test_cos, test_sin, test_mask)
            traced_out = traced(test_hidden, test_cos, test_sin, test_mask)
            diff = (ref_out - traced_out).abs().max().item()

        print(f"  Max diff (traced vs eager): {diff:.6e}")
        print(f"  Output shape: {ref_out.shape} (expect [1, 1, {VOCAB_SIZE}])")
        if diff > 1e-3:
            print(f"  WARNING: Large divergence! Check tracing compatibility.")
        else:
            print(f"  OK — traced model matches eager mode")

    # ---- Step 7: Convert to CoreML ----
    print("\nConverting to CoreML...")
    import coremltools as ct

    print(f"  coremltools version: {ct.__version__}")

    query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)

    inputs = [
        ct.TensorType("hidden_states", shape=(1, query_length, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("position_cos", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("position_sin", shape=(1, query_length, HEAD_DIM), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, query_length, end_step_dim), dtype=np.float32),
    ]

    # Output is now logits [1, 1, VOCAB_SIZE] — always last position only
    outputs = [
        ct.TensorType("logits", dtype=np.float32),
    ]

    # 56 state buffers: 28 layers x (K + V)
    states = []
    for i in range(NUM_LAYERS):
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float16
                ),
                name=f"k_cache_{i}",
            )
        )
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float16
                ),
                name=f"v_cache_{i}",
            )
        )

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

    # ---- Step 8: Save ----
    output_path = output_dir / "qwen3_asr_decoder_stateful.mlpackage"
    mlmodel.save(str(output_path))
    print(f"\nSaved to: {output_path}")

    # ---- Step 9: Quick CoreML validation ----
    print("\nValidating CoreML model...")
    try:
        state = mlmodel.make_state()
        test_input = {
            "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
            "position_cos": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "position_sin": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
        }
        output = mlmodel.predict(test_input, state=state)
        out_arr = output["logits"]
        print(f"  Decode output shape: {out_arr.shape} (expect (1, 1, {VOCAB_SIZE}))")
        print(f"  Output range: [{out_arr.min():.4f}, {out_arr.max():.4f}]")

        # Test with prefill shape (Q=5, end_step=5)
        test_prefill = {
            "hidden_states": np.random.randn(1, 5, HIDDEN_SIZE).astype(np.float32),
            "position_cos": np.random.randn(1, 5, HEAD_DIM).astype(np.float32),
            "position_sin": np.random.randn(1, 5, HEAD_DIM).astype(np.float32),
            "attention_mask": np.triu(np.full((1, 1, 5, 5), -1e9, dtype=np.float32), k=1),
        }
        state2 = mlmodel.make_state()
        output2 = mlmodel.predict(test_prefill, state=state2)
        out_prefill = output2["logits"]
        print(f"  Prefill output shape: {out_prefill.shape} (expect (1, 1, {VOCAB_SIZE}))")
        print("  CoreML validation passed!")
    except Exception as e:
        print(f"  CoreML validation failed: {e}")
        print("  The model was saved but may need debugging.")

    print("\nDone!")


if __name__ == "__main__":
    main()
