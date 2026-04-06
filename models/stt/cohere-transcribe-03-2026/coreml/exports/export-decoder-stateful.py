#!/usr/bin/env python3
"""Export Cohere Transcribe decoder with stateful KV cache (Qwen3 approach).

This implements GPU-resident KV cache using register_buffer(), eliminating the
marshaling overhead of passing cache tensors in/out at each decode step.

Based on Qwen3's proven stateful approach, adapted for Cohere's architecture:
- 8 layers (vs Qwen3's 28)
- Standard attention (vs GQA)
- Simple position encoding lookup (vs RoPE)
- Both self-attention and cross-attention per layer

KEY: Infers current position from attention_mask shape (like Qwen3), avoiding
     the .item() tracing issue that causes constants to be baked in.

Usage:
    uv run export-decoder-stateful.py --output-dir build
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSpeechSeq2Seq

# Cohere decoder architecture
NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
MAX_SEQ_LEN = 108


class StatefulCohereDecoder(nn.Module):
    """Cohere decoder with stateful KV cache for CoreML export.

    Implements Qwen3's stateful cache approach:
    - Register fp16 buffers for KV cache (GPU-resident)
    - In-place cache updates via slice assignment
    - Manual self-attention computation
    - Pass-through cross-attention (no cache needed)
    - Infer position from attention_mask shape (avoids .item() tracing issue)
    """

    def __init__(self, decoder_wrapper, lm_head, max_seq_len=108):
        super().__init__()

        # Store original modules
        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.lm_head = lm_head
        self.num_layers = len(self.layers)
        self.max_seq_len = max_seq_len

        # Register 16 state buffers (8 layers × K/V for self-attention only)
        # CoreML states MUST be fp16
        for i in range(self.num_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, NUM_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, NUM_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16),
            )

    def forward(
        self,
        input_id: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder with in-place KV cache updates.

        Args:
            input_id: [1, 1] - current token ID
            encoder_hidden_states: [1, 438, 1024] - from encoder (3500 frames @ 35s)
            cross_attention_mask: [1, 1, 1, 438] - encoder mask
            attention_mask: [1, 1, 1, end_step] - self-attention mask
                The size of attention_mask determines the current position:
                - end_step = attention_mask.shape[-1]
                - past_kv_len = end_step - 1
            position_ids: [1, 1] - current position for embedding lookup

        Returns:
            logits: [1, 16384] - token logits
        """
        # Infer cache position from attention mask shape (Qwen3 approach)
        # This avoids .item() which would get traced as a constant
        q_len = input_id.shape[1]  # Should be 1 (single token)
        end_step = attention_mask.shape[-1]  # Total sequence length
        past_kv_len = end_step - q_len  # How many tokens already in cache

        # 1. Get embeddings (includes position encoding lookup)
        hidden_states = self.embedding(input_id, position_ids)

        # 2. Process through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            k_cache = getattr(self, f"k_cache_{layer_idx}")
            v_cache = getattr(self, f"v_cache_{layer_idx}")

            # --- Self-attention with stateful cache ---
            residual = hidden_states
            hidden_states = layer.layer_norm_1(hidden_states)

            hidden_states = self._manual_self_attention(
                hidden_states=hidden_states,
                attention_module=layer.first_sub_layer,
                k_cache=k_cache,
                v_cache=v_cache,
                attention_mask=attention_mask,
                past_kv_len=past_kv_len,
                end_step=end_step,
            )
            hidden_states = residual + hidden_states

            # --- Cross-attention (no cache, encoder is static) ---
            residual = hidden_states
            hidden_states = layer.layer_norm_2(hidden_states)

            # Use original cross-attention module (no cache needed)
            cross_out = layer.second_sub_layer(
                hidden_states=hidden_states,
                context_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                past_key_values=None,
                cache_position=None,
                is_cross_attention=True,
                kv_seq_len=None,
            )
            hidden_states = residual + cross_out

            # --- FFN ---
            residual = hidden_states
            hidden_states = layer.layer_norm_3(hidden_states)
            hidden_states = residual + layer.third_sub_layer(hidden_states)

        # 3. Final norm and projection to logits
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits.squeeze(1)  # [1, 16384]

    def _manual_self_attention(
        self,
        hidden_states: torch.Tensor,
        attention_module: nn.Module,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        past_kv_len: int,
        end_step: int,
    ) -> torch.Tensor:
        """Manually compute self-attention with stateful KV cache.

        This is the critical part adapted from Qwen3:
        - Project Q, K, V from current token
        - Update cache in-place (CoreML detects as state mutation)
        - Read full valid cache entries
        - Compute attention using PyTorch's built-in
        """
        # 1. Project Q, K, V
        query = attention_module.query_net(hidden_states)
        key = attention_module.key_net(hidden_states)
        value = attention_module.value_net(hidden_states)

        # 2. Reshape to multi-head: [1, 1, 1024] -> [1, 8, 1, 128]
        query = attention_module._reshape(query)
        key = attention_module._reshape(key)
        value = attention_module._reshape(value)

        # 3. In-place KV cache update (CoreML detects as state mutation)
        # Qwen3 approach: slice assignment with computed indices
        # Cast fp32 -> fp16 for storage (CoreML states must be fp16)
        k_cache[:, :, past_kv_len:end_step, :] = key.half()
        v_cache[:, :, past_kv_len:end_step, :] = value.half()

        # 4. Read valid cache entries and cast back to fp32 for attention
        k_full = k_cache[:, :, :end_step, :].float()  # [1, 8, end_step, 128]
        v_full = v_cache[:, :, :end_step, :].float()  # [1, 8, end_step, 128]

        # 5. Scaled dot-product attention (use PyTorch's built-in, same as Cohere)
        attn_output = F.scaled_dot_product_attention(
            query,
            k_full,
            v_full,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=attention_module.scale,
        )

        # 6. Reshape and project output: [1, 8, 1, 128] -> [1, 1, 1024]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.shape[0], hidden_states.shape[1], attention_module.hidden_size)
        )

        return attention_module.out_projection(attn_output)


def main():
    parser = argparse.ArgumentParser(description="Export Cohere stateful decoder")
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--max-seq-len", type=int, default=108)
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Transcribe Stateful Decoder Export (Qwen3 Interface)")
    print("="*70)
    print(f"Model: {args.model_id}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Output: {output_dir}")
    print()

    # ---- Step 1: Load model ----
    print("[1/6] Loading model...")
    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

    # ---- Step 2: Extract components ----
    print(f"\n[2/6] Extracting decoder components...")
    decoder_wrapper = model.transf_decoder
    lm_head = model.log_softmax.mlp.layer0

    print(f"   Decoder layers: {len(decoder_wrapper._decoder.layers)}")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Num heads: {NUM_HEADS}")
    print(f"   Head dim: {HEAD_DIM}")
    print(f"   LM head: {lm_head.in_features} -> {lm_head.out_features}")

    # Verify architecture
    layer0 = decoder_wrapper._decoder.layers[0]
    print(f"   Self-attention module: {type(layer0.first_sub_layer).__name__}")
    print(f"   Cross-attention module: {type(layer0.second_sub_layer).__name__}")
    print(f"   FFN module: {type(layer0.third_sub_layer).__name__}")

    # ---- Step 3: Create stateful wrapper ----
    print(f"\n[3/6] Creating stateful decoder (max_seq_len={args.max_seq_len})...")
    stateful_decoder = StatefulCohereDecoder(
        decoder_wrapper,
        lm_head,
        max_seq_len=args.max_seq_len
    )
    stateful_decoder.eval()
    print(f"   ✓ Created with {stateful_decoder.num_layers} layers")
    print(f"   ✓ Registered {stateful_decoder.num_layers * 2} state buffers")

    # ---- Step 4: Trace ----
    print("\n[4/6] Tracing model...")

    # Trace inputs (single token decode at step 0)
    input_id = torch.tensor([[13764]], dtype=torch.long)  # Start token
    encoder_hidden = torch.randn(1, 438, 1024)  # 3500 frames @ 35s
    cross_mask = torch.ones(1, 1, 1, 438)
    # Attention mask: [1, 1, 1, 1] for first token (position 0)
    attention_mask = torch.zeros(1, 1, 1, 1)
    # Position IDs: [1, 1] with value 0 for first token
    position_ids = torch.tensor([[0]], dtype=torch.long)

    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(
            stateful_decoder,
            (input_id, encoder_hidden, cross_mask, attention_mask, position_ids)
        )
    traced.eval()
    print(f"   ✓ Traced in {time.time() - t0:.1f}s")

    # ---- Step 5: Validate traced model ----
    if not args.skip_validation:
        print("\n[5/6] Validating traced vs eager...")

        # Create fresh instance
        stateful_ref = StatefulCohereDecoder(
            decoder_wrapper,
            lm_head,
            max_seq_len=args.max_seq_len
        )
        stateful_ref.eval()

        test_input_id = torch.tensor([[13764]], dtype=torch.long)
        test_encoder = torch.randn(1, 438, 1024)  # 3500 frames @ 35s
        test_cross_mask = torch.ones(1, 1, 1, 438)
        test_attn_mask = torch.zeros(1, 1, 1, 1)
        test_position_ids = torch.tensor([[0]], dtype=torch.long)

        with torch.no_grad():
            ref_out = stateful_ref(test_input_id, test_encoder, test_cross_mask, test_attn_mask, test_position_ids)
            traced_out = traced(test_input_id, test_encoder, test_cross_mask, test_attn_mask, test_position_ids)
            diff = (ref_out - traced_out).abs().max().item()

        print(f"   Max diff (traced vs eager): {diff:.6e}")
        if diff > 1e-3:
            print(f"   ⚠️  WARNING: Large divergence! Check tracing compatibility.")
        else:
            print(f"   ✓ Traced model matches eager mode")
    else:
        print("\n[5/6] Skipping validation")

    # ---- Step 6: Convert to CoreML ----
    print("\n[6/6] Converting to CoreML...")
    import coremltools as ct

    print(f"   coremltools version: {ct.__version__}")

    # Define inputs with RangeDim for attention_mask
    # attention_mask grows from [1,1,1,1] to [1,1,1,MAX_SEQ_LEN] as we generate
    attn_mask_dim = ct.RangeDim(lower_bound=1, upper_bound=args.max_seq_len, default=1)

    inputs = [
        ct.TensorType("input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("encoder_hidden_states", shape=(1, 438, 1024), dtype=np.float16),
        ct.TensorType("cross_attention_mask", shape=(1, 1, 1, 438), dtype=np.float16),
        ct.TensorType("attention_mask", shape=(1, 1, 1, attn_mask_dim), dtype=np.float16),
        ct.TensorType("position_ids", shape=(1, 1), dtype=np.int32),
    ]

    outputs = [
        ct.TensorType("logits", dtype=np.float16),
    ]

    # Define state buffers (16 total: 8 layers × K + V)
    states = []
    for i in range(NUM_LAYERS):
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, NUM_HEADS, args.max_seq_len, HEAD_DIM),
                    dtype=np.float16
                ),
                name=f"k_cache_{i}",
            )
        )
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, NUM_HEADS, args.max_seq_len, HEAD_DIM),
                    dtype=np.float16
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
        minimum_deployment_target=ct.target.macOS15,  # Requires macOS 15 for State API
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    print(f"   ✓ Converted in {time.time() - t0:.1f}s")

    # Save
    # Include max_seq_len in filename if not default (108)
    if args.max_seq_len == 108:
        output_path = output_dir / "cohere_decoder_stateful.mlpackage"
    else:
        output_path = output_dir / f"cohere_decoder_stateful_{args.max_seq_len}.mlpackage"

    mlmodel.save(str(output_path))
    print(f"\n✓ Saved to: {output_path}")

    # ---- Step 7: Validate CoreML ----
    print("\n[7/7] Validating CoreML model...")
    try:
        state = mlmodel.make_state()
        test_input = {
            "input_id": np.array([[13764]], dtype=np.int32),
            "encoder_hidden_states": np.random.randn(1, 438, 1024).astype(np.float16),
            "cross_attention_mask": np.ones((1, 1, 1, 438), dtype=np.float16),
            "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float16),
            "position_ids": np.array([[0]], dtype=np.int32),
        }
        output = mlmodel.predict(test_input, state=state)
        logits = output["logits"]
        print(f"   Output shape: {logits.shape}")
        print(f"   Output range: [{logits.min():.2f}, {logits.max():.2f}]")
        print(f"   Max logit token: {np.argmax(logits[0])}")

        # Test multi-step inference with growing attention mask
        print(f"\n   Testing multi-step inference...")
        state = mlmodel.make_state()
        for i in range(3):
            # Attention mask grows: [1,1,1,1] -> [1,1,1,2] -> [1,1,1,3]
            # Position IDs match current position
            test_input["attention_mask"] = np.zeros((1, 1, 1, i+1), dtype=np.float16)
            test_input["position_ids"] = np.array([[i]], dtype=np.int32)
            output = mlmodel.predict(test_input, state=state)
            next_token = int(np.argmax(output["logits"][0]))
            print(f"     Step {i}: attn_mask_size={i+1}, position={i}, token={next_token}")

        print("   ✓ CoreML validation passed!")
    except Exception as e:
        print(f"   ❌ CoreML validation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Export Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Test: python tests/test-stateful-decoder.py")
    print("2. Benchmark: python tests/test-librispeech.py")
    print("3. If cache-length bug appears, implement cache padding")


if __name__ == "__main__":
    main()
