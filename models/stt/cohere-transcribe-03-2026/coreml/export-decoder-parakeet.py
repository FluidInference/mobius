#!/usr/bin/env python3
"""Export Cohere Transcribe decoder using Parakeet's cache-external pattern.

Brandon's approach: "for parakeet we just passed it in manually each loop and
tracked the state outside of the coreml decoder"

Key differences from other approaches:
- Stateless: No cache at all, O(n²) complexity
- Stateful: Cache managed INSIDE CoreML with register_buffer() (Qwen3 pattern)
- Parakeet: Cache passed IN as inputs, returned OUT as outputs, managed in Swift

Advantages:
- ✅ Works on macOS 14 (no State API needed)
- ✅ Can compile to .mlmodelc
- ✅ O(n) complexity (efficient)
- ✅ Simple Swift-side cache management
- ✅ Full visibility into cache state for debugging

This is the recommended approach per Brandon.

Usage:
    uv run export-decoder-parakeet.py --output-dir build
"""

import argparse
import time
from pathlib import Path

import coremltools as ct
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
VOCAB_SIZE = 16384
MAX_SEQ_LEN = 108


class ParakeetStyleCohereDecoder(nn.Module):
    """Cohere decoder with cache-external pattern (Parakeet approach).

    The CoreML model is STATELESS - it just:
    1. Takes current token + current cache as inputs
    2. Returns logits + updated cache as outputs

    Swift code manages cache lifetime and passes it in/out each step.
    """

    def __init__(self, decoder_wrapper, lm_head):
        super().__init__()

        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.lm_head = lm_head
        self.num_layers = len(self.layers)

    def forward(
        self,
        input_id: torch.Tensor,  # [1, 1] - current token
        position_id: torch.Tensor,  # [1, 1] - current position
        encoder_hidden_states: torch.Tensor,  # [1, 438, 1024]
        cross_attention_mask: torch.Tensor,  # [1, 1, 1, 438]
        # KV cache inputs (16 total: 8 layers × K/V)
        k_cache_0: torch.Tensor,  # [1, 8, 108, 128]
        v_cache_0: torch.Tensor,
        k_cache_1: torch.Tensor,
        v_cache_1: torch.Tensor,
        k_cache_2: torch.Tensor,
        v_cache_2: torch.Tensor,
        k_cache_3: torch.Tensor,
        v_cache_3: torch.Tensor,
        k_cache_4: torch.Tensor,
        v_cache_4: torch.Tensor,
        k_cache_5: torch.Tensor,
        v_cache_5: torch.Tensor,
        k_cache_6: torch.Tensor,
        v_cache_6: torch.Tensor,
        k_cache_7: torch.Tensor,
        v_cache_7: torch.Tensor,
        past_kv_len: torch.Tensor,  # [1] - scalar, how many positions filled
    ):
        """Process one token with cache passed in/out.

        Returns:
            Tuple of (logits, k_cache_0_out, v_cache_0_out, ..., k_cache_7_out, v_cache_7_out)
        """

        # Collect input caches
        k_caches = [
            k_cache_0, k_cache_1, k_cache_2, k_cache_3,
            k_cache_4, k_cache_5, k_cache_6, k_cache_7,
        ]
        v_caches = [
            v_cache_0, v_cache_1, v_cache_2, v_cache_3,
            v_cache_4, v_cache_5, v_cache_6, v_cache_7,
        ]

        # CRITICAL: Do NOT use .item() - it gets traced as a constant!
        # Instead, Swift will manage which position to write to
        # We receive the FULL cache with the new K/V already written at the right position
        # This is simpler: Swift does the bookkeeping, we just process

        # 1. Get embedding
        hidden_states = self.embedding(input_id, position_id)  # [1, 1, 1024]

        # Get current sequence length from past_kv_len
        # This works because past_kv_len is passed as a tensor input
        current_seq_len = past_kv_len + 1  # Tensor addition (no .item()!)

        # Output caches (will be updated)
        output_k_caches = []
        output_v_caches = []

        # 2. Process through layers
        for layer_idx, layer in enumerate(self.layers):
            k_cache = k_caches[layer_idx]
            v_cache = v_caches[layer_idx]

            # --- Self-attention with cache ---
            residual = hidden_states
            hidden_states = layer.layer_norm_1(hidden_states)

            # Manually compute self-attention
            query = layer.first_sub_layer.query_net(hidden_states)
            key = layer.first_sub_layer.key_net(hidden_states)
            value = layer.first_sub_layer.value_net(hidden_states)

            # Reshape to multi-head
            query = layer.first_sub_layer._reshape(query)  # [1, 8, 1, 128]
            key = layer.first_sub_layer._reshape(key)      # [1, 8, 1, 128]
            value = layer.first_sub_layer._reshape(value)  # [1, 8, 1, 128]

            # Concatenate new K/V with cache
            # Input cache has shape [1, 8, 108, 128] with past_kv_len positions filled
            # We append the new K/V and create output cache
            # Swift will slice out the valid portion [:, :, :current_seq_len, :]
            k_cache_updated = k_cache.clone()
            v_cache_updated = v_cache.clone()

            # Use torch.cat to append - this avoids indexing with past_kv_len.item()
            # But we need to write at a specific position...
            # Actually, let's use scatter - no, that also needs indices

            # SOLUTION: Swift pre-writes the cache at the right position!
            # We just use the cross-attention approach - read the filled portion
            # Swift does: cache[:, :, step, :] = new_kv BEFORE calling model
            # We just need to read cache[:, :, :current_seq_len, :]

            # For this to work, we need to change the contract:
            # Input cache ALREADY has the new K/V written at position past_kv_len
            # We DON'T write it here - Swift did it already

            # Actually, that's backwards. Let me think...
            # We MUST write it here because the model needs to see the new token's K/V

            # Real solution: Accept that we need per-layer outputs
            # Each layer appends to its cache, we return the concatenated result
            # Use torch.narrow to read valid cache, torch.cat to append

            # Read existing valid cache entries
            # This uses tensor indexing (not .item())
            # But slicing with past_kv_len[:] still uses .item() internally...

            # ACTUAL solution from Qwen3: Use attention mask size to infer position!
            # But we don't have attention mask input for self-attention...

            # Let me simplify: Just read ALL of cache and use attention mask
            # Swift passes an attention mask that tells us which positions are valid

            # For now: Accept the slicing issue will be fixed by converting to EnumeratedShapes
            # But that won't work either...

            # REAL real solution: Make Swift write the cache first, we just read it
            # Input k_cache at step N ALREADY contains positions 0..N
            # We output the same cache (unchanged)
            # Swift manages writing new K/V after getting output

            # NO - that doesn't work because we need K/V for attention IN THIS FORWARD PASS

            # Final answer: Use scatter operations with fancy indexing
            # Create index tensor for the current position
            idx = past_kv_len.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, NUM_HEADS, 1, HEAD_DIM)

            k_cache_updated = k_cache.scatter(2, idx, key)
            v_cache_updated = v_cache.scatter(2, idx, value)

            # Read valid portion using mask instead of slicing
            # Create a mask: positions < current_seq_len are valid
            pos_range = torch.arange(MAX_SEQ_LEN, dtype=torch.int32).unsqueeze(0)  # [1, 108]
            valid_mask = pos_range < current_seq_len  # [1, 108]
            # Expand to [1, 8, 108, 128]
            valid_mask = valid_mask.unsqueeze(1).unsqueeze(3).expand(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)

            # Zero out invalid positions
            k_valid = torch.where(valid_mask, k_cache_updated, torch.zeros_like(k_cache_updated))
            v_valid = torch.where(valid_mask, v_cache_updated, torch.zeros_like(v_cache_updated))

            # Create causal mask for self-attention [1, 1, 1, 108]
            # Mask out positions >= current_seq_len
            pos_range_2d = torch.arange(MAX_SEQ_LEN, dtype=torch.int32).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 108]
            self_attn_mask = torch.where(
                pos_range_2d < current_seq_len,
                torch.zeros_like(pos_range_2d, dtype=torch.float32),
                torch.full_like(pos_range_2d, float("-inf"), dtype=torch.float32),
            )

            # Scaled dot-product attention
            # Use full cache (108 positions) but mask will hide invalid ones
            attn_output = F.scaled_dot_product_attention(
                query,  # [1, 8, 1, 128]
                k_valid,  # [1, 8, 108, 128] (zeros beyond current_seq_len)
                v_valid,  # [1, 8, 108, 128] (zeros beyond current_seq_len)
                attn_mask=self_attn_mask,  # [1, 1, 1, 108] (-inf beyond current_seq_len)
                dropout_p=0.0,
                scale=layer.first_sub_layer.scale,
            )

            # Reshape and project
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(1, 1, HIDDEN_SIZE)
            )
            attn_output = layer.first_sub_layer.out_projection(attn_output)

            hidden_states = residual + attn_output

            # Store updated caches for output
            output_k_caches.append(k_cache_updated)
            output_v_caches.append(v_cache_updated)

            # --- Cross-attention (no cache needed) ---
            residual = hidden_states
            hidden_states = layer.layer_norm_2(hidden_states)

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

        # 3. Final norm and logits
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)  # [1, 1, 16384]
        logits = logits.squeeze(1)  # [1, 16384]

        # Return logits + all updated caches
        return (
            logits,
            output_k_caches[0], output_v_caches[0],
            output_k_caches[1], output_v_caches[1],
            output_k_caches[2], output_v_caches[2],
            output_k_caches[3], output_v_caches[3],
            output_k_caches[4], output_v_caches[4],
            output_k_caches[5], output_v_caches[5],
            output_k_caches[6], output_v_caches[6],
            output_k_caches[7], output_v_caches[7],
        )


def main():
    parser = argparse.ArgumentParser(description="Export Cohere decoder (Parakeet pattern)")
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--output-dir", default="build")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Transcribe Decoder Export (Parakeet Cache-External Pattern)")
    print("="*70)
    print(f"Model: {args.model_id}")
    print(f"Output: {output_dir}")
    print()
    print("Approach: Brandon's recommendation")
    print("  • Cache managed in Swift (outside CoreML)")
    print("  • Model takes cache as inputs")
    print("  • Model returns updated cache as outputs")
    print("  • No State API - works on macOS 14")
    print()

    # Load model
    print("[1/4] Loading model...")
    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

    # Extract components
    print("\n[2/4] Extracting decoder components...")
    decoder_wrapper = model.transf_decoder
    lm_head = model.log_softmax.mlp.layer0

    print(f"   Decoder layers: {len(decoder_wrapper._decoder.layers)}")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Num heads: {NUM_HEADS}")
    print(f"   Head dim: {HEAD_DIM}")

    # Create wrapper
    print("\n[3/4] Creating Parakeet-style wrapper...")
    parakeet_decoder = ParakeetStyleCohereDecoder(decoder_wrapper, lm_head)
    parakeet_decoder.eval()
    print("   ✓ Created cache-external decoder")

    # Trace
    print("\n[4/4] Tracing and converting to CoreML...")

    # Example inputs for first token (position 0)
    input_id = torch.tensor([[4]], dtype=torch.long)
    position_id = torch.tensor([[0]], dtype=torch.long)
    encoder_hidden = torch.randn(1, 438, HIDDEN_SIZE, dtype=torch.float32)
    cross_mask = torch.ones(1, 1, 1, 438, dtype=torch.float32)

    # Empty caches (all zeros initially)
    k_caches = [
        torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32)
        for _ in range(NUM_LAYERS)
    ]
    v_caches = [
        torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM, dtype=torch.float32)
        for _ in range(NUM_LAYERS)
    ]
    past_kv_len = torch.tensor([0], dtype=torch.int32)

    trace_inputs = (
        input_id,
        position_id,
        encoder_hidden,
        cross_mask,
        *k_caches,  # k_cache_0 through k_cache_7
        *v_caches,  # v_cache_0 through v_cache_7
        past_kv_len,
    )

    print("   Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(parakeet_decoder, trace_inputs)

    print("   Converting to CoreML...")

    # Define inputs
    inputs = [
        ct.TensorType("input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("position_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("encoder_hidden_states", shape=(1, 438, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("cross_attention_mask", shape=(1, 1, 1, 438), dtype=np.float32),
    ]

    # Add KV cache inputs
    for i in range(NUM_LAYERS):
        inputs.append(
            ct.TensorType(
                f"k_cache_{i}",
                shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM),
                dtype=np.float32,
            )
        )
        inputs.append(
            ct.TensorType(
                f"v_cache_{i}",
                shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM),
                dtype=np.float32,
            )
        )

    inputs.append(ct.TensorType("past_kv_len", shape=(1,), dtype=np.int32))

    # Define outputs
    outputs = [ct.TensorType("logits", dtype=np.float32)]
    for i in range(NUM_LAYERS):
        outputs.append(ct.TensorType(f"k_cache_{i}_out", dtype=np.float32))
        outputs.append(ct.TensorType(f"v_cache_{i}_out", dtype=np.float32))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,  # Works on macOS 14!
    )

    # Add metadata
    mlmodel.author = "FluidInference"
    mlmodel.license = "Apache 2.0"
    mlmodel.short_description = "Cohere Transcribe decoder (Parakeet cache-external pattern)"
    mlmodel.version = "1.0"

    # Save
    output_path = output_dir / "cohere_decoder_parakeet.mlpackage"
    mlmodel.save(str(output_path))

    print(f"\n✅ Saved to: {output_path}")

    # Print size
    import subprocess
    size_mb = subprocess.check_output(["du", "-sh", str(output_path)]).decode().split()[0]
    print(f"   Model size: {size_mb}")

    print()
    print("="*70)
    print("Export Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Create CohereDecoderState struct in Swift")
    print("  2. Implement runDecoder() that passes cache in/out")
    print("  3. Test inference loop")
    print()
    print("Swift will manage:")
    print("  • k_cache_0..7 and v_cache_0..7 (16 MLMultiArrays)")
    print("  • past_kv_len counter")
    print("  • Passing them in and extracting updated versions each step")
    print()


if __name__ == "__main__":
    main()
