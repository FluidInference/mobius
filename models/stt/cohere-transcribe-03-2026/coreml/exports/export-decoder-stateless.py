#!/usr/bin/env python3
"""Export Cohere Transcribe decoder with STATELESS decoding (Parakeet approach).

This is the SIMPLE approach - no KV cache management, no State API complexity.
Just reprocess all tokens each step, like Parakeet TDT decoder.

Key advantages over stateful decoder:
- ✅ Works on macOS 14 (no State API requirement)
- ✅ Can compile to .mlmodelc for better ANE optimization
- ✅ Much simpler code - just forward pass
- ✅ No cache management bugs
- ✅ Proven approach (Parakeet, Qwen3 non-stateful)

Trade-off:
- O(n²) complexity vs O(n) for stateful
- But with 108 token limit, this is totally acceptable
- ~10x more compute per step at end, but ANE is fast

Usage:
    uv run export-decoder-stateless.py --output-dir build

    # Then compile to .mlmodelc (like Parakeet!)
    xcrun coremlcompiler compile build/cohere_decoder_stateless.mlpackage build/
"""

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq

# Cohere decoder architecture
NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
VOCAB_SIZE = 16384


class StatelessCohereDecoder(nn.Module):
    """Cohere decoder WITHOUT cache - reprocess all tokens each step.

    This is the Parakeet approach:
    - No state management
    - Just forward pass through decoder
    - Simpler, more debuggable, works on macOS 14

    For 108 token limit, O(n²) complexity is acceptable.
    """

    def __init__(self, decoder_wrapper, lm_head):
        super().__init__()

        # Store original modules - NO cache buffers!
        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor,  # [1, seq_len] - ALL tokens so far
        encoder_hidden_states: torch.Tensor,  # [1, 438, 1024]
        cross_attention_mask: torch.Tensor,  # [1, 1, 1, 438]
    ) -> torch.Tensor:
        """Run decoder on all tokens (stateless).

        Args:
            input_ids: [1, seq_len] - ALL tokens generated so far (not just new one)
            encoder_hidden_states: [1, 438, 1024] - from encoder
            cross_attention_mask: [1, 1, 1, 438] - encoder mask

        Returns:
            logits: [1, seq_len, 16384] - logits for all positions
        """
        seq_len = input_ids.shape[1]

        # Create position IDs for all tokens
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)  # [1, seq_len]

        # Create causal attention mask (lower triangular)
        # This ensures token i can only attend to tokens 0..i
        attention_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
        )
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        # Convert to additive mask (0 for attend, -inf for mask)
        attention_mask = torch.where(
            attention_mask,
            torch.zeros_like(attention_mask, dtype=torch.float32),
            torch.full_like(attention_mask, float("-inf"), dtype=torch.float32),
        )

        # 1. Get embeddings (includes position encoding)
        hidden_states = self.embedding(input_ids, position_ids)  # [1, seq_len, 1024]

        # 2. Process through decoder layers
        # Each layer does:
        #   - Self-attention (on all previous tokens)
        #   - Cross-attention (on encoder outputs)
        #   - FFN
        for layer in self.layers:
            # Self-attention
            residual = hidden_states
            hidden_states = layer.layer_norm_1(hidden_states)

            # Use original self-attention module with use_cache=False
            # This computes attention over ALL tokens (no cache)
            self_attn_out = layer.first_sub_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,  # No cache!
                cache_position=None,
                is_cross_attention=False,
                kv_seq_len=None,
            )
            hidden_states = residual + self_attn_out

            # Cross-attention (on encoder - no cache needed)
            residual = hidden_states
            hidden_states = layer.layer_norm_2(hidden_states)

            cross_attn_out = layer.second_sub_layer(
                hidden_states=hidden_states,
                context_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                past_key_values=None,
                cache_position=None,
                is_cross_attention=True,
                kv_seq_len=None,
            )
            hidden_states = residual + cross_attn_out

            # FFN
            residual = hidden_states
            hidden_states = layer.layer_norm_3(hidden_states)
            hidden_states = residual + layer.third_sub_layer(hidden_states)

        # 3. Final norm and projection to logits
        hidden_states = self.final_norm(hidden_states)  # [1, seq_len, 1024]
        logits = self.lm_head(hidden_states)  # [1, seq_len, 16384]

        return logits


def main():
    parser = argparse.ArgumentParser(description="Export Cohere stateless decoder")
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Transcribe STATELESS Decoder Export (Parakeet Approach)")
    print("="*70)
    print(f"Model: {args.model_id}")
    print(f"Output: {output_dir}")
    print()
    print("Advantages:")
    print("  ✅ Works on macOS 14 (no State API)")
    print("  ✅ Can compile to .mlmodelc for better ANE optimization")
    print("  ✅ Much simpler code - just forward pass")
    print("  ✅ No cache management complexity")
    print()
    print("Trade-off:")
    print("  ⚠️  O(n²) complexity (but acceptable for 108 tokens)")
    print()

    # ---- Step 1: Load model ----
    print("[1/5] Loading model...")
    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

    # ---- Step 2: Extract components ----
    print(f"\n[2/5] Extracting decoder components...")
    decoder_wrapper = model.transf_decoder
    lm_head = model.log_softmax.mlp.layer0

    print(f"   Decoder layers: {len(decoder_wrapper._decoder.layers)}")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Num heads: {NUM_HEADS}")
    print(f"   Head dim: {HEAD_DIM}")
    print(f"   LM head: {lm_head.in_features} -> {lm_head.out_features}")

    # ---- Step 3: Create stateless wrapper ----
    print(f"\n[3/5] Creating stateless decoder wrapper...")
    stateless_decoder = StatelessCohereDecoder(decoder_wrapper, lm_head)
    stateless_decoder.eval()
    print("   ✓ No cache buffers - just forward pass!")

    # ---- Step 4: Trace with example inputs ----
    print(f"\n[4/5] Tracing model...")

    # Example inputs for tracing
    # Use sequence length of 10 as example
    example_seq_len = 10
    example_input_ids = torch.randint(0, VOCAB_SIZE, (1, example_seq_len), dtype=torch.long)
    example_encoder_hidden = torch.randn(1, 438, HIDDEN_SIZE, dtype=torch.float32)
    example_cross_mask = torch.ones(1, 1, 1, 438, dtype=torch.float32)

    print(f"   Tracing with sequence length: {example_seq_len}")
    print(f"   Input IDs: {example_input_ids.shape}")
    print(f"   Encoder hidden: {example_encoder_hidden.shape}")

    with torch.no_grad():
        traced_model = torch.jit.trace(
            stateless_decoder,
            (example_input_ids, example_encoder_hidden, example_cross_mask),
        )
    print("   ✓ Model traced successfully")

    # ---- Step 5: Convert to CoreML ----
    print(f"\n[5/5] Converting to CoreML...")

    # Use flexible shapes for input_ids (sequence can vary)
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(shape=(1, ct.RangeDim(1, 108))),  # Flexible seq length
                dtype=np.int32,
            ),
            ct.TensorType(
                name="encoder_hidden_states",
                shape=(1, 438, HIDDEN_SIZE),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cross_attention_mask",
                shape=(1, 1, 1, 438),
                dtype=np.float32,
            ),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32)
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,  # Works on macOS 14!
    )

    # Add metadata
    mlmodel.author = "FluidInference"
    mlmodel.license = "Apache 2.0"
    mlmodel.short_description = "Cohere Transcribe stateless decoder (Parakeet approach)"
    mlmodel.version = "1.0"

    # Save
    output_path = output_dir / "cohere_decoder_stateless.mlpackage"
    mlmodel.save(str(output_path))
    print(f"   ✓ Saved to: {output_path}")

    # Print size
    import subprocess
    size_mb = subprocess.check_output(["du", "-sh", str(output_path)]).decode().split()[0]
    print(f"   Model size: {size_mb}")

    print()
    print("="*70)
    print("✅ Export complete!")
    print("="*70)
    print()
    print("Next steps:")
    print(f"  1. Test with Python:")
    print(f"     python test_stateless_decoder.py")
    print()
    print(f"  2. Compile to .mlmodelc for better ANE optimization:")
    print(f"     xcrun coremlcompiler compile {output_path} {output_dir}/")
    print()
    print(f"  3. Compare performance vs stateful decoder")
    print()
    print("Key differences from stateful:")
    print("  • Works on macOS 14 (not just 15+)")
    print("  • Can compile to .mlmodelc")
    print("  • Simpler architecture (like Parakeet)")
    print("  • ~10x more compute at step 108, but ANE should handle it")
    print()


if __name__ == "__main__":
    main()
