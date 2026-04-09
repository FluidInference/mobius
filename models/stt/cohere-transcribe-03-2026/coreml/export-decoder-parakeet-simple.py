#!/usr/bin/env python3
"""Export Cohere Transcribe decoder - TRUE Parakeet approach.

Simplified contract (like Parakeet TDT):
1. Swift manages KV cache arrays
2. Swift writes new K/V to cache at position [step] BEFORE calling model
3. Model receives cache + mask indicating valid positions
4. Model returns ONLY logits (no cache outputs - Swift already has them!)

This is cleaner than trying to update cache inside the model.

Usage:
    uv run export-decoder-parakeet-simple.py --output-dir build
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


class SimpleParakeetCohereDecoder(nn.Module):
    """TRUE Parakeet pattern: Swift manages cache, model just reads it.

    Swift does:
    1. Project new token to K/V using projection weights
    2. Write K/V to cache at position [step]
    3. Call model with cache + attention_mask
    4. Model returns logits
    5. Swift extracts next token, repeat

    Model does:
    - Read cache (Swift already wrote new K/V)
    - Apply attention with mask
    - Return logits

    NO cache in outputs! Swift maintains cache state.
    """

    def __init__(self, decoder_wrapper, lm_head):
        super().__init__()

        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.lm_head = lm_head

    def forward(
        self,
        input_id: torch.Tensor,  # [1, 1]
        position_id: torch.Tensor,  # [1, 1]
        encoder_hidden_states: torch.Tensor,  # [1, 438, 1024]
        cross_attention_mask: torch.Tensor,  # [1, 1, 1, 438]
        attention_mask: torch.Tensor,  # [1, 1, 1, 108] - causal mask for self-attention
        # KV caches (Swift manages these)
        k_cache_0: torch.Tensor,
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
    ) -> torch.Tensor:
        """Process one token. Swift handles cache updates externally.

        Returns:
            logits: [1, 16384]
        """

        k_caches = [k_cache_0, k_cache_1, k_cache_2, k_cache_3,
                    k_cache_4, k_cache_5, k_cache_6, k_cache_7]
        v_caches = [v_cache_0, v_cache_1, v_cache_2, v_cache_3,
                    v_cache_4, v_cache_5, v_cache_6, v_cache_7]

        # 1. Get embedding
        hidden_states = self.embedding(input_id, position_id)

        # 2. Process through layers
        for layer_idx, layer in enumerate(self.layers):
            k_cache = k_caches[layer_idx]  # [1, 8, 108, 128] - Swift filled it
            v_cache = v_caches[layer_idx]

            # --- Self-attention ---
            residual = hidden_states
            hidden_states = layer.layer_norm_1(hidden_states)

            # Project Q (K/V already in cache from Swift)
            query = layer.first_sub_layer.query_net(hidden_states)
            query = layer.first_sub_layer._reshape(query)  # [1, 8, 1, 128]

            # Attention over full cache (mask handles valid positions)
            attn_output = F.scaled_dot_product_attention(
                query,
                k_cache,  # [1, 8, 108, 128]
                v_cache,  # [1, 8, 108, 128]
                attn_mask=attention_mask,  # [1, 1, 1, 108]
                dropout_p=0.0,
                scale=layer.first_sub_layer.scale,
            )

            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(1, 1, HIDDEN_SIZE)
            )
            attn_output = layer.first_sub_layer.out_projection(attn_output)
            hidden_states = residual + attn_output

            # --- Cross-attention ---
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
        logits = self.lm_head(hidden_states).squeeze(1)  # [1, 16384]

        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--output-dir", default="build")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Cohere Decoder Export - Simple Parakeet Pattern")
    print("="*70)
    print()
    print("Contract:")
    print("  • Swift manages KV cache arrays (16 total)")
    print("  • Swift projects new token to K/V and writes to cache")
    print("  • Model reads cache + attention mask")
    print("  • Model returns ONLY logits")
    print("  • Swift handles all cache bookkeeping")
    print()

    # Load
    print("[1/3] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    # Extract
    print("\n[2/3] Creating wrapper...")
    decoder = SimpleParakeetCohereDecoder(model.transf_decoder, model.log_softmax.mlp.layer0)
    decoder.eval()

    # Trace
    print("\n[3/3] Tracing and converting...")

    input_id = torch.tensor([[4]], dtype=torch.long)
    position_id = torch.tensor([[0]], dtype=torch.long)
    encoder_hidden = torch.randn(1, 438, HIDDEN_SIZE)
    cross_mask = torch.ones(1, 1, 1, 438)
    attention_mask = torch.zeros(1, 1, 1, MAX_SEQ_LEN)  # All positions valid initially
    attention_mask[:, :, :, 1:] = float("-inf")  # Mask out all but position 0

    k_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]
    v_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]

    with torch.no_grad():
        traced = torch.jit.trace(decoder, (
            input_id, position_id, encoder_hidden, cross_mask, attention_mask,
            *k_caches, *v_caches
        ))

    # Convert
    inputs = [
        ct.TensorType("input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("position_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("encoder_hidden_states", shape=(1, 438, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("cross_attention_mask", shape=(1, 1, 1, 438), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, 1, MAX_SEQ_LEN), dtype=np.float32),
    ]

    for i in range(NUM_LAYERS):
        inputs.append(ct.TensorType(f"k_cache_{i}", shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32))
        inputs.append(ct.TensorType(f"v_cache_{i}", shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType("logits", dtype=np.float32)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    output_path = output_dir / "cohere_decoder_parakeet.mlpackage"
    mlmodel.save(str(output_path))

    print(f"\n✅ Saved to: {output_path}")
    print("\n" + "="*70)
    print("Next: Implement Swift side with CohereDecoderState + runDecoder()")
    print("="*70)


if __name__ == "__main__":
    main()
