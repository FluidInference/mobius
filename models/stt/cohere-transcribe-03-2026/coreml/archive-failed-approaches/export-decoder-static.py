#!/usr/bin/env python3
"""Export Cohere decoder with manual static cache management."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, Cache


class StaticSizeCache(Cache):
    """Cache that maintains fixed size but reports variable sequence length."""

    def __init__(self, static_cache_k, static_cache_v, current_step):
        """
        Args:
            static_cache_k: (num_layers, num_heads, max_seq_len, head_dim)
            static_cache_v: (num_layers, num_heads, max_seq_len, head_dim)
            current_step: (1,) tensor - how many tokens are currently cached
        """
        super().__init__()
        self.key_cache = []
        self.value_cache = []
        self.num_layers = static_cache_k.shape[0]
        self.current_step = current_step

        # Store cache for each layer with batch dimension
        for layer_idx in range(self.num_layers):
            self.key_cache.append(static_cache_k[layer_idx:layer_idx+1, :, :, :])
            self.value_cache.append(static_cache_v[layer_idx:layer_idx+1, :, :, :])

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Update cache by replacing the stored cache with new states."""
        # The decoder will pass updated cache here (with appended token)
        # Store it for this layer
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        """Return current step as sequence length."""
        # Convert step tensor to int for the transformers library
        # This will cause a tracer warning but is necessary
        return int(self.current_step.item())

    def get_max_length(self):
        """Return max length."""
        return self.key_cache[0].shape[2]


class StaticCacheDecoderWrapper(nn.Module):
    """Manually manage cache as static tensors - no Cache classes."""

    def __init__(self, full_model, max_seq_len=109):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        dec_config = full_model.config.transf_decoder["config_dict"]
        self.num_layers = dec_config["num_layers"]
        self.num_heads = dec_config["num_attention_heads"]
        self.hidden_size = dec_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len

    def forward(self, input_id, encoder_hidden_states, cache_k, cache_v, step, cross_attention_mask):
        """
        Manual static cache: pre-allocated tensors, update by indexing.

        Args:
            cache_k: (num_layers, num_heads, max_seq_len, head_dim) - pre-allocated
            cache_v: (num_layers, num_heads, max_seq_len, head_dim) - pre-allocated
            step: (1,) - current position (0-indexed)
        """
        batch_size = 1
        device = input_id.device
        dtype = encoder_hidden_states.dtype

        # Use StaticSizeCache which reports correct sequence length
        self_attention_cache = StaticSizeCache(cache_k, cache_v, step)
        cross_attention_cache = DynamicCache()

        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Position tensor
        positions_input = step.view(1, 1).long()

        # Self-attention mask for attention mechanism
        # StaticSizeCache reports seq_len=step, so cache has positions 0..step-1
        # We're adding a token at position step
        # Mask should allow positions 0..step (step+1 total positions)
        mask_len = self.max_seq_len
        pos_range_mask = torch.arange(mask_len, device=device).view(1, 1, 1, -1)
        step_exp_mask = (step + 1).view(1, 1, 1, 1)  # Allow up to step (inclusive)
        should_mask = pos_range_mask >= step_exp_mask

        self_attention_mask = torch.where(
            should_mask,
            torch.full((1, 1, 1, mask_len), float("-inf"), device=device, dtype=dtype),
            torch.zeros((1, 1, 1, mask_len), device=device, dtype=dtype)
        )

        # Cross attention mask
        cross_mask_reshaped = cross_attention_mask.squeeze(1).squeeze(1)

        # Decoder forward
        decoder_outputs, updated_cache = self.decoder(
            input_ids=input_id,
            positions=positions_input,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_mask_reshaped,
            past_key_values=past_key_values,
            cache_position=None,
            kv_seq_len=None,
        )

        # Get logits
        logits = self.log_softmax(decoder_outputs).squeeze(1)

        # Extract updated cache from StaticSizeCache
        # The cache was modified in-place by the decoder
        self_attn_cache = updated_cache.self_attention_cache

        new_cache_k_list = []
        new_cache_v_list = []

        for layer_idx in range(self.num_layers):
            # StaticSizeCache stores with batch dimension: (1, num_heads, max_seq_len, head_dim)
            updated_k = self_attn_cache.key_cache[layer_idx]
            updated_v = self_attn_cache.value_cache[layer_idx]

            # Remove batch dimension: (num_heads, max_seq_len, head_dim)
            layer_k = updated_k.squeeze(0)
            layer_v = updated_v.squeeze(0)

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)  # (num_layers, num_heads, max_seq_len, head_dim)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_static(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - Manual Static Cache")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("   ✓ Loaded")

    print("\n[2/5] Wrapping decoder...")
    max_seq_len = 109  # 108 for history + 1 for current = 109 total
    wrapped = StaticCacheDecoderWrapper(model, max_seq_len=max_seq_len)
    wrapped.eval()
    print("   ✓ Wrapped")

    print("\n[3/5] Creating inputs...")
    example_input_id = torch.tensor([[13764]], dtype=torch.long)
    example_encoder_hidden = torch.randn(1, 376, 1024)
    # Static cache: pre-allocated to max_seq_len
    example_cache_k = torch.zeros(8, 8, max_seq_len, 128)
    example_cache_v = torch.zeros(8, 8, max_seq_len, 128)
    example_step = torch.tensor([0], dtype=torch.int32)
    example_cross_mask = torch.ones(1, 1, 1, 376)

    print("\n[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped,
            (example_input_id, example_encoder_hidden, example_cache_k, example_cache_v, example_step, example_cross_mask),
            check_trace=False,
        )

    logits, k, v = traced(example_input_id, example_encoder_hidden, example_cache_k, example_cache_v, example_step, example_cross_mask)
    print(f"   Output: logits={logits.shape}, cache={k.shape}")

    print(f"\n[5/5] Converting to CoreML ({precision})...")
    inputs = [
        ct.TensorType(name="input_id", shape=example_input_id.shape, dtype=np.int32),
        ct.TensorType(name="encoder_hidden_states", shape=example_encoder_hidden.shape, dtype=np.float32),
        ct.TensorType(name="cache_k", shape=example_cache_k.shape, dtype=np.float32),
        ct.TensorType(name="cache_v", shape=example_cache_v.shape, dtype=np.float32),
        ct.TensorType(name="step", shape=example_step.shape, dtype=np.int32),
        ct.TensorType(name="cross_attention_mask", shape=example_cross_mask.shape, dtype=np.float32),
    ]

    compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="new_cache_k"),
            ct.TensorType(name="new_cache_v"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
    )

    output_path = output_dir / "cohere_decoder_static.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_static(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
