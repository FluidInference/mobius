#!/usr/bin/env python3
"""Export Cohere decoder with fixed cache handling - no sliding window."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class FixedCachedDecoderWrapper(nn.Module):
    """Fixed cache handling - use stable positions, keep FIRST 108 tokens."""

    def __init__(self, full_model, max_seq_len=108):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        self.config = full_model.config
        dec_config = full_model.config.transf_decoder["config_dict"]
        self.num_layers = dec_config["num_layers"]
        self.num_heads = dec_config["num_attention_heads"]
        self.hidden_size = dec_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len

    def forward(self, input_id, encoder_hidden_states, cache_k, cache_v, step, cross_attention_mask):
        """
        Fixed cache handling: slice cache to only filled positions (0:step).
        """
        batch_size = 1
        device = input_id.device
        dtype = encoder_hidden_states.dtype

        # Build cache with ONLY the filled positions (0 to step-1)
        # NOTE: Using .item() causes tracer warnings but is necessary for dynamic slicing
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        step_int = int(step.item())

        for layer_idx in range(self.num_layers):
            # Slice to only include filled positions (0 to step-1)
            # At step=0, this is empty (0:0 slice)
            # At step=10, this is positions 0:10
            if step_int > 0:
                layer_k = cache_k[layer_idx:layer_idx+1, :, :step_int, :]
                layer_v = cache_v[layer_idx:layer_idx+1, :, :step_int, :]
                self_attention_cache.update(layer_k, layer_v, layer_idx)

        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Position tensor
        positions_input = step.view(1, 1).long()

        # Self-attention mask
        # At step=N: cache has N positions (0..N-1), we're adding position N
        # Total positions after this step: N+1
        # Mask should cover these N+1 positions and allow all of them
        mask_len = step_int + 1  # Total positions: 0..step
        pos_range = torch.arange(mask_len, device=device).view(1, 1, 1, -1)

        # Don't mask anything - all positions 0..step should be visible
        # (causal masking is already handled by the decoder)
        self_attention_mask = torch.zeros((1, 1, 1, mask_len), device=device, dtype=dtype)

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

        # Extract updated cache and pad to max_seq_len
        self_attn_cache = updated_cache.self_attention_cache

        new_cache_k_list = []
        new_cache_v_list = []

        for layer_idx in range(self.num_layers):
            updated_k = self_attn_cache.key_cache[layer_idx].squeeze(0)  # (num_heads, step+1, head_dim)
            updated_v = self_attn_cache.value_cache[layer_idx].squeeze(0)

            # Pad to max_seq_len (always pad at the end)
            current_len = updated_k.shape[1]
            if current_len < self.max_seq_len:
                pad_len = self.max_seq_len - current_len
                layer_k = torch.nn.functional.pad(updated_k, (0, 0, 0, pad_len))
                layer_v = torch.nn.functional.pad(updated_v, (0, 0, 0, pad_len))
            else:
                # If we've exceeded max_seq_len, keep the LAST max_seq_len tokens
                layer_k = updated_k[:, -self.max_seq_len:, :]
                layer_v = updated_v[:, -self.max_seq_len:, :]

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_fixed(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - Fixed Cache Handling")
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
    wrapped = FixedCachedDecoderWrapper(model, max_seq_len=108)
    wrapped.eval()
    print("   ✓ Wrapped")

    print("\n[3/5] Creating inputs...")
    example_input_id = torch.tensor([[13764]], dtype=torch.long)
    example_encoder_hidden = torch.randn(1, 376, 1024)
    example_cache_k = torch.zeros(8, 8, 108, 128)
    example_cache_v = torch.zeros(8, 8, 108, 128)
    example_step = torch.tensor([0], dtype=torch.int32)
    example_cross_mask = torch.ones(1, 1, 1, 376)

    print("\n[4/5] Scripting (using torch.jit.script for control flow support)...")
    with torch.no_grad():
        scripted = torch.jit.script(wrapped)

    logits, k, v = scripted(example_input_id, example_encoder_hidden, example_cache_k, example_cache_v, example_step, example_cross_mask)
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
        scripted,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="new_cache_k"),
            ct.TensorType(name="new_cache_v"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
    )

    output_path = output_dir / "cohere_decoder_fixed.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print("\n" + "="*70)
    print("EXPORT COMPLETE - Fixed cache handling (no sliding window)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_fixed(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
