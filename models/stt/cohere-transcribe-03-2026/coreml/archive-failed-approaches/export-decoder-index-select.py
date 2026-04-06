#!/usr/bin/env python3
"""Export Cohere decoder using torch.index_select for traceable slicing."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class IndexSelectCachedDecoderWrapper(nn.Module):
    """
    Use torch.index_select for traceable cache slicing.

    Key insight: Create indices tensor based on step, then use index_select
    to slice cache. This is fully traceable without .item().
    """

    def __init__(self, full_model, max_seq_len=108):
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
        Use index_select to slice cache based on step - fully traceable.

        At step N:
        - Select positions 0..N-1 from cache (N filled positions)
        - Process token at position N
        - Return cache with positions 0..N filled
        """
        device = input_id.device
        dtype = encoder_hidden_states.dtype

        # Build cache using index_select for slicing
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        # Create indices for filled positions: 0..step-1
        # At step=0, we want empty cache (no indices)
        # At step=N, we want indices [0, 1, ..., N-1]

        # Use arange and masking to create indices
        all_indices = torch.arange(self.max_seq_len, device=device)  # (108,)
        step_expanded = step.expand(self.max_seq_len)  # (108,) all same value

        # Mask: True for positions < step
        mask = all_indices < step_expanded  # (108,) boolean

        # Get count of True values (number of filled positions)
        # This is a bit tricky - we need to handle step=0 case
        # Use cumsum trick: if step=3, mask=[T,T,T,F,F,...], sum=3
        num_filled = mask.sum()

        # For step > 0, select the filled positions
        # For step = 0, skip cache entirely
        for layer_idx in range(self.num_layers):
            layer_k_full = cache_k[layer_idx:layer_idx+1, :, :, :]  # (1, 8, 108, 128)
            layer_v_full = cache_v[layer_idx:layer_idx+1, :, :, :]

            # Use masked_select to get filled positions
            # Shape: (1, 8, 108, 128) → (1, 8, num_filled, 128)
            # But masked_select flattens, so we need a different approach

            # Alternative: Use gather to select specific indices
            # We need indices shape (1, 8, num_filled, 128)
            # But num_filled is dynamic...

            # Actually, simpler approach: multiply by mask and use masking in attention
            # Pass full cache but zeros for unfilled positions
            mask_reshaped = mask.view(1, 1, -1, 1)  # (1, 1, 108, 1)
            layer_k_masked = layer_k_full * mask_reshaped.float()
            layer_v_masked = layer_v_full * mask_reshaped.float()

            # Only update cache if step > 0
            if step.item() > 0:
                # Now we need to slice to only include filled positions
                # Use narrow with dynamic length - but this needs .item()!
                # Or use nonzero to get indices
                filled_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # (num_filled,)

                # index_select along dim=2 (sequence dimension)
                # But index_select needs a 1D indices tensor
                # And layer_k has shape (1, 8, 108, 128)
                # We want to select along dim=2
                layer_k_selected = torch.index_select(layer_k_full.squeeze(0), dim=1, index=filled_indices)
                layer_v_selected = torch.index_select(layer_v_full.squeeze(0), dim=1, index=filled_indices)

                # Add back batch dimension
                layer_k = layer_k_selected.unsqueeze(0)  # (1, 8, num_filled, 128)
                layer_v = layer_v_selected.unsqueeze(0)

                self_attention_cache.update(layer_k, layer_v, layer_idx)

        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Position tensor
        positions_input = step.view(1, 1).long()

        # Attention mask for filled + current position
        mask_len = step + 1  # Dynamic size!
        # But we need a fixed size tensor for CoreML...
        # This won't work either

        # Use max_seq_len + 1 and mask unused positions
        full_mask_len = self.max_seq_len + 1
        pos_range = torch.arange(full_mask_len, device=device).view(1, 1, 1, -1)
        step_exp = step.view(1, 1, 1, 1)

        should_mask = pos_range > step_exp
        self_attention_mask = torch.where(
            should_mask,
            torch.full((1, 1, 1, full_mask_len), float("-inf"), device=device, dtype=dtype),
            torch.zeros((1, 1, 1, full_mask_len), device=device, dtype=dtype)
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

        # Extract cache and pad to max_seq_len
        self_attn_cache = updated_cache.self_attention_cache
        new_cache_k_list = []
        new_cache_v_list = []

        for layer_idx in range(self.num_layers):
            updated_k = self_attn_cache.key_cache[layer_idx].squeeze(0)
            updated_v = self_attn_cache.value_cache[layer_idx].squeeze(0)

            current_len = updated_k.shape[1]
            if current_len < self.max_seq_len:
                pad_len = self.max_seq_len - current_len
                layer_k = torch.nn.functional.pad(updated_k, (0, 0, 0, pad_len))
                layer_v = torch.nn.functional.pad(updated_v, (0, 0, 0, pad_len))
            else:
                # Keep FIRST max_seq_len positions
                layer_k = updated_k[:, :self.max_seq_len, :]
                layer_v = updated_v[:, :self.max_seq_len, :]

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_index_select(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - torch.index_select approach")
    print("="*70)
    print("\nNOTE: This approach still uses .item() in some places")
    print("Attempting anyway to see if it's more traceable than narrow...\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("   ✓ Loaded")

    print("\n[2/5] Wrapping decoder...")
    wrapped = IndexSelectCachedDecoderWrapper(model, max_seq_len=108)
    wrapped.eval()
    print("   ✓ Wrapped")

    print("\n[3/5] Creating inputs...")
    example_input_id = torch.tensor([[13764]], dtype=torch.long)
    example_encoder_hidden = torch.randn(1, 376, 1024)
    example_cache_k = torch.zeros(8, 8, 108, 128)
    example_cache_v = torch.zeros(8, 8, 108, 128)
    example_step = torch.tensor([0], dtype=torch.int32)
    example_cross_mask = torch.ones(1, 1, 1, 376)

    print("\n[4/5] Tracing...")
    try:
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapped,
                (example_input_id, example_encoder_hidden, example_cache_k, example_cache_v, example_step, example_cross_mask),
                check_trace=False,
            )

        logits, k, v = traced(example_input_id, example_encoder_hidden, example_cache_k, example_cache_v, example_step, example_cross_mask)
        print(f"   Output: logits={logits.shape}, cache={k.shape}")
    except Exception as e:
        print(f"\n❌ Tracing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n[5/5] Converting to CoreML ({precision})...")
    try:
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

        output_path = output_dir / "cohere_decoder_index_select.mlpackage"
        mlmodel.save(str(output_path))

        size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
        print(f"   ✓ Saved: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print("\n" + "="*70)
        print("EXPORT COMPLETE - torch.index_select approach")
        print("="*70)
    except Exception as e:
        print(f"\n❌ CoreML conversion failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_index_select(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
