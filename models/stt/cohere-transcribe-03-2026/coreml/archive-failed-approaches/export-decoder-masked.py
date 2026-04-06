#!/usr/bin/env python3
"""Export Cohere decoder using attention masking - NO slicing, fully traceable."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class MaskedCachedDecoderWrapper(nn.Module):
    """
    Use attention masking instead of cache slicing - fully traceable.

    Key insight: Pass full 108-position cache to DynamicCache, but use attention
    mask to hide positions >= step. This avoids any need for .item() or slicing.
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
        Fully traceable forward pass using attention masking.

        At step=N:
        - Cache positions 0..N-1 contain previous tokens
        - We mask positions >= N in the attention
        - Decoder processes position N (current token)
        - Output cache has positions 0..N filled
        """
        device = input_id.device
        dtype = encoder_hidden_states.dtype

        # Build cache - pass full cache to DynamicCache
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        for layer_idx in range(self.num_layers):
            # Pass FULL cache (all 108 positions)
            layer_k = cache_k[layer_idx:layer_idx+1, :, :, :]
            layer_v = cache_v[layer_idx:layer_idx+1, :, :, :]
            self_attention_cache.update(layer_k, layer_v, layer_idx)

        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Position tensor
        positions_input = step.view(1, 1).long()

        # Attention mask: Hide positions >= step
        # Create mask for all possible positions (max_seq_len + 1 for appending)
        mask_len = self.max_seq_len + 1  # 109 positions (0..108)
        pos_range = torch.arange(mask_len, device=device).view(1, 1, 1, -1)  # (1, 1, 1, 109)
        step_exp = step.view(1, 1, 1, 1)  # (1, 1, 1, 1)

        # Mask positions < step: visible (0.0)
        # Mask positions >= step: hidden (-inf), EXCEPT position step itself (current token)
        # So positions 0..step-1 are visible, position step is visible, positions > step are masked

        # Actually, wait. At step=N:
        # - Cache has positions 0..N-1 (N positions)
        # - We're adding position N (1 new position)
        # - Total: N+1 positions (0..N)
        # - We should mask positions > N (i.e., positions >= N+1)

        should_mask = pos_range > step_exp  # Mask positions > step (allow 0..step)

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

        # Extract cache and ensure it's exactly max_seq_len positions
        self_attn_cache = updated_cache.self_attention_cache
        new_cache_k_list = []
        new_cache_v_list = []

        for layer_idx in range(self.num_layers):
            updated_k = self_attn_cache.key_cache[layer_idx].squeeze(0)  # (num_heads, seq_len, head_dim)
            updated_v = self_attn_cache.value_cache[layer_idx].squeeze(0)

            # Ensure exactly max_seq_len positions
            current_len = updated_k.shape[1]

            # Pad or truncate to max_seq_len
            if current_len < self.max_seq_len:
                # Pad at the end
                pad_len = self.max_seq_len - current_len
                layer_k = torch.nn.functional.pad(updated_k, (0, 0, 0, pad_len))
                layer_v = torch.nn.functional.pad(updated_v, (0, 0, 0, pad_len))
            elif current_len > self.max_seq_len:
                # Keep FIRST max_seq_len positions (not last!)
                # This keeps positions 0..107, drops 108+
                layer_k = updated_k[:, :self.max_seq_len, :]
                layer_v = updated_v[:, :self.max_seq_len, :]
            else:
                layer_k = updated_k
                layer_v = updated_v

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_masked(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - Attention Masking (Fully Traceable)")
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
    wrapped = MaskedCachedDecoderWrapper(model, max_seq_len=108)
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

    output_path = output_dir / "cohere_decoder_masked.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print("\n" + "="*70)
    print("EXPORT COMPLETE - Attention Masking (No slicing, fully traceable)")
    print("="*70)
    print("\nKey difference: Uses attention mask to hide unused cache positions")
    print("instead of slicing. Should work correctly after CoreML conversion!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_masked(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
