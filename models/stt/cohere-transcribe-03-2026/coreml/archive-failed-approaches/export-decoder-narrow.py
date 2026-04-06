#!/usr/bin/env python3
"""Export Cohere decoder using torch.narrow for traceable slicing."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class NarrowCachedDecoderWrapper(nn.Module):
    """Use torch.narrow for traceable slicing."""

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
        Use torch.narrow to slice cache - should be traceable.

        torch.narrow(tensor, dim, start, length) creates a view without .item()
        """
        device = input_id.device
        dtype = encoder_hidden_states.dtype

        # Build cache using torch.narrow to slice to filled positions
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        # At step=N, we want positions 0..N-1 (length=N)
        # torch.narrow(tensor, dim=2, start=0, length=step)
        # But length must be > 0, so handle step=0 specially

        for layer_idx in range(self.num_layers):
            layer_k_full = cache_k[layer_idx:layer_idx+1, :, :, :]  # (1, 8, 108, 128)
            layer_v_full = cache_v[layer_idx:layer_idx+1, :, :, :]

            # Use narrow to slice dim=2 (sequence dimension) from 0 to step
            # torch.narrow requires length > 0, so add 1 and check
            slice_len = step + 1  # At step=0, slice_len=1 (will slice 0:1, giving 1 position)

            # But we want 0:step (step positions), not 0:step+1
            # So use max(step, 1) to avoid zero-length, then slice accordingly
            actual_len = torch.clamp(step, min=torch.tensor(1, device=device))

            # Narrow from position 0 with length=actual_len
            layer_k = torch.narrow(layer_k_full, dim=2, start=0, length=int(actual_len.item()))
            layer_v = torch.narrow(layer_v_full, dim=2, start=0, length=int(actual_len.item()))

            # Only update cache if step > 0 (has previous tokens)
            if step.item() > 0:
                self_attention_cache.update(layer_k, layer_v, layer_idx)

        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Position tensor
        positions_input = step.view(1, 1).long()

        # Attention mask
        mask_len = int(step.item()) + 1  # Positions 0..step
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

        # Extract and pad cache
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
                layer_k = updated_k[:, :self.max_seq_len, :]
                layer_v = updated_v[:, :self.max_seq_len, :]

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_narrow(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - torch.narrow approach")
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
    wrapped = NarrowCachedDecoderWrapper(model, max_seq_len=108)
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

    output_path = output_dir / "cohere_decoder_narrow.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print("\n" + "="*70)
    print("EXPORT COMPLETE - torch.narrow approach")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_narrow(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
