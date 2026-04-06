#!/usr/bin/env python3
"""Export Cohere decoder that accepts pre-computed cross-attention K/V.

This optimized decoder uses the cross-KV projector output, avoiding
redundant cross-attention projection computation at every decode step.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache


class OptimizedCachedDecoderWrapper(nn.Module):
    """Decoder wrapper that accepts pre-computed cross-attention K/V."""

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

    def forward(self, input_id, cross_k, cross_v, cache_k, cache_v, step, cross_attention_mask):
        """
        Optimized decoder with pre-computed cross-attention K/V.

        Args:
            input_id: (1, 1) int64 - current token
            cross_k: (num_layers, num_heads, encoder_seq_len, head_dim) - pre-computed cross K
            cross_v: (num_layers, num_heads, encoder_seq_len, head_dim) - pre-computed cross V
            cache_k: (num_layers, num_heads, max_seq_len, head_dim) - self-attention key cache
            cache_v: (num_layers, num_heads, max_seq_len, head_dim) - self-attention value cache
            step: (1,) int32 - current decoding step (0-indexed)
            cross_attention_mask: (1, 1, 1, encoder_seq_len) - encoder attention mask

        Returns:
            logits: (1, vocab_size) - next token logits
            new_cache_k: (num_layers, num_heads, max_seq_len, head_dim)
            new_cache_v: (num_layers, num_heads, max_seq_len, head_dim)
        """
        batch_size = 1

        # Build self-attention cache with masking
        positions = torch.arange(self.max_seq_len, device=input_id.device).view(1, 1, -1, 1)
        step_expanded = step.view(1, 1, 1, 1)
        valid_mask = (positions < step_expanded).float()  # (1, 1, max_seq_len, 1)

        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        for layer_idx in range(self.num_layers):
            # Self-attention cache with masking
            layer_k = cache_k[layer_idx].unsqueeze(0)  # (1, num_heads, max_seq_len, head_dim)
            layer_v = cache_v[layer_idx].unsqueeze(0)

            # Zero out positions >= step
            layer_k_masked = layer_k * valid_mask
            layer_v_masked = layer_v * valid_mask

            self_attention_cache.update(layer_k_masked, layer_v_masked, layer_idx)

            # Cross-attention cache - use pre-computed K/V
            layer_cross_k = cross_k[layer_idx].unsqueeze(0)  # (1, num_heads, enc_seq_len, head_dim)
            layer_cross_v = cross_v[layer_idx].unsqueeze(0)

            cross_attention_cache.update(layer_cross_k, layer_cross_v, layer_idx)

        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Position tensor
        positions_input = step.view(1, 1).long()

        # Self-attention mask (extended for appending)
        mask_len = self.max_seq_len + 1  # 109 to handle appending
        pos_range = torch.arange(mask_len, device=input_id.device).view(1, 1, 1, -1)
        step_exp = step.view(1, 1, 1, 1)
        should_mask = pos_range >= step_exp

        self_attention_mask = torch.where(
            should_mask,
            torch.full((1, 1, 1, mask_len), float("-inf"), device=input_id.device, dtype=cross_k.dtype),
            torch.zeros((1, 1, 1, mask_len), device=input_id.device, dtype=cross_k.dtype)
        )

        # Cross-attention mask
        cross_mask_reshaped = cross_attention_mask.squeeze(1).squeeze(1)

        # Decoder forward pass
        decoder_outputs, updated_cache = self.decoder(
            input_ids=input_id,
            positions=positions_input,
            encoder_hidden_states=None,  # Not needed - using pre-computed cross K/V
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_mask_reshaped,
            past_key_values=past_key_values,
            cache_position=None,
            kv_seq_len=None,
        )

        # Get logits
        logits = self.log_softmax(decoder_outputs).squeeze(1)

        # Extract and pad self-attention cache (cross-attention cache remains unchanged)
        self_attn_cache = updated_cache.self_attention_cache
        new_cache_k_list = []
        new_cache_v_list = []

        for layer_idx in range(self.num_layers):
            layer_k = self_attn_cache.key_cache[layer_idx].squeeze(0)
            layer_v = self_attn_cache.value_cache[layer_idx].squeeze(0)

            # Pad to max_seq_len
            current_len = layer_k.shape[1]
            if current_len < self.max_seq_len:
                pad_len = self.max_seq_len - current_len
                layer_k = torch.nn.functional.pad(layer_k, (0, 0, 0, pad_len))
                layer_v = torch.nn.functional.pad(layer_v, (0, 0, 0, pad_len))
            elif current_len > self.max_seq_len:
                layer_k = layer_k[:, :self.max_seq_len, :]
                layer_v = layer_v[:, :self.max_seq_len, :]

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_with_cross_kv(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - Optimized with Pre-computed Cross K/V")
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
    wrapped = OptimizedCachedDecoderWrapper(model, max_seq_len=108)
    wrapped.eval()
    print("   ✓ Wrapped")

    print("\n[3/5] Creating inputs...")
    example_input_id = torch.tensor([[13764]], dtype=torch.long)
    example_cross_k = torch.randn(8, 8, 376, 128)  # Pre-computed from encoder
    example_cross_v = torch.randn(8, 8, 376, 128)
    example_cache_k = torch.zeros(8, 8, 108, 128)
    example_cache_v = torch.zeros(8, 8, 108, 128)
    example_step = torch.tensor([0], dtype=torch.int32)
    example_cross_mask = torch.ones(1, 1, 1, 376)

    print("\n[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped,
            (example_input_id, example_cross_k, example_cross_v, example_cache_k,
             example_cache_v, example_step, example_cross_mask),
            check_trace=False,
        )

    logits, k, v = traced(example_input_id, example_cross_k, example_cross_v,
                          example_cache_k, example_cache_v, example_step, example_cross_mask)
    print(f"   Output: logits={logits.shape}, cache={k.shape}")

    print(f"\n[5/5] Converting to CoreML ({precision})...")
    inputs = [
        ct.TensorType(name="input_id", shape=example_input_id.shape, dtype=np.int32),
        ct.TensorType(name="cross_k", shape=example_cross_k.shape, dtype=np.float32),
        ct.TensorType(name="cross_v", shape=example_cross_v.shape, dtype=np.float32),
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

    output_path = output_dir / "cohere_decoder_optimized.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")

    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    print("1. Run encoder: encoder_hidden = encoder(mel)")
    print("2. Project ONCE: cross_k, cross_v = projector(encoder_hidden)")
    print("3. Decode loop:")
    print("   for step in range(max_tokens):")
    print("       logits, cache_k, cache_v = decoder(")
    print("           token, cross_k, cross_v, cache_k, cache_v, step, mask)")
    print("\n   cross_k and cross_v are reused every step (computed only once!)")

    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_with_cross_kv(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
