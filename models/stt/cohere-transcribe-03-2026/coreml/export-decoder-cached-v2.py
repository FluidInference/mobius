#!/usr/bin/env python3
"""Export Cohere Transcribe decoder (cached version) to CoreML - Version 2.

This version doesn't use dynamic slicing - instead it relies on proper masking and cache management.
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


class SimplifiedCachedDecoderWrapperV2(nn.Module):
    """
    Simplified decoder wrapper without dynamic slicing to avoid CoreML shape issues.
    """

    def __init__(self, full_model, max_seq_len=108):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        self.config = full_model.config

        # Cache dimensions
        dec_config = self.config.transf_decoder["config_dict"]
        self.num_layers = dec_config["num_layers"]
        self.num_heads = dec_config["num_attention_heads"]
        self.hidden_size = dec_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_id,
        encoder_hidden_states,
        cache_k,
        cache_v,
        step,
        cross_attention_mask,
    ):
        """
        Single-step autoregressive decoding without dynamic slicing.

        Args:
            input_id: (1, 1) int64 - current token
            encoder_hidden_states: (1, encoder_frames, decoder_hidden) - encoder output
            cache_k: (num_layers, num_heads, max_seq_len, head_dim) - key cache
            cache_v: (num_layers, num_heads, max_seq_len, head_dim) - value cache
            step: (1,) int32 - current decoding step (0-indexed)
            cross_attention_mask: (1, 1, 1, encoder_frames) - encoder attention mask

        Returns:
            logits: (1, vocab_size) - next token logits
            new_cache_k: (num_layers, num_heads, max_seq_len, head_dim)
            new_cache_v: (num_layers, num_heads, max_seq_len, head_dim)
        """
        batch_size = 1

        # Build cache from input tensors
        # Key insight: Don't slice! Pass full cache and let attention masking handle it
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()

        for layer_idx in range(self.num_layers):
            layer_k = cache_k[layer_idx].unsqueeze(0)  # (1, num_heads, max_seq_len, head_dim)
            layer_v = cache_v[layer_idx].unsqueeze(0)

            # Don't truncate - pass full cache
            # The attention mechanism will use cache_position to know which parts are valid
            self_attention_cache.update(layer_k, layer_v, layer_idx)

        # Create EncoderDecoderCache
        past_key_values = EncoderDecoderCache(self_attention_cache, cross_attention_cache)

        # Create position tensor using only tensor operations
        positions = step.view(1, 1).long()

        # Create attention mask that covers all positions up to max_seq_len
        # Positions beyond current step will be masked
        cache_position = torch.arange(self.max_seq_len, device=input_id.device).unsqueeze(0)  # (1, max_seq_len)

        # Mask positions that are beyond the current sequence length
        # self_attention_mask: (1, 1, 1, max_seq_len)
        # Mask = -inf where we should NOT attend
        valid_mask = cache_position <= step.view(1, 1)  # (1, max_seq_len) - positions <= current step are valid
        self_attention_mask = torch.where(
            valid_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, 1, max_seq_len)
            torch.zeros(1, 1, 1, self.max_seq_len, device=input_id.device, dtype=encoder_hidden_states.dtype),
            torch.full((1, 1, 1, self.max_seq_len), float("-inf"), device=input_id.device, dtype=encoder_hidden_states.dtype)
        )

        # Cross attention mask
        cross_mask_reshaped = cross_attention_mask.squeeze(1).squeeze(1) if cross_attention_mask is not None else None

        # Call decoder
        decoder_outputs, updated_cache = self.decoder(
            input_ids=input_id,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_mask_reshaped,
            past_key_values=past_key_values,
            cache_position=cache_position,
            kv_seq_len=None,
        )

        # Project to vocab
        logits = self.log_softmax(decoder_outputs)
        logits = logits.squeeze(1)

        # Extract updated cache and pad to max_seq_len
        self_attn_cache = updated_cache.self_attention_cache
        new_cache_k_list = []
        new_cache_v_list = []

        for layer_idx in range(self.num_layers):
            layer_k = self_attn_cache.key_cache[layer_idx].squeeze(0)  # (num_heads, seq_len, head_dim)
            layer_v = self_attn_cache.value_cache[layer_idx].squeeze(0)

            # Pad using F.pad (no conditionals)
            current_len = layer_k.shape[1]
            pad_len = self.max_seq_len - current_len
            layer_k = torch.nn.functional.pad(layer_k, (0, 0, 0, pad_len))
            layer_v = torch.nn.functional.pad(layer_v, (0, 0, 0, pad_len))

            new_cache_k_list.append(layer_k)
            new_cache_v_list.append(layer_v)

        new_cache_k = torch.stack(new_cache_k_list, dim=0)
        new_cache_v = torch.stack(new_cache_v_list, dim=0)

        return logits, new_cache_k, new_cache_v


def export_decoder_cached(output_dir: Path, precision: str = "float16"):
    """Export the cached Cohere decoder to CoreML."""
    print("="*70)
    print("Cohere Transcribe Decoder (Cached) Export - V2")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("   ✓ Model loaded")

    dec_config = model.config.transf_decoder["config_dict"]
    num_layers = dec_config["num_layers"]
    num_heads = dec_config["num_attention_heads"]
    max_seq_len = 108
    hidden_size = dec_config["hidden_size"]
    head_dim = hidden_size // num_heads

    print("\n[2/5] Wrapping decoder...")
    wrapped_decoder = SimplifiedCachedDecoderWrapperV2(model, max_seq_len=max_seq_len)
    wrapped_decoder.eval()
    print(f"   ✓ Decoder wrapped (max_seq_len={max_seq_len})")

    print("\n[3/5] Creating example inputs...")
    batch_size = 1
    encoder_frames = 376
    decoder_hidden_size = dec_config["hidden_size"]

    example_input_id = torch.tensor([[13764]], dtype=torch.long)
    example_encoder_hidden = torch.randn(batch_size, encoder_frames, decoder_hidden_size)
    example_cache_k = torch.zeros(num_layers, num_heads, max_seq_len, head_dim)
    example_cache_v = torch.zeros(num_layers, num_heads, max_seq_len, head_dim)
    example_step = torch.tensor([0], dtype=torch.int32)
    example_cross_mask = torch.ones(batch_size, 1, 1, encoder_frames)

    print(f"   Shapes: input_id={example_input_id.shape}, cache={example_cache_k.shape}")

    print("\n[4/5] Tracing decoder...")
    with torch.no_grad():
        traced_decoder = torch.jit.trace(
            wrapped_decoder,
            (example_input_id, example_encoder_hidden, example_cache_k, example_cache_v, example_step, example_cross_mask),
            check_trace=False,
        )

    logits, new_k, new_v = traced_decoder(example_input_id, example_encoder_hidden, example_cache_k, example_cache_v, example_step, example_cross_mask)
    print(f"   Logits: {logits.shape}, Cache: {new_k.shape}")

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
        traced_decoder,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="new_cache_k"),
            ct.TensorType(name="new_cache_v"),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
    )

    output_path = output_dir / "cohere_decoder_cached.mlpackage"
    mlmodel.save(str(output_path))

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2:.1f} MB")
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Export Cohere decoder (cached) to CoreML V2")
    parser.add_argument("--output-dir", type=Path, default=Path("build"), help="Output directory")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16", help="Precision")
    args = parser.parse_args()

    try:
        export_decoder_cached(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
