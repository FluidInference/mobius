#!/usr/bin/env python3
"""Export Cohere decoder with stateless approach - no cache, reprocess all tokens."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq


class StatelessDecoderWrapper(nn.Module):
    """
    Stateless decoder: reprocess all tokens at each step (no cache).

    Simpler than cache management and fully traceable. At step N:
    - Input: all N tokens (0..N-1) and encoder hidden states
    - Output: logits for position N-1

    O(n^2) complexity but simpler and avoids all cache-related issues.
    """

    def __init__(self, full_model, max_seq_len=108):
        super().__init__()
        self.decoder = full_model.transf_decoder
        self.log_softmax = full_model.log_softmax
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, encoder_hidden_states, cross_attention_mask):
        """
        Process all tokens without cache.

        Args:
            input_ids: All tokens so far, shape (1, seq_len)
            encoder_hidden_states: Encoder output, shape (1, enc_len, hidden_dim)
            cross_attention_mask: Mask for encoder, shape (1, 1, 1, enc_len)

        Returns:
            logits: Log probabilities for the last token, shape (1, vocab_size)
        """
        device = input_ids.device
        dtype = encoder_hidden_states.dtype
        seq_len = input_ids.shape[1]

        # Position IDs for all tokens
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        # Causal attention mask (already handled by decoder, pass None)
        self_attention_mask = None

        # Cross attention mask
        cross_mask_reshaped = cross_attention_mask.squeeze(1).squeeze(1)  # (1, enc_len)

        # Decoder forward - no cache
        decoder_outputs, _ = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_mask_reshaped,
            past_key_values=None,  # No cache!
            cache_position=None,
            kv_seq_len=None,
        )

        # Get logits for the LAST token
        last_hidden = decoder_outputs[:, -1:, :]  # (1, 1, hidden_dim)
        logits = self.log_softmax(last_hidden).squeeze(1)  # (1, vocab_size)

        return logits


def export_decoder_stateless(output_dir: Path, precision: str = "float16"):
    print("="*70)
    print("Cohere Decoder Export - Stateless (No Cache)")
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
    wrapped = StatelessDecoderWrapper(model, max_seq_len=108)
    wrapped.eval()
    print("   ✓ Wrapped")

    print("\n[3/5] Creating inputs...")
    # Start with just the BOS token
    example_input_ids = torch.tensor([[13764]], dtype=torch.long)  # (1, 1)
    example_encoder_hidden = torch.randn(1, 438, 1024)  # 3500 frames @ 35s -> 438 outputs
    example_cross_mask = torch.ones(1, 1, 1, 438)

    print("\n[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapped,
            (example_input_ids, example_encoder_hidden, example_cross_mask),
            check_trace=False,
        )

    logits = traced(example_input_ids, example_encoder_hidden, example_cross_mask)
    print(f"   Output: logits={logits.shape}")

    # Test with 2 tokens
    example_input_ids_2 = torch.tensor([[13764, 7]], dtype=torch.long)
    logits_2 = traced(example_input_ids_2, example_encoder_hidden, example_cross_mask)
    print(f"   Output (2 tokens): logits={logits_2.shape}")

    print(f"\n[5/5] Converting to CoreML ({precision})...")

    # We need to use flexible shapes for input_ids since seq_len varies
    # CoreML supports enumerated shapes
    inputs = [
        ct.TensorType(
            name="input_ids",
            shape=ct.EnumeratedShapes(shapes=[[1, i] for i in range(1, 109)]),  # 1 to 108 tokens
            dtype=np.int32
        ),
        ct.TensorType(name="encoder_hidden_states", shape=example_encoder_hidden.shape, dtype=np.float32),
        ct.TensorType(name="cross_attention_mask", shape=example_cross_mask.shape, dtype=np.float32),
    ]

    # Neural Network format requires iOS 14 or lower (iOS 15+ forces ML Program)
    # Note: Neural Network doesn't support compute_precision, FP16 conversion happens differently
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[
            ct.TensorType(name="logits"),
        ],
        minimum_deployment_target=ct.target.iOS14,
        convert_to="neuralnetwork",  # Force Neural Network format for .mlmodelc support
    )

    # Convert to FP16 for Neural Network format
    if precision == "float16":
        from coremltools.models.neural_network import quantization_utils
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    output_path = output_dir / "cohere_decoder_stateless.mlpackage"
    mlmodel.save(str(output_path))

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**2
    print(f"   ✓ Saved: {output_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print("\n" + "="*70)
    print("EXPORT COMPLETE - Stateless (No Cache, O(n^2))")
    print("="*70)
    print("\nKey difference: Reprocesses all tokens at each step")
    print("Simpler, fully traceable, but slower O(n^2) complexity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    try:
        export_decoder_stateless(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
