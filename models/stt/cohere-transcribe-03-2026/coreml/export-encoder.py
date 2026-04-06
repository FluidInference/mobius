#!/usr/bin/env python3
"""Export Cohere Transcribe encoder (with projection) to CoreML.

This exports the Conformer encoder + encoder_decoder_proj layer as a single model.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq


class EncoderWrapper(nn.Module):
    """Wrapper that combines encoder + projection layer."""

    def __init__(self, encoder, encoder_decoder_proj):
        super().__init__()
        self.encoder = encoder
        self.encoder_decoder_proj = encoder_decoder_proj

    def forward(self, input_features, feature_length):
        """
        Args:
            input_features: (batch, n_mels, n_frames) mel spectrogram
            feature_length: (batch,) int32 - actual length before padding

        Returns:
            hidden_states: (batch, encoded_frames, decoder_hidden_size) - encoder output after projection
        """
        encoder_outputs = self.encoder(
            input_features=input_features,
            lengths=feature_length,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Apply projection if it exists
        if self.encoder_decoder_proj is not None:
            hidden_states = self.encoder_decoder_proj(hidden_states)

        return hidden_states


def export_encoder(output_dir: Path, precision: str = "float16"):
    """Export the Cohere encoder to CoreML."""
    print("="*70)
    print("Cohere Transcribe Encoder Export")
    print("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full model
    print("\n[1/5] Loading model from HuggingFace...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("   ✓ Model loaded")

    # Wrap encoder + projection
    print("\n[2/5] Wrapping encoder...")
    wrapped_encoder = EncoderWrapper(model.encoder, model.encoder_decoder_proj)
    wrapped_encoder.eval()
    print("   ✓ Encoder wrapped")

    # Create example inputs
    print("\n[3/5] Creating example inputs...")
    batch_size = 1
    n_mels = 128
    max_frames = 3001  # From manifest

    example_input_features = torch.randn(batch_size, n_mels, max_frames)
    example_feature_length = torch.tensor([max_frames], dtype=torch.int32)

    print(f"   Input features: {example_input_features.shape}")
    print(f"   Feature length: {example_feature_length.shape}")

    # Trace the model
    print("\n[4/5] Tracing encoder...")
    with torch.no_grad():
        traced_encoder = torch.jit.trace(
            wrapped_encoder,
            (example_input_features, example_feature_length),
            check_trace=False,  # Disable due to conditional logic
        )

    # Test traced model
    output = traced_encoder(example_input_features, example_feature_length)
    print(f"   Output shape: {output.shape}")

    # Convert to CoreML
    print(f"\n[5/5] Converting to CoreML ({precision})...")

    # Define inputs
    inputs = [
        ct.TensorType(name="input_features", shape=example_input_features.shape, dtype=np.float32),
        ct.TensorType(name="feature_length", shape=example_feature_length.shape, dtype=np.int32),
    ]

    # Set compute precision
    compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    # Convert
    mlmodel = ct.convert(
        traced_encoder,
        inputs=inputs,
        outputs=[ct.TensorType(name="hidden_states")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
    )

    # Save
    output_path = output_dir / "cohere_encoder.mlpackage"
    mlmodel.save(str(output_path))

    print(f"   ✓ Saved to: {output_path}")
    print(f"   Model size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**3:.2f} GB")

    print("\n" + "="*70)
    print("ENCODER EXPORT COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"\nModel inputs:")
    print(f"  - input_features: (1, 128, 3001) float32 - mel spectrogram")
    print(f"  - feature_length: (1,) int32 - actual length before padding")
    print(f"\nModel output:")
    print(f"  - hidden_states: (1, 376, 1024) float16/32 - encoder output after projection")
    print()


def main():
    parser = argparse.ArgumentParser(description="Export Cohere encoder to CoreML")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build"),
        help="Output directory for CoreML models"
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Model precision (default: float16)"
    )

    args = parser.parse_args()

    try:
        export_encoder(args.output_dir, args.precision)
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
