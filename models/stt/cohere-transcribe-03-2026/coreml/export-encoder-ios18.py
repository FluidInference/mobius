#!/usr/bin/env python3
"""Export Cohere Transcribe encoder (with projection) to CoreML.

This exports the Conformer encoder + encoder_decoder_proj layer as a single model.

The `--target-frames` flag controls the fixed input length baked into the exported
.mlpackage. Cohere's reference configuration is 3500 frames (35 s at
hop_length=160, sr=16000). Smaller buckets (e.g. 500 / 1000 / 2000) are used by
the ANE-bucket plan — see docs/ENCODER_BUCKETS_PLAN.md — to avoid paying 35 s
of encoder compute for short audio.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq

# Conformer subsampling stride. target_frames must be divisible by this or the
# traced encoder will produce a fractional last subsampled block.
CONFORMER_SUBSAMPLE = 4

# Cohere reference input length (35 s at hop_length=160, sr=16000).
DEFAULT_TARGET_FRAMES = 3500


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
            length=feature_length,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Apply projection if it exists
        if self.encoder_decoder_proj is not None:
            hidden_states = self.encoder_decoder_proj(hidden_states)

        return hidden_states


def export_encoder(
    output_dir: Path,
    precision: str = "float16",
    target_frames: int = DEFAULT_TARGET_FRAMES,
):
    """Export the Cohere encoder to CoreML.

    Args:
        output_dir: directory to write the .mlpackage into.
        precision: "float16" or "float32" compute precision for the converter.
        target_frames: fixed input length (mel frames). Must be > 0 and
            divisible by the Conformer subsampling stride.
    """
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    if target_frames % CONFORMER_SUBSAMPLE != 0:
        raise ValueError(
            f"target_frames ({target_frames}) must be divisible by "
            f"{CONFORMER_SUBSAMPLE} (Conformer subsampling stride)"
        )

    duration_s = target_frames * 160 / 16000

    print("=" * 70)
    print("Cohere Transcribe Encoder Export")
    print("=" * 70)
    print(f"Target frames: {target_frames} ({duration_s:.2f} s at 10 ms/frame)")

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
    max_frames = target_frames

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
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=compute_precision,
    )

    # Save — encode target_frames into the filename so bucket variants
    # coexist in the same output dir without overwriting each other.
    # The default 3500-frame export keeps the legacy "cohere_encoder.mlpackage"
    # name for backwards compatibility with downstream quantization/compile
    # scripts; smaller buckets get a `_fN` suffix.
    if target_frames == DEFAULT_TARGET_FRAMES:
        output_path = output_dir / "cohere_encoder.mlpackage"
    else:
        output_path = output_dir / f"cohere_encoder_f{target_frames}.mlpackage"
    mlmodel.save(str(output_path))

    size_gb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024**3
    print(f"   ✓ Saved to: {output_path}")
    print(f"   Model size: {size_gb:.2f} GB")

    # The Conformer encoder runs 4× subsampling, so encoded frames =
    # target_frames / 4 (integer because we validated divisibility above).
    encoded_frames = target_frames // CONFORMER_SUBSAMPLE

    print("\n" + "=" * 70)
    print("ENCODER EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    print(f"\nModel inputs:")
    print(
        f"  - input_features: (1, 128, {target_frames}) float32 - "
        f"mel spectrogram ({duration_s:.1f}s max)"
    )
    print(f"  - feature_length: (1,) int32 - actual length before padding")
    print(f"\nModel output:")
    print(
        f"  - hidden_states: (1, {encoded_frames}, 1024) float16/32 - "
        f"encoder output after projection"
    )
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
    parser.add_argument(
        "--target-frames",
        type=int,
        default=DEFAULT_TARGET_FRAMES,
        help=(
            "Fixed mel-frame input length to bake into the exported model. "
            f"Default {DEFAULT_TARGET_FRAMES} = Cohere reference (35 s). "
            "Smaller values (e.g. 500/1000/2000) are used by the ANE-bucket "
            "plan to avoid paying 35 s of encoder compute for short audio. "
            f"Must be divisible by {CONFORMER_SUBSAMPLE}."
        ),
    )

    args = parser.parse_args()

    try:
        export_encoder(args.output_dir, args.precision, args.target_frames)
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
