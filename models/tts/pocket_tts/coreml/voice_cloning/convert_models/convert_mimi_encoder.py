"""Convert Mimi encoder to CoreML for voice cloning.

Creates a CoreML model that encodes audio to voice conditioning embeddings.

Input:  audio [1, 1, T] at 24kHz (T should be multiple of 1920)
Output: conditioning [1, num_frames, 1024]

Usage:
    python convert_mimi_encoder.py
    python convert_mimi_encoder.py --output mimi_encoder.mlpackage
"""
import argparse
import os
import sys
from pathlib import Path

# Disable beartype before importing pocket_tts (interferes with JIT tracing)
# Monkey-patch beartype to be a no-op decorator
import beartype
_original_beartype = beartype.beartype
beartype.beartype = lambda func=None, **kwargs: func if func else (lambda f: f)

import numpy as np
import torch
import coremltools as ct

# Add project paths
SCRIPT_DIR = Path(__file__).parent.absolute()
CONVERT_MODELS_DIR = SCRIPT_DIR.parent
COREML_DIR = CONVERT_MODELS_DIR.parent
PROJECT_DIR = COREML_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(CONVERT_MODELS_DIR / "traceable"))

from traceable_mimi_encoder import TraceableMimiEncoderSimple


def convert(output_path: str = "mimi_encoder.mlpackage", audio_seconds: float = 10.0):
    """Convert Mimi encoder to CoreML.

    Args:
        output_path: Output .mlpackage path
        audio_seconds: Example audio duration for tracing (affects shape flexibility)
    """
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    if not model.has_voice_cloning:
        print("ERROR: Voice cloning not available. Accept terms at:")
        print("  https://huggingface.co/kyutai/pocket-tts")
        print("Then run: huggingface-cli login")
        sys.exit(1)

    print("Creating traceable Mimi encoder...")
    traceable = TraceableMimiEncoderSimple.from_tts_model(model)
    traceable.eval()

    # Create example input (10 seconds of audio at 24kHz)
    sample_rate = 24000
    frame_size = 1920  # 80ms frames
    num_samples = int(audio_seconds * sample_rate)
    # Pad to frame boundary
    num_samples = ((num_samples + frame_size - 1) // frame_size) * frame_size

    print(f"Tracing with example audio: {num_samples} samples ({num_samples / sample_rate:.1f}s)")
    example_audio = torch.randn(1, 1, num_samples)

    # Test forward pass
    with torch.no_grad():
        output = traceable(example_audio)
        print(f"Output shape: {output.shape}")
        expected_frames = num_samples // frame_size
        print(f"Expected frames: {expected_frames}")

    # Trace
    print("\nTracing model...")
    traced = torch.jit.trace(traceable, example_audio)

    # Convert to CoreML
    print("Converting to CoreML...")

    # Use flexible shape for audio length
    audio_shape = ct.Shape(
        shape=(1, 1, ct.RangeDim(lower_bound=frame_size, upper_bound=sample_rate * 60))  # Up to 60s
    )

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="audio", shape=audio_shape, dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="conditioning", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    # Save
    output_path = COREML_DIR / output_path
    print(f"Saving to {output_path}...")
    mlmodel.save(str(output_path))

    # Also export speaker projection weight separately for verification
    speaker_proj_path = COREML_DIR / "constants" / "speaker_proj_weight.bin"
    speaker_proj_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_proj = model.flow_lm.speaker_proj_weight.data.cpu().numpy().astype(np.float32)
    speaker_proj.tofile(str(speaker_proj_path))
    print(f"Saved speaker projection weight to {speaker_proj_path}")
    print(f"  Shape: {speaker_proj.shape}")

    print("\n✅ Conversion complete!")
    print(f"Model: {output_path}")

    # Verify
    print("\nVerifying CoreML model...")
    loaded = ct.models.MLModel(str(output_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    test_audio = np.random.randn(1, 1, frame_size * 10).astype(np.float32)  # 10 frames
    result = loaded.predict({"audio": test_audio})
    coreml_out = result["conditioning"]
    print(f"CoreML output shape: {coreml_out.shape}")

    # Compare with PyTorch
    with torch.no_grad():
        torch_out = traceable(torch.from_numpy(test_audio)).numpy()
    print(f"PyTorch output shape: {torch_out.shape}")

    diff = np.abs(coreml_out - torch_out).max()
    print(f"Max difference: {diff:.6f}")

    if diff < 0.01:
        print("✅ Verification passed!")
    else:
        print("⚠️  Large difference detected, check model")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert Mimi encoder to CoreML")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="mimi_encoder.mlpackage",
        help="Output .mlpackage path"
    )
    parser.add_argument(
        "--audio-seconds",
        type=float,
        default=10.0,
        help="Example audio duration for tracing"
    )
    args = parser.parse_args()

    convert(args.output, args.audio_seconds)


if __name__ == "__main__":
    main()
