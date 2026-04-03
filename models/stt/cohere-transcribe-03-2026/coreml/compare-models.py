#!/usr/bin/env python3
"""Compare PyTorch vs CoreML outputs for Cohere Transcribe 03-2026.

Validates numerical parity between the original PyTorch model and CoreML conversion.

Usage:
  uv run python compare-models.py --audio-file test.wav --coreml-dir ./build/cohere-transcribe
  uv run python compare-models.py --audio-file test.wav --coreml-dir ./build/cohere-transcribe --language en
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

DEFAULT_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
SAMPLE_RATE = 16000


def _load_audio(audio_file: Path, max_duration: float = 30.0) -> np.ndarray:
    """Load and preprocess audio file."""
    import librosa

    data, sr = sf.read(str(audio_file), dtype="float32")

    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        typer.echo(f"  Resampling from {sr} Hz to {SAMPLE_RATE} Hz")
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Truncate if too long
    max_samples = int(max_duration * SAMPLE_RATE)
    if len(data) > max_samples:
        typer.echo(f"  Truncating audio to {max_duration}s")
        data = data[:max_samples]

    return data


def _compare_tensors(
    pytorch_output: np.ndarray,
    coreml_output: np.ndarray,
    name: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Compare two tensors and report statistics."""
    typer.echo(f"\n{name}:")
    typer.echo(f"  PyTorch shape: {pytorch_output.shape}")
    typer.echo(f"  CoreML shape:  {coreml_output.shape}")

    if pytorch_output.shape != coreml_output.shape:
        typer.echo("  ❌ SHAPE MISMATCH")
        return False

    # Compute error metrics
    abs_diff = np.abs(pytorch_output - coreml_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)

    max_abs = np.max(abs_diff)
    max_rel = np.max(rel_diff)
    mean_abs = np.mean(abs_diff)
    mean_rel = np.mean(rel_diff)

    typer.echo(f"  Max absolute error: {max_abs:.6f}")
    typer.echo(f"  Max relative error: {max_rel:.6f}")
    typer.echo(f"  Mean absolute error: {mean_abs:.6f}")
    typer.echo(f"  Mean relative error: {mean_rel:.6f}")

    # Check if within tolerance
    close = np.allclose(pytorch_output, coreml_output, rtol=rtol, atol=atol)

    if close:
        typer.echo("  ✓ Within tolerance")
    else:
        typer.echo(f"  ❌ Outside tolerance (rtol={rtol}, atol={atol})")

    return close


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def compare(
    audio_file: Path = typer.Option(
        ...,
        "--audio-file",
        exists=True,
        help="Path to test audio file (WAV, 16kHz)",
    ),
    coreml_dir: Path = typer.Option(
        ...,
        "--coreml-dir",
        exists=True,
        help="Directory containing CoreML packages",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="HuggingFace model ID",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Language code (en, fr, de, etc.)",
    ),
    rtol: float = typer.Option(
        1e-3,
        "--rtol",
        help="Relative tolerance for comparison",
    ),
    atol: float = typer.Option(
        1e-5,
        "--atol",
        help="Absolute tolerance for comparison",
    ),
) -> None:
    """Compare PyTorch and CoreML outputs."""

    typer.echo("=" * 60)
    typer.echo("Cohere Transcribe: PyTorch vs CoreML Comparison")
    typer.echo("=" * 60)

    # Load audio
    typer.echo(f"\nLoading audio: {audio_file}")
    audio_data = _load_audio(audio_file)
    typer.echo(f"  Duration: {len(audio_data) / SAMPLE_RATE:.2f}s")
    typer.echo(f"  Samples: {len(audio_data)}")

    # Load PyTorch model
    typer.echo(f"\nLoading PyTorch model: {model_id}")
    try:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    except ImportError:
        typer.echo("ERROR: transformers not found.")
        typer.echo("Requires: pip install 'transformers>=4.57.0'")
        raise typer.Exit(1)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Run PyTorch inference
    typer.echo("\n[PyTorch] Running inference...")
    with torch.no_grad():
        # Prepare inputs
        inputs = processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )

        # Run audio encoder
        input_features = inputs["input_features"]
        typer.echo(f"  Input features shape: {input_features.shape}")

        # Pad to fixed shape (matching CoreML conversion)
        fixed_time_frames = 3000  # From conversion script
        actual_time_frames = input_features.shape[2]

        if actual_time_frames < fixed_time_frames:
            typer.echo(f"  Padding PyTorch input from {actual_time_frames} to {fixed_time_frames} frames")
            pad_width = ((0, 0), (0, 0), (0, fixed_time_frames - actual_time_frames))
            input_features_padded = torch.from_numpy(
                np.pad(input_features.numpy(), pad_width, mode='constant', constant_values=0)
            )
        else:
            input_features_padded = input_features

        # Create length tensor with fixed value (matching conversion wrapper)
        batch_size = input_features_padded.shape[0]
        length = torch.full((batch_size,), fixed_time_frames, dtype=torch.int64)

        # Run encoder (model.encoder, not model.audio_encoder)
        encoder_output, _ = model.encoder(
            input_features=input_features_padded,
            length=length
        )
        pytorch_encoder_output = encoder_output.numpy()

        typer.echo(f"  Encoder output shape: {pytorch_encoder_output.shape}")

    # Load CoreML models
    typer.echo("\n[CoreML] Loading models...")

    audio_encoder_path = coreml_dir / "cohere_audio_encoder.mlpackage"
    if not audio_encoder_path.exists():
        typer.echo(f"ERROR: Audio encoder not found at {audio_encoder_path}")
        raise typer.Exit(1)

    coreml_encoder = ct.models.MLModel(str(audio_encoder_path))
    typer.echo(f"  Loaded: {audio_encoder_path.name}")

    # Run CoreML inference
    typer.echo("\n[CoreML] Running inference...")

    # Prepare input for CoreML (needs to match traced shape)
    coreml_input = input_features.numpy().astype(np.float32)

    # Pad to fixed shape if needed (CoreML model expects fixed size)
    expected_time_frames = 3000  # From conversion script
    actual_time_frames = coreml_input.shape[2]

    if actual_time_frames != expected_time_frames:
        typer.echo(f"  Padding input from {actual_time_frames} to {expected_time_frames} frames")
        pad_width = ((0, 0), (0, 0), (0, expected_time_frames - actual_time_frames))
        coreml_input = np.pad(coreml_input, pad_width, mode='constant', constant_values=0)

    typer.echo(f"  CoreML input shape: {coreml_input.shape}")

    encoder_result = coreml_encoder.predict({"input_features": coreml_input})
    coreml_encoder_output = encoder_result["encoder_output"]

    typer.echo(f"  CoreML output shape: {coreml_encoder_output.shape}")

    # Compare outputs
    typer.echo("\n" + "=" * 60)
    typer.echo("Comparison Results")
    typer.echo("=" * 60)

    # Compare full outputs (both PyTorch and CoreML now use same padding)
    encoder_match = _compare_tensors(
        pytorch_encoder_output,
        coreml_encoder_output,
        "Audio Encoder Output (Full)",
        rtol=rtol,
        atol=atol,
    )

    # Also compare just the valid portion (before padding)
    # Calculate valid frames from original audio length
    valid_time_frames = actual_time_frames  # Original time frames before padding
    valid_output_frames = pytorch_encoder_output.shape[1] * valid_time_frames // fixed_time_frames
    typer.echo(f"\nNote: Original audio had {actual_time_frames}/{fixed_time_frames} frames")
    typer.echo(f"      Corresponding to ~{valid_output_frames}/{pytorch_encoder_output.shape[1]} output frames")

    # Summary
    typer.echo("\n" + "=" * 60)
    typer.echo("Summary")
    typer.echo("=" * 60)

    results = {
        "audio_encoder": encoder_match,
    }

    all_passed = all(results.values())

    for component, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        typer.echo(f"  {component}: {status}")

    if all_passed:
        typer.echo("\n✓ All components within tolerance!")
    else:
        typer.echo("\n❌ Some components failed validation")
        raise typer.Exit(1)

    # Save comparison results
    results_path = coreml_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "audio_file": str(audio_file),
            "language": language,
            "rtol": rtol,
            "atol": atol,
            "results": results,
            "all_passed": all_passed,
        }, f, indent=2)

    typer.echo(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    app()
