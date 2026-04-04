#!/usr/bin/env python3
"""Validate Japanese CTC head CoreML model against NeMo implementation.

Compares CoreML inference output with the original NeMo model to ensure conversion accuracy.

Usage:
    uv run python validate-ctc-ja.py --audio-file test.wav --model-name nvidia/parakeet-tdt_ctc-0.6b-ja
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

import nemo.collections.asr as nemo_asr

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def decode_ctc_greedy(log_probs: np.ndarray, vocab: list[str], blank_id: int) -> str:
    """Greedy CTC decoding: collapse repeats and remove blanks."""
    # Get best path
    best_path = np.argmax(log_probs, axis=-1)  # [T]

    # Collapse repeats
    collapsed = []
    prev = None
    for token_id in best_path:
        if token_id != prev and token_id != blank_id:
            collapsed.append(int(token_id))
        prev = token_id

    # Convert to text
    tokens = [vocab[i] for i in collapsed]
    text = "".join(tokens).replace("▁", " ").strip()
    return text


def pad_or_truncate_encoder_output(encoder_output: np.ndarray, target_time_steps: int) -> np.ndarray:
    """Pad or truncate encoder output to match CoreML model's expected input shape.

    Args:
        encoder_output: Shape [1, encoder_dim, time_steps]
        target_time_steps: Target time dimension (e.g., 188)

    Returns:
        Padded/truncated array of shape [1, encoder_dim, target_time_steps]
    """
    current_time_steps = encoder_output.shape[2]

    if current_time_steps == target_time_steps:
        return encoder_output
    elif current_time_steps > target_time_steps:
        # Truncate
        return encoder_output[:, :, :target_time_steps]
    else:
        # Pad with zeros
        pad_width = ((0, 0), (0, 0), (0, target_time_steps - current_time_steps))
        return np.pad(encoder_output, pad_width, mode='constant', constant_values=0)


@app.command()
def validate(
    audio_file: Path = typer.Option(
        ...,
        "--audio-file",
        exists=True,
        resolve_path=True,
        help="Test audio file (WAV, 16kHz mono)",
    ),
    model_name: str = typer.Option(
        "nvidia/parakeet-tdt_ctc-0.6b-ja",
        "--model-name",
        help="HuggingFace model name or path to .nemo checkpoint",
    ),
    coreml_dir: Path = typer.Option(
        Path("build"),
        help="Directory containing CoreML model",
    ),
) -> None:
    """Validate CoreML model against NeMo reference."""

    # Load vocabulary and metadata
    vocab_path = coreml_dir / "vocab.json"
    vocab = json.loads(vocab_path.read_text())
    blank_id = len(vocab)

    metadata_path = coreml_dir / "ctc_head_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    target_time_steps = metadata["time_steps"]

    typer.echo(f"Vocabulary: {len(vocab)} tokens + blank")
    typer.echo(f"Target time steps: {target_time_steps}")

    # Load NeMo model
    typer.echo(f"\nLoading NeMo model from {model_name}...")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_name, map_location="cpu"
        )
    except Exception as e:
        typer.echo(f"Failed to load from HuggingFace, trying as local .nemo: {e}")
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            str(model_name), map_location="cpu"
        )

    asr_model.eval()

    # Load audio
    typer.echo(f"\nLoading audio: {audio_file}")
    audio, sr = sf.read(str(audio_file), dtype="float32")

    if sr != 16000:
        typer.echo(f"  Warning: Expected 16kHz, got {sr}Hz. Resampling may be needed.")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    audio_length = torch.tensor([audio.shape[0]], dtype=torch.int32)

    typer.echo(f"  Duration: {len(audio)/sr:.2f}s")
    typer.echo(f"  Samples: {len(audio)}")

    # NeMo inference
    typer.echo("\nRunning NeMo inference...")
    with torch.inference_mode():
        # Preprocessor
        mel, mel_length = asr_model.preprocessor(
            input_signal=audio_tensor, length=audio_length.long()
        )

        # Encoder
        encoded, encoded_length = asr_model.encoder(
            audio_signal=mel, length=mel_length.long()
        )

        # CTC decoder
        ctc_logits_nemo = asr_model.ctc_decoder(encoder_output=encoded)

    typer.echo(f"  Encoder output shape: {list(encoded.shape)}")
    typer.echo(f"  CTC logits shape: {list(ctc_logits_nemo.shape)}")

    # Decode NeMo output
    nemo_log_probs = torch.nn.functional.log_softmax(ctc_logits_nemo[0], dim=-1).numpy()
    nemo_text = decode_ctc_greedy(nemo_log_probs, vocab, blank_id)
    typer.echo(f"\n  NeMo transcription: {nemo_text}")

    # Load CoreML model
    typer.echo(f"\nLoading CoreML model from {coreml_dir}...")
    mlmodel_path = coreml_dir / "CtcHeadJa.mlmodelc"

    if not mlmodel_path.exists():
        typer.echo(f"  CoreML model not found. Compiling...")
        mlpackage = coreml_dir / "CtcHeadJa.mlpackage"
        typer.echo(f"  xcrun coremlcompiler compile {mlpackage} {coreml_dir}/")
        import subprocess
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(coreml_dir)],
            check=True,
        )

    mlmodel = ct.models.MLModel(str(mlmodel_path), compute_units=ct.ComputeUnit.ALL)

    # CoreML inference
    typer.echo("\nRunning CoreML inference...")
    encoder_output_np = encoded.numpy()  # [1, 1024, T]

    # Pad or truncate to match CoreML model's expected shape
    encoder_output_padded = pad_or_truncate_encoder_output(encoder_output_np, target_time_steps)

    coreml_input = {"encoder_output": encoder_output_padded}
    coreml_output = mlmodel.predict(coreml_input)

    ctc_logits_coreml = coreml_output["ctc_logits"]  # [1, T, V+1]
    typer.echo(f"  CoreML output shape: {list(ctc_logits_coreml.shape)}")

    # Decode CoreML output
    coreml_log_probs = torch.nn.functional.log_softmax(
        torch.from_numpy(ctc_logits_coreml[0]), dim=-1
    ).numpy()
    coreml_text = decode_ctc_greedy(coreml_log_probs, vocab, blank_id)
    typer.echo(f"\n  CoreML transcription: {coreml_text}")

    # Compare outputs
    typer.echo("\n=== Validation Results ===")

    # Numerical comparison (only compare valid time steps)
    actual_time_steps = ctc_logits_nemo.shape[1]
    ctc_logits_nemo_np = ctc_logits_nemo.numpy()[:, :actual_time_steps, :]
    ctc_logits_coreml_trimmed = ctc_logits_coreml[:, :actual_time_steps, :]

    max_diff = np.abs(ctc_logits_nemo_np - ctc_logits_coreml_trimmed).max()
    mean_diff = np.abs(ctc_logits_nemo_np - ctc_logits_coreml_trimmed).mean()

    typer.echo(f"Max logit diff:  {max_diff:.6e}")
    typer.echo(f"Mean logit diff: {mean_diff:.6e}")

    # Text comparison
    if nemo_text == coreml_text:
        typer.echo("✓ Transcriptions match!")
    else:
        typer.echo("✗ Transcriptions differ:")
        typer.echo(f"  NeMo:   '{nemo_text}'")
        typer.echo(f"  CoreML: '{coreml_text}'")

        # Character-level diff
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, nemo_text, coreml_text).ratio()
        typer.echo(f"  Similarity: {ratio*100:.1f}%")

    # Numerical tolerance check
    if max_diff < 1e-3:
        typer.echo("✓ Numerical accuracy: EXCELLENT (< 1e-3)")
    elif max_diff < 1e-2:
        typer.echo("✓ Numerical accuracy: GOOD (< 1e-2)")
    elif max_diff < 1e-1:
        typer.echo("⚠ Numerical accuracy: ACCEPTABLE (< 1e-1)")
    else:
        typer.echo("✗ Numerical accuracy: POOR (>= 1e-1)")


if __name__ == "__main__":
    app()
