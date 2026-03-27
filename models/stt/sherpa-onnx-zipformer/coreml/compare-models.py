#!/usr/bin/env python3
"""Compare PyTorch and CoreML outputs for a Sherpa-ONNX Zipformer2 transducer.

Loads the original icefall checkpoint and the exported CoreML .mlpackage files,
runs both on the same audio, and reports numerical accuracy (encoder outputs)
plus transcription quality (greedy RNNT decode).

Usage:
    uv run python compare-models.py \
        --checkpoint /Volumes/hdd/models/vosk/vosk-model-en-0.62-atc/am/epoch-56-avg-4.pt \
        --tokens /Volumes/hdd/models/vosk/vosk-model-en-0.62-atc/lang/tokens.txt \
        --coreml-dir ./build/vosk-0.62-atc \
        --audio-file sample_16khz.wav
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np
import torch
import typer

# Local imports (same directory)
from mel import compute_fbank_features
from rnnt_decode import (
    greedy_decode_coreml,
    greedy_decode_pytorch,
    tokens_to_text,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return float(dot / max(norm, 1e-8))


@app.command()
def compare(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", exists=True, resolve_path=True,
        help="Path to icefall .pt checkpoint.",
    ),
    tokens: Path = typer.Option(
        ..., "--tokens", exists=True, resolve_path=True,
        help="Path to tokens.txt.",
    ),
    coreml_dir: Path = typer.Option(
        ..., "--coreml-dir", exists=True, resolve_path=True,
        help="Directory with exported .mlpackage files and metadata.json.",
    ),
    audio_file: Path = typer.Option(
        ..., "--audio-file", exists=True, resolve_path=True,
        help="16 kHz WAV file to transcribe.",
    ),
    reference: Optional[str] = typer.Option(
        None, "--reference",
        help="Reference transcription (prints WER if provided).",
    ),
) -> None:
    """Compare PyTorch vs CoreML encoder outputs and transcriptions."""
    # Import from hyphenated filename via importlib
    import importlib
    convert_mod = importlib.import_module("convert-coreml")
    DecoderForExport = convert_mod.DecoderForExport
    EncoderForExport = convert_mod.EncoderForExport
    JoinerForExport = convert_mod.JoinerForExport
    convert_scaled_for_coreml = convert_mod.convert_scaled_for_coreml
    load_model = convert_mod.load_model
    load_tokens = convert_mod.load_tokens

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    meta_path = coreml_dir / "metadata.json"
    if not meta_path.exists():
        typer.echo(f"ERROR: metadata.json not found in {coreml_dir}", err=True)
        raise typer.Exit(1)
    metadata = json.loads(meta_path.read_text())

    mel_frames = metadata["mel_frames"]
    blank_id = metadata["blank_id"]
    context_size = metadata["context_size"]
    joiner_dim = metadata["joiner_dim"]
    vocab_size = metadata["vocab_size"]

    # ------------------------------------------------------------------
    # 2. Load PyTorch model
    # ------------------------------------------------------------------
    typer.echo("Loading PyTorch checkpoint...")
    ckpt, encoder_embed, encoder, decoder, joiner = load_model(checkpoint)
    for module in [encoder_embed, encoder, decoder, joiner]:
        convert_scaled_for_coreml(module)

    enc_pt = EncoderForExport(encoder_embed, encoder, joiner.encoder_proj).eval()
    dec_pt = DecoderForExport(decoder, joiner.decoder_proj).eval()
    join_pt = JoinerForExport(joiner.output_linear).eval()

    # ------------------------------------------------------------------
    # 3. Load CoreML models
    # ------------------------------------------------------------------
    typer.echo("Loading CoreML models...")
    cu = ct.ComputeUnit.ALL
    enc_ml = ct.models.MLModel(str(coreml_dir / "encoder.mlpackage"), compute_units=cu)
    dec_ml = ct.models.MLModel(str(coreml_dir / "decoder.mlpackage"), compute_units=cu)
    join_ml = ct.models.MLModel(str(coreml_dir / "joiner.mlpackage"), compute_units=cu)

    # ------------------------------------------------------------------
    # 4. Compute features
    # ------------------------------------------------------------------
    typer.echo(f"Computing mel features from {audio_file.name}...")
    features = compute_fbank_features(audio_file)  # (T, 80)
    T_actual = features.shape[0]

    # Pad or truncate to mel_frames
    if T_actual > mel_frames:
        typer.echo(f"  Truncating {T_actual} -> {mel_frames} frames")
        features = features[:mel_frames]
    elif T_actual < mel_frames:
        pad = np.zeros((mel_frames - T_actual, 80), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
    T_used = min(T_actual, mel_frames)

    x = torch.from_numpy(features).unsqueeze(0)  # (1, mel_frames, 80)
    # Encoder expects fixed input size; use mel_frames for both PyTorch and CoreML
    x_lens = torch.tensor([mel_frames], dtype=torch.int64)

    # ------------------------------------------------------------------
    # 5. Compare encoder outputs
    # ------------------------------------------------------------------
    typer.echo("\n=== Encoder Comparison ===")
    with torch.no_grad():
        pt_enc_out, pt_enc_lens = enc_pt(x, x_lens)
    pt_enc_np = pt_enc_out.numpy()

    ml_enc_pred = enc_ml.predict({
        "x": features[np.newaxis].astype(np.float32),
        "x_lens": np.array([mel_frames], dtype=np.int32),
    })
    ml_enc_out = ml_enc_pred["encoder_out"]
    ml_enc_lens = ml_enc_pred["encoder_out_lens"]

    typer.echo(f"  PyTorch encoder shape: {pt_enc_np.shape}")
    typer.echo(f"  CoreML  encoder shape: {ml_enc_out.shape}")
    typer.echo(f"  PyTorch encoder_lens:  {pt_enc_lens.item()}")
    typer.echo(f"  CoreML  encoder_lens:  {int(ml_enc_lens.item())}")

    cos_sim = _cosine_similarity(pt_enc_np, ml_enc_out)
    max_abs = float(np.max(np.abs(pt_enc_np - ml_enc_out)))
    mean_abs = float(np.mean(np.abs(pt_enc_np - ml_enc_out)))

    typer.echo(f"  Cosine similarity:     {cos_sim:.6f}")
    typer.echo(f"  Max absolute error:    {max_abs:.6f}")
    typer.echo(f"  Mean absolute error:   {mean_abs:.6f}")

    if cos_sim > 0.999:
        typer.echo("  -> PASS (cosine > 0.999)")
    elif cos_sim > 0.99:
        typer.echo("  -> WARN (cosine 0.99-0.999, minor drift)")
    else:
        typer.echo("  -> FAIL (cosine < 0.99, significant divergence)")

    # ------------------------------------------------------------------
    # 6. Compare transcriptions
    # ------------------------------------------------------------------
    typer.echo("\n=== Transcription Comparison ===")

    # Load vocab
    vocab_path = coreml_dir / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
    else:
        token_map = load_tokens(tokens)
        vocab = [token_map.get(i, "") for i in range(vocab_size)]

    # PyTorch decode
    pt_tokens = greedy_decode_pytorch(
        pt_enc_out, dec_pt, join_pt, blank_id, context_size, joiner_dim,
    )
    pt_text = tokens_to_text(pt_tokens, vocab)

    # CoreML decode
    ml_tokens = greedy_decode_coreml(
        ml_enc_out, dec_ml, join_ml, blank_id, context_size,
    )
    ml_text = tokens_to_text(ml_tokens, vocab)

    typer.echo(f"  PyTorch: {pt_text}")
    typer.echo(f"  CoreML:  {ml_text}")

    if pt_text == ml_text:
        typer.echo("  -> MATCH")
    else:
        typer.echo("  -> MISMATCH (greedy paths diverged)")

    # ------------------------------------------------------------------
    # 7. WER (optional)
    # ------------------------------------------------------------------
    if reference:
        try:
            from jiwer import wer as compute_wer

            pt_wer = compute_wer(reference, pt_text)
            ml_wer = compute_wer(reference, ml_text)
            typer.echo(f"\n=== Word Error Rate ===")
            typer.echo(f"  Reference: {reference}")
            typer.echo(f"  PyTorch WER: {pt_wer:.2%}")
            typer.echo(f"  CoreML  WER: {ml_wer:.2%}")
        except ImportError:
            typer.echo("\n  (install jiwer for WER: pip install jiwer)")


if __name__ == "__main__":
    app()
