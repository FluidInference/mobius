#!/usr/bin/env python3
"""CLI for exporting Parakeet CTC Japanese (nvidia/parakeet-tdt_ctc-0.6b-ja) to CoreML.

This script exports all components of the hybrid TDT+CTC model:
- Preprocessor (audio -> mel spectrogram)
- Encoder (mel -> encoder features)
- CTC Decoder (encoder features -> CTC log-probabilities)
- Fused variants (MelEncoder, FullPipeline)

Usage:
    uv run python convert-parakeet-ja.py --output-dir ./build
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import (
    CTCDecoderWrapper,
    EncoderWrapper,
    ExportSettings,
    MelEncoderCTCWrapper,
    MelEncoderWrapper,
    PreprocessorWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt_ctc-0.6b-ja"
AUTHOR = "Fluid Inference"

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _parse_compute_units(name: str) -> ct.ComputeUnit:
    """Parse compute units string."""
    normalized = str(name).strip().upper()
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute units '{name}'. Choose from: {', '.join(mapping.keys())}"
        )
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    """Parse compute precision string."""
    if name is None or name.strip() == "":
        return None
    normalized = str(name).strip().upper()
    mapping = {
        "FLOAT32": ct.precision.FLOAT32,
        "FLOAT16": ct.precision.FLOAT16,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown precision '{name}'. Choose from: {', '.join(mapping.keys())}"
        )
    return mapping[normalized]


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    """Save CoreML model with metadata."""
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"✓ Saved: {path}")


@app.command()
def convert(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="HuggingFace model ID or path to .nemo file",
    ),
    output_dir: Path = typer.Option(
        Path("build"), help="Output directory for CoreML models"
    ),
    max_audio_seconds: float = typer.Option(
        15.0, "--max-audio-seconds", help="Fixed audio window duration"
    ),
    compute_units: str = typer.Option(
        "CPU_ONLY",
        "--compute-units",
        help="Compute units: ALL, CPU_ONLY, CPU_AND_NE",
    ),
    compute_precision: Optional[str] = typer.Option(
        None,
        "--compute-precision",
        help="Precision: FLOAT32 (default) or FLOAT16",
    ),
    export_individual: bool = typer.Option(
        True, "--export-individual/--no-individual", help="Export individual components"
    ),
    export_fused: bool = typer.Option(
        True, "--export-fused/--no-fused", help="Export fused pipelines"
    ),
) -> None:
    """Export Parakeet CTC Japanese model components to CoreML."""

    # Setup
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=_parse_compute_units(compute_units),
        deployment_target=ct.target.iOS17,
        compute_precision=_parse_compute_precision(compute_precision),
        max_audio_seconds=max_audio_seconds,
    )

    typer.echo("=== Parakeet CTC Japanese → CoreML ===\n")
    typer.echo(f"Model: {model_id}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Settings: {asdict(settings)}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    typer.echo("Loading NeMo model...")
    if Path(model_id).exists():
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            str(model_id), map_location="cpu"
        )
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_id, map_location="cpu"
        )
    asr_model.eval()

    # Extract model info
    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    vocab_size = int(asr_model.tokenizer.vocab_size)
    max_samples = int(round(max_audio_seconds * sample_rate))

    typer.echo(f"✓ Loaded: {type(asr_model).__name__}")
    typer.echo(f"  Sample rate: {sample_rate} Hz")
    typer.echo(f"  Vocab size: {vocab_size}")
    typer.echo(f"  Max samples: {max_samples} ({max_audio_seconds}s)\n")

    # Run reference forward pass to get shapes
    typer.echo("Running reference forward pass...")
    torch.manual_seed(42)
    dummy_audio = torch.randn(1, max_samples, dtype=torch.float32)
    dummy_length = torch.tensor([max_samples], dtype=torch.int32)

    with torch.inference_mode():
        mel, mel_length = asr_model.preprocessor(
            input_signal=dummy_audio, length=dummy_length.long()
        )
        encoded, encoded_length = asr_model.encoder(
            audio_signal=mel, length=mel_length.long()
        )
        ctc_logits = asr_model.ctc_decoder(encoder_output=encoded)

    mel_features = int(mel.shape[1])
    mel_frames = int(mel.shape[2])
    encoder_dim = int(encoded.shape[1])
    time_steps = int(encoded.shape[2])

    typer.echo(f"  Audio: [1, {max_samples}]")
    typer.echo(f"  Mel: [1, {mel_features}, {mel_frames}]")
    typer.echo(f"  Encoded: [1, {encoder_dim}, {time_steps}]")
    typer.echo(f"  CTC logits: {list(ctc_logits.shape)}\n")

    # Save vocabulary
    vocab_path = output_dir / "vocab.json"
    sp = asr_model.tokenizer.tokenizer
    vocab_list = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    vocab_path.write_text(json.dumps(vocab_list, ensure_ascii=False, indent=2))
    typer.echo(f"✓ Saved vocabulary ({len(vocab_list)} tokens): {vocab_path}\n")

    # Wrap components
    preprocessor = PreprocessorWrapper(asr_model.preprocessor)
    encoder = EncoderWrapper(asr_model.encoder)
    ctc_decoder = CTCDecoderWrapper(asr_model.ctc_decoder)

    # ===== Export Individual Components =====
    if export_individual:
        typer.echo("=== Exporting Individual Components ===\n")

        # 1. Preprocessor
        typer.echo("1. Preprocessor (audio -> mel)...")
        with torch.inference_mode():
            prep_traced = torch.jit.trace(
                preprocessor, (dummy_audio, dummy_length), strict=False
            )
        prep_traced.eval()

        prep_model = _coreml_convert(
            prep_traced,
            inputs=[
                ct.TensorType(name="audio_signal", shape=(1, max_samples), dtype=np.float32),
                ct.TensorType(name="length", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="mel_features", dtype=np.float32),
                ct.TensorType(name="mel_length", dtype=np.int32),
            ],
            settings=settings,
        )
        _save_mlpackage(
            prep_model,
            output_dir / "Preprocessor.mlpackage",
            f"Parakeet Japanese Preprocessor (audio -> mel, {sample_rate}Hz)",
        )

        # 2. Encoder
        typer.echo("2. Encoder (mel -> encoder features)...")
        with torch.inference_mode():
            enc_traced = torch.jit.trace(
                encoder, (mel, mel_length.to(torch.int32)), strict=False
            )
        enc_traced.eval()

        enc_model = _coreml_convert(
            enc_traced,
            inputs=[
                ct.TensorType(name="mel_features", shape=(1, mel_features, mel_frames), dtype=np.float32),
                ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="encoder_output", dtype=np.float32),
                ct.TensorType(name="encoder_length", dtype=np.int32),
            ],
            settings=settings,
        )
        _save_mlpackage(
            enc_model,
            output_dir / "Encoder.mlpackage",
            f"Parakeet Japanese Encoder (mel -> features, dim={encoder_dim})",
        )

        # 3. CTC Decoder
        typer.echo("3. CTC Decoder (encoder -> logits)...")
        with torch.inference_mode():
            ctc_traced = torch.jit.trace(ctc_decoder, (encoded,), strict=False)
        ctc_traced.eval()

        ctc_model = _coreml_convert(
            ctc_traced,
            inputs=[
                ct.TensorType(
                    name="encoder_output",
                    shape=(1, encoder_dim, time_steps),
                    dtype=np.float32,
                ),
            ],
            outputs=[
                ct.TensorType(name="ctc_logits", dtype=np.float32),
            ],
            settings=settings,
        )
        _save_mlpackage(
            ctc_model,
            output_dir / "CtcDecoder.mlpackage",
            f"Parakeet Japanese CTC Decoder - RAW logits (vocab={vocab_size}+1 blank)",
        )

        typer.echo()

    # ===== Export Fused Components =====
    if export_fused:
        typer.echo("=== Exporting Fused Components ===\n")

        # 4. Mel+Encoder
        typer.echo("4. MelEncoder (audio -> encoder features)...")
        mel_encoder = MelEncoderWrapper(preprocessor, encoder)
        with torch.inference_mode():
            melenc_traced = torch.jit.trace(
                mel_encoder, (dummy_audio, dummy_length), strict=False
            )
        melenc_traced.eval()

        melenc_model = _coreml_convert(
            melenc_traced,
            inputs=[
                ct.TensorType(name="audio_signal", shape=(1, max_samples), dtype=np.float32),
                ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="encoder_output", dtype=np.float32),
                ct.TensorType(name="encoder_length", dtype=np.int32),
            ],
            settings=settings,
        )
        _save_mlpackage(
            melenc_model,
            output_dir / "MelEncoder.mlpackage",
            f"Parakeet Japanese Mel+Encoder (audio -> features)",
        )

        # 5. Full Pipeline (audio -> CTC logits)
        typer.echo("5. FullPipeline (audio -> logits)...")
        full_pipeline = MelEncoderCTCWrapper(preprocessor, encoder, ctc_decoder)
        with torch.inference_mode():
            full_traced = torch.jit.trace(
                full_pipeline, (dummy_audio, dummy_length), strict=False
            )
        full_traced.eval()

        full_model = _coreml_convert(
            full_traced,
            inputs=[
                ct.TensorType(name="audio_signal", shape=(1, max_samples), dtype=np.float32),
                ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="ctc_logits", dtype=np.float32),
                ct.TensorType(name="encoder_length", dtype=np.int32),
            ],
            settings=settings,
        )
        _save_mlpackage(
            full_model,
            output_dir / "FullPipeline.mlpackage",
            f"Parakeet Japanese Full Pipeline - RAW CTC logits (audio -> logits)",
        )

        typer.echo()

    # Save metadata
    metadata = {
        "model": "parakeet-tdt_ctc-0.6b-ja",
        "language": "ja (Japanese)",
        "source": model_id,
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "max_samples": max_samples,
        "vocab_size": vocab_size,
        "blank_id": vocab_size,
        "mel_features": mel_features,
        "mel_frames": mel_frames,
        "encoder_dim": encoder_dim,
        "time_steps": time_steps,
        "components": {
            "preprocessor": {"input": [1, max_samples], "output": [1, mel_features, mel_frames]},
            "encoder": {"input": [1, mel_features, mel_frames], "output": [1, encoder_dim, time_steps]},
            "ctc_decoder": {"input": [1, encoder_dim, time_steps], "output": [1, time_steps, vocab_size + 1]},
        },
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    typer.echo(f"✓ Saved metadata: {meta_path}\n")

    typer.echo("=== Conversion Complete ===\n")
    typer.echo("To compile models:")
    typer.echo(f"  cd {output_dir}")
    typer.echo(f"  xcrun coremlcompiler compile Preprocessor.mlpackage .")
    typer.echo(f"  xcrun coremlcompiler compile Encoder.mlpackage .")
    typer.echo(f"  xcrun coremlcompiler compile CtcDecoder.mlpackage .")
    typer.echo(f"  xcrun coremlcompiler compile MelEncoder.mlpackage .")
    typer.echo(f"  xcrun coremlcompiler compile FullPipeline.mlpackage .")
    typer.echo()


if __name__ == "__main__":
    app()
