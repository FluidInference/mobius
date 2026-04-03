#!/usr/bin/env python3
"""CLI for exporting Cohere Transcribe 03-2026 components to CoreML.

Architecture:
  CohereAsrForConditionalGeneration
    ├── audio_encoder     → AudioEncoderWrapper      → cohere_audio_encoder.mlpackage
    │   ├── mel frontend  (log-mel spectrogram)
    │   └── conformer     (2B param encoder)
    ├── decoder           → DecoderWrapper           → cohere_decoder.mlpackage
    │   └── transformer   (with KV cache)
    └── lm_head           → LMHeadWrapper            → cohere_lm_head.mlpackage

Usage:
  uv run python convert-cohere-transcribe.py
  uv run python convert-cohere-transcribe.py --output-dir ./build/cohere-transcribe
  uv run python convert-cohere-transcribe.py --components audio_encoder
  uv run python convert-cohere-transcribe.py --quantize int8
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import typer

DEFAULT_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
AUTHOR = "Fluid Inference"
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30.0  # max audio duration for tracing


@dataclass
class ExportSettings:
    """Settings for CoreML export."""
    sample_rate: int
    max_audio_seconds: float
    max_samples: int

    @classmethod
    def default(cls) -> ExportSettings:
        max_samples = int(MAX_AUDIO_SECONDS * SAMPLE_RATE)
        return cls(
            sample_rate=SAMPLE_RATE,
            max_audio_seconds=MAX_AUDIO_SECONDS,
            max_samples=max_samples,
        )


class AudioEncoderWrapper(nn.Module):
    """Wraps Cohere encoder (conformer on mel features) for tracing."""

    def __init__(self, model, fixed_length: int):
        super().__init__()
        self.encoder = model.encoder
        self.fixed_length = fixed_length

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: (batch, n_mels, time) - mel spectrogram features
        Returns:
            encoder_output: (batch, seq_len, hidden_dim) - acoustic embeddings
        """
        # Use fixed length to avoid dynamic shape issues during CoreML conversion
        batch_size = input_features.shape[0]
        length = torch.full((batch_size,), self.fixed_length, dtype=torch.int64)

        # Encoder returns (output, output_length)
        encoder_output, _ = self.encoder(
            input_features=input_features,
            length=length
        )
        return encoder_output


class DecoderWrapper(nn.Module):
    """Wraps Cohere decoder for tracing with fixed input shapes."""

    def __init__(self, model, max_seq_len: int = 512):
        super().__init__()
        self.transf_decoder = model.transf_decoder
        self.encoder_decoder_proj = model.encoder_decoder_proj
        self.max_seq_len = max_seq_len

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) - token IDs
            positions: (batch, seq_len) - position indices
            encoder_hidden_states: (batch, audio_seq_len, hidden_dim)
        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
        """
        # Project encoder outputs to decoder dimension
        encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)

        # Run decoder (returns tuple of (hidden_states, past_key_values))
        decoder_output, _ = self.transf_decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
        )
        return decoder_output


class LMHeadWrapper(nn.Module):
    """Wraps language model head for token prediction."""

    def __init__(self, model):
        super().__init__()
        self.log_softmax = model.log_softmax

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.log_softmax(hidden_states)


def _quantize_weights(model: ct.models.MLModel, dtype: str) -> ct.models.MLModel:
    """Apply post-training weight quantization to a CoreML model."""
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    dtype_map = {
        "int8": "int8",
        "int4": "int4",
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported quantization dtype: {dtype}. Use 'int8' or 'int4'.")

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(dtype=dtype_map[dtype])
    )
    return linear_quantize_weights(model, config=config)


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    """Save CoreML model as mlpackage with metadata."""
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"  Saved: {path}")


def _load_model(model_id: str):
    """Load Cohere Transcribe model from HuggingFace."""
    typer.echo(f"Loading model: {model_id}")

    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    typer.echo("  Loading with trust_remote_code=True (model uses custom classes)...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    typer.echo(f"  Loaded: {model.__class__.__name__}")

    # Check model structure
    if hasattr(model, 'audio_encoder'):
        typer.echo(f"  Audio encoder: {model.audio_encoder.__class__.__name__}")
    if hasattr(model, 'decoder'):
        typer.echo(f"  Decoder: {model.decoder.__class__.__name__}")

    return model, processor


def _export_audio_encoder(
    model,
    settings: ExportSettings,
    output_dir: Path,
    quantize: Optional[str] = None,
) -> None:
    """Export audio encoder (conformer on mel features) to CoreML."""
    typer.echo("\n[1/3] Exporting Audio Encoder...")

    # Create dummy mel spectrogram input (batch, n_mels, time_frames)
    # For 30s audio at 16kHz with subsampling factor 8, expect ~3000 frames
    n_mels = 128  # From model config
    time_frames = 3000  # Approximate for 30s audio
    dummy_input_features = torch.randn(1, n_mels, time_frames, dtype=torch.float32)

    wrapper = AudioEncoderWrapper(model, fixed_length=time_frames)
    wrapper.eval()

    typer.echo(f"  Input features shape: {dummy_input_features.shape}")
    typer.echo(f"  Fixed length: {time_frames}")
    typer.echo("  Tracing model...")

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy_input_features,),
            strict=False,  # Allow minor graph differences due to dropout
            check_trace=False,  # Skip trace verification due to nondeterminism
        )

    typer.echo("  Converting to CoreML...")

    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_features", shape=dummy_input_features.shape),
        ],
        outputs=[ct.TensorType(name="encoder_output")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,  # Trace on CPU, runtime can use ANE
    )

    if quantize:
        typer.echo(f"  Quantizing to {quantize}...")
        coreml_model = _quantize_weights(coreml_model, quantize)

    output_path = output_dir / "cohere_audio_encoder.mlpackage"
    _save_mlpackage(
        coreml_model,
        output_path,
        "Cohere Transcribe 03-2026 Audio Encoder (Mel + Conformer)",
    )


def _export_decoder(
    model,
    settings: ExportSettings,
    output_dir: Path,
    quantize: Optional[str] = None,
) -> None:
    """Export decoder to CoreML."""
    typer.echo("\n[2/3] Exporting Decoder...")

    wrapper = DecoderWrapper(model, max_seq_len=512)
    wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    seq_len = 10
    audio_seq_len = 1500  # Typical for 30s audio
    hidden_dim = model.config.encoder["d_model"]  # Encoder output dimension

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    dummy_positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    dummy_encoder_output = torch.randn(batch_size, audio_seq_len, hidden_dim, dtype=torch.float32)

    typer.echo(f"  Input IDs shape: {dummy_input_ids.shape}")
    typer.echo(f"  Positions shape: {dummy_positions.shape}")
    typer.echo(f"  Encoder output shape: {dummy_encoder_output.shape}")
    typer.echo("  Tracing model...")

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy_input_ids, dummy_positions, dummy_encoder_output),
            strict=False,
            check_trace=False,
        )

    typer.echo("  Converting to CoreML...")

    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=dummy_input_ids.shape, dtype=np.int32),
            ct.TensorType(name="positions", shape=dummy_positions.shape, dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=dummy_encoder_output.shape),
        ],
        outputs=[ct.TensorType(name="hidden_states")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    if quantize:
        typer.echo(f"  Quantizing to {quantize}...")
        coreml_model = _quantize_weights(coreml_model, quantize)

    output_path = output_dir / "cohere_decoder.mlpackage"
    _save_mlpackage(
        coreml_model,
        output_path,
        "Cohere Transcribe 03-2026 Decoder (Transformer)",
    )


def _export_lm_head(
    model,
    output_dir: Path,
    quantize: Optional[str] = None,
) -> None:
    """Export LM head to CoreML."""
    typer.echo("\n[3/3] Exporting LM Head...")

    wrapper = LMHeadWrapper(model)
    wrapper.eval()

    # Create dummy input
    batch_size = 1
    seq_len = 10
    hidden_dim = model.config.transf_decoder["config_dict"]["hidden_size"]  # Decoder output dimension

    dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)

    typer.echo(f"  Hidden states shape: {dummy_hidden_states.shape}")
    typer.echo("  Tracing model...")

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy_hidden_states,),
            strict=False,
            check_trace=False,
        )

    typer.echo("  Converting to CoreML...")

    coreml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=dummy_hidden_states.shape)],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    if quantize:
        typer.echo(f"  Quantizing to {quantize}...")
        coreml_model = _quantize_weights(coreml_model, quantize)

    output_path = output_dir / "cohere_lm_head.mlpackage"
    _save_mlpackage(
        coreml_model,
        output_path,
        "Cohere Transcribe 03-2026 LM Head",
    )


def _save_metadata(
    model,
    processor,
    settings: ExportSettings,
    output_dir: Path,
) -> None:
    """Save model metadata as JSON."""
    config = model.config

    metadata = {
        "model_id": DEFAULT_MODEL_ID,
        "sample_rate": settings.sample_rate,
        "max_audio_seconds": settings.max_audio_seconds,
        "max_samples": settings.max_samples,
        "vocab_size": config.vocab_size,
        "encoder_hidden_size": config.encoder["d_model"],
        "decoder_hidden_size": config.transf_decoder["config_dict"]["hidden_size"],
        "lm_head_hidden_size": config.head["hidden_size"],
        "num_encoder_layers": config.encoder["n_layers"],
        "num_decoder_layers": config.num_hidden_layers,
        "num_languages": len(config.supported_languages),
        "languages": config.supported_languages,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    typer.echo(f"\n  Metadata saved: {metadata_path}")


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="HuggingFace model ID",
    ),
    output_dir: Path = typer.Option(
        Path("./build/cohere-transcribe"),
        "--output-dir",
        help="Output directory for CoreML packages",
    ),
    components: Optional[str] = typer.Option(
        None,
        "--components",
        help="Comma-separated components to export (audio_encoder,decoder,lm_head). Default: all",
    ),
    quantize: Optional[str] = typer.Option(
        None,
        "--quantize",
        help="Quantization type: int8, int4",
    ),
) -> None:
    """Convert Cohere Transcribe 03-2026 to CoreML."""

    typer.echo("=" * 60)
    typer.echo("Cohere Transcribe 03-2026 → CoreML Conversion")
    typer.echo("=" * 60)

    # Parse components
    all_components = ["audio_encoder", "decoder", "lm_head"]
    if components:
        selected = [c.strip() for c in components.split(",")]
        invalid = [c for c in selected if c not in all_components]
        if invalid:
            typer.echo(f"ERROR: Invalid components: {invalid}")
            typer.echo(f"Valid: {all_components}")
            raise typer.Exit(1)
    else:
        selected = all_components

    typer.echo(f"Components: {', '.join(selected)}")
    if quantize:
        typer.echo(f"Quantization: {quantize}")

    # Load model
    model, processor = _load_model(model_id)
    settings = ExportSettings.default()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export selected components
    if "audio_encoder" in selected:
        _export_audio_encoder(model, settings, output_dir, quantize)

    if "decoder" in selected:
        _export_decoder(model, settings, output_dir, quantize)

    if "lm_head" in selected:
        _export_lm_head(model, output_dir, quantize)

    # Save metadata
    _save_metadata(model, processor, settings, output_dir)

    typer.echo("\n" + "=" * 60)
    typer.echo("Conversion complete!")
    typer.echo("=" * 60)
    typer.echo(f"Output directory: {output_dir}")
    typer.echo("\nNext steps:")
    typer.echo("  1. Validate with: uv run python compare-models.py --coreml-dir " + str(output_dir))
    typer.echo("  2. Profile with: cd ../../tools/coreml-cli && uv run coreml-cli " + str(output_dir / "cohere_audio_encoder.mlmodelc"))


if __name__ == "__main__":
    app()
