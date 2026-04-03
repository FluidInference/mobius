#!/usr/bin/env python3
"""Export full CoreML pipeline: Preprocessor + Encoder + CTC Decoder for zh-CN.

Converts the complete Parakeet CTC zh-CN model to CoreML with optional fp16 precision
on the encoder for reduced memory footprint. For int8 quantization, use quantize-encoder-advanced.py separately.
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

import nemo.collections.asr as nemo_asr

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


class PreprocessorWrapper(torch.nn.Module):
    """Wrapper for NeMo preprocessor (audio -> mel spectrogram)."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, audio_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.module(
            input_signal=audio_signal, length=length.to(dtype=torch.long)
        )
        return mel, mel_length


class EncoderWrapper(torch.nn.Module):
    """Wrapper for NeMo FastConformer encoder (mel -> encoder features)."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, features: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_lengths = self.module(
            audio_signal=features, length=length.to(dtype=torch.long)
        )
        return encoded, encoded_lengths


class CTCDecoderWrapper(torch.nn.Module):
    """CTC decoder head: encoder_output -> log-probabilities."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.module(encoder_output=encoder_output)


def _prepare_audio(audio_path: Path, sample_rate: int, max_samples: int) -> torch.Tensor:
    """Load and prepare audio for tracing."""
    data, sr = sf.read(str(audio_path), dtype="float32")
    if sr != sample_rate:
        raise ValueError(f"Audio sample rate {sr} does not match model rate {sample_rate}")

    if data.ndim > 1:
        data = data[:, 0]  # Mono

    if data.size == 0:
        raise ValueError("Audio is empty")

    # Pad or truncate to max_samples
    if data.size < max_samples:
        pad_width = max_samples - data.size
        data = np.pad(data, (0, pad_width))
    elif data.size > max_samples:
        data = data[:max_samples]

    audio = torch.from_numpy(data).unsqueeze(0).to(dtype=torch.float32)
    return audio


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    """Save CoreML model with metadata."""
    model.short_description = description
    model.author = "Fluid Inference"
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"✓ Saved: {path}")


@app.command()
def convert(
    nemo_path: Path = typer.Option(
        ...,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to parakeet-ctc-zh-cn .nemo checkpoint",
    ),
    output_dir: Path = typer.Option(
        Path("build-full"),
        "--output-dir",
        help="Output directory for CoreML models",
    ),
    audio_path: Optional[Path] = typer.Option(
        None,
        "--audio-path",
        help="Path to 15-second audio for tracing (random if not provided)",
    ),
    max_audio_seconds: float = typer.Option(
        15.0,
        "--max-audio-seconds",
        help="Maximum audio duration in seconds",
    ),
    quantize_encoder: bool = typer.Option(
        True,
        "--quantize-encoder/--no-quantize-encoder",
        help="Apply fp16 precision to encoder (for int8, use quantize-encoder-advanced.py separately)",
    ),
) -> None:
    """Convert full Parakeet CTC zh-CN pipeline to CoreML.

    Exports three models:
    1. Preprocessor.mlpackage - Audio -> mel spectrogram (CPU-only)
    2. Encoder.mlpackage - Mel -> encoder features [1, 1024, 188] (ANE, optionally fp16)
    3. CtcHeadZhCn.mlpackage - Encoder features -> CTC logits [1, 188, 7001] (ANE)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NeMo model
    typer.echo(f"Loading NeMo model from {nemo_path}...")
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        str(nemo_path), map_location="cpu"
    )
    asr_model.eval()

    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = int(max_audio_seconds * sample_rate)

    # Prepare trace audio
    if audio_path is not None and audio_path.exists():
        typer.echo(f"Using trace audio: {audio_path}")
        audio_tensor = _prepare_audio(audio_path, sample_rate, max_samples)
    else:
        typer.echo("Using random audio for tracing")
        audio_tensor = torch.randn(1, max_samples, dtype=torch.float32)

    audio_length = torch.tensor([max_samples], dtype=torch.int32)

    # Wrap components
    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder = EncoderWrapper(asr_model.encoder.eval())
    ctc_decoder = CTCDecoderWrapper(asr_model.ctc_decoder.eval())

    # ========== Export Preprocessor ==========
    typer.echo("\n[1/3] Exporting Preprocessor...")
    with torch.inference_mode():
        mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)

    mel_ref = mel_ref.clone()
    mel_length_ref = mel_length_ref.clone().to(dtype=torch.int32)

    traced_preprocessor = torch.jit.trace(
        preprocessor.cpu(),
        (audio_tensor.cpu(), audio_length.cpu()),
        strict=False,
    )
    traced_preprocessor.eval()

    preprocessor_inputs = [
        ct.TensorType(
            name="audio_signal",
            shape=(1, max_samples),  # Fixed-length audio (15 seconds)
            dtype=np.float32,
        ),
        ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
    ]
    preprocessor_outputs = [
        ct.TensorType(name="mel", dtype=np.float32),
        ct.TensorType(name="mel_length", dtype=np.int32),
    ]

    preprocessor_model = ct.convert(
        traced_preprocessor,
        inputs=preprocessor_inputs,
        outputs=preprocessor_outputs,
        compute_units=ct.ComputeUnit.CPU_ONLY,  # Preprocessor is CPU-only
        minimum_deployment_target=ct.target.iOS17,
    )

    preprocessor_path = output_dir / "Preprocessor.mlpackage"
    _save_mlpackage(
        preprocessor_model,
        preprocessor_path,
        "Parakeet zh-CN Preprocessor: audio -> mel spectrogram",
    )

    # ========== Export Encoder ==========
    typer.echo("\n[2/3] Exporting Encoder...")
    with torch.inference_mode():
        encoder_ref, encoder_length_ref = encoder(mel_ref, mel_length_ref)

    encoder_ref = encoder_ref.clone()
    encoder_length_ref = encoder_length_ref.clone().to(dtype=torch.int32)

    traced_encoder = torch.jit.trace(
        encoder.cpu(),
        (mel_ref.cpu(), mel_length_ref.cpu()),
        strict=False,
    )
    traced_encoder.eval()

    # Get actual mel shape from reference output
    mel_shape = tuple(mel_ref.shape)

    encoder_inputs = [
        ct.TensorType(name="audio_signal", shape=mel_shape, dtype=np.float32),
        ct.TensorType(name="length", shape=tuple(mel_length_ref.shape), dtype=np.int32),
    ]
    encoder_outputs = [
        ct.TensorType(name="encoder_output", dtype=np.float32),
        ct.TensorType(name="encoded_length", dtype=np.int32),
    ]

    # Convert with float16 precision if quantization requested
    if quantize_encoder:
        typer.echo("  Converting with float16 precision...")
        encoder_model = ct.convert(
            traced_encoder,
            inputs=encoder_inputs,
            outputs=encoder_outputs,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
        )
    else:
        encoder_model = ct.convert(
            traced_encoder,
            inputs=encoder_inputs,
            outputs=encoder_outputs,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS17,
        )

    encoder_path = output_dir / "Encoder.mlpackage"
    quantization_label = "(fp16)" if quantize_encoder else ""
    _save_mlpackage(
        encoder_model,
        encoder_path,
        f"Parakeet zh-CN Encoder: mel -> encoder features {quantization_label}",
    )

    # ========== Export CTC Decoder Head ==========
    typer.echo("\n[3/3] Exporting CTC Decoder Head...")
    with torch.inference_mode():
        # CTC decoder expects [B, D, T] format
        ctc_logits_ref = ctc_decoder(encoder_ref)

    ctc_logits_ref = ctc_logits_ref.clone()

    traced_ctc = torch.jit.trace(
        ctc_decoder.cpu(),
        (encoder_ref.cpu(),),
        strict=False,
    )
    traced_ctc.eval()

    ctc_inputs = [
        ct.TensorType(
            name="encoder_output",
            shape=encoder_ref.shape,  # Fixed shape [1, 1024, 188]
            dtype=np.float32,
        )
    ]
    ctc_outputs = [
        ct.TensorType(name="ctc_logits", dtype=np.float32)
    ]

    ctc_model = ct.convert(
        traced_ctc,
        inputs=ctc_inputs,
        outputs=ctc_outputs,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS17,
    )

    ctc_path = output_dir / "CtcHeadZhCn.mlpackage"
    _save_mlpackage(
        ctc_model,
        ctc_path,
        "Parakeet zh-CN CTC Decoder Head: encoder features -> CTC logits",
    )

    # ========== Save vocabulary and metadata ==========
    typer.echo("\nSaving vocabulary and metadata...")

    # Extract vocabulary
    vocab = {i: token for i, token in enumerate(asr_model.tokenizer.ids_to_tokens())}
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    typer.echo(f"✓ Saved vocabulary: {vocab_path}")

    # Save metadata
    metadata = {
        "model": "parakeet-ctc-0.6b-zh-cn-full-pipeline",
        "language": "zh-CN (Mandarin Chinese Simplified)",
        "source": str(nemo_path),
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,
        "encoder_dim": 1024,
        "time_steps": 188,
        "vocab_size": len(vocab),
        "blank_id": len(vocab),
        "quantized_encoder": quantize_encoder,
        "components": {
            "preprocessor": {
                "input": {"audio_signal": [1, f"1-{max_samples}"], "audio_length": [1]},
                "output": {"mel": "variable", "mel_length": [1]},
            },
            "encoder": {
                "input": {"audio_signal": "variable", "length": [1]},
                "output": {"encoder_output": [1, 1024, 188], "encoded_length": [1]},
                "quantization": "float16" if quantize_encoder else "float32",
            },
            "ctc_head": {
                "input": {"encoder_output": [1, 1024, 188]},
                "output": {"ctc_logits": [1, 188, len(vocab) + 1]},  # +1 for blank token
            },
        },
    }

    metadata_path = output_dir / "pipeline_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    typer.echo(f"✓ Saved metadata: {metadata_path}")

    typer.echo("\n" + "="*50)
    typer.echo("✓ Full pipeline conversion complete!")
    typer.echo(f"  Output directory: {output_dir}")
    typer.echo(f"  Preprocessor: {preprocessor_path.name}")
    typer.echo(f"  Encoder: {encoder_path.name} {'(fp16 precision)' if quantize_encoder else ''}")
    typer.echo(f"  CTC Head: {ctc_path.name}")
    typer.echo(f"  Vocabulary: {vocab_path.name} ({len(vocab)} tokens)")
    if quantize_encoder:
        typer.echo("\nNote: For int8 quantization, run: uv run python conversion/quantize-encoder-advanced.py")


if __name__ == "__main__":
    app()
