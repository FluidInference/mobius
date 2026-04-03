#!/usr/bin/env python3
"""Convert Cohere Transcribe via ONNX intermediate format.

This approach bypasses torch.jit.trace() issues by using ONNX export,
which handles dynamic operations differently.

Based on successful community conversion (love4cristiano, 2026-03-28):
- 6-bit quantization
- 15-35x RTF on M3 Pro
- GPU target (not ANE - ANE has overhead issues)
- FP16 preferred over INT8

Usage:
  uv run python convert-via-onnx.py --component encoder
  uv run python convert-via-onnx.py --component decoder
"""
import torch
import torch.nn as nn
import onnx
import coremltools as ct
from onnx_coreml import convert as onnx_to_coreml
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq
import typer

app = typer.Typer()


class StatelessEncoderWrapper(nn.Module):
    """Stateless encoder wrapper that removes dynamic length handling."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_features):
        """
        Args:
            input_features: (batch, n_mels, time) - mel spectrogram
        Returns:
            encoder_output: (batch, seq_len, hidden_dim)
        """
        # Assume full-length input (no dynamic masking)
        batch_size, n_mels, time_frames = input_features.shape
        length = torch.full((batch_size,), time_frames, dtype=torch.int64)

        # Call encoder
        output, _ = self.encoder(
            input_features=input_features,
            length=length
        )
        return output


@app.command()
def export_encoder_onnx(
    output_dir: Path = typer.Option(
        Path("./build/onnx"),
        "--output-dir",
        help="Output directory for ONNX model",
    ),
):
    """Export encoder to ONNX format."""
    typer.echo("Loading Cohere model...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    typer.echo("Wrapping encoder...")
    wrapper = StatelessEncoderWrapper(model.encoder)
    wrapper.eval()

    # Create dummy input
    n_mels = 128
    time_frames = 3000
    dummy_input = torch.randn(1, n_mels, time_frames, dtype=torch.float32)

    output_path = output_dir / "cohere_encoder.onnx"
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Exporting to ONNX: {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["input_features"],
            output_names=["encoder_output"],
            dynamic_axes={
                "input_features": {0: "batch", 2: "time"},
                "encoder_output": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,  # Use legacy exporter (may handle dynamic ops better)
        )

    typer.echo(f"✓ ONNX model saved: {output_path}")

    # Validate ONNX
    typer.echo("Validating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    typer.echo("✓ ONNX model is valid")

    return output_path


@app.command()
def convert_onnx_to_coreml(
    onnx_path: Path = typer.Option(
        ...,
        "--onnx-path",
        exists=True,
        help="Path to ONNX model",
    ),
    output_dir: Path = typer.Option(
        Path("./build/coreml"),
        "--output-dir",
        help="Output directory for CoreML model",
    ),
    compute_precision: str = typer.Option(
        "FLOAT16",
        "--precision",
        help="Compute precision: FLOAT32 or FLOAT16",
    ),
):
    """Convert ONNX model to CoreML."""
    typer.echo(f"Converting {onnx_path} to CoreML...")

    # Load ONNX model
    onnx_model = onnx.load(str(onnx_path))

    # Convert ONNX to CoreML using onnx-coreml
    typer.echo("Converting ONNX to CoreML (this may take several minutes)...")

    coreml_model = onnx_to_coreml(
        onnx_model,
        minimum_ios_deployment_target='17',
    )

    output_path = output_dir / "cohere_encoder.mlpackage"
    output_dir.mkdir(parents=True, exist_ok=True)

    coreml_model.save(str(output_path))
    typer.echo(f"✓ CoreML model saved: {output_path}")

    return output_path


@app.command()
def full_pipeline(
    output_dir: Path = typer.Option(
        Path("./build"),
        "--output-dir",
        help="Base output directory",
    ),
):
    """Run full ONNX → CoreML conversion pipeline."""
    typer.echo("=" * 60)
    typer.echo("Cohere Transcribe: ONNX → CoreML Pipeline")
    typer.echo("=" * 60)

    # Step 1: Export to ONNX
    typer.echo("\n[1/2] Exporting to ONNX...")
    onnx_path = export_encoder_onnx(output_dir=output_dir / "onnx")

    # Step 2: Convert to CoreML
    typer.echo("\n[2/2] Converting to CoreML...")
    coreml_path = convert_onnx_to_coreml(
        onnx_path=onnx_path,
        output_dir=output_dir / "coreml",
        compute_precision="FLOAT16",  # Based on community success
    )

    typer.echo("\n" + "=" * 60)
    typer.echo("Conversion complete!")
    typer.echo("=" * 60)
    typer.echo(f"ONNX: {onnx_path}")
    typer.echo(f"CoreML: {coreml_path}")
    typer.echo("\nNext steps:")
    typer.echo("  1. Profile with coreml-cli to verify GPU assignment")
    typer.echo("  2. Test inference speed (expect 15-35x RTF)")
    typer.echo("  3. Apply 6-bit quantization if needed")


if __name__ == "__main__":
    app()
