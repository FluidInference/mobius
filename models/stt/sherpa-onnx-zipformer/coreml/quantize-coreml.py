#!/usr/bin/env python3
"""Quantize CoreML mlpackages for Sherpa-ONNX Zipformer2 transducer.

Produces int8 variants of all components (encoder, decoder, joiner).
The encoder is the largest component and benefits most from quantization.

Usage:
    uv run python quantize-coreml.py \
        --input-dir ./build/vosk-0.62-atc \
        --output-dir ./build/vosk-0.62-atc-int8
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import coremltools as ct
import typer
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpLinearQuantizerConfig,
    linear_quantize_weights,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

BYTES_IN_MB = 1024 * 1024


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / BYTES_IN_MB


def _quantize_int8(model: ct.models.MLModel) -> ct.models.MLModel:
    """Apply int8 per-channel symmetric linear quantization."""
    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
        )
    )
    return linear_quantize_weights(model, config=config)


@app.command()
def quantize(
    input_dir: Path = typer.Option(
        ..., "--input-dir", exists=True, resolve_path=True,
        help="Source model directory with .mlpackage files.",
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", resolve_path=True,
        help="Output directory for quantized models.",
    ),
    precision: str = typer.Option(
        "int8", "--precision",
        help="Quantization precision: int8.",
    ),
) -> None:
    """Quantize all mlpackage files in a model directory."""
    if precision != "int8":
        typer.echo(
            f"Only int8 is supported for post-training quantization. "
            f"For FP16, re-export with --float16 flag in convert-coreml.py."
        )
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Input:  {input_dir}")
    typer.echo(f"Output: {output_dir}")
    typer.echo(f"Precision: {precision}\n")

    results = {}

    for item in sorted(input_dir.iterdir()):
        if item.suffix == ".mlpackage":
            component = item.stem
            typer.echo(f"Quantizing {component}...")

            model = ct.models.MLModel(str(item), compute_units=ct.ComputeUnit.ALL)
            original_size = _dir_size_mb(item)

            quantized = _quantize_int8(model)

            out_path = output_dir / item.name
            if out_path.exists():
                shutil.rmtree(out_path)
            quantized.save(str(out_path))
            quantized_size = _dir_size_mb(out_path)

            ratio = original_size / quantized_size if quantized_size > 0 else 0
            results[component] = {
                "original_mb": round(original_size, 1),
                "quantized_mb": round(quantized_size, 1),
                "compression": round(ratio, 2),
            }
            typer.echo(
                f"  {original_size:.1f} MB -> {quantized_size:.1f} MB "
                f"({ratio:.2f}x compression)"
            )

        elif item.is_file():
            # Copy non-model files (metadata.json, vocab.json)
            shutil.copy2(item, output_dir / item.name)

    # Save summary
    summary_path = output_dir / "quantization_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    typer.echo(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    app()
