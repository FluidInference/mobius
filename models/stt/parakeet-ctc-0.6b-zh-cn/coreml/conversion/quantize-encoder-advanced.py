#!/usr/bin/env python3
"""Advanced int8 quantization for Encoder using coremltools.optimize."""
from pathlib import Path
import typer
import coremltools as ct
import numpy as np

app = typer.Typer(add_completion=False)


@app.command()
def quantize(
    input_model: Path = typer.Option(
        Path("build-full/Encoder.mlpackage"),
        "--input",
        help="Input encoder model path",
    ),
    output_model: Path = typer.Option(
        Path("build-full/Encoder-int8.mlpackage"),
        "--output",
        help="Output quantized model path",
    ),
) -> None:
    """Apply int8 post-training quantization to encoder."""

    typer.echo(f"Loading model from {input_model}...")
    model = ct.models.MLModel(str(input_model))

    typer.echo("\nTrying int8 quantization with coremltools.optimize...")

    try:
        from coremltools.optimize.coreml import (
            linear_quantize_weights,
            OptimizationConfig,
            OpLinearQuantizerConfig,
        )

        # Configure linear quantization
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",  # Symmetric quantization
                dtype="int8",             # 8-bit integers
                granularity="per_channel", # Per-channel for better accuracy
                weight_threshold=512,      # Only quantize large weight tensors
            )
        )

        typer.echo("Applying linear quantization...")
        quantized_model = linear_quantize_weights(model, config=config)

        typer.echo("✓ Int8 quantization successful!")
        typer.echo(f"Saving to {output_model}...")
        quantized_model.save(str(output_model))

    except Exception as e:
        typer.echo(f"❌ Optimization API failed: {e}")
        typer.echo("\nTrying palettization (weight clustering) instead...")

        try:
            from coremltools.optimize.coreml import (
                palettize_weights,
                OptimizationConfig,
                OpPalettizerConfig,
            )

            # 256-level palettization (8-bit equivalent)
            config = OptimizationConfig(
                global_config=OpPalettizerConfig(
                    mode="kmeans",
                    nbits=8,  # 256 unique values (2^8)
                    weight_threshold=512,
                )
            )

            typer.echo("Applying 8-bit palettization...")
            palettized_model = palettize_weights(model, config=config)

            typer.echo("✓ 8-bit palettization successful!")
            typer.echo(f"Saving to {output_model}...")
            palettized_model.save(str(output_model))

        except Exception as e2:
            typer.echo(f"❌ Palettization also failed: {e2}")
            typer.echo("\nTrying pruning + float16 as fallback...")

            try:
                from coremltools.optimize.coreml import (
                    prune_weights,
                    OptimizationConfig,
                    OpMagnitudePrunerConfig,
                )

                # Prune small weights + fp16
                config = OptimizationConfig(
                    global_config=OpMagnitudePrunerConfig(
                        target_sparsity=0.5,  # Remove 50% of weights
                        weight_threshold=512,
                    )
                )

                typer.echo("Applying 50% magnitude pruning...")
                pruned_model = prune_weights(model, config=config)

                typer.echo("✓ Pruning successful! (50% sparsity)")
                typer.echo(f"Saving to {output_model}...")
                pruned_model.save(str(output_model))

            except Exception as e3:
                typer.echo(f"❌ All optimization methods failed:")
                typer.echo(f"  Linear quantization: {e}")
                typer.echo(f"  Palettization: {e2}")
                typer.echo(f"  Pruning: {e3}")
                raise typer.Exit(1)

    # Compare sizes
    input_size = sum(f.stat().st_size for f in input_model.rglob('*') if f.is_file())
    output_size = sum(f.stat().st_size for f in output_model.rglob('*') if f.is_file())

    typer.echo("\n" + "="*60)
    typer.echo(f"✓ Compression complete!")
    typer.echo(f"  Original size:  {input_size / (1024**3):.2f} GB")
    typer.echo(f"  Compressed:     {output_size / (1024**3):.2f} GB")
    typer.echo(f"  Reduction:      {(1 - output_size/input_size)*100:.1f}%")
    typer.echo(f"  Compression:    {input_size / output_size:.2f}x")
    typer.echo(f"  Output:         {output_model}")


if __name__ == "__main__":
    app()
