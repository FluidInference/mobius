#!/usr/bin/env python3
"""Quantize the Encoder model to int8 or float16."""
from pathlib import Path
import typer
import coremltools as ct

app = typer.Typer(add_completion=False)


@app.command()
def quantize(
    input_model: Path = typer.Option(
        Path("build-full/Encoder.mlpackage"),
        "--input",
        help="Input encoder model path",
    ),
    output_model: Path = typer.Option(
        Path("build-full/Encoder-quantized.mlpackage"),
        "--output",
        help="Output quantized model path",
    ),
    mode: str = typer.Option(
        "int8",
        "--mode",
        help="Quantization mode: int8, int4, or float16",
    ),
) -> None:
    """Quantize CoreML encoder model."""

    typer.echo(f"Loading model from {input_model}...")
    model = ct.models.MLModel(str(input_model))

    if mode == "float16":
        typer.echo("Converting to float16...")
        # Float16 precision
        model_fp16 = ct.models.neural_network.quantization_utils.quantize_weights(
            model, nbits=16
        )
        typer.echo(f"✓ Converted to float16")
        typer.echo(f"Saving to {output_model}...")
        model_fp16.save(str(output_model))

    elif mode == "int8":
        typer.echo("Attempting int8 quantization...")
        try:
            # Try new optimize API
            import coremltools.optimize.coreml as cto

            config = cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    weight_threshold=512,
                )
            )

            compressed_model = cto.linear_quantize_weights(model, config=config)
            typer.echo("✓ Int8 quantization successful (new API)")
            typer.echo(f"Saving to {output_model}...")
            compressed_model.save(str(output_model))

        except Exception as e:
            typer.echo(f"New API failed: {e}")
            typer.echo("Trying legacy API...")

            try:
                # Try legacy API
                model_int8 = ct.models.neural_network.quantization_utils.quantize_weights(
                    model, nbits=8
                )
                typer.echo("✓ Int8 quantization successful (legacy API)")
                typer.echo(f"Saving to {output_model}...")
                model_int8.save(str(output_model))

            except Exception as e2:
                typer.echo(f"❌ Both APIs failed:")
                typer.echo(f"  New API: {e}")
                typer.echo(f"  Legacy API: {e2}")
                raise typer.Exit(1)

    elif mode == "int4":
        typer.echo("Attempting int4 quantization...")
        try:
            import coremltools.optimize.coreml as cto

            config = cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    weight_threshold=512,
                    dtype="int4",
                )
            )

            compressed_model = cto.linear_quantize_weights(model, config=config)
            typer.echo("✓ Int4 quantization successful")
            typer.echo(f"Saving to {output_model}...")
            compressed_model.save(str(output_model))

        except Exception as e:
            typer.echo(f"❌ Int4 quantization failed: {e}")
            raise typer.Exit(1)

    else:
        typer.echo(f"Unknown mode: {mode}")
        raise typer.Exit(1)

    # Compare sizes
    input_size = sum(f.stat().st_size for f in input_model.rglob('*') if f.is_file())
    output_size = sum(f.stat().st_size for f in output_model.rglob('*') if f.is_file())

    typer.echo("\n" + "="*60)
    typer.echo(f"✓ Quantization complete!")
    typer.echo(f"  Original size:  {input_size / (1024**3):.2f} GB")
    typer.echo(f"  Quantized size: {output_size / (1024**3):.2f} GB")
    typer.echo(f"  Compression:    {input_size / output_size:.2f}x")
    typer.echo(f"  Output:         {output_model}")


if __name__ == "__main__":
    app()
