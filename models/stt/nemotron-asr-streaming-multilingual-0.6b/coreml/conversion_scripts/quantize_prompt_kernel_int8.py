"""L1: Quantize prompt_kernel weights to INT8 (weight-only, post-conversion).

The prompt_kernel is a 2-layer MLP applied at the END of the encoder
that conditions the encoded features on language hint (post-encoder
language adapter). Currently FP16. Weight-only INT8 quantization is
in-rule (no calibration data, no retraining) and should be cheap.

Targets:
    prompt_kernel_0_weight_to_fp16: [2048, 1152]
    prompt_kernel_2_weight_to_fp16: [1024, 2048]
    (~8.5 MB FP16 → ~4.25 MB INT8)

Three variants tested:
    1. global INT8 (everything in encoder → INT8 weights)
    2. selective prompt_kernel-only INT8

If global INT8 survives WER, we ship the bigger win. If not, we ship
selective.
"""
from __future__ import annotations

from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def quantize(
    input_pkg: Path = typer.Option(..., "--input"),
    output_pkg: Path = typer.Option(..., "--output"),
    mode: str = typer.Option("selective", "--mode", help="selective | global"),
):
    """Quantize encoder weights to INT8.

    selective: only prompt_kernel ops
    global:    all weights above threshold
    """
    typer.echo(f"Loading {input_pkg} (skip_model_load=True)...")
    mlm = ct.models.MLModel(str(input_pkg), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)

    if mode == "global":
        # Quantize all weights of size >= threshold
        # threshold = small enough to catch FF/attn but skip tiny scalar params
        op_cfg = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=1024,
        )
        config = cto.OptimizationConfig(global_config=op_cfg)
    elif mode == "selective":
        # Quantize ONLY ops named with 'prompt_kernel' in them.
        # The trick: coremltools' op_name_configs accepts regex patterns.
        # However, we don't have a direct op-name reference yet — we'll use
        # the weight metadata to find the ops that consume prompt_kernel weights.
        meta = cto.get_weights_metadata(
            ct.models.MLModel(str(input_pkg), compute_units=ct.ComputeUnit.CPU_ONLY),
            weight_threshold=1024,
        )
        prompt_op_names = []
        for w_name, w_meta in meta.items():
            if "prompt_kernel" in w_name and "weight" in w_name:
                # Each WeightMetadata has child_ops listing consumers
                for child in w_meta.child_ops:
                    prompt_op_names.append(child.name)
        prompt_op_names = sorted(set(prompt_op_names))
        typer.echo(f"  Targeting {len(prompt_op_names)} prompt_kernel ops:")
        for n in prompt_op_names:
            typer.echo(f"    {n}")

        op_cfg = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=1024,
        )
        config = cto.OptimizationConfig(
            global_config=None,
            op_name_configs={name: op_cfg for name in prompt_op_names},
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    typer.echo(f"Applying linear_quantize_weights ({mode})...")
    mlm_loaded = ct.models.MLModel(str(input_pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    compressed = cto.linear_quantize_weights(mlm_loaded, config=config)

    output_pkg.parent.mkdir(parents=True, exist_ok=True)
    if output_pkg.exists():
        import shutil
        if output_pkg.is_dir():
            shutil.rmtree(output_pkg)
    compressed.save(str(output_pkg))
    typer.echo(f"Saved {output_pkg}")

    # Report size delta
    import subprocess
    def du(p):
        return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]
    typer.echo(f"  Input size:  {du(input_pkg)}")
    typer.echo(f"  Output size: {du(output_pkg)}")


if __name__ == "__main__":
    app()
