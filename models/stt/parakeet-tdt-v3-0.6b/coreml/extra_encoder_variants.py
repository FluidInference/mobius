#!/usr/bin/env python3
"""Extra encoder-scoped quantization variants for Parakeet-TDT-v3.

Builds on top of ``quantize_coreml.py`` by adding encoder-only variants that
were not part of the original sweep:

  * ``enc8bit-palettize``        - 8-bit kmeans palettize (quality reference vs 6-bit)
  * ``enc-prune+int8``           - 50% threshold prune followed by int8-per-channel
  * ``enc-int4-linear-per-channel``    - int4 symmetric, per-output-channel
  * ``enc-int4-palettize-grouped-16``  - int4 kmeans palette, grouped-channel-16
  * ``enc-int4-linear-per-block-32``   - int4 symmetric, per-block (block_size=32)
  * ``enc-prune+int4-block``     - 50% prune then int4-per-block-32

Quality is measured against a baseline encoder run (max-abs / max-rel /
normalized L2 against the fp16 baseline), latency is timed on the requested
compute units, and offline mlpackage->mlmodelc compile time is captured for
each variant.

Notes
-----

* int4 weights require ``MILSpec`` version 9 (iOS18 / macOS 15). After the
  optimization pass, the spec version is bumped explicitly so that the
  produced mlpackage loads on the iOS18 runtime.
* Decoder, joint, joint_decision, mel_encoder, preprocessor are NOT touched
  here - this script intentionally targets the standalone ``parakeet_encoder``
  component so the variants slot directly into the existing v3 model layout.
* The default audio is reused from ``audio/yc_first_minute_16k_15s.wav`` if
  present, identical to ``quantize_coreml.py``.
* Run via ``uv run python extra_encoder_variants.py run`` (typer-driven CLI).

"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import typer

import coremltools as ct
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpLinearQuantizerConfig,
    OpPalettizerConfig,
    OpThresholdPrunerConfig,
    linear_quantize_weights,
    palettize_weights,
    prune_weights,
)

# Reuse helpers from the canonical quantization script. They live next to this
# file in ``quantize_coreml.py``.
from quantize_coreml import (  # type: ignore[import-not-found]
    BYTES_IN_MB,
    _chip_spec_string,
    _dir_size_bytes,
    _max_abs_rel,
    _offline_compile_time_ms,
    _prepare_audio,
    _predict_latency,
    _save_mlpackage,
)


BASE_DIR = Path(__file__).resolve().parent
ENCODER_FILENAME = "parakeet_encoder.mlpackage"


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@dataclass
class EncoderVariant:
    """One encoder-only quantization recipe.

    ``steps`` is a list of (op_kind, OptimizationConfig) tuples applied in
    order. ``bump_to_ios18`` forces the output mlpackage's spec version to
    9 (iOS18 / macOS 15) which is required for int4 weight payloads.
    """

    name: str
    steps: List[Tuple[str, OptimizationConfig]]
    bump_to_ios18: bool = False
    description: str = ""


def _default_variants() -> List[EncoderVariant]:
    return [
        EncoderVariant(
            name="enc8bit-palettize",
            steps=[(
                "palettize",
                OptimizationConfig(
                    global_config=OpPalettizerConfig(mode="kmeans", nbits=8),
                ),
            )],
            description="Encoder: 8-bit kmeans palettize",
        ),
        EncoderVariant(
            name="enc-prune+int8",
            steps=[
                (
                    "prune",
                    OptimizationConfig(
                        global_config=OpThresholdPrunerConfig(
                            threshold=1e-3,
                            minimum_sparsity_percentile=0.5,
                        )
                    ),
                ),
                (
                    "linear",
                    OptimizationConfig(
                        global_config=OpLinearQuantizerConfig(
                            mode="linear_symmetric",
                            dtype="int8",
                            granularity="per_channel",
                            weight_threshold=512,
                        )
                    ),
                ),
            ],
            description="Encoder: 50% prune + int8 per-channel symmetric",
        ),
        EncoderVariant(
            name="enc-int4-linear-per-channel",
            steps=[(
                "linear",
                OptimizationConfig(
                    global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric",
                        dtype="int4",
                        granularity="per_channel",
                        weight_threshold=512,
                    )
                ),
            )],
            bump_to_ios18=True,
            description="Encoder: int4 symmetric, per-channel (iOS18)",
        ),
        EncoderVariant(
            name="enc-int4-palettize-grouped-16",
            steps=[(
                "palettize",
                OptimizationConfig(
                    global_config=OpPalettizerConfig(
                        mode="kmeans",
                        nbits=4,
                        granularity="per_grouped_channel",
                        group_size=16,
                        weight_threshold=512,
                    )
                ),
            )],
            bump_to_ios18=True,
            description="Encoder: int4 grouped-channel palette, group_size=16 (iOS18)",
        ),
        EncoderVariant(
            name="enc-int4-linear-per-block-32",
            steps=[(
                "linear",
                OptimizationConfig(
                    global_config=OpLinearQuantizerConfig(
                        mode="linear_symmetric",
                        dtype="int4",
                        granularity="per_block",
                        block_size=32,
                        weight_threshold=512,
                    )
                ),
            )],
            bump_to_ios18=True,
            description="Encoder: int4 symmetric, per-block (block_size=32, iOS18)",
        ),
        EncoderVariant(
            name="enc-prune+int4-block",
            steps=[
                (
                    "prune",
                    OptimizationConfig(
                        global_config=OpThresholdPrunerConfig(
                            threshold=1e-3,
                            minimum_sparsity_percentile=0.5,
                        )
                    ),
                ),
                (
                    "linear",
                    OptimizationConfig(
                        global_config=OpLinearQuantizerConfig(
                            mode="linear_symmetric",
                            dtype="int4",
                            granularity="per_block",
                            block_size=32,
                            weight_threshold=512,
                        )
                    ),
                ),
            ],
            bump_to_ios18=True,
            description="Encoder: 50% prune + int4 per-block-32 (iOS18)",
        ),
    ]


def _apply_steps(model: ct.models.MLModel, variant: EncoderVariant) -> ct.models.MLModel:
    out = model
    for kind, cfg in variant.steps:
        if kind == "linear":
            out = linear_quantize_weights(out, cfg)
        elif kind == "palettize":
            out = palettize_weights(out, cfg)
        elif kind == "prune":
            out = prune_weights(out, cfg)
        else:
            raise ValueError(f"Unknown step kind: {kind!r}")
    return out


def _bump_spec_to_ios18(model: ct.models.MLModel) -> ct.models.MLModel:
    """Rebuild the MLModel with specificationVersion=9 (iOS18 / macOS 15).

    The optimize.coreml passes preserve the source target (iOS17 = spec 8)
    even when emitting int4 ops, so we must explicitly bump the version
    before saving or the runtime will refuse to load the package.
    """
    spec = model.get_spec()
    spec.specificationVersion = max(int(spec.specificationVersion), 9)
    weights_dir = getattr(model, "weights_dir", None)
    if weights_dir is None:
        return ct.models.MLModel(spec)
    return ct.models.MLModel(spec, weights_dir=weights_dir)


def _compute_unit_for(name: str) -> ct.ComputeUnit:
    return ct.ComputeUnit.CPU_AND_GPU if name == "preprocessor" else ct.ComputeUnit.CPU_AND_NE


def _quality_norm_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Returns 1 - L2(ref - cand) / L2(ref) clamped to [0, 1]."""
    a = np.asarray(reference, dtype=np.float32)
    b = np.asarray(candidate, dtype=np.float32)
    if a.size == 0:
        return 0.0
    l2_ref = float(np.linalg.norm(a))
    l2_err = float(np.linalg.norm(a - b))
    if l2_ref <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - l2_err / (l2_ref + 1e-8)))


@app.command()
def run(
    input_dir: Path = typer.Option(
        Path("parakeet_coreml"),
        help="Baseline directory with parakeet_*.mlpackage and metadata.json",
    ),
    output_root: Path = typer.Option(
        Path("parakeet_coreml_encoder_only"),
        help="Output root for encoder-only quantized variants.",
    ),
    validation_audio: Optional[Path] = typer.Option(
        None,
        exists=True,
        resolve_path=True,
        help="Optional 15s, 16kHz wav for evaluation (defaults to bundled audio if present).",
    ),
    compute_units: str = typer.Option(
        "CPU_AND_NE",
        help="Compute units used to evaluate the encoder. Preprocessor is forced to CPU_AND_GPU.",
    ),
    runs: int = typer.Option(10, help="Timed runs per model"),
    only: Optional[List[str]] = typer.Option(
        None,
        "--only",
        help="Restrict to a subset of variant names (repeatable).",
    ),
) -> None:
    """Generate the extra encoder variants and report quality/latency/size."""
    input_dir = input_dir.resolve()
    output_root = output_root.resolve()

    meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        raise typer.BadParameter(f"Expected metadata.json in {input_dir}")
    meta = json.loads(meta_path.read_text())

    sr = int(meta.get("sample_rate", 16000))
    seconds = float(meta.get("max_audio_seconds", 15.0))

    encoder_src = input_dir / ENCODER_FILENAME
    if not encoder_src.exists():
        raise typer.BadParameter(f"Missing baseline encoder mlpackage: {encoder_src}")

    pre_src = input_dir / "parakeet_preprocessor.mlpackage"
    if not pre_src.exists():
        raise typer.BadParameter(f"Missing baseline preprocessor: {pre_src}")

    # Default audio (matches quantize_coreml.py default)
    default_audio = (BASE_DIR / "audio" / "yc_first_minute_16k_15s.wav").resolve()
    audio_path = validation_audio if validation_audio is not None else (
        default_audio if default_audio.exists() else None
    )
    if audio_path is not None and validation_audio is None:
        typer.echo(f"Using default validation audio: {audio_path}")

    audio, audio_len = _prepare_audio(seconds, sr, audio_path)

    # Baseline references
    pre_base = ct.models.MLModel(str(pre_src), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    pre_out = pre_base.predict({"audio_signal": audio, "audio_length": audio_len})
    mel_ref = np.array(pre_out["mel"], dtype=np.float32, copy=True)
    mel_len = np.array(pre_out["mel_length"], dtype=np.int32, copy=True)
    enc_inputs = {"mel": mel_ref, "mel_length": mel_len}

    enc_base = ct.models.MLModel(str(encoder_src), compute_units=_compute_unit_for("encoder"))
    enc_base_out = enc_base.predict(enc_inputs)
    encoder_ref = np.array(enc_base_out["encoder"], dtype=np.float32, copy=True)
    enc_base_ms, _ = _predict_latency(enc_base, enc_inputs, runs=runs)
    enc_base_size = _dir_size_bytes(encoder_src)
    enc_base_compile_ms = _offline_compile_time_ms(encoder_src)

    typer.echo(
        f"Baseline encoder: size={enc_base_size / BYTES_IN_MB:.2f} MB, "
        f"latency={enc_base_ms:.2f} ms, compile={enc_base_compile_ms:.0f} ms"
    )

    # Variants (optionally filtered)
    variants = _default_variants()
    if only:
        keep = {name.strip() for name in only if name.strip()}
        variants = [v for v in variants if v.name in keep]
        if not variants:
            raise typer.BadParameter(
                f"No variants matched --only={sorted(keep)}; available: "
                f"{[v.name for v in _default_variants()]}"
            )

    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, object]] = {
        "_meta": {
            "host": _chip_spec_string(compute_units),
            "input_dir": str(input_dir),
            "output_root": str(output_root),
            "audio": str(audio_path) if audio_path else None,
        },
        "baseline": {
            "size_mb": float(enc_base_size) / BYTES_IN_MB,
            "size_bytes": float(enc_base_size),
            "latency_ms": enc_base_ms,
            "compile_ms": enc_base_compile_ms,
            "quality": 1.0,
            "compression": 1.0,
        },
        "variants": {},
    }

    for variant in variants:
        out_dir = output_root / variant.name
        out_dir.mkdir(parents=True, exist_ok=True)
        dst_path = out_dir / ENCODER_FILENAME
        if dst_path.exists():
            shutil.rmtree(dst_path)

        typer.echo(f"\n=== {variant.name} ===")
        typer.echo(f"  {variant.description}")
        t0 = time.perf_counter()
        try:
            base_for_optim = ct.models.MLModel(
                str(encoder_src), compute_units=_compute_unit_for("encoder")
            )
            # int4 weight payloads require iOS18 (spec version 9). The
            # optimize.coreml int4 pass refuses to run if the source spec is
            # still pinned to iOS17, so bump *before* applying steps.
            try:
                target = ct.target.iOS18 if variant.bump_to_ios18 else ct.target.iOS17
                base_for_optim.minimum_deployment_target = target
            except Exception:
                pass
            if variant.bump_to_ios18:
                base_for_optim = _bump_spec_to_ios18(base_for_optim)
            optimized = _apply_steps(base_for_optim, variant)
            if variant.bump_to_ios18:
                # Defensive: ensure the optimized output is still iOS18.
                optimized = _bump_spec_to_ios18(optimized)
        except Exception as e:
            typer.echo(f"  ! Quantization failed: {e}")
            summary["variants"][variant.name] = {"error": repr(e)}
            continue
        optim_s = time.perf_counter() - t0

        _save_mlpackage(optimized, dst_path, f"{variant.name} (encoder-only)")

        # Reload at requested compute units for measurement
        try:
            measured = ct.models.MLModel(
                str(dst_path), compute_units=_compute_unit_for("encoder")
            )
            enc_q_out = measured.predict(enc_inputs)
            encoder_q = np.array(enc_q_out["encoder"], dtype=np.float32, copy=True)
            quality = _quality_norm_l2(encoder_ref, encoder_q)
            max_abs, max_rel = _max_abs_rel(encoder_ref, encoder_q)
            latency_ms, latency_std = _predict_latency(measured, enc_inputs, runs=runs)
        except Exception as e:
            typer.echo(f"  ! Reload/predict failed: {e}")
            summary["variants"][variant.name] = {"error": repr(e)}
            continue

        size_bytes = _dir_size_bytes(dst_path)
        compile_ms = _offline_compile_time_ms(dst_path)

        record = {
            "size_bytes": float(size_bytes),
            "size_mb": float(size_bytes) / BYTES_IN_MB,
            "compression": float(enc_base_size) / float(max(size_bytes, 1)),
            "latency_ms": latency_ms,
            "latency_std_ms": latency_std,
            "compile_ms": compile_ms,
            "quality_norm_l2": quality,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "optimize_seconds": optim_s,
            "spec_version_bumped_to_ios18": variant.bump_to_ios18,
            "description": variant.description,
        }
        summary["variants"][variant.name] = record

        typer.echo(
            f"  size={record['size_mb']:.2f} MB ({record['compression']:.2f}x), "
            f"latency={latency_ms:.2f}±{latency_std:.2f} ms, "
            f"quality={quality:.4f}, compile={compile_ms:.0f} ms"
        )

    summary_path = output_root / "encoder_variants_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\nWrote {summary_path}")


if __name__ == "__main__":
    app()
