#!/usr/bin/env python3
"""Quantize CoreML mlpackages and compare quality, compression, latency, and compile time.

This script focuses on the fused models:
  - parakeet_mel_encoder.mlpackage (waveform -> encoder)
  - parakeet_joint_decision.mlpackage (joint + softmax/argmax)

It also quantizes the rest of the components in the directory for completeness.

Variants tried by default (examples):
  - int8-linear (per-channel)
  - palettize 4/6/8-bit (global and Mel-only)
  - jd-only variants (int8 and palette)
  - prune + int8

Outputs:
  - A new directory per variant under <output_root>/<variant>/ with quantized mlpackages
  - quantization_summary.json with aggregate metrics (baseline vs each variant)
  - Plots saved under <output_root>/plots/ and mirrored to <repo>/plots/quantize/<compute_units_lower>/
    - fused_quality.png / fused_latency.png / fused_compression.png / fused_size.png (latency chart also shows compile time)
    - all_components_quality.png / all_components_latency.png / all_components_compression.png / all_components_size.png (latency chart also shows compile time)

Notes:
  - Uses CoreMLTools optimize.coreml linear/palettize/prune for MLProgram models.
  - Keeps the fixed 15-second window shapes as per context/coreml_component_io.md.
  - Run via `uv run` to ensure reproducible environment.
  - Sets minimum deployment target to iOS 17 for all outputs and loads models with `compute_units=ALL` by default to enable ANE.
  - Tracks offline compile time (mlpackage->mlmodelc) and includes that metric in plots.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf
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


# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


BASE_DIR = Path(__file__).resolve().parent
BYTES_IN_MB = 1024 * 1024


DISPLAY_LABEL_OVERRIDES = {
    "mel6bit-palettize": "mel6bit",
    "mel4bit-palettize": "mel4bit",
    "mel8bit-palettize": "mel8bit",
    "all8bit-palettize": "all8bit",
    "int8-linear": "int8 linear",
    "int8-linear_symmetric-pt": "int8 linear pt",
}


DEFAULT_TARGET_COMPONENTS: Tuple[str, str] = ("mel_encoder", "joint_decision")


# Formatting helpers -------------------------------------------------------


def _format_labels(labels: List[str]) -> List[str]:
    formatted: List[str] = []
    for label in labels:
        pretty = DISPLAY_LABEL_OVERRIDES.get(label, label)
        pretty = pretty.replace("_", " ")
        pretty = pretty.replace("-", "\n")
        formatted.append(pretty)
    return formatted


def _friendly_component_name(name: str) -> str:
    return name.replace("_", " ").title()


def _get_metric_series(
    metrics: Dict[str, Dict[str, List[float]]],
    component: str,
    key: str,
    length: int,
) -> List[float]:
    values = metrics.get(component, {}).get(key)
    if values is None:
        return [float("nan")] * length
    if len(values) >= length:
        return list(values[:length])
    padded = list(values)
    padded.extend([float("nan")] * (length - len(values)))
    return padded


def _plot_bar_rows(
    out_path: Path,
    display_labels: List[str],
    rows: List[Dict[str, object]],
    title_suffix: str,
) -> None:
    if not HAS_MPL:
        return
    if not rows or not display_labels:
        return

    n_labels = len(display_labels)
    fig_width = max(8.0, 1.3 * n_labels)
    fig_height = max(2.6 * len(rows), 3.0)

    fig, axes = plt.subplots(len(rows), 1, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()
    x = np.arange(n_labels)

    for ax, row in zip(axes, rows):
        values = np.asarray(row.get("values", [float("nan")] * n_labels), dtype=np.float64)
        color = row.get("color")
        bars = ax.bar(x, values, width=0.6, color=color)
        ax.set_title(str(row.get("title", "")))
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=25, ha="right")
        ylim = row.get("ylim")
        if isinstance(ylim, tuple) and len(ylim) == 2:
            ax.set_ylim(ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        # Add value labels for bars
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _plot_fused_category_charts(
    plot_dir: Path,
    labels: List[str],
    mel_quality: List[float],
    mel_latency_ms: List[float],
    mel_compression: List[float],
    mel_size_mb: List[float],
    jd_acc: List[float],
    jd_latency_ms: List[float],
    jd_compression: List[float],
    jd_size_mb: List[float],
    mel_compile_ms: Optional[List[float]] = None,
    jd_compile_ms: Optional[List[float]] = None,
    title_suffix: str = "",
) -> List[Path]:
    if not HAS_MPL:
        return []
    outputs: List[Path] = []
    display = _format_labels(labels)
    prefix = f"Fused Components — {title_suffix}" if title_suffix else "Fused Components"

    quality_path = plot_dir / "fused_quality.png"
    _plot_bar_rows(
        quality_path,
        display,
        [
            {
                "title": "MelEncoder quality (1 - norm err)",
                "values": mel_quality,
                "color": "C0",
                "ylim": (0.0, 1.05),
            },
            {
                "title": "JointDecision token-id match rate",
                "values": jd_acc,
                "color": "C0",
                "ylim": (0.0, 1.05),
            },
        ],
        f"{prefix} — Quality",
    )
    outputs.append(quality_path)

    compression_path = plot_dir / "fused_compression.png"
    _plot_bar_rows(
        compression_path,
        display,
        [
            {
                "title": "MelEncoder compression ratio",
                "values": mel_compression,
                "color": "C2",
            },
            {
                "title": "JointDecision compression ratio",
                "values": jd_compression,
                "color": "C2",
            },
        ],
        f"{prefix} — Compression",
    )
    outputs.append(compression_path)

    size_path = plot_dir / "fused_size.png"
    _plot_bar_rows(
        size_path,
        display,
        [
            {
                "title": "MelEncoder size (MB)",
                "values": mel_size_mb,
                "color": "C4",
            },
            {
                "title": "JointDecision size (MB)",
                "values": jd_size_mb,
                "color": "C4",
            },
        ],
        f"{prefix} — Size",
    )
    outputs.append(size_path)

    latency_rows = [
        {
            "title": "MelEncoder latency (ms)",
            "values": mel_latency_ms,
            "color": "C1",
        },
        {
            "title": "JointDecision latency (ms)",
            "values": jd_latency_ms,
            "color": "C1",
        },
    ]
    if mel_compile_ms is not None and jd_compile_ms is not None:
        latency_rows.extend(
            [
                {
                    "title": "MelEncoder compile (ms)",
                    "values": mel_compile_ms,
                    "color": "C3",
                },
                {
                    "title": "JointDecision compile (ms)",
                    "values": jd_compile_ms,
                    "color": "C3",
                },
            ]
        )

    latency_path = plot_dir / "fused_latency.png"
    _plot_bar_rows(
        latency_path,
        display,
        latency_rows,
        f"{prefix} — Latency",
    )
    outputs.append(latency_path)

    return outputs


def _plot_all_component_category_charts(
    plot_dir: Path,
    labels: List[str],
    metrics: Dict[str, Dict[str, List[float]]],
    title_suffix: str = "",
) -> List[Path]:
    if not HAS_MPL:
        return []
    outputs: List[Path] = []
    display = _format_labels(labels)
    prefix = f"Component Breakdown — {title_suffix}" if title_suffix else "Component Breakdown"
    comp_order = [
        "preprocessor",
        "encoder",
        "mel_encoder",
        "decoder",
        "joint",
        "joint_decision",
    ]
    n = len(labels)

    quality_rows: List[Dict[str, object]] = []
    for comp in comp_order:
        friendly = _friendly_component_name(comp)
        if comp == "joint_decision":
            values = _get_metric_series(metrics, comp, "acc", n)
            title = f"{friendly} token-id match rate"
        else:
            values = _get_metric_series(metrics, comp, "quality", n)
            title = f"{friendly} quality (1 - norm err)"
        quality_rows.append({"title": title, "values": values, "color": "C0", "ylim": (0.0, 1.05)})

    compression_rows: List[Dict[str, object]] = []
    for comp in comp_order:
        friendly = _friendly_component_name(comp)
        compression_rows.append(
            {
                "title": f"{friendly} compression ratio",
                "values": _get_metric_series(metrics, comp, "compression", n),
                "color": "C2",
            }
        )

    size_rows: List[Dict[str, object]] = []
    for comp in comp_order:
        friendly = _friendly_component_name(comp)
        size_rows.append(
            {
                "title": f"{friendly} size (MB)",
                "values": _get_metric_series(metrics, comp, "size_mb", n),
                "color": "C4",
            }
        )

    latency_rows: List[Dict[str, object]] = []
    for comp in comp_order:
        friendly = _friendly_component_name(comp)
        latency_rows.append(
            {
                "title": f"{friendly} latency (ms)",
                "values": _get_metric_series(metrics, comp, "latency_ms", n),
                "color": "C1",
            }
        )

    compile_rows: List[Dict[str, object]] = []
    for comp in comp_order:
        friendly = _friendly_component_name(comp)
        compile_rows.append(
            {
                "title": f"{friendly} compile (ms)",
                "values": _get_metric_series(metrics, comp, "compile_ms", n),
                "color": "C3",
            }
        )

    quality_path = plot_dir / "all_components_quality.png"
    _plot_bar_rows(quality_path, display, quality_rows, f"{prefix} — Quality")
    outputs.append(quality_path)

    compression_path = plot_dir / "all_components_compression.png"
    _plot_bar_rows(compression_path, display, compression_rows, f"{prefix} — Compression")
    outputs.append(compression_path)

    size_path = plot_dir / "all_components_size.png"
    _plot_bar_rows(size_path, display, size_rows, f"{prefix} — Size")
    outputs.append(size_path)

    latency_path = plot_dir / "all_components_latency.png"
    _plot_bar_rows(latency_path, display, latency_rows, f"{prefix} — Latency")
    outputs.append(latency_path)

    compile_path = plot_dir / "all_components_compile.png"
    _plot_bar_rows(compile_path, display, compile_rows, f"{prefix} — Compile")
    outputs.append(compile_path)

    return outputs


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@dataclass
class VariantConfig:
    name: str
    # List of (op_kind, opt_config) steps applied in order; op_kind in {'linear','palettize','prune'}
    steps: List[Tuple[str, OptimizationConfig]]
    category: str
    whitelist: Optional[List[str]] = None  # component names to apply; others copied


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _load_metadata(input_dir: Path) -> Dict[str, object]:
    meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        raise typer.BadParameter(f"Expected metadata.json in {input_dir}")
    return json.loads(meta_path.read_text())


def _prepare_audio(
    seconds: float,
    sample_rate: int,
    audio_path: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    max_samples = int(round(seconds * sample_rate))
    if audio_path is None:
        # random but deterministic sample
        rng = np.random.default_rng(1234)
        audio = rng.standard_normal(size=(1, max_samples), dtype=np.float32).astype(np.float32)
    else:
        data, sr = sf.read(str(audio_path), dtype="float32")
        if sr != sample_rate:
            raise typer.BadParameter(
                f"Validation audio sample rate {sr} != expected {sample_rate}"
            )
        if data.ndim > 1:
            data = data[:, 0]
        if data.size < max_samples:
            data = np.pad(data, (0, max_samples - data.size))
        elif data.size > max_samples:
            data = data[:max_samples]
        audio = data.reshape(1, -1).astype(np.float32, copy=False)
    length = np.array([max_samples], dtype=np.int32)
    return audio, length


def _predict_latency(model: ct.models.MLModel, inputs: Dict[str, np.ndarray], runs: int = 10, warmup: int = 3) -> Tuple[float, float]:
    # Warmup
    for _ in range(max(0, warmup)):
        _ = model.predict(inputs)
    # Timed runs
    times: List[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        _ = model.predict(inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    arr = np.array(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def _max_abs_rel(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0:
        return 0.0, 0.0
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom == 0.0, 0.0, diff / denom)
    max_rel = float(rel.max())
    return max_abs, max_rel


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure iOS 17 target for proper MLProgram ops (e.g., blockwise shift/scale)
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = "Fluid Inference"
    model.save(str(path))


def _offline_compile_time_ms(model_path: Path) -> float:
    """Compile mlpackage -> mlmodelc and return wall time in ms (host offline compile).

    Returns NaN on failure.
    """
    compiled_dir: Optional[Path] = None
    expected_dir = model_path.with_suffix(".mlmodelc")
    try:
        # Delete any prior compile artifact in-place so we can measure a fresh build
        if expected_dir.exists():
            shutil.rmtree(expected_dir, ignore_errors=True)

        t0 = time.perf_counter()
        compiled_path = ct.utils.compile_model(str(model_path))
        t1 = time.perf_counter()
        compiled_dir = Path(compiled_path)
        return (t1 - t0) * 1000.0
    except Exception:
        return float("nan")
    finally:
        if (
            compiled_dir is not None
            and compiled_dir.exists()
            and compiled_dir == expected_dir
        ):
            shutil.rmtree(compiled_dir, ignore_errors=True)

def _chip_spec_string(compute_units: str) -> str:
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        chip = platform.processor() or platform.machine()
    mac_ver = platform.mac_ver()[0] or platform.platform()
    return f"Host: {chip} • macOS {mac_ver} • CoreMLTools {ct.__version__} • ComputeUnits={compute_units} • Min Target: iOS17"


def _quantize_dir(
    input_dir: Path,
    output_dir: Path,
    variant: VariantConfig,
    global_whitelist: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Quantize all mlpackages in input_dir into output_dir using given variant config.

    Returns a map of component name -> saved relative path.
    """
    meta = _load_metadata(input_dir)
    comps = meta.get("components", {})
    saved: Dict[str, str] = {}
    for name, cfg in comps.items():
        src_name = cfg.get("path")
        if not src_name:
            continue
        src_path = input_dir / src_name
        if not src_path.exists():
            continue
        dst_path = output_dir / src_name
        base_model = ct.models.MLModel(str(src_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        # Target iOS17 when running optimizations so the right ops are chosen
        try:
            base_model.minimum_deployment_target = ct.target.iOS17
        except Exception:
            pass

        if name == "decoder":
            typer.echo(f"[{variant.name}] Skipping decoder quantization; copying baseline: {src_name}")
            _save_mlpackage(base_model, dst_path, "Baseline copy (decoder quantization disabled) - decoder")
            saved[name] = dst_path.name
            continue

        skip_reasons: List[str] = []
        if variant.whitelist is not None and name not in variant.whitelist:
            skip_reasons.append("not targeted by variant")
        if global_whitelist is not None and name not in global_whitelist:
            skip_reasons.append("not in requested components")

        if skip_reasons:
            reason = "; ".join(skip_reasons)
            typer.echo(f"[{variant.name}] Skipping {name} ({reason}); copying baseline: {src_name}")
            _save_mlpackage(base_model, dst_path, f"Baseline copy ({reason}) - {name}")
            saved[name] = dst_path.name
            continue

        typer.echo(f"[{variant.name}] Quantizing {name}: {src_name}")

        try:
            q_model = base_model
            for step_kind, step_cfg in variant.steps:
                if step_kind == 'linear':
                    q_model = linear_quantize_weights(q_model, step_cfg)
                elif step_kind == 'palettize':
                    q_model = palettize_weights(q_model, step_cfg)
                elif step_kind == 'prune':
                    q_model = prune_weights(q_model, step_cfg)
                else:
                    raise ValueError(f"Unknown variant step: {step_kind}")
        except Exception as e:
            # If quantization fails (e.g., unsupported op), fall back to copying baseline.
            typer.echo(f"  ! Failed to quantize {name} with {variant.name}: {e}. Copying baseline.")
            _save_mlpackage(base_model, dst_path, f"Baseline copy (failed to quantize) - {name}")
        else:
            _save_mlpackage(q_model, dst_path, f"{variant.name} quantized - {name}")
        saved[name] = dst_path.name
    # Persist a variant metadata shim
    out_meta = {
        "variant": variant.name,
        "base_dir": str(input_dir.resolve()),
        "components": saved,
    }
    (output_dir / "quantization_metadata.json").write_text(json.dumps(out_meta, indent=2))
    return saved


@app.command()
def quantize(
    input_dir: Path = typer.Option(Path("parakeet_coreml"), help="Directory containing baseline mlpackages + metadata.json"),
    output_root: Path = typer.Option(Path("parakeet_coreml_quantized"), help="Root output dir for quantized variants"),
    validation_audio: Optional[Path] = typer.Option(None, exists=True, resolve_path=True, help="Optional 15s, 16kHz wav for evaluation (defaults to bundled audio if present)"),
    compute_units: str = typer.Option("ALL", help="Compute units for evaluation: CPU_ONLY, CPU_AND_NE, ALL (use ALL for ANE)"),
    runs: int = typer.Option(10, help="Timed runs per model for latency measurement"),
    categories: Optional[List[str]] = typer.Option(
        None,
        "--category",
        "-c",
        help="Only run quantization variants in these categories (e.g., linear, mel-palettize). Can be repeated.",
    ),
    components: Optional[List[str]] = typer.Option(
        None,
        "--component",
        "-m",
        help="Component names to quantize. Defaults to mel_encoder and joint_decision. Use 'all' to keep every component enabled.",
    ),
) -> None:
    """Quantize models, then compare quality, compression, latency, and compile time.

    Variants include int8-linear (per-channel/per-tensor/block), palettization (4/6/8-bit),
    jd-only probes, and prune+int8. Baseline is the pre-converted models.
    """
    meta = _load_metadata(input_dir)
    sr = int(meta.get("sample_rate", 16000))
    seconds = float(meta.get("max_audio_seconds", 15.0))

    components_meta = meta.get("components", {})
    component_lookup = {name.lower(): name for name in components_meta.keys()}

    component_filter: Optional[Set[str]]
    if components:
        normalized_components = [comp.strip().lower() for comp in components if comp.strip()]
        if any(comp == "all" for comp in normalized_components):
            component_filter = None
        else:
            resolved: List[str] = []
            invalid: List[str] = []
            for comp in normalized_components:
                match = component_lookup.get(comp)
                if match is None:
                    invalid.append(comp)
                else:
                    resolved.append(match)
            if invalid:
                available = ", ".join(sorted(component_lookup.values())) or "none"
                bad = ", ".join(sorted(set(invalid)))
                raise typer.BadParameter(
                    f"Unknown component(s) for --component: {bad}. Available components: {available}."
                )
            component_filter = set(resolved)
    else:
        resolved_default = [component_lookup.get(comp.lower()) for comp in DEFAULT_TARGET_COMPONENTS]
        component_filter = {comp for comp in resolved_default if comp is not None}
        if not component_filter:
            component_filter = None

    if component_filter is None:
        typer.echo("Quantizing components: all components (no whitelist)")
    else:
        typer.echo(
            "Quantizing components: "
            + ", ".join(sorted(component_filter))
        )

    # Default audio if present
    default_audio = (BASE_DIR / "audio" / "yc_first_minute_16k_15s.wav").resolve()
    audio_path = validation_audio if validation_audio is not None else (default_audio if default_audio.exists() else None)
    if audio_path is not None and validation_audio is None:
        typer.echo(f"Using default validation audio: {audio_path}")
    audio, audio_len = _prepare_audio(seconds, sr, audio_path)

    # Load baseline models and helpers for inputs
    # Baseline models for input preparation
    pre_base = ct.models.MLModel(str(input_dir / "parakeet_preprocessor.mlpackage"), compute_units=ct.ComputeUnit[compute_units])
    enc_base = ct.models.MLModel(str(input_dir / "parakeet_encoder.mlpackage"), compute_units=ct.ComputeUnit[compute_units])
    mel_encoder_base = ct.models.MLModel(str(input_dir / "parakeet_mel_encoder.mlpackage"), compute_units=ct.ComputeUnit[compute_units])
    decoder_base = ct.models.MLModel(str(input_dir / "parakeet_decoder.mlpackage"), compute_units=ct.ComputeUnit[compute_units])
    joint_decision_base = ct.models.MLModel(str(input_dir / "parakeet_joint_decision.mlpackage"), compute_units=ct.ComputeUnit[compute_units])
    joint_base = ct.models.MLModel(str(input_dir / "parakeet_joint.mlpackage"), compute_units=ct.ComputeUnit[compute_units])

    # Prepare typical inputs once using baseline models
    pre_out = pre_base.predict({"audio_signal": audio, "audio_length": audio_len})
    mel_ref = np.array(pre_out["mel"], dtype=np.float32, copy=True)
    mel_len = np.array(pre_out["mel_length"], dtype=np.int32, copy=True)

    enc_out = enc_base.predict({"mel": mel_ref, "mel_length": mel_len})
    encoder_ref = np.array(enc_out["encoder"], dtype=np.float32, copy=True)
    encoder_len = np.array(enc_out["encoder_length"], dtype=np.int32, copy=True)

    # Decoder inputs from metadata
    dec_in = meta["components"]["decoder"]["inputs"]
    targets_shape = tuple(int(x) for x in dec_in["targets"])  # e.g., (1, 256)
    h_shape = tuple(int(x) for x in dec_in["h_in"])         # e.g., (2, 1, 640)
    # Use zeros as targets for reproducibility without needing blank idx
    targets = np.zeros(targets_shape, dtype=np.int32)
    target_length = np.array([targets_shape[1]], dtype=np.int32)
    h0 = np.zeros(h_shape, dtype=np.float32)
    c0 = np.zeros(h_shape, dtype=np.float32)
    dec_out = decoder_base.predict({
        "targets": targets,
        "target_length": target_length,
        "h_in": h0,
        "c_in": c0,
    })
    decoder_ref = np.array(dec_out["decoder"], dtype=np.float32, copy=True)

    # Baseline sizes per component
    pre_base_size = _dir_size_bytes(input_dir / "parakeet_preprocessor.mlpackage")
    enc_base_size = _dir_size_bytes(input_dir / "parakeet_encoder.mlpackage")
    mel_base_size = _dir_size_bytes(input_dir / "parakeet_mel_encoder.mlpackage")
    dec_base_size = _dir_size_bytes(input_dir / "parakeet_decoder.mlpackage")
    joint_base_size = _dir_size_bytes(input_dir / "parakeet_joint.mlpackage")
    jd_base_size = _dir_size_bytes(input_dir / "parakeet_joint_decision.mlpackage")

    # Baseline latencies
    pre_base_inputs = {"audio_signal": audio, "audio_length": audio_len}
    enc_base_inputs = {"mel": mel_ref, "mel_length": mel_len}
    mel_base_inputs = {"audio_signal": audio, "audio_length": audio_len}
    dec_base_inputs = {"targets": targets, "target_length": target_length, "h_in": h0, "c_in": c0}
    joint_base_inputs = {"encoder": encoder_ref, "decoder": decoder_ref}
    jd_base_inputs = {"encoder": encoder_ref, "decoder": decoder_ref}
    pre_base_ms, _ = _predict_latency(pre_base, pre_base_inputs, runs=runs)
    enc_base_ms, _ = _predict_latency(enc_base, enc_base_inputs, runs=runs)
    mel_base_ms, _ = _predict_latency(mel_encoder_base, mel_base_inputs, runs=runs)
    dec_base_ms, _ = _predict_latency(decoder_base, dec_base_inputs, runs=runs)
    joint_base_ms, _ = _predict_latency(joint_base, joint_base_inputs, runs=runs)
    jd_base_ms, _ = _predict_latency(joint_decision_base, jd_base_inputs, runs=runs)
    # Cache baseline joint logits for comparisons
    logits_base = np.array(joint_base.predict(joint_base_inputs)["logits"], dtype=np.float32, copy=True)

    # Variants
    variants: List[VariantConfig] = [
        VariantConfig(
            name="int8-linear",
            steps=[(
                "linear",
                OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear", granularity="per_channel")),
            )],
            category="linear",
        ),
        # Per-tensor flavor (sometimes faster to compile)
        VariantConfig(
            name="int8-linear_symmetric-pt",
            steps=[(
                "linear",
                OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_tensor")),
            )],
            category="linear",
        ),
        # 6-bit palettization for MelEncoder only
        VariantConfig(
            name="mel6bit-palettize",
            steps=[(
                "palettize",
                OptimizationConfig(
                    global_config=OpPalettizerConfig(
                        mode="kmeans",
                        nbits=6,
                        granularity="per_channel",
                    )
                ),
            )],
            category="mel-palettize",
            whitelist=["mel_encoder"],
        ),
        VariantConfig(
            name="mel4bit-palettize",
            steps=[(
                "palettize",
                OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4)),
            )],
            category="mel-palettize",
            whitelist=["mel_encoder"],
        ),
        # 8-bit palettization for MelEncoder only
        VariantConfig(
            name="mel8bit-palettize",
            steps=[(
                "palettize",
                OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=8)),
            )],
            category="mel-palettize",
            whitelist=["mel_encoder"],
        ),
        # Global palettization flavors (8/4-bit) across all components
        VariantConfig(
            name="all8bit-palettize",
            steps=[(
                "palettize",
                OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=8)),
            )],
            category="global-palettize",
        ),
    ]

    available_categories = {variant.category for variant in variants}
    category_lookup = {cat.lower(): cat for cat in available_categories}
    selected_categories: Set[str]
    if categories:
        normalized_categories = [cat.strip().lower() for cat in categories if cat.strip()]
        if normalized_categories:
            invalid_categories = [cat for cat in normalized_categories if cat not in category_lookup]
            if invalid_categories:
                available_str = ", ".join(sorted(available_categories)) or "none"
                bad = ", ".join(sorted(set(invalid_categories)))
                raise typer.BadParameter(
                    f"Unknown category (--category): {bad}. Available categories: {available_str}."
                )
            selected_categories = {category_lookup[cat] for cat in normalized_categories}
            variants = [variant for variant in variants if variant.category in selected_categories]
        else:
            selected_categories = available_categories
    else:
        selected_categories = available_categories

    if not variants:
        typer.echo("No quantization variants matched the requested categories; nothing to do.")
        raise typer.Exit(code=0)

    typer.echo("Running variant categories: " + ", ".join(sorted(selected_categories)))

    # Aggregate results (baseline + variants)
    summary: Dict[str, Dict[str, object]] = {}
    variants_names: List[str] = []
    # Build arrays including baseline as first label
    fused_labels: List[str] = ["baseline"]
    mel_quality_scores: List[float] = [1.0]
    mel_latency_ms: List[float] = [mel_base_ms]
    mel_compression: List[float] = [1.0]
    mel_size_mb: List[float] = [float(mel_base_size) / BYTES_IN_MB]
    # Offline compile time (host) for fused models
    mel_compile_ms: List[float] = [_offline_compile_time_ms(input_dir / "parakeet_mel_encoder.mlpackage")]
    jd_accuracy: List[float] = [1.0]
    jd_latency_ms: List[float] = [jd_base_ms]
    jd_compression: List[float] = [1.0]
    jd_size_mb: List[float] = [float(jd_base_size) / BYTES_IN_MB]
    jd_compile_ms: List[float] = [_offline_compile_time_ms(input_dir / "parakeet_joint_decision.mlpackage")]

    # For the all-components chart, collect per-component metrics similarly
    all_metrics: Dict[str, Dict[str, List[float]]] = {
        "preprocessor": {
            "quality": [1.0],
            "compression": [1.0],
            "latency_ms": [pre_base_ms],
            "compile_ms": [_offline_compile_time_ms(input_dir / "parakeet_preprocessor.mlpackage")],
            "size_mb": [float(pre_base_size) / BYTES_IN_MB],
        },
        "encoder": {
            "quality": [1.0],
            "compression": [1.0],
            "latency_ms": [enc_base_ms],
            "compile_ms": [_offline_compile_time_ms(input_dir / "parakeet_encoder.mlpackage")],
            "size_mb": [float(enc_base_size) / BYTES_IN_MB],
        },
        "mel_encoder": {
            "quality": [1.0],
            "compression": [1.0],
            "latency_ms": [mel_base_ms],
            "compile_ms": [mel_compile_ms[0]],
            "size_mb": [float(mel_base_size) / BYTES_IN_MB],
        },
        "decoder": {
            "quality": [1.0],
            "compression": [1.0],
            "latency_ms": [dec_base_ms],
            "compile_ms": [_offline_compile_time_ms(input_dir / "parakeet_decoder.mlpackage")],
            "size_mb": [float(dec_base_size) / BYTES_IN_MB],
        },
        "joint": {
            "quality": [1.0],
            "compression": [1.0],
            "latency_ms": [joint_base_ms],
            "compile_ms": [_offline_compile_time_ms(input_dir / "parakeet_joint.mlpackage")],
            "size_mb": [float(joint_base_size) / BYTES_IN_MB],
        },
        "joint_decision": {
            "acc": [1.0],
            "compression": [1.0],
            "latency_ms": [jd_base_ms],
            "compile_ms": [jd_compile_ms[0]],
            "size_mb": [float(jd_base_size) / BYTES_IN_MB],
        },
    }

    # Populate baseline entry
    summary["baseline"] = {
        "components": {
            "preprocessor": {
                "quality": 1.0,
                "latency_ms": pre_base_ms,
                "size_bytes": float(pre_base_size),
                "size_mb": float(pre_base_size) / BYTES_IN_MB,
                "compression_ratio": 1.0,
                "compile_ms": all_metrics["preprocessor"]["compile_ms"][0],
            },
            "encoder": {
                "quality": 1.0,
                "latency_ms": enc_base_ms,
                "size_bytes": float(enc_base_size),
                "size_mb": float(enc_base_size) / BYTES_IN_MB,
                "compression_ratio": 1.0,
                "compile_ms": all_metrics["encoder"]["compile_ms"][0],
            },
            "mel_encoder": {
                "quality": 1.0,
                "latency_ms": mel_base_ms,
                "size_bytes": float(mel_base_size),
                "size_mb": float(mel_base_size) / BYTES_IN_MB,
                "compression_ratio": 1.0,
                "compile_ms": all_metrics["mel_encoder"]["compile_ms"][0],
            },
            "decoder": {
                "quality": 1.0,
                "latency_ms": dec_base_ms,
                "size_bytes": float(dec_base_size),
                "size_mb": float(dec_base_size) / BYTES_IN_MB,
                "compression_ratio": 1.0,
                "compile_ms": all_metrics["decoder"]["compile_ms"][0],
            },
            "joint": {
                "quality": 1.0,
                "latency_ms": joint_base_ms,
                "size_bytes": float(joint_base_size),
                "size_mb": float(joint_base_size) / BYTES_IN_MB,
                "compression_ratio": 1.0,
                "compile_ms": all_metrics["joint"]["compile_ms"][0],
            },
            "joint_decision": {
                "acc": 1.0,
                "latency_ms": jd_base_ms,
                "size_bytes": float(jd_base_size),
                "size_mb": float(jd_base_size) / BYTES_IN_MB,
                "compression_ratio": 1.0,
                "compile_ms": all_metrics["joint_decision"]["compile_ms"][0],
            },
        }
    }

    for var in variants:
        variants_names.append(var.name)
        out_dir = output_root / var.name
        out_dir_exists = out_dir.exists()
        out_dir.mkdir(parents=True, exist_ok=True)

        expected_components = []
        for comp_cfg in meta.get("components", {}).values():
            rel = comp_cfg.get("path")
            if rel:
                expected_components.append(out_dir / rel)

        missing = [p for p in expected_components if not p.exists()]
        if out_dir_exists and not missing:
            typer.echo(f"[{var.name}] Output already present at {out_dir}; skipping quantization step.")
        else:
            if out_dir_exists and missing:
                missing_names = ", ".join(sorted(p.name for p in missing)) or "unknown"
                typer.echo(f"[{var.name}] Output directory exists but is incomplete (missing: {missing_names}). Re-quantizing.")
                shutil.rmtree(out_dir, ignore_errors=True)
                out_dir.mkdir(parents=True, exist_ok=True)
            _quantize_dir(input_dir, out_dir, var, component_filter)

        # Load quantized models and pre-compute offline compile time
        pre_q_path = out_dir / "parakeet_preprocessor.mlpackage"
        enc_q_path = out_dir / "parakeet_encoder.mlpackage"
        mel_q_path = out_dir / "parakeet_mel_encoder.mlpackage"
        dec_q_path = out_dir / "parakeet_decoder.mlpackage"
        joint_q_path = out_dir / "parakeet_joint.mlpackage"
        jd_q_path = out_dir / "parakeet_joint_decision.mlpackage"

        pre_compile_ms = _offline_compile_time_ms(pre_q_path)
        enc_compile_ms = _offline_compile_time_ms(enc_q_path)
        mel_compile_q_ms = _offline_compile_time_ms(mel_q_path)
        dec_compile_ms = _offline_compile_time_ms(dec_q_path)
        joint_compile_ms = _offline_compile_time_ms(joint_q_path)
        jd_compile_q_ms = _offline_compile_time_ms(jd_q_path)

        pre_q = ct.models.MLModel(str(pre_q_path), compute_units=ct.ComputeUnit[compute_units])
        enc_q = ct.models.MLModel(str(enc_q_path), compute_units=ct.ComputeUnit[compute_units])
        mel_q = ct.models.MLModel(str(mel_q_path), compute_units=ct.ComputeUnit[compute_units])
        dec_q = ct.models.MLModel(str(dec_q_path), compute_units=ct.ComputeUnit[compute_units])
        joint_q = ct.models.MLModel(str(joint_q_path), compute_units=ct.ComputeUnit[compute_units])
        jd_q = ct.models.MLModel(str(jd_q_path), compute_units=ct.ComputeUnit[compute_units])

        # Preprocessor quality vs baseline
        pre_q_out = pre_q.predict(pre_base_inputs)
        mel_q_in = np.array(pre_q_out["mel"], dtype=np.float32, copy=True)
        a_pre, r_pre = _max_abs_rel(mel_ref, mel_q_in)
        l2_pre_ref = float(np.linalg.norm(mel_ref))
        l2_pre_err = float(np.linalg.norm(mel_ref - mel_q_in))
        pre_norm_err = (l2_pre_err / (l2_pre_ref + 1e-8)) if l2_pre_ref > 0 else 0.0
        pre_quality = float(max(0.0, 1.0 - pre_norm_err))
        pre_q_ms, _ = _predict_latency(pre_q, pre_base_inputs, runs=runs)
        pre_q_size = _dir_size_bytes(out_dir / "parakeet_preprocessor.mlpackage")
        all_metrics["preprocessor"]["quality"].append(pre_quality)
        all_metrics["preprocessor"]["latency_ms"].append(pre_q_ms)
        all_metrics["preprocessor"]["compression"].append(float(pre_base_size) / float(pre_q_size if pre_q_size > 0 else 1))
        all_metrics["preprocessor"].setdefault("compile_ms", []).append(pre_compile_ms)
        all_metrics["preprocessor"]["size_mb"].append(float(pre_q_size) / BYTES_IN_MB)

        # Encoder quality vs baseline (feed baseline mel to both)
        enc_q_out = enc_q.predict({"mel": mel_ref, "mel_length": mel_len})
        encoder_q = np.array(enc_q_out["encoder"], dtype=np.float32, copy=True)
        l2_enc_ref = float(np.linalg.norm(encoder_ref))
        l2_enc_err = float(np.linalg.norm(encoder_ref - encoder_q))
        enc_norm_err = (l2_enc_err / (l2_enc_ref + 1e-8)) if l2_enc_ref > 0 else 0.0
        enc_quality = float(max(0.0, 1.0 - enc_norm_err))
        enc_q_ms, _ = _predict_latency(enc_q, enc_base_inputs, runs=runs)
        enc_q_size = _dir_size_bytes(out_dir / "parakeet_encoder.mlpackage")
        all_metrics["encoder"]["quality"].append(enc_quality)
        all_metrics["encoder"]["latency_ms"].append(enc_q_ms)
        all_metrics["encoder"]["compression"].append(float(enc_base_size) / float(enc_q_size if enc_q_size > 0 else 1))
        all_metrics["encoder"].setdefault("compile_ms", []).append(enc_compile_ms)
        all_metrics["encoder"]["size_mb"].append(float(enc_q_size) / BYTES_IN_MB)

        # MelEncoder quality
        mel_q_out = mel_q.predict(mel_base_inputs)
        enc_q_fused = np.array(mel_q_out["encoder"], dtype=np.float32, copy=True)
        a_mel, r_mel = _max_abs_rel(encoder_ref, enc_q_fused)
        # Normalize error into a [0,1] quality score: 1 / (1 + normalized L2)
        # Use relative measure derived from L2 norms as more stable than max.
        l2_ref = float(np.linalg.norm(encoder_ref))
        l2_err = float(np.linalg.norm(encoder_ref - enc_q_fused))
        norm_err = (l2_err / (l2_ref + 1e-8)) if l2_ref > 0 else 0.0
        mel_quality = float(max(0.0, 1.0 - norm_err))
        mel_q_ms, _ = _predict_latency(mel_q, mel_base_inputs, runs=runs)
        mel_q_size = _dir_size_bytes(out_dir / "parakeet_mel_encoder.mlpackage")
        mel_ratio = float(mel_base_size) / float(mel_q_size if mel_q_size > 0 else 1)
        mel_size_mb.append(float(mel_q_size) / BYTES_IN_MB)
        all_metrics["mel_encoder"]["quality"].append(mel_quality)
        all_metrics["mel_encoder"]["latency_ms"].append(mel_q_ms)
        all_metrics["mel_encoder"]["compression"].append(float(mel_base_size) / float(mel_q_size if mel_q_size > 0 else 1))
        all_metrics["mel_encoder"].setdefault("compile_ms", []).append(mel_compile_q_ms)
        all_metrics["mel_encoder"]["size_mb"].append(float(mel_q_size) / BYTES_IN_MB)

        # JointDecision quality: token-id and duration match rates
        jd_base_out = joint_decision_base.predict(jd_base_inputs)
        token_id_base = np.array(jd_base_out["token_id"], dtype=np.int32, copy=True)
        duration_base = np.array(jd_base_out["duration"], dtype=np.int32, copy=True)
        token_prob_base = np.array(jd_base_out["token_prob"], dtype=np.float32, copy=True)

        jd_q_out = jd_q.predict(jd_base_inputs)
        token_id_q = np.array(jd_q_out["token_id"], dtype=np.int32, copy=True)
        duration_q = np.array(jd_q_out["duration"], dtype=np.int32, copy=True)
        token_prob_q = np.array(jd_q_out["token_prob"], dtype=np.float32, copy=True)

        # Accuracy metrics
        id_match = float((token_id_q == token_id_base).mean())
        dur_match = float((duration_q == duration_base).mean())
        # Aggregate a single "accuracy" number as token-id match rate (primary)
        jd_acc = id_match
        jd_q_ms, _ = _predict_latency(jd_q, jd_base_inputs, runs=runs)
        jd_q_size = _dir_size_bytes(out_dir / "parakeet_joint_decision.mlpackage")
        jd_ratio = float(jd_base_size) / float(jd_q_size if jd_q_size > 0 else 1)
        jd_size_mb.append(float(jd_q_size) / BYTES_IN_MB)
        all_metrics["joint_decision"].setdefault("acc", []).append(jd_acc)
        all_metrics["joint_decision"]["latency_ms"].append(jd_q_ms)
        all_metrics["joint_decision"]["compression"].append(float(jd_base_size) / float(jd_q_size if jd_q_size > 0 else 1))
        all_metrics["joint_decision"].setdefault("compile_ms", []).append(jd_compile_q_ms)
        all_metrics["joint_decision"]["size_mb"].append(float(jd_q_size) / BYTES_IN_MB)

        # Decoder quality vs baseline
        dec_q_out = dec_q.predict(dec_base_inputs)
        decoder_q = np.array(dec_q_out["decoder"], dtype=np.float32, copy=True)
        l2_dec_ref = float(np.linalg.norm(decoder_ref))
        l2_dec_err = float(np.linalg.norm(decoder_ref - decoder_q))
        dec_norm_err = (l2_dec_err / (l2_dec_ref + 1e-8)) if l2_dec_ref > 0 else 0.0
        dec_quality = float(max(0.0, 1.0 - dec_norm_err))
        dec_q_ms, _ = _predict_latency(dec_q, dec_base_inputs, runs=runs)
        dec_q_size = _dir_size_bytes(out_dir / "parakeet_decoder.mlpackage")
        all_metrics["decoder"]["quality"].append(dec_quality)
        all_metrics["decoder"]["latency_ms"].append(dec_q_ms)
        all_metrics["decoder"]["compression"].append(float(dec_base_size) / float(dec_q_size if dec_q_size > 0 else 1))
        all_metrics["decoder"].setdefault("compile_ms", []).append(dec_compile_ms)
        all_metrics["decoder"]["size_mb"].append(float(dec_q_size) / BYTES_IN_MB)

        # Joint quality vs baseline (compare logits)
        joint_q_out = joint_q.predict(joint_base_inputs)
        logits_q = np.array(joint_q_out["logits"], dtype=np.float32, copy=True)
        l2_joint_ref = float(np.linalg.norm(logits_base))
        l2_joint_err = float(np.linalg.norm(logits_base - logits_q))
        joint_norm_err = (l2_joint_err / (l2_joint_ref + 1e-8)) if l2_joint_ref > 0 else 0.0
        joint_quality = float(max(0.0, 1.0 - joint_norm_err))
        joint_q_ms, _ = _predict_latency(joint_q, joint_base_inputs, runs=runs)
        joint_q_size = _dir_size_bytes(out_dir / "parakeet_joint.mlpackage")
        all_metrics["joint"]["quality"].append(joint_quality)
        all_metrics["joint"]["latency_ms"].append(joint_q_ms)
        all_metrics["joint"]["compression"].append(float(joint_base_size) / float(joint_q_size if joint_q_size > 0 else 1))
        all_metrics["joint"].setdefault("compile_ms", []).append(joint_compile_ms)
        all_metrics["joint"]["size_mb"].append(float(joint_q_size) / BYTES_IN_MB)

        # Decoder deltas for JSON
        a_dec, r_dec = _max_abs_rel(decoder_ref, decoder_q)
        # Joint deltas for JSON
        a_joint, r_joint = _max_abs_rel(logits_base, logits_q)

        # Store metrics
        summary[var.name] = {
            "components": {
                "preprocessor": {
                    "quality": pre_quality,
                    "latency_ms": pre_q_ms,
                    "size_bytes": float(pre_q_size),
                    "size_mb": float(pre_q_size) / BYTES_IN_MB,
                    "compression_ratio": float(pre_base_size) / float(pre_q_size if pre_q_size > 0 else 1),
                    "max_abs": a_pre,
                    "max_rel": r_pre,
                    "compile_ms": pre_compile_ms,
                },
                "encoder": {
                    "quality": enc_quality,
                    "latency_ms": enc_q_ms,
                    "size_bytes": float(enc_q_size),
                    "size_mb": float(enc_q_size) / BYTES_IN_MB,
                    "compression_ratio": float(enc_base_size) / float(enc_q_size if enc_q_size > 0 else 1),
                    "compile_ms": enc_compile_ms,
                },
                "mel_encoder": {
                    "quality": mel_quality,
                    "latency_ms": mel_q_ms,
                    "size_bytes": float(mel_q_size),
                    "size_mb": float(mel_q_size) / BYTES_IN_MB,
                    "compression_ratio": mel_ratio,
                    "max_abs": a_mel,
                    "max_rel": r_mel,
                    "compile_ms": mel_compile_q_ms,
                },
                "decoder": {
                    "quality": dec_quality,
                    "latency_ms": dec_q_ms,
                    "size_bytes": float(dec_q_size),
                    "size_mb": float(dec_q_size) / BYTES_IN_MB,
                    "compression_ratio": float(dec_base_size) / float(dec_q_size if dec_q_size > 0 else 1),
                    "max_abs": a_dec,
                    "max_rel": r_dec,
                    "compile_ms": dec_compile_ms,
                },
                "joint": {
                    "quality": joint_quality,
                    "latency_ms": joint_q_ms,
                    "size_bytes": float(joint_q_size),
                    "size_mb": float(joint_q_size) / BYTES_IN_MB,
                    "compression_ratio": float(joint_base_size) / float(joint_q_size if joint_q_size > 0 else 1),
                    "max_abs": a_joint,
                    "max_rel": r_joint,
                    "compile_ms": joint_compile_ms,
                },
                "joint_decision": {
                    "acc": jd_acc,
                    "duration_match": dur_match,
                    "prob_mae": float(np.mean(np.abs(token_prob_q - token_prob_base))),
                    "latency_ms": jd_q_ms,
                    "size_bytes": float(jd_q_size),
                    "size_mb": float(jd_q_size) / BYTES_IN_MB,
                    "compression_ratio": jd_ratio,
                    "compile_ms": jd_compile_q_ms,
                },
            }
        }

        fused_labels.append(var.name)
        mel_quality_scores.append(mel_quality)
        mel_latency_ms.append(mel_q_ms)
        mel_compression.append(mel_ratio)
        jd_accuracy.append(jd_acc)
        jd_latency_ms.append(jd_q_ms)
        jd_compression.append(jd_ratio)
        mel_compile_ms.append(mel_compile_q_ms)
        jd_compile_ms.append(jd_compile_q_ms)

    # Write summary JSON
    out_root = output_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "quantization_summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    plot_dir = out_root / "plots"
    title_suffix = _chip_spec_string(compute_units)
    fused_paths = _plot_fused_category_charts(
        plot_dir,
        fused_labels,
        mel_quality_scores,
        mel_latency_ms,
        mel_compression,
        mel_size_mb,
        jd_accuracy,
        jd_latency_ms,
        jd_compression,
        jd_size_mb,
        mel_compile_ms,
        jd_compile_ms,
        title_suffix,
    )
    component_paths = _plot_all_component_category_charts(
        plot_dir,
        fused_labels,
        all_metrics,
        title_suffix,
    )

    typer.echo(f"Wrote summary JSON: {out_root / 'quantization_summary.json'}")
    if HAS_MPL:
        all_plot_paths = fused_paths + component_paths
        repo_plot_dir = BASE_DIR / "plots" / "quantize" / compute_units.lower()
        repo_plot_dir.mkdir(parents=True, exist_ok=True)
        for path in all_plot_paths:
            typer.echo(f"Wrote plot: {path}")
            if path.exists():
                dest = repo_plot_dir / path.name
                shutil.copy2(path, dest)
                typer.echo(f"Mirrored plot: {dest}")
    else:
        typer.echo("matplotlib unavailable; skipped plotting.")


if __name__ == "__main__":
    app()
