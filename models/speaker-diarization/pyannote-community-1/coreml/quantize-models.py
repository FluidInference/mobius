#!/usr/bin/env python3
"""Generate and benchmark quantized CoreML model variants."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import coremltools as ct
import coremltools.optimize.coreml as cto
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


SAMPLE_RATE = 16_000
SEGMENTATION_WINDOW = 160_000  # 10 s @ 16 kHz
EMBEDDING_WINDOW = 160_000  # 10 s @ 16 kHz (matches embedding Core ML input expectation)
DEFAULT_AUDIO = Path("../../../../longconvo-30m.wav")
DEFAULT_COREML_DIR = Path(__file__).resolve().parent / "coreml_models"
WARMUP_RUNS = 1

_COMPUTE_UNIT_ORDER = [
    "CPU_ONLY",
    "CPU_AND_GPU",
    "CPU_AND_NE",
    "ALL",
]


def _default_compute_units() -> list[ct.ComputeUnit]:
    units: list[ct.ComputeUnit] = []
    for name in _COMPUTE_UNIT_ORDER:
        if hasattr(ct.ComputeUnit, name):
            units.append(getattr(ct.ComputeUnit, name))
    if not units:
        raise RuntimeError("No Core ML compute units are available in coremltools")
    return units


DEFAULT_COMPUTE_UNITS = _default_compute_units()


def compute_unit_key(compute_unit: ct.ComputeUnit) -> str:
    return compute_unit.name


def compute_unit_display(compute_unit: ct.ComputeUnit) -> str:
    title_cased = compute_unit.name.replace("_", " ").title()
    return title_cased.replace("Cpu", "CPU").replace("Gpu", "GPU").replace("Ne", "NE")


@dataclass
class VariantSpec:
    """Specification for a model variant."""

    name: str
    suffix: str
    description: str
    transform: str = "quantize"


# Define all optimization variants
MODEL_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="int8-per-channel",
        suffix="-int8-per-channel",
        description="INT8 per-channel quantization",
    ),
    VariantSpec(
        name="int8-per-block",
        suffix="-int8-per-block",
        description="INT8 per-block quantization (block_size=128)",
    ),
    VariantSpec(
        name="int4-per-block",
        suffix="-int4-per-block",
        description="INT4 per-block quantization (block_size=128)",
    ),
    VariantSpec(
        name="palettized-6bit",
        suffix="-palettized-6bit",
        description="6-bit weight palettization (k-means, per-channel)",
        transform="palettize",
    ),
)
ALL_VARIANTS = MODEL_VARIANTS


@dataclass
class ComparisonResult:
    mse: float
    mae: float
    max_abs: float
    corr: float
    cosine: float | None = None


def load_audio(audio_path: Path) -> torch.Tensor:
    """Load mono audio at SAMPLE_RATE."""
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    return waveform.float()


def chunk_waveform(waveform: torch.Tensor, window: int) -> Iterable[torch.Tensor]:
    """Yield contiguous windows padded with zeros to the requested length."""
    total_samples = waveform.shape[-1]
    num_chunks = max(1, math.ceil(total_samples / window))
    for index in range(num_chunks):
        start = index * window
        end = start + window
        chunk = waveform[:, start:end]
        if chunk.shape[-1] < window:
            pad = torch.zeros(chunk.shape[0], window - chunk.shape[-1], dtype=chunk.dtype)
            chunk = torch.cat([chunk, pad], dim=-1)
        yield chunk.unsqueeze(0)


def compute_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    diff = cand - ref
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    ref_std = float(np.std(ref))
    cand_std = float(np.std(cand))
    if ref_std == 0.0 or cand_std == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(ref, cand)[0, 1])
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "corr": corr}


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    denom = np.linalg.norm(ref) * np.linalg.norm(cand)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(ref, cand) / denom)


def get_model_size_mb(model_path: Path) -> float:
    """Get total size of .mlpackage in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def create_quantization_variant(
    base_model: ct.models.MLModel, variant_spec: VariantSpec
) -> ct.models.MLModel:
    """Create a quantized variant of the model."""
    print(f"  Creating {variant_spec.name}...")

    if "int8-per-channel" in variant_spec.name:
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype=np.int8, granularity="per_channel", weight_threshold=512
            )
        )
    elif "int8-per-block" in variant_spec.name:
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
                granularity="per_block",
                block_size=128,
                weight_threshold=512,
            )
        )
    elif "int4-per-block" in variant_spec.name:
        config = cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int4",
                granularity="per_block",
                block_size=128,
                weight_threshold=512,
            )
        )
    else:
        raise ValueError(f"Unknown quantization variant: {variant_spec.name}")

    return cto.linear_quantize_weights(base_model, config=config)


def create_palettized_variant(
    base_model: ct.models.MLModel, variant_spec: VariantSpec
) -> ct.models.MLModel:
    """Create a palettized variant of the model."""
    print(f"  Creating {variant_spec.name}...")

    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            mode="kmeans",
            nbits=6,
            granularity="per_tensor",
            group_size=0,
            weight_threshold=512,
        )
    )

    return cto.palettize_weights(base_model, config=config)

def generate_variant(
    base_model_path: Path, variant_spec: VariantSpec, output_dir: Path
) -> Path:
    """Generate a single optimization variant."""
    # Load base model
    base_model = ct.models.MLModel(str(base_model_path))

    # Create variant
    if variant_spec.transform == "quantize":
        variant_model = create_quantization_variant(base_model, variant_spec)
    elif variant_spec.transform == "palettize":
        variant_model = create_palettized_variant(base_model, variant_spec)
    else:
        raise ValueError(f"Unsupported transform '{variant_spec.transform}' for variant {variant_spec.name}")

    # Save variant
    base_name = base_model_path.stem
    output_path = output_dir / f"{base_name}{variant_spec.suffix}.mlpackage"
    variant_model.save(str(output_path))
    print(f"  Saved to {output_path}")

    return output_path


def benchmark_model_variant(
    model_path: Path,
    baseline_outputs: list[np.ndarray],
    input_batches: list[dict[str, np.ndarray]],
    compute_units: Iterable[ct.ComputeUnit],
    is_embedding: bool = False,
) -> dict[str, dict]:
    """Benchmark a model variant across all compute units."""
    results: dict[str, dict] = {}

    for compute_unit in compute_units:
        unit_key = compute_unit_key(compute_unit)

        try:
            # Measure compilation time
            compile_start = time.perf_counter()
            model = ct.models.MLModel(str(model_path), compute_units=compute_unit)
            compile_time = time.perf_counter() - compile_start

            # Warmup
            warmup_elapsed = 0.0
            if WARMUP_RUNS > 0 and input_batches:
                warmup_input = input_batches[0]
                for _ in range(WARMUP_RUNS):
                    warm_start = time.perf_counter()
                    model.predict(warmup_input)
                    warmup_elapsed += time.perf_counter() - warm_start

            # Benchmark inference
            variant_outputs: list[np.ndarray] = []
            inference_times: list[float] = []

            for batch_inputs in input_batches:
                infer_start = time.perf_counter()
                output = model.predict(batch_inputs)
                infer_elapsed = time.perf_counter() - infer_start

                output_arr = next(iter(output.values()))
                variant_outputs.append(output_arr)
                inference_times.append(infer_elapsed)

            # Compute accuracy metrics
            stacked_baseline = np.concatenate(baseline_outputs, axis=0)
            stacked_variant = np.concatenate(variant_outputs, axis=0)

            aggregate_metrics = compute_error_metrics(stacked_baseline, stacked_variant)
            aggregate_cos = cosine_similarity(stacked_baseline, stacked_variant) if is_embedding else None

            aggregate = ComparisonResult(**aggregate_metrics, cosine=aggregate_cos)

            # Timing statistics
            total_inference = float(np.sum(inference_times))
            avg_inference = float(np.mean(inference_times))
            median_inference = float(np.median(inference_times))
            p95_inference = float(np.percentile(inference_times, 95))
            p99_inference = float(np.percentile(inference_times, 99))

            results[unit_key] = {
                "compute_unit": compute_unit_display(compute_unit),
                "compile_time": compile_time,
                "warmup_time": warmup_elapsed,
                "inference_times": inference_times,
                "total_inference": total_inference,
                "avg_inference": avg_inference,
                "median_inference": median_inference,
                "p95_inference": p95_inference,
                "p99_inference": p99_inference,
                "aggregate": aggregate,
                "output_shape": stacked_variant.shape,
            }

        except Exception as exc:
            results[unit_key] = {
                "error": str(exc),
                "compute_unit": compute_unit_display(compute_unit),
            }

    return results


def build_plot_signature(base_dir: Path) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    git_hash = "unknown"
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=base_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass
    return f"Generated {timestamp} | git {git_hash}"


def annotate_figure(fig: plt.Figure, signature: str) -> None:
    if not signature:
        return
    fig.text(
        0.99,
        0.02,
        signature,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )


def plot_compilation_time(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
    output_path: Path,
    signature: str,
) -> None:
    """Plot compilation time comparison across variants and compute units."""
    # Collect data
    entries = []

    for variant_name, variant_data in seg_results.items():
        for unit_key, unit_data in variant_data.items():
            if "error" in unit_data:
                continue
            entries.append(
                {
                    "model": "Segmentation",
                    "variant": variant_name,
                    "compute_unit": unit_data.get("compute_unit", unit_key),
                    "compile_time": unit_data.get("compile_time", 0.0),
                }
            )

    for variant_name, variant_data in emb_results.items():
        for unit_key, unit_data in variant_data.items():
            if "error" in unit_data:
                continue
            entries.append(
                {
                    "model": "Embedding",
                    "variant": variant_name,
                    "compute_unit": unit_data.get("compute_unit", unit_key),
                    "compile_time": unit_data.get("compile_time", 0.0),
                }
            )

    if not entries:
        return

    # Create grouped bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    def plot_model_compilation(ax, model_name, model_entries):
        # Group by variant
        variant_names = sorted(set(e["variant"] for e in model_entries))
        compute_units = sorted(set(e["compute_unit"] for e in model_entries))

        x = np.arange(len(variant_names))
        width = 0.8 / len(compute_units)

        for i, cu in enumerate(compute_units):
            times = []
            for variant in variant_names:
                matching = [e for e in model_entries if e["variant"] == variant and e["compute_unit"] == cu]
                times.append(matching[0]["compile_time"] if matching else 0.0)

            ax.bar(x + i * width, times, width, label=cu)

        ax.set_xlabel("Variant")
        ax.set_ylabel("Compilation Time (seconds)")
        ax.set_title(f"{model_name} - Compilation Time by Variant")
        ax.set_xticks(x + width * (len(compute_units) - 1) / 2)
        ax.set_xticklabels(variant_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    seg_entries = [e for e in entries if e["model"] == "Segmentation"]
    emb_entries = [e for e in entries if e["model"] == "Embedding"]

    if seg_entries:
        plot_model_compilation(ax1, "Segmentation", seg_entries)
    else:
        ax1.axis("off")

    if emb_entries:
        plot_model_compilation(ax2, "Embedding", emb_entries)
    else:
        ax2.axis("off")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    annotate_figure(fig, signature)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_latency_timeseries(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
    window_seconds: dict[str, float],
    output_path: Path,
    signature: str,
) -> None:
    """Plot inference latency over time for all variants."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    def plot_model_latency(ax, model_name, model_results, window_sec):
        # Use ALL compute unit for clearest comparison
        unit_key = "ALL"

        colors = plt.cm.tab20.colors
        line_styles = ["-", "--", "-.", ":"]

        color_idx = 0
        for variant_name, variant_data in sorted(model_results.items()):
            if unit_key not in variant_data or "error" in variant_data[unit_key]:
                continue

            times = variant_data[unit_key].get("inference_times", [])
            if not times:
                continue

            times_ms = np.array(times) * 1000.0
            x_axis = np.arange(len(times)) * window_sec

            style = line_styles[color_idx % len(line_styles)]
            color = colors[color_idx % len(colors)]

            ax.plot(x_axis, times_ms, linestyle=style, color=color, linewidth=1.5, label=variant_name)
            color_idx += 1

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{model_name} - Inference Latency Over Time (ALL compute units)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plot_model_latency(ax1, "Segmentation", seg_results, window_seconds.get("segmentation", 10.0))
    plot_model_latency(ax2, "Embedding", emb_results, window_seconds.get("embedding", 5.0))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    annotate_figure(fig, signature)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_accuracy_grid(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
    output_path: Path,
    signature: str,
) -> None:
    """Plot accuracy metrics in a heatmap grid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    def plot_model_accuracy(ax, model_name, model_results, include_cosine=False):
        # Use ALL compute unit
        unit_key = "ALL"

        variant_names = []
        metrics_matrix = []

        for variant_name, variant_data in sorted(model_results.items()):
            if unit_key not in variant_data or "error" in variant_data[unit_key]:
                continue

            aggregate = variant_data[unit_key].get("aggregate")
            if not aggregate:
                continue

            variant_names.append(variant_name)
            row = [aggregate.mse, aggregate.mae, aggregate.max_abs, aggregate.corr]
            if include_cosine and aggregate.cosine is not None:
                row.append(aggregate.cosine)
            metrics_matrix.append(row)

        if not metrics_matrix:
            ax.axis("off")
            return

        metrics_matrix = np.array(metrics_matrix)
        metric_names = ["MSE", "MAE", "Max Abs", "Corr"]
        if include_cosine:
            metric_names.append("Cosine")

        # Create heatmap
        im = ax.imshow(metrics_matrix.T, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")

        # Set ticks
        ax.set_xticks(np.arange(len(variant_names)))
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_xticklabels(variant_names, rotation=45, ha="right")
        ax.set_yticklabels(metric_names)

        # Add values to cells
        for i in range(len(metric_names)):
            for j in range(len(variant_names)):
                value = metrics_matrix[j, i]
                if metric_names[i] in ["Corr", "Cosine"]:
                    text = ax.text(j, i, f"{value:.4f}", ha="center", va="center", color="black", fontsize=8)
                else:
                    text = ax.text(j, i, f"{value:.2e}", ha="center", va="center", color="black", fontsize=8)

        ax.set_title(f"{model_name} - Accuracy Metrics (ALL compute units)")
        fig.colorbar(im, ax=ax)

    plot_model_accuracy(ax1, "Segmentation", seg_results, include_cosine=False)
    plot_model_accuracy(ax2, "Embedding", emb_results, include_cosine=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    annotate_figure(fig, signature)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_summary(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
    baseline_sizes: dict[str, float],
    variant_sizes: dict[str, dict[str, float]],
    baseline_times: dict[str, float],
    output_path: Path,
    signature: str,
) -> None:
    """Plot summary with speedup, size reduction, RTF, and accuracy vs size tradeoff."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax_speedup = fig.add_subplot(gs[0, 0])
    ax_size = fig.add_subplot(gs[0, 1])
    ax_rtf = fig.add_subplot(gs[1, 0])
    ax_tradeoff = fig.add_subplot(gs[1, 1])

    # Use ALL compute unit
    unit_key = "ALL"

    # Collect data for all models
    speedup_data = []
    size_data = []
    rtf_data = []
    tradeoff_data = []

    for model_name, model_results, baseline_size, baseline_time in [
        ("Seg", seg_results, baseline_sizes.get("segmentation", 0), baseline_times.get("segmentation", 0)),
        ("Emb", emb_results, baseline_sizes.get("embedding", 0), baseline_times.get("embedding", 0)),
    ]:
        for variant_name, variant_data in sorted(model_results.items()):
            if unit_key not in variant_data or "error" in variant_data[unit_key]:
                continue

            unit_data = variant_data[unit_key]
            variant_time = unit_data.get("avg_inference", 0.0)
            speedup = baseline_time / variant_time if variant_time > 0 else 0

            variant_size = variant_sizes.get(model_name.lower() + "mentation", {}).get(variant_name, 0)
            size_reduction = (1 - variant_size / baseline_size) * 100 if baseline_size > 0 else 0

            aggregate = unit_data.get("aggregate")
            accuracy = 1.0 - aggregate.mae if aggregate else 0

            speedup_data.append({"label": f"{model_name}-{variant_name}", "speedup": speedup})
            size_data.append({"label": f"{model_name}-{variant_name}", "reduction": size_reduction})
            tradeoff_data.append({"label": f"{model_name}-{variant_name}", "size_mb": variant_size, "accuracy": accuracy})

    # Plot 1: Speedup
    if speedup_data:
        labels = [d["label"] for d in speedup_data]
        speedups = [d["speedup"] for d in speedup_data]
        ax_speedup.barh(labels, speedups, color="steelblue")
        ax_speedup.axvline(x=1.0, color="red", linestyle="--", linewidth=1, label="Baseline")
        ax_speedup.set_xlabel("Speedup (x)")
        ax_speedup.set_title("Inference Speedup vs FP16 Baseline")
        ax_speedup.legend()
        ax_speedup.grid(True, alpha=0.3, axis="x")

    # Plot 2: Size Reduction
    if size_data:
        labels = [d["label"] for d in size_data]
        reductions = [d["reduction"] for d in size_data]
        colors = ["green" if r > 0 else "red" for r in reductions]
        ax_size.barh(labels, reductions, color=colors)
        ax_size.set_xlabel("Size Reduction (%)")
        ax_size.set_title("Model Size Reduction vs FP16 Baseline")
        ax_size.grid(True, alpha=0.3, axis="x")

    # Plot 3: RTF (placeholder - would need audio duration)
    ax_rtf.text(0.5, 0.5, "RTF calculation requires audio duration metadata", ha="center", va="center")
    ax_rtf.axis("off")

    # Plot 4: Accuracy vs Size Tradeoff
    if tradeoff_data:
        sizes = [d["size_mb"] for d in tradeoff_data]
        accuracies = [d["accuracy"] for d in tradeoff_data]
        labels = [d["label"] for d in tradeoff_data]

        ax_tradeoff.scatter(sizes, accuracies, s=100, alpha=0.6)
        for i, label in enumerate(labels):
            ax_tradeoff.annotate(label, (sizes[i], accuracies[i]), fontsize=7, ha="right")

        ax_tradeoff.set_xlabel("Model Size (MB)")
        ax_tradeoff.set_ylabel("Accuracy (1 - MAE)")
        ax_tradeoff.set_title("Accuracy vs Size Tradeoff")
        ax_tradeoff.grid(True, alpha=0.3)

    fig.suptitle("Optimization Summary", fontsize=16)
    annotate_figure(fig, signature)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and benchmark quantized CoreML models")
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=DEFAULT_AUDIO,
        help="Path to evaluation audio",
    )
    parser.add_argument(
        "--coreml-dir",
        type=Path,
        default=DEFAULT_COREML_DIR,
        help="Directory containing Core ML packages",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip variant generation, only benchmark existing models",
    )
    return parser.parse_args()


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()

    if not args.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if not args.coreml_dir.exists():
        raise FileNotFoundError(f"Core ML directory not found: {args.coreml_dir}")

    # Load audio
    print(f"Loading audio: {args.audio_path}")
    waveform = load_audio(args.audio_path)
    print(f"Audio shape: {tuple(waveform.shape)}, sample_rate={SAMPLE_RATE}")

    # Base model paths
    seg_base_path = args.coreml_dir / "segmentation-community-1.mlpackage"
    emb_base_path = args.coreml_dir / "embedding-community-1.mlpackage"
    fbank_base_path = args.coreml_dir / "fbank-community-1.mlpackage"

    if not seg_base_path.exists() or not emb_base_path.exists() or not fbank_base_path.exists():
        raise FileNotFoundError("Base Core ML models not found. Run convert-coreml.py first.")

    emb_spec = ct.utils.load_spec(str(emb_base_path))
    emb_weight_length: int | None = None
    fbank_shape: tuple[int, ...] | None = None
    for input_desc in emb_spec.description.input:  # type: ignore[attr-defined]
        name = getattr(input_desc, "name", "")
        array_type = getattr(input_desc.type, "multiArrayType", None)
        if array_type is None:
            continue
        shape = tuple(int(dim) for dim in getattr(array_type, "shape", []))
        if not shape:
            continue
        if name == "weights":
            emb_weight_length = int(shape[-1])
        elif name == "fbank_features":
            fbank_shape = shape
    if emb_weight_length is None or emb_weight_length <= 0:
        raise RuntimeError("Embedding Core ML model is missing a valid 'weights' input shape")
    if fbank_shape is None:
        raise RuntimeError("Embedding Core ML model is missing a valid 'fbank_features' input shape")

    # Generate variants
    if not args.skip_generation:
        print("\n" + "=" * 80)
        print("GENERATING SEGMENTATION VARIANTS")
        print("=" * 80)
        for variant_spec in ALL_VARIANTS:
            try:
                generate_variant(seg_base_path, variant_spec, args.coreml_dir)
            except Exception as exc:
                print(f"  ERROR generating {variant_spec.name}: {exc}")

        print("\n" + "=" * 80)
        print("GENERATING EMBEDDING VARIANTS")
        print("=" * 80)
        for variant_spec in ALL_VARIANTS:
            try:
                generate_variant(emb_base_path, variant_spec, args.coreml_dir)
            except Exception as exc:
                print(f"  ERROR generating {variant_spec.name}: {exc}")

    # Prepare baseline outputs
    print("\n" + "=" * 80)
    print("COMPUTING BASELINE OUTPUTS")
    print("=" * 80)

    from pyannote.audio import Model

    print("Loading segmentation baseline...")
    seg_torch = Model.from_pretrained(str(seg_base_path.parent.parent / "pyannote-speaker-diarization-community-1" / "segmentation")).eval()
    seg_baseline_outputs = []
    seg_input_batches = []

    with torch.inference_mode():
        for chunk in chunk_waveform(waveform, SEGMENTATION_WINDOW):
            array = np.array(chunk.cpu().numpy(), dtype=np.float32, copy=True)
            seg_input_batches.append({"audio": array})
            seg_baseline_outputs.append(seg_torch(chunk).cpu().numpy())

    print("Loading embedding baseline...")
    emb_torch = Model.from_pretrained(str(emb_base_path.parent.parent / "pyannote-speaker-diarization-community-1" / "embedding")).eval()
    emb_baseline_outputs = []
    emb_input_batches = []

    try:
        fbank_ml = ct.models.MLModel(str(fbank_base_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Core ML FBANK model: {exc}") from exc

    with torch.inference_mode():
        for chunk in chunk_waveform(waveform, EMBEDDING_WINDOW):
            array = np.array(chunk.cpu().numpy(), dtype=np.float32, copy=True)
            emb_baseline_outputs.append(emb_torch(chunk).cpu().numpy())
            features = fbank_ml.predict({"audio": array})
            feature_array = np.array(features["fbank_features"], dtype=np.float32, copy=True)
            if feature_array.shape[1:] != tuple(fbank_shape[1:]):
                raise RuntimeError(
                    f"FBANK feature shape mismatch: expected (*, {fbank_shape[1:]}), got {feature_array.shape}"
                )
            weights = np.ones((1, emb_weight_length), dtype=np.float32)
            emb_input_batches.append({
                "fbank_features": feature_array,
                "weights": weights,
            })

    # Benchmark baseline
    print("\nBenchmarking FP16 baseline...")
    seg_baseline_results = benchmark_model_variant(
        seg_base_path,
        seg_baseline_outputs,
        seg_input_batches,
        DEFAULT_COMPUTE_UNITS,
        is_embedding=False,
    )
    emb_baseline_results = benchmark_model_variant(
        emb_base_path,
        emb_baseline_outputs,
        emb_input_batches,
        DEFAULT_COMPUTE_UNITS,
        is_embedding=True,
    )

    # Collect baseline times for speedup calculation
    baseline_times = {
        "segmentation": seg_baseline_results.get("ALL", {}).get("avg_inference", 0.0),
        "embedding": emb_baseline_results.get("ALL", {}).get("avg_inference", 0.0),
    }

    # Collect baseline sizes
    baseline_sizes = {
        "segmentation": get_model_size_mb(seg_base_path),
        "embedding": get_model_size_mb(emb_base_path),
    }

    # Benchmark all variants
    print("\n" + "=" * 80)
    print("BENCHMARKING VARIANTS")
    print("=" * 80)

    seg_results = {"baseline-fp16": seg_baseline_results}
    emb_results = {"baseline-fp16": emb_baseline_results}

    variant_sizes = {"segmentation": {}, "embedding": {}}

    for variant_spec in ALL_VARIANTS:
        print(f"\nBenchmarking {variant_spec.name}...")

        # Segmentation
        seg_variant_path = args.coreml_dir / f"segmentation-community-1{variant_spec.suffix}.mlpackage"
        if seg_variant_path.exists():
            seg_results[variant_spec.name] = benchmark_model_variant(
                seg_variant_path,
                seg_baseline_outputs,
                seg_input_batches,
                DEFAULT_COMPUTE_UNITS,
                is_embedding=False,
            )
            variant_sizes["segmentation"][variant_spec.name] = get_model_size_mb(seg_variant_path)

        # Embedding
        emb_variant_path = args.coreml_dir / f"embedding-community-1{variant_spec.suffix}.mlpackage"
        if emb_variant_path.exists():
            emb_results[variant_spec.name] = benchmark_model_variant(
                emb_variant_path,
                emb_baseline_outputs,
                emb_input_batches,
                DEFAULT_COMPUTE_UNITS,
                is_embedding=True,
            )
            variant_sizes["embedding"][variant_spec.name] = get_model_size_mb(emb_variant_path)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plots_dir = args.coreml_dir.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    signature = build_plot_signature(args.coreml_dir.parent)

    print("Generating compilation time plot...")
    plot_compilation_time(seg_results, emb_results, plots_dir / "quantization_compilation_time.png", signature)

    print("Generating latency timeseries plot...")
    plot_latency_timeseries(
        seg_results,
        emb_results,
        {"segmentation": SEGMENTATION_WINDOW / SAMPLE_RATE, "embedding": EMBEDDING_WINDOW / SAMPLE_RATE},
        plots_dir / "quantization_latency_timeseries.png",
        signature,
    )

    print("Generating accuracy grid plot...")
    plot_accuracy_grid(seg_results, emb_results, plots_dir / "quantization_accuracy_grid.png", signature)

    print("Generating summary plot...")
    plot_summary(
        seg_results,
        emb_results,
        baseline_sizes,
        variant_sizes,
        baseline_times,
        plots_dir / "quantization_summary.png",
        signature,
    )

    # Save comprehensive results
    print("\nSaving results to JSON...")
    results = {
        "metadata": {
            "audio_path": str(args.audio_path),
            "audio_duration_seconds": float(waveform.shape[-1]) / SAMPLE_RATE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_sizes_mb": baseline_sizes,
        },
        "segmentation": seg_results,
        "embedding": emb_results,
        "variant_sizes_mb": variant_sizes,
    }

    results_path = args.coreml_dir / "quantization-results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print(f"Plots saved to {plots_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
