#!/usr/bin/env python3
"""Compare PyTorch vs. Core ML outputs for pyannote community-1 segmentation and embedding."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import coremltools as ct
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import torch
import torchaudio
from scipy.linalg import eigh
from pyannote.audio import Model, Pipeline
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from coreml_wrappers import DEFAULT_EMBEDDING_WINDOW, wrap_pipeline_with_coreml

try:  # Optional, used for machine spec reporting.
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None

SAMPLE_RATE = 16_000
SEGMENTATION_WINDOW = 160_000  # 10 s @ 16 kHz
EMBEDDING_WINDOW = DEFAULT_EMBEDDING_WINDOW
DEFAULT_AUDIO = Path("../../../../longconvo-30m-last5m.wav")
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parent / "pyannote-speaker-diarization-community-1"
DEFAULT_COREML_DIR = Path(__file__).resolve().parent / "coreml_models"
WARMUP_RUNS = 2  # Increased from 1 to ensure stable measurements and complete JIT compilation

_COMPUTE_UNIT_ORDER = [
    "CPU_ONLY",
    "CPU_AND_GPU",
    "CPU_AND_NE",
    "ALL",
]

_MODEL_ORDER = ["segmentation", "embedding"]

_PIPELINE_DEVICE_ORDER = ["TORCH_MPS", *_COMPUTE_UNIT_ORDER]


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
    return (
        title_cased.replace("Cpu", "CPU")
        .replace("Gpu", "GPU")
        .replace("Ne", "NE")
    )


@dataclass
class PLDAPipeline:
    xvec_tf: callable
    plda_tf: callable
    phi: np.ndarray
    lda_dim: int

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        transformed = self.xvec_tf(np.asarray(embeddings))
        return self.plda_tf(transformed, lda_dim=self.lda_dim)

    def rho(self, embeddings: np.ndarray) -> np.ndarray:
        fea = self.transform(embeddings)
        return fea * np.sqrt(self.phi[: self.lda_dim])


def _l2_norm(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        denom = np.linalg.norm(array)
        if denom == 0.0:
            return array
        return array / denom
    denom = np.linalg.norm(array, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return array / denom


def _build_plda_pipeline(mean1, mean2, lda, mu, tr, psi) -> PLDAPipeline:
    mean1 = np.asarray(mean1, dtype=np.float64)
    mean2 = np.asarray(mean2, dtype=np.float64)
    lda = np.asarray(lda, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    tr = np.asarray(tr, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)

    lda_dim = lda.shape[1]

    # Within/between class matrices
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w_matrix = np.linalg.inv(tr.T @ tr)
        b_matrix = np.linalg.inv((tr.T / psi).dot(tr))
        eigenvalues, eigenvectors = eigh(b_matrix, w_matrix)
    plda_psi = eigenvalues[::-1]
    plda_tr = eigenvectors.T[::-1]

    def xvec_tf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        centered = _l2_norm(x - mean1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = (np.sqrt(lda.shape[0]) * centered) @ lda
        shifted = projected - mean2
        normalized = _l2_norm(shifted)
        return np.sqrt(lda_dim) * normalized

    def plda_tf(x0: np.ndarray, lda_dim: int = lda.shape[1]) -> np.ndarray:
        x0 = np.asarray(x0, dtype=np.float64)
        return ((x0 - mu) @ plda_tr.T)[:, :lda_dim]

    return PLDAPipeline(xvec_tf=xvec_tf, plda_tf=plda_tf, phi=plda_psi, lda_dim=lda_dim)



def _load_json_tensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    tensors = {}
    for name, meta in payload.get("tensors", {}).items():
        encoded = meta.get("data_base64", "")
        raw = base64.b64decode(encoded)
        dtype_str = meta.get("dtype", "float32")
        dtype = np.dtype(dtype_str)
        array = np.frombuffer(raw, dtype=dtype).reshape(meta.get("shape", []))
        tensors[name] = array
    return tensors


def load_plda_pipeline_from_npz(model_root: Path) -> PLDAPipeline:
    transform_npz = np.load(model_root / "plda" / "xvec_transform.npz")
    plda_npz = np.load(model_root / "plda" / "plda.npz")
    return _build_plda_pipeline(
        transform_npz["mean1"],
        transform_npz["mean2"],
        transform_npz["lda"],
        plda_npz["mu"],
        plda_npz["tr"],
        plda_npz["psi"],
    )


def load_plda_pipeline_from_json(coreml_dir: Path) -> PLDAPipeline:
    transform_json = _load_json_tensors(coreml_dir / "resources" / "xvector-transform.json")
    plda_json = _load_json_tensors(coreml_dir / "resources" / "plda-parameters.json")
    return _build_plda_pipeline(
        transform_json["mean1"],
        transform_json["mean2"],
        transform_json["lda"],
        plda_json["mu"],
        plda_json["tr"],
        plda_json["psi"],
    )


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


def normalize_speaker_labels(annotation: Annotation) -> Annotation:
    """Normalize speaker labels by first appearance time.

    Ensures consistent speaker labeling across different runs by assigning
    SPEAKER_00 to the first speaker who appears, SPEAKER_01 to the second, etc.
    This makes visual comparison of diarization plots meaningful even when
    clustering algorithms assign arbitrary cluster IDs.
    """
    # Collect all segments sorted by start time
    segments_by_time: list[tuple[Segment, str]] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        segments_by_time.append((segment, label))

    if not segments_by_time:
        return annotation

    # Sort by segment start time to determine first appearance order
    segments_by_time.sort(key=lambda x: x[0].start)

    # Build mapping from original labels to normalized labels
    seen_labels: list[str] = []
    label_mapping: dict[str, str] = {}

    for segment, original_label in segments_by_time:
        if original_label not in label_mapping:
            new_index = len(seen_labels)
            label_mapping[original_label] = f"SPEAKER_{new_index:02d}"
            seen_labels.append(original_label)

    # Apply the mapping
    return annotation.rename_labels(mapping=label_mapping)


def batch_chunks(chunks: list[np.ndarray], batch_size: int = 32) -> list[tuple[np.ndarray, int]]:
    """
    Batch chunks into groups of batch_size, padding the last batch if needed.

    Returns list of (batched_array, actual_count) tuples where:
    - batched_array: shape (batch_size, *inner_shape)
    - actual_count: number of real chunks in this batch (rest is padding)
    """
    batches = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks_list = chunks[i:i+batch_size]
        actual_count = len(batch_chunks_list)

        # Squeeze first dimension if present (chunks come as (1, ...))
        squeezed_chunks = [chunk.squeeze(0) if chunk.shape[0] == 1 else chunk for chunk in batch_chunks_list]

        # Pad to batch_size if needed
        if actual_count < batch_size:
            padding_count = batch_size - actual_count
            padding_shape = (padding_count, *squeezed_chunks[0].shape)
            padding = np.zeros(padding_shape, dtype=squeezed_chunks[0].dtype)
            batch_array = np.concatenate([np.stack(squeezed_chunks), padding], axis=0)
        else:
            batch_array = np.stack(squeezed_chunks)

        batches.append((batch_array, actual_count))

    return batches


def compute_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    diff = cand - ref
    mse = float(np.mean(diff ** 2))
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


def make_comparison_result(reference: np.ndarray, candidate: np.ndarray) -> ComparisonResult:
    metrics = compute_error_metrics(reference, candidate)
    cos = cosine_similarity(reference, candidate)
    return ComparisonResult(**metrics, cosine=cos)


@dataclass
class ComparisonResult:
    mse: float
    mae: float
    max_abs: float
    corr: float
    cosine: float | None = None
    baseline_time: float | None = None
    candidate_time: float | None = None


def compare_segmentation(
    model_root: Path,
    coreml_dir: Path,
    audio: torch.Tensor,
    compute_units: Iterable[ct.ComputeUnit],
) -> dict[str, dict]:
    segmentation_torch = Model.from_pretrained(str(model_root / "segmentation")).eval()

    chunk_arrays: list[np.ndarray] = []
    torch_outputs: list[np.ndarray] = []
    torch_times: list[float] = []
    audio_seconds = float(audio.shape[-1]) / SAMPLE_RATE

    with torch.inference_mode():
        for chunk in chunk_waveform(audio, SEGMENTATION_WINDOW):
            array = np.array(chunk.cpu().numpy(), dtype=np.float32, copy=True)
            chunk_arrays.append(array)

            torch_start = time.perf_counter()
            torch_out = segmentation_torch(chunk).cpu().numpy()
            torch_elapsed = time.perf_counter() - torch_start

            torch_outputs.append(torch_out)
            torch_times.append(torch_elapsed)

    if not torch_outputs:
        return {}

    stacked_torch = np.concatenate(torch_outputs, axis=0)
    total_torch = float(np.sum(torch_times)) if torch_times else float("nan")
    avg_torch = float(np.mean(torch_times)) if torch_times else float("nan")
    processed_seconds = len(torch_times) * (SEGMENTATION_WINDOW / SAMPLE_RATE)

    results: dict[str, dict] = {}

    for compute_unit in [ct.ComputeUnit.ALL]:
        unit_key = compute_unit_key(compute_unit)
        compile_start = time.perf_counter()
        try:
            segmentation_ml = ct.models.MLModel(
                str(coreml_dir / "segmentation-community-1.mlpackage"),
                compute_units=compute_unit,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            results[unit_key] = {
                "error": str(exc),
                "compute_unit": compute_unit_display(compute_unit),
            }
            continue
        compile_time = time.perf_counter() - compile_start

        warmup_elapsed = 0.0
        if WARMUP_RUNS > 0 and chunk_arrays:
            warmup_input = {"audio": chunk_arrays[0]}
            for _ in range(WARMUP_RUNS):
                warm_start = time.perf_counter()
                segmentation_ml.predict(warmup_input)
                warmup_elapsed += time.perf_counter() - warm_start

        coreml_outputs: list[np.ndarray] = []
        per_chunk: list[ComparisonResult] = []
        coreml_times: list[float] = []

        # Process one chunk at a time (segmentation model uses batch size 1)
        for idx, chunk_array in enumerate(chunk_arrays):
            torch_out = torch_outputs[idx]
            torch_elapsed = torch_times[idx]

            core_start = time.perf_counter()
            core_out = segmentation_ml.predict({"audio": chunk_array})
            core_elapsed = time.perf_counter() - core_start

            core_arr = next(iter(core_out.values()))

            metrics = compute_error_metrics(torch_out, core_arr)

            per_chunk.append(
                ComparisonResult(
                    **metrics,
                    cosine=None,
                    baseline_time=torch_elapsed,
                    candidate_time=core_elapsed,
                )
            )
            coreml_outputs.append(core_arr)
            coreml_times.append(core_elapsed)

        if not coreml_outputs:
            results[unit_key] = {
                "error": "No Core ML outputs produced",
                "compute_unit": compute_unit_display(compute_unit),
            }
            continue

        stacked_core = np.concatenate(coreml_outputs, axis=0)
        aggregate = ComparisonResult(**compute_error_metrics(stacked_torch, stacked_core), cosine=None)

        per_class_corr = []
        for cls in range(stacked_torch.shape[-1]):
            ref = stacked_torch[..., cls].reshape(-1)
            cand = stacked_core[..., cls].reshape(-1)
            ref_std = float(np.std(ref))
            cand_std = float(np.std(cand))
            if ref_std == 0.0 or cand_std == 0.0:
                per_class_corr.append(float("nan"))
            else:
                per_class_corr.append(float(np.corrcoef(ref, cand)[0, 1]))

        total_core = float(np.sum(coreml_times)) if coreml_times else float("nan")
        avg_core = float(np.mean(coreml_times)) if coreml_times else float("nan")
        setup_time = compile_time + warmup_elapsed
        total_with_setup = (
            setup_time + total_core if math.isfinite(total_core) else float("nan")
        )
        speedup = (
            total_torch / total_core
            if math.isfinite(total_core) and total_core > 0.0
            else float("nan")
        )
        speedup_with_setup = (
            total_torch / total_with_setup
            if math.isfinite(total_with_setup) and total_with_setup > 0.0
            else float("nan")
        )

        results[unit_key] = {
            "aggregate": aggregate,
            "chunks": per_chunk,
            "per_class_corr": per_class_corr,
            "core_output_shape": stacked_core.shape,
            "timings": {
                "baseline": list(torch_times),
                "coreml": coreml_times,
                "setup": {
                    "compile_seconds": compile_time,
                    "warmup_seconds": warmup_elapsed,
                    "warmup_runs": WARMUP_RUNS,
                    "total_seconds": setup_time,
                },
                "stats": {
                    "baseline_total": total_torch,
                    "coreml_total": total_core,
                    "baseline_avg": avg_torch,
                    "coreml_avg": avg_core,
                    "coreml_setup": setup_time,
                    "coreml_total_with_setup": total_with_setup,
                    "compile_time": compile_time,
                    "warmup_time": warmup_elapsed,
                    "warmup_runs": WARMUP_RUNS,
                    "speedup": speedup,
                    "speedup_including_setup": speedup_with_setup,
                    "audio_seconds": audio_seconds,
                    "processed_audio_seconds": processed_seconds,
                },
            },
            "window_seconds": SEGMENTATION_WINDOW / SAMPLE_RATE,
            "compute_unit": compute_unit_display(compute_unit),
        }

    return results


def compare_embedding(
    model_root: Path,
    coreml_dir: Path,
    audio: torch.Tensor,
    compute_units: Iterable[ct.ComputeUnit],
) -> dict[str, dict]:
    embedding_torch = Model.from_pretrained(str(model_root / "embedding")).eval()

    chunk_inputs: list[dict[str, np.ndarray]] = []
    torch_outputs: list[np.ndarray] = []
    torch_times: list[float] = []
    audio_seconds = float(audio.shape[-1]) / SAMPLE_RATE

    spec = ct.utils.load_spec(str(coreml_dir / "embedding-community-1.mlpackage"))
    weight_length: int | None = None
    fbank_shape: tuple[int, ...] | None = None
    for input_desc in spec.description.input:  # type: ignore[attr-defined]
        name = getattr(input_desc, "name", "")
        array_type = getattr(input_desc.type, "multiArrayType", None)
        if array_type is None:
            continue
        shape = tuple(int(dim) for dim in getattr(array_type, "shape", []))
        if not shape:
            continue
        if name == "fbank_features":
            fbank_shape = shape
        elif name == "weights":
            weight_length = int(shape[-1])
    if fbank_shape is None:
        raise RuntimeError(
            "Core ML embedding model is missing a valid 'fbank_features' input shape"
        )
    if weight_length is None or weight_length <= 0:
        raise RuntimeError(
            "Core ML embedding model is missing a valid 'weights' input shape"
        )

    try:
        fbank_ml = ct.models.MLModel(
            str(coreml_dir / "fbank-community-1.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
    except Exception as exc:  # pragma: no cover - runtime specific
        raise RuntimeError(f"Failed to load Core ML FBANK model: {exc}") from exc

    with torch.inference_mode():
        for chunk in chunk_waveform(audio, EMBEDDING_WINDOW):
            array = np.array(chunk.cpu().numpy(), dtype=np.float32, copy=True)
            weights = np.ones((1, weight_length), dtype=np.float32)
            features = fbank_ml.predict({"audio": array})
            feature_array = np.array(features["fbank_features"], dtype=np.float32, copy=True)
            if feature_array.shape[1:] != tuple(fbank_shape[1:]):
                raise RuntimeError(
                    f"FBANK feature shape mismatch: expected (*, {fbank_shape[1:]}), got {feature_array.shape}"
                )
            chunk_inputs.append({
                "fbank_features": feature_array,
                "weights": weights,
            })

            torch_start = time.perf_counter()
            torch_out = embedding_torch(chunk).cpu().numpy()
            torch_elapsed = time.perf_counter() - torch_start

            torch_outputs.append(torch_out)
            torch_times.append(torch_elapsed)

    if not torch_outputs:
        return {}

    stacked_torch = np.concatenate(torch_outputs, axis=0)
    total_torch = float(np.sum(torch_times)) if torch_times else float("nan")
    avg_torch = float(np.mean(torch_times)) if torch_times else float("nan")
    processed_seconds = len(torch_times) * (EMBEDDING_WINDOW / SAMPLE_RATE)

    plda_resources: tuple[PLDAPipeline, PLDAPipeline] | None = None
    try:
        plda_resources = (
            load_plda_pipeline_from_npz(model_root),
            load_plda_pipeline_from_json(coreml_dir),
        )
    except FileNotFoundError:
        plda_resources = None

    results: dict[str, dict] = {}

    for compute_unit in compute_units:
        unit_key = compute_unit_key(compute_unit)
        compile_start = time.perf_counter()
        try:
            embedding_ml = ct.models.MLModel(
                str(coreml_dir / "embedding-community-1.mlpackage"),
                compute_units=compute_unit,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            results[unit_key] = {
                "error": str(exc),
                "compute_unit": compute_unit_display(compute_unit),
            }
            continue
        compile_time = time.perf_counter() - compile_start

        warmup_elapsed = 0.0
        if WARMUP_RUNS > 0 and chunk_inputs:
            warmup_input = chunk_inputs[0]
            for _ in range(WARMUP_RUNS):
                warm_start = time.perf_counter()
                embedding_ml.predict(warmup_input)
                warmup_elapsed += time.perf_counter() - warm_start

        coreml_outputs: list[np.ndarray] = []
        per_chunk: list[ComparisonResult] = []
        cosines: list[float] = []
        coreml_times: list[float] = []

        # Process one chunk at a time to mirror PyTorch timings (Core ML model now supports batching)
        for idx, coreml_input in enumerate(chunk_inputs):
            torch_out = torch_outputs[idx]
            torch_elapsed = torch_times[idx]

            core_start = time.perf_counter()
            core_out = embedding_ml.predict(coreml_input)
            core_elapsed = time.perf_counter() - core_start

            core_arr = core_out["embedding"]

            metrics = compute_error_metrics(torch_out, core_arr)
            cos = cosine_similarity(torch_out, core_arr)

            per_chunk.append(
                ComparisonResult(
                    **metrics,
                    cosine=cos,
                    baseline_time=torch_elapsed,
                    candidate_time=core_elapsed,
                )
            )
            cosines.append(cos)
            coreml_outputs.append(core_arr)
            coreml_times.append(core_elapsed)

        if not coreml_outputs:
            results[unit_key] = {
                "error": "No Core ML outputs produced",
                "compute_unit": compute_unit_display(compute_unit),
            }
            continue

        stacked_core = np.concatenate(coreml_outputs, axis=0)

        plda_checks: dict[str, ComparisonResult] = {}
        if plda_resources is not None:
            plda_npz, plda_json = plda_resources

            torch_npz = plda_npz.transform(stacked_torch)
            core_npz = plda_npz.transform(stacked_core)
            torch_json = plda_json.transform(stacked_torch)
            core_json = plda_json.transform(stacked_core)

            rho_torch_json = plda_json.rho(stacked_torch)
            rho_core_json = plda_json.rho(stacked_core)
            rho_torch_npz = plda_npz.rho(stacked_torch)
            rho_core_npz = plda_npz.rho(stacked_core)

            score_torch_json = rho_torch_json @ rho_torch_json.T
            score_core_json = rho_core_json @ rho_core_json.T
            score_torch_npz = rho_torch_npz @ rho_torch_npz.T
            score_core_npz = rho_core_npz @ rho_core_npz.T

            plda_checks = {
                "phi_alignment": make_comparison_result(
                    plda_npz.phi[: plda_npz.lda_dim], plda_json.phi[: plda_json.lda_dim]
                ),
                "json_features_torch_vs_coreml": make_comparison_result(
                    torch_json, core_json
                ),
                "npz_features_torch_vs_coreml": make_comparison_result(torch_npz, core_npz),
                "resource_parity_torch": make_comparison_result(torch_npz, torch_json),
                "resource_parity_coreml": make_comparison_result(core_npz, core_json),
                "rho_json_torch_vs_coreml": make_comparison_result(
                    rho_torch_json, rho_core_json
                ),
                "rho_resource_parity_torch": make_comparison_result(
                    rho_torch_npz, rho_torch_json
                ),
                "score_matrix_alignment": make_comparison_result(
                    score_torch_json, score_core_json
                ),
                "score_resource_parity_torch": make_comparison_result(
                    score_torch_npz, score_torch_json
                ),
                "score_resource_parity_core": make_comparison_result(
                    score_core_npz, score_core_json
                ),
            }

        aggregate_metrics = compute_error_metrics(stacked_torch, stacked_core)
        aggregate_cos = cosine_similarity(stacked_torch, stacked_core)
        aggregate = ComparisonResult(**aggregate_metrics, cosine=aggregate_cos)

        per_dim_corr = []
        for dim in range(stacked_torch.shape[-1]):
            ref = stacked_torch[:, dim]
            cand = stacked_core[:, dim]
            ref_std = float(np.std(ref))
            cand_std = float(np.std(cand))
            if ref_std == 0.0 or cand_std == 0.0:
                per_dim_corr.append(float("nan"))
            else:
                per_dim_corr.append(float(np.corrcoef(ref, cand)[0, 1]))

        total_core = float(np.sum(coreml_times)) if coreml_times else float("nan")
        avg_core = float(np.mean(coreml_times)) if coreml_times else float("nan")
        setup_time = compile_time + warmup_elapsed
        total_with_setup = (
            setup_time + total_core if math.isfinite(total_core) else float("nan")
        )
        speedup = (
            total_torch / total_core
            if math.isfinite(total_core) and total_core > 0.0
            else float("nan")
        )
        speedup_with_setup = (
            total_torch / total_with_setup
            if math.isfinite(total_with_setup) and total_with_setup > 0.0
            else float("nan")
        )

        results[unit_key] = {
            "aggregate": aggregate,
            "chunks": per_chunk,
            "per_dim_corr": per_dim_corr,
            "core_output_shape": stacked_core.shape,
            "chunk_cosines": cosines,
            "timings": {
                "baseline": list(torch_times),
                "coreml": coreml_times,
                "setup": {
                    "compile_seconds": compile_time,
                    "warmup_seconds": warmup_elapsed,
                    "warmup_runs": WARMUP_RUNS,
                    "total_seconds": setup_time,
                },
                "stats": {
                    "baseline_total": total_torch,
                    "coreml_total": total_core,
                    "baseline_avg": avg_torch,
                    "coreml_avg": avg_core,
                    "coreml_setup": setup_time,
                    "coreml_total_with_setup": total_with_setup,
                    "compile_time": compile_time,
                    "warmup_time": warmup_elapsed,
                    "warmup_runs": WARMUP_RUNS,
                    "speedup": speedup,
                    "speedup_including_setup": speedup_with_setup,
                    "audio_seconds": audio_seconds,
                    "processed_audio_seconds": processed_seconds,
                },
            },
            "window_seconds": EMBEDDING_WINDOW / SAMPLE_RATE,
            "plda_checks": plda_checks,
            "compute_unit": compute_unit_display(compute_unit),
        }

    return results


def compare_plda(
    model_root: Path,
    coreml_dir: Path,
    embeddings: np.ndarray,
    compute_units: Iterable[ct.ComputeUnit],
) -> dict[str, dict]:
    """Compare PLDA CoreML model against reference NPZ implementation."""

    # Load reference PLDA pipeline
    plda_npz = load_plda_pipeline_from_npz(model_root)

    # Check embedding normalization
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    print(f"PLDA input embeddings: shape={embeddings.shape}, "
          f"norm mean={embedding_norms.mean():.6f}, norm std={embedding_norms.std():.6f}, "
          f"norm min={embedding_norms.min():.6f}, norm max={embedding_norms.max():.6f}")

    # Reference outputs
    ref_features = plda_npz.transform(embeddings)
    print(f"PLDA features: shape={ref_features.shape}, "
          f"mean={ref_features.mean():.6f}, std={ref_features.std():.6f}")

    ref_rho = plda_npz.rho(embeddings)
    print(f"PLDA rho: shape={ref_rho.shape}, "
          f"mean={ref_rho.mean():.6f}, std={ref_rho.std():.6f}")

    ref_scores = ref_rho @ ref_rho.T
    print(f"PLDA scores: shape={ref_scores.shape}, "
          f"mean={ref_scores.mean():.6f}, std={ref_scores.std():.6f}, "
          f"min={ref_scores.min():.6f}, max={ref_scores.max():.6f}")

    # Pre-compute scaling factors used by rho transformation
    phi = plda_npz.phi[: plda_npz.lda_dim].astype(np.float64)
    phi_sqrt = np.sqrt(phi)
    phi_sqrt_safe = np.where(phi_sqrt > 0.0, phi_sqrt, 1.0)
    zero_phi_mask = phi_sqrt == 0.0

    results: dict[str, dict] = {}

    for compute_unit in compute_units:
        unit_key = compute_unit_key(compute_unit)

        # Prefer the plda_rho Core ML export, fall back to legacy plda bundle if needed
        plda_model_path = coreml_dir / "plda_rho-community-1.mlpackage"
        output_key = "rho"
        if not plda_model_path.exists():
            plda_model_path = coreml_dir / "plda-community-1.mlpackage"
            output_key = "plda_features"

        if not plda_model_path.exists():
            results[unit_key] = {
                "error": "PLDA rho CoreML model not found",
                "compute_unit": compute_unit_display(compute_unit),
            }
            continue

        try:
            plda_ml = ct.models.MLModel(str(plda_model_path), compute_units=compute_unit)
        except Exception as exc:
            results[unit_key] = {
                "error": str(exc),
                "compute_unit": compute_unit_display(compute_unit),
            }
            continue

        # Run Core ML predictions with batching
        coreml_outputs = []
        coreml_times = []
        ref_times = []
        missing_output_error: str | None = None

        # Convert embeddings to list for batch_chunks
        embedding_list = [embeddings[i:i+1].astype(np.float32) for i in range(embeddings.shape[0])]
        batched_embeddings = batch_chunks(embedding_list, batch_size=32)

        emb_idx = 0
        for batch_array, actual_count in batched_embeddings:
            # Single CoreML call for the batch
            coreml_start = time.perf_counter()
            result = plda_ml.predict({"embeddings": batch_array})
            coreml_elapsed = time.perf_counter() - coreml_start

            output_value = result.get(output_key)
            if output_value is None:
                available = ", ".join(sorted(result.keys())) or "<none>"
                missing_output_error = (
                    f"Expected Core ML output '{output_key}' but received [{available}]"
                )
                break

            batch_results = np.asarray(output_value, dtype=np.float64)

            # Process each embedding in the batch
            for i in range(actual_count):
                emb = embeddings[emb_idx:emb_idx+1]

                # Time reference transform (feature path)
                ref_start = time.perf_counter()
                _ = plda_npz.transform(emb)
                ref_elapsed = time.perf_counter() - ref_start
                ref_times.append(ref_elapsed)

                # Divide batch time evenly across embeddings
                per_emb_time = coreml_elapsed / actual_count
                coreml_times.append(per_emb_time)

                # Extract result for this embedding (keep 2D shape)
                core_arr = batch_results[i:i+1]
                coreml_outputs.append(core_arr)

                emb_idx += 1

        if missing_output_error is not None:
            results[unit_key] = {
                "error": missing_output_error,
                "compute_unit": compute_unit_display(compute_unit),
            }
            continue

        coreml_stacked = np.concatenate(coreml_outputs, axis=0)

        if output_key == "rho":
            coreml_rho = coreml_stacked
            coreml_features = coreml_rho / phi_sqrt_safe
            if np.any(zero_phi_mask):
                coreml_features[:, zero_phi_mask] = 0.0
            print(f"CoreML output key: {output_key} (rho directly from model)")
        else:
            coreml_features = coreml_stacked
            coreml_rho = coreml_features * phi_sqrt
            if np.any(zero_phi_mask):
                coreml_rho[:, zero_phi_mask] = 0.0
            print(f"CoreML output key: {output_key} (features, computing rho)")

        print(f"CoreML features: shape={coreml_features.shape}, "
              f"mean={coreml_features.mean():.6f}, std={coreml_features.std():.6f}")
        print(f"CoreML rho: shape={coreml_rho.shape}, "
              f"mean={coreml_rho.mean():.6f}, std={coreml_rho.std():.6f}")

        coreml_scores = coreml_rho @ coreml_rho.T
        print(f"CoreML scores: shape={coreml_scores.shape}, "
              f"mean={coreml_scores.mean():.6f}, std={coreml_scores.std():.6f}, "
              f"min={coreml_scores.min():.6f}, max={coreml_scores.max():.6f}")

        # Compute comparison metrics
        features_comparison = make_comparison_result(ref_features, coreml_features)
        rho_comparison = make_comparison_result(ref_rho, coreml_rho)
        scores_comparison = make_comparison_result(ref_scores, coreml_scores)

        # Compute per-dimension correlation on feature space
        per_dim_corr = []
        for dim in range(ref_features.shape[-1]):
            ref_dim = ref_features[:, dim]
            coreml_dim = coreml_features[:, dim]
            ref_std = float(np.std(ref_dim))
            coreml_std = float(np.std(coreml_dim))
            if ref_std == 0.0 or coreml_std == 0.0:
                per_dim_corr.append(float("nan"))
            else:
                per_dim_corr.append(float(np.corrcoef(ref_dim, coreml_dim)[0, 1]))

        total_ref = float(np.sum(ref_times)) if ref_times else float("nan")
        total_coreml = float(np.sum(coreml_times)) if coreml_times else float("nan")
        avg_ref = float(np.mean(ref_times)) if ref_times else float("nan")
        avg_coreml = float(np.mean(coreml_times)) if coreml_times else float("nan")
        speedup = (
            total_ref / total_coreml
            if math.isfinite(total_coreml) and total_coreml > 0.0
            else float("nan")
        )

        results[unit_key] = {
            "features_comparison": features_comparison,
            "rho_comparison": rho_comparison,
            "scores_comparison": scores_comparison,
            "per_dim_corr": per_dim_corr,
            "num_embeddings": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1],
            "plda_dim": ref_features.shape[1],
            "timings": {
                "reference": list(ref_times),
                "coreml": coreml_times,
                "stats": {
                    "reference_total": total_ref,
                    "coreml_total": total_coreml,
                    "reference_avg": avg_ref,
                    "coreml_avg": avg_coreml,
                    "speedup": speedup,
                },
            },
            "compute_unit": compute_unit_display(compute_unit),
            "output_key": output_key,
        }

    return results


def compare_pipeline_end_to_end(
    model_root: Path,
    coreml_dir: Path,
    audio_path: Path,
    compute_units: Iterable[ct.ComputeUnit],
    waveform: torch.Tensor | None = None,
    audio_duration_seconds: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the diarization pipeline with PyTorch vs. Core ML components."""

    baseline_pipeline: Pipeline = Pipeline.from_pretrained(str(model_root))
    # undo this, hard coding for faster accuracy check..
    baseline_pipeline.to(torch.device("mps"))
    def _make_file_dict() -> dict[str, Any]:
        if waveform is not None:
            return {
                "waveform": waveform.detach().cpu(),
                "sample_rate": SAMPLE_RATE,
                "uri": audio_path.stem,
            }
        return {"audio": str(audio_path), "uri": audio_path.stem}

    print("Running baseline pipeline (PyTorch)")

    audio_seconds = audio_duration_seconds
    if audio_seconds is None:
        if waveform is not None and waveform.shape[-1] > 0:
            audio_seconds = float(waveform.shape[-1]) / SAMPLE_RATE
        else:
            audio_seconds = float("nan")

    baseline_start = time.perf_counter()
    baseline_output = baseline_pipeline(_make_file_dict())
    baseline_elapsed = time.perf_counter() - baseline_start
    baseline_rtf = (
        baseline_elapsed / audio_seconds
        if audio_seconds is not None
        and math.isfinite(audio_seconds)
        and audio_seconds > 0.0
        else float("nan")
    )
    baseline_rtfx = (
        1.0 / baseline_rtf
        if math.isfinite(baseline_rtf) and baseline_rtf > 0.0
        else float("nan")
    )
    baseline_annotation, baseline_exclusive = _extract_pipeline_annotations(
        baseline_output
    )

    # Normalize speaker labels by first appearance for consistent visual comparison
    baseline_annotation = normalize_speaker_labels(baseline_annotation)
    if baseline_exclusive is not None:
        baseline_exclusive = normalize_speaker_labels(baseline_exclusive)

    # print(
    #     json.dumps(
    #         _serialize_diarization_output(baseline_output),
    #         indent=2,
    #         sort_keys=True,
    #     )
    # )
    baseline_segments = _collect_annotation_segments(baseline_annotation)
    baseline_exclusive_segments = (
        _collect_annotation_segments(baseline_exclusive)
        if baseline_exclusive is not None
        else []
    )
    baseline_speaker_labels: set[str] = {segment[2] for segment in baseline_segments}
    baseline_exclusive_speakers = sorted({seg[2] for seg in baseline_exclusive_segments})
    if baseline_exclusive_segments:
        baseline_speaker_labels.update(seg[2] for seg in baseline_exclusive_segments)
    baseline_speakers = sorted(baseline_speaker_labels)

    baseline_details: list[str] = []
    if math.isfinite(baseline_rtf):
        baseline_details.append(f"RTF {baseline_rtf:.3f}")
    if math.isfinite(baseline_rtfx):
        baseline_details.append(f"RTFx {baseline_rtfx:.2f}")
    baseline_detail_str = f" ({' | '.join(baseline_details)})" if baseline_details else ""
    print(
        "Baseline pipeline runtime: "
        f"{baseline_elapsed:.3f}s"
        f"{baseline_detail_str}"
    )

    results: dict[str, dict[str, Any]] = {
        "baseline": {
            "annotation": baseline_annotation,
            "exclusive_annotation": baseline_exclusive,
            "segments": baseline_segments,
            "exclusive_segments": baseline_exclusive_segments,
            "speakers": baseline_speakers,
            "exclusive_speakers": baseline_exclusive_speakers,
            "alignment": [],
            "exclusive_alignment": [],
            "mismatched_windows": [],
            "exclusive_mismatched_windows": [],
            "der_vs_baseline": 0.0,
            "exclusive_der_vs_baseline": float("nan"),
            "jer_vs_baseline": 0.0,
            "exclusive_jer_vs_baseline": float("nan"),
            "compute_unit": "PyTorch baseline (mps)",
            "timing": {
                "runtime_seconds": baseline_elapsed,
                "rtf": baseline_rtf,
                "rtfx": baseline_rtfx,
                "audio_seconds": audio_seconds,
            },
        }
    }

    def _register_candidate_result(
        *,
        key: str,
        label: str,
        annotation: Annotation,
        exclusive: Annotation | None,
        elapsed: float,
        compile_time: float = 0.0,
        warmup_time: float = 0.0,
    ) -> None:
        # Normalize speaker labels for consistent visual comparison
        annotation = normalize_speaker_labels(annotation)
        if exclusive is not None:
            exclusive = normalize_speaker_labels(exclusive)

        candidate_segments = _collect_annotation_segments(annotation)
        candidate_exclusive_segments = (
            _collect_annotation_segments(exclusive) if exclusive is not None else []
        )
        candidate_speaker_labels: set[str] = {seg[2] for seg in candidate_segments}
        candidate_exclusive_speakers = sorted({seg[2] for seg in candidate_exclusive_segments})
        if candidate_exclusive_segments:
            candidate_speaker_labels.update(seg[2] for seg in candidate_exclusive_segments)
        candidate_speakers = sorted(candidate_speaker_labels)

        der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)

        der = float(der_metric(baseline_annotation, annotation))
        jer = float(jer_metric(baseline_annotation, annotation))
        alignment = build_alignment_windows(baseline_annotation, annotation)
        mismatches = [window for window in alignment if not window.get("match", False)]

        exclusive_alignment: list[dict[str, Any]] = []
        exclusive_mismatches: list[dict[str, Any]] = []
        exclusive_der = float("nan")
        exclusive_jer = float("nan")
        if baseline_exclusive is not None and exclusive is not None:
            exclusive_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
            exclusive_jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)
            exclusive_der = float(exclusive_metric(baseline_exclusive, exclusive))
            exclusive_jer = float(exclusive_jer_metric(baseline_exclusive, exclusive))
            exclusive_alignment = build_alignment_windows(baseline_exclusive, exclusive)
            exclusive_mismatches = [
                window
                for window in exclusive_alignment
                if not window.get("match", False)
            ]

        candidate_rtf = (
            elapsed / audio_seconds
            if audio_seconds is not None
            and math.isfinite(audio_seconds)
            and audio_seconds > 0.0
            else float("nan")
        )
        candidate_rtfx = (
            1.0 / candidate_rtf
            if math.isfinite(candidate_rtf) and candidate_rtf > 0.0
            else float("nan")
        )
        rtf_delta = (
            candidate_rtf - baseline_rtf
            if math.isfinite(candidate_rtf) and math.isfinite(baseline_rtf)
            else float("nan")
        )
        rtf_ratio = (
            baseline_rtf / candidate_rtf
            if math.isfinite(candidate_rtf)
            and math.isfinite(baseline_rtf)
            and candidate_rtf != 0.0
            else float("nan")
        )

        # Calculate total runtime including setup overhead
        setup_time = compile_time + warmup_time
        total_with_setup = (
            elapsed + setup_time if math.isfinite(elapsed) else float("nan")
        )
        rtf_with_setup = (
            total_with_setup / audio_seconds
            if audio_seconds is not None
            and math.isfinite(audio_seconds)
            and audio_seconds > 0.0
            and math.isfinite(total_with_setup)
            else float("nan")
        )
        speedup = (
            baseline_elapsed / elapsed
            if math.isfinite(baseline_elapsed) and math.isfinite(elapsed) and elapsed > 0.0
            else float("nan")
        )
        speedup_with_setup = (
            baseline_elapsed / total_with_setup
            if math.isfinite(baseline_elapsed)
            and math.isfinite(total_with_setup)
            and total_with_setup > 0.0
            else float("nan")
        )

        detail_parts: list[str] = []
        if math.isfinite(candidate_rtf):
            detail_parts.append(f"RTF {candidate_rtf:.3f}")
        if math.isfinite(candidate_rtfx):
            detail_parts.append(f"RTFx {candidate_rtfx:.2f}")
        if math.isfinite(rtf_delta):
            detail_parts.append(f"ΔRTF {rtf_delta:+.3f}")
        detail_str = f" ({' | '.join(detail_parts)})" if detail_parts else ""
        print(f"  {label} runtime: {elapsed:.3f}s{detail_str}")

        if math.isfinite(der) or math.isfinite(jer):
            metrics_parts = []
            if math.isfinite(der):
                metrics_parts.append(f"DER {der * 100:.2f}%")
            if math.isfinite(jer):
                metrics_parts.append(f"JER {jer * 100:.2f}%")
            if metrics_parts:
                print(f"    {' | '.join(metrics_parts)}")
        if math.isfinite(exclusive_der) or math.isfinite(exclusive_jer):
            exclusive_parts = []
            if math.isfinite(exclusive_der):
                exclusive_parts.append(f"exclusive DER {exclusive_der * 100:.2f}%")
            if math.isfinite(exclusive_jer):
                exclusive_parts.append(f"exclusive JER {exclusive_jer * 100:.2f}%")
            if exclusive_parts:
                print(f"    {' | '.join(exclusive_parts)}")

        results[key] = {
            "annotation": annotation,
            "exclusive_annotation": exclusive,
            "segments": candidate_segments,
            "exclusive_segments": candidate_exclusive_segments,
            "speakers": candidate_speakers,
            "exclusive_speakers": candidate_exclusive_speakers,
            "alignment": alignment,
            "exclusive_alignment": exclusive_alignment,
            "mismatched_windows": mismatches,
            "exclusive_mismatched_windows": exclusive_mismatches,
            "der_vs_baseline": der,
            "exclusive_der_vs_baseline": exclusive_der,
            "jer_vs_baseline": jer,
            "exclusive_jer_vs_baseline": exclusive_jer,
            "compute_unit": label,
            "timing": {
                "runtime_seconds": elapsed,
                "compile_time": compile_time,
                "warmup_time": warmup_time,
                "setup_time": setup_time,
                "runtime_with_setup": total_with_setup,
                "rtf": candidate_rtf,
                "rtf_with_setup": rtf_with_setup,
                "rtf_delta_vs_baseline": rtf_delta,
                "rtf_ratio_vs_baseline": rtf_ratio,
                "rtfx": candidate_rtfx,
                "speedup": speedup,
                "speedup_with_setup": speedup_with_setup,
                "audio_seconds": audio_seconds,
            },
        }

    # try:
    #     print("Running PyTorch pipeline on MPS backend")
    #     candidate_pipeline = Pipeline.from_pretrained(str(model_root))
    #     candidate_pipeline.to(torch.device("mps"))
    #     candidate_start = time.perf_counter()
    #     candidate_output = candidate_pipeline(_make_file_dict())
    #     if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
    #         torch.mps.synchronize()  # type: ignore[attr-defined]
    #     candidate_elapsed = time.perf_counter() - candidate_start
    #     candidate_annotation, candidate_exclusive = _extract_pipeline_annotations(
    #         candidate_output
    #     )
    #     _register_candidate_result(
    #         key="TORCH_MPS",
    #         label="PyTorch MPS",
    #         annotation=candidate_annotation,
    #         exclusive=candidate_exclusive,
    #         elapsed=candidate_elapsed,
    #     )
    # except Exception as exc:  # pragma: no cover - runtime specific
    #     print(f"  PyTorch MPS pipeline failed: {exc}")
    #     results["TORCH_MPS"] = {
    #         "error": str(exc),
    #         "compute_unit": "PyTorch MPS",
    #     }

    for compute_unit in compute_units:
        print(f"Running for COREML {compute_unit.name}")
        unit_key = compute_unit_key(compute_unit)
        try:
            candidate_pipeline: Pipeline = Pipeline.from_pretrained(str(model_root))

            # Time CoreML model compilation/loading
            compile_start = time.perf_counter()
            wrap_pipeline_with_coreml(candidate_pipeline, coreml_dir, compute_unit)
            compile_time = time.perf_counter() - compile_start

            # Warmup runs to eliminate cold start overhead
            warmup_elapsed = 0.0
            if WARMUP_RUNS > 0:
                for _ in range(WARMUP_RUNS):
                    warm_start = time.perf_counter()
                    candidate_pipeline(_make_file_dict())
                    warmup_elapsed += time.perf_counter() - warm_start

            # Actual timed inference (excluding setup)
            candidate_start = time.perf_counter()
            candidate_output = candidate_pipeline(_make_file_dict())
            candidate_elapsed = time.perf_counter() - candidate_start

            candidate_annotation, candidate_exclusive = _extract_pipeline_annotations(
                candidate_output
            )
            _register_candidate_result(
                key=unit_key,
                label=compute_unit_display(compute_unit),
                annotation=candidate_annotation,
                exclusive=candidate_exclusive,
                elapsed=candidate_elapsed,
                compile_time=compile_time,
                warmup_time=warmup_elapsed,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            results[unit_key] = {
                "error": str(exc),
                "compute_unit": compute_unit_display(compute_unit),
            }

    return results


def _collect_segment_boundaries(*annotations: Annotation) -> list[float]:
    boundaries: set[float] = set()
    for annotation in annotations:
        if annotation is None:
            continue
        for segment in annotation.itersegments():
            if math.isfinite(segment.start):
                boundaries.add(float(segment.start))
            if math.isfinite(segment.end):
                boundaries.add(float(segment.end))
    ordered = sorted(boundaries)
    return ordered


def _labels_for_interval(annotation: Annotation, segment: Segment) -> list[str]:
    if annotation is None:
        return []
    cropped = annotation.crop(segment, mode="intersection")
    if len(cropped) == 0:
        return []
    labels = {label for _, _, label in cropped.itertracks(yield_label=True)}
    return sorted(str(label) for label in labels)


def build_alignment_windows(
    baseline_ann: Annotation,
    candidate_ann: Annotation,
) -> list[dict[str, Any]]:
    """Generate aligned timeline windows comparing baseline vs candidate labels."""

    boundaries = _collect_segment_boundaries(baseline_ann, candidate_ann)
    if len(boundaries) < 2:
        return []

    windows: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start <= 1e-6:
            continue
        window = Segment(start, end)
        baseline_labels = _labels_for_interval(baseline_ann, window)
        candidate_labels = _labels_for_interval(candidate_ann, window)
        windows.append(
            {
                "start": float(start),
                "end": float(end),
                "baseline_labels": baseline_labels,
                "candidate_labels": candidate_labels,
                "match": baseline_labels == candidate_labels,
            }
        )
    return windows


def log_pipeline_alignment(
    baseline_ann: Annotation,
    candidate_ann: Annotation,
    display_label: str,
    *,
    only_mismatches: bool = False,
) -> None:
    windows = build_alignment_windows(baseline_ann, candidate_ann)
    if not windows:
        print("  Timeline alignment -> insufficient boundaries")
        return

    header = "  Timeline mismatches vs. baseline:" if only_mismatches else "  Timeline alignment vs. baseline:"
    print(header)

    printed_any = False
    for window in windows:
        start = float(window["start"])
        end = float(window["end"])
        if end - start <= 1e-6:
            continue
        base_labels = window.get("baseline_labels", [])
        cand_labels = window.get("candidate_labels", [])
        status = "match" if window.get("match", False) else "diff"
        if only_mismatches and status == "match":
            continue
        base_str = ",".join(base_labels) if base_labels else "-"
        cand_str = ",".join(cand_labels) if cand_labels else "-"
        print(
            f"    {start:6.2f}-{end:6.2f}s | baseline={base_str:<12} "
            f"candidate={cand_str:<12} [{status}]"
        )
        printed_any = True

    if only_mismatches and not printed_any:
        print("    (perfect alignment)")


def _collect_annotation_segments(annotation: Annotation) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        start = float(getattr(segment, "start", float("nan")))
        end = float(getattr(segment, "end", float("nan")))
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end <= start:
            continue
        segments.append((start, end, str(label)))
    segments.sort(key=lambda item: (item[0], item[1], item[2]))
    return segments


def _extract_pipeline_annotations(output: Any) -> tuple[Annotation, Annotation | None]:
    """Return speaker_diarization and exclusive_speaker_diarization annotations."""

    if isinstance(output, Annotation):
        return output, None

    speaker_ann = getattr(output, "speaker_diarization", None)
    if not isinstance(speaker_ann, Annotation):
        raise ValueError("Pipeline output does not include speaker_diarization Annotation")

    exclusive_ann = getattr(output, "exclusive_speaker_diarization", None)
    if not isinstance(exclusive_ann, Annotation):
        exclusive_ann = None

    return speaker_ann, exclusive_ann


def _annotation_to_json(annotation: Annotation | None) -> list[dict[str, Any]]:
    if annotation is None:
        return []
    serialized: list[dict[str, Any]] = []
    for start, end, label in _collect_annotation_segments(annotation):
        serialized.append({
            "end": end,
            "label": label,
            "start": start,
        })
    return serialized


def _jsonify(value: Any) -> Any:
    if isinstance(value, Annotation):
        return _annotation_to_json(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


def _serialize_diarization_output(output: Any) -> Any:
    if isinstance(output, Annotation):
        return {
            "speaker_diarization": _annotation_to_json(output),
        }

    if hasattr(output, "_asdict"):
        raw_dict = output._asdict()
    elif hasattr(output, "__dict__"):
        raw_dict = vars(output)
    else:
        return str(output)

    serialized: dict[str, Any] = {}
    for key, value in raw_dict.items():
        serialized[key] = _jsonify(value)
    return serialized


def log_annotation_segments(annotation: Annotation, heading: str, indent: str = "  ") -> None:
    print(f"{indent}{heading}")
    segments = _collect_annotation_segments(annotation)
    if not segments:
        print(f"{indent}  (no segments)")
        return
    for index, (start, end, label) in enumerate(segments, start=1):
        duration = end - start
        duration_str = human_readable_duration(duration)
        print(
            f"{indent}  #{index:02d}: {start:7.2f}-{end:7.2f}s "
            f"({duration_str}) -> {label}"
        )


def ensure_plots_dir(base: Path) -> Path:
    plots_dir = base / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def _metrics_from_chunks(chunks: list[ComparisonResult]) -> dict[str, np.ndarray]:
    if not chunks:
        return {"mse": np.array([]), "mae": np.array([]), "max_abs": np.array([]), "corr": np.array([])}
    return {
        "mse": np.array([chunk.mse for chunk in chunks], dtype=np.float32),
        "mae": np.array([chunk.mae for chunk in chunks], dtype=np.float32),
        "max_abs": np.array([chunk.max_abs for chunk in chunks], dtype=np.float32),
        "corr": np.array([chunk.corr for chunk in chunks], dtype=np.float32),
    }


def _collect_active_compute_units(*results: dict[str, dict]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _maybe_add(key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        ordered.append(key)

    for candidate in _COMPUTE_UNIT_ORDER:
        if any(candidate in result for result in results):
            _maybe_add(candidate)

    for result in results:
        for key, entry in result.items():
            if not isinstance(entry, dict):
                continue
            if "chunks" not in entry and "timings" not in entry:
                continue
            _maybe_add(key)

    return ordered


def build_compute_unit_visuals(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
) -> tuple[
    list[str],
    dict[str, tuple[float, float, float, float]],
    dict[str, str],
    dict[str, str],
]:
    active_keys = _collect_active_compute_units(seg_results, emb_results)
    if not active_keys:
        return [], {}, {}, {}

    cmap = plt.get_cmap("tab10")
    style_cycle = ["-", "--", "-.", ":"]
    compute_colors: dict[str, tuple[float, float, float, float]] = {}
    compute_styles: dict[str, str] = {}
    compute_labels: dict[str, str] = {}

    def _resolve_label(unit_key: str) -> str:
        for entries in (seg_results, emb_results):
            entry = entries.get(unit_key)
            if isinstance(entry, dict):
                label = entry.get("compute_unit")
                if label:
                    return str(label)
        return unit_key

    for idx, key in enumerate(active_keys):
        compute_colors[key] = cmap(idx % cmap.N)
        compute_styles[key] = style_cycle[idx % len(style_cycle)]
        compute_labels[key] = _resolve_label(key)

    return active_keys, compute_colors, compute_styles, compute_labels


def build_plot_signature(
    base_dir: Path,
    *,
    specs: dict[str, Any] | None = None,
    audio_seconds: float | None = None,
) -> str:
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
    lines = [f"Generated {timestamp} | git {git_hash}"]

    if specs:
        system_desc = specs.get("system") or "Unknown OS"
        machine_desc = specs.get("machine") or "unknown"
        cpu_desc = specs.get("cpu") or "Unknown CPU"
        cpu_count = specs.get("cpu_count")
        memory_gb = specs.get("memory_gb")

        machine_line = f"Machine: {system_desc} ({machine_desc})"
        cpu_line = cpu_desc
        if isinstance(cpu_count, int) and cpu_count > 0:
            cpu_line += f" (cores: {cpu_count})"
        details = [machine_line, f"CPU: {cpu_line}"]
        if isinstance(memory_gb, (int, float)) and math.isfinite(memory_gb):
            details.append(f"Memory: {memory_gb:.1f} GB")
        lines.append(" | ".join(details))

    if audio_seconds is not None and math.isfinite(audio_seconds) and audio_seconds > 0.0:
        lines.append(
            f"Audio: {human_readable_duration(audio_seconds)} @ {SAMPLE_RATE} Hz"
        )

    return "\n".join(lines)


def extract_model_metadata(coreml_dir: Path) -> dict[str, dict[str, str]]:
    """Extract input/output shapes and precision from CoreML models."""
    metadata = {}

    model_files = {
        "segmentation": coreml_dir / "segmentation-community-1.mlpackage",
        "embedding": coreml_dir / "embedding-community-1.mlpackage",
        "plda_rho": coreml_dir / "plda_rho-community-1.mlpackage",
    }

    for model_name, model_path in model_files.items():
        if not model_path.exists():
            continue

        try:
            model = ct.models.MLModel(str(model_path))
            spec = model.get_spec()

            inputs = []
            for inp in spec.description.input:
                name = inp.name
                if hasattr(inp.type, 'multiArrayType'):
                    arr_type = inp.type.multiArrayType
                    shape = list(arr_type.shape) if hasattr(arr_type, 'shape') else []
                    dtype = str(arr_type.dataType) if hasattr(arr_type, 'dataType') else 'unknown'
                    inputs.append(f"{name}: {shape} ({dtype})")

            outputs = []
            for outp in spec.description.output:
                name = outp.name
                if hasattr(outp.type, 'multiArrayType'):
                    arr_type = outp.type.multiArrayType
                    shape = list(arr_type.shape) if hasattr(arr_type, 'shape') else []
                    dtype = str(arr_type.dataType) if hasattr(arr_type, 'dataType') else 'unknown'
                    outputs.append(f"{name}: {shape} ({dtype})")

            metadata[model_name] = {
                "inputs": ", ".join(inputs) if inputs else "N/A",
                "outputs": ", ".join(outputs) if outputs else "N/A",
            }
        except Exception as e:
            metadata[model_name] = {"error": str(e)}

    return metadata


def annotate_figure(fig: plt.Figure, signature: str, model_metadata: dict[str, dict[str, str]] | None = None) -> None:
    if not signature and not model_metadata:
        return

    text_parts = []
    if signature:
        text_parts.append(signature)

    if model_metadata:
        metadata_lines = []
        for model_name, info in model_metadata.items():
            if "error" in info:
                continue
            inputs = info.get("inputs", "N/A")
            outputs = info.get("outputs", "N/A")
            metadata_lines.append(f"{model_name}: in={inputs}, out={outputs}")

        if metadata_lines:
            text_parts.append("Model Info: " + " | ".join(metadata_lines))

    if text_parts:
        fig.text(
            0.99,
            0.02,
            "\n".join(text_parts),
            ha="right",
            va="bottom",
            fontsize=7,
            color="#666666",
        )


def plot_combined_metric_grid(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
    output_path: Path,
    signature: str,
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    metrics = [("MSE", "mse"), ("Corr", "corr")]

    active_keys, compute_colors, compute_styles, compute_labels = build_compute_unit_visuals(
        seg_results,
        emb_results,
    )
    if not active_keys:
        return

    def _collect_series(
        results: dict[str, dict],
        default_window: float,
    ) -> dict[str, dict[str, Any]]:
        series: dict[str, dict[str, Any]] = {}
        for key, result in results.items():
            if not isinstance(result, dict) or "chunks" not in result:
                continue
            metrics_dict = _metrics_from_chunks(result["chunks"])
            if not metrics_dict["mse"].size:
                continue
            window_seconds = float(result.get("window_seconds", default_window) or default_window)
            if not math.isfinite(window_seconds) or window_seconds <= 0.0:
                window_seconds = default_window
            series[key] = {
                "metrics": metrics_dict,
                "window": window_seconds,
                "label": str(result.get("compute_unit", key)),
            }
        return series

    seg_series = _collect_series(seg_results, SEGMENTATION_WINDOW / SAMPLE_RATE)
    emb_series = _collect_series(emb_results, EMBEDDING_WINDOW / SAMPLE_RATE)

    component_linewidth = {"Segmentation": 1.9, "Embedding": 1.25}
    component_alpha = {"Segmentation": 1.0, "Embedding": 0.75}

    fig, axes = plt.subplots(len(metrics), 1, figsize=(11.5, 5.6 * len(metrics)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Keep track of legend handles for compute unit styles and component colors
    unit_handles: dict[str, Line2D] = {}
    component_handles: dict[str, Line2D] = {}

    for ax, (metric_label, metric_key) in zip(axes, metrics):
        plotted_any = False
        for unit_key in active_keys:
            color = compute_colors.get(unit_key)
            style = compute_styles.get(unit_key, "-")
            if color is None:
                continue
            for component_name, series_map in (
                ("Segmentation", seg_series),
                ("Embedding", emb_series),
            ):
                entry = series_map.get(unit_key)
                if not entry:
                    continue

                values = np.asarray(entry["metrics"].get(metric_key), dtype=np.float32)
                if values.size == 0:
                    continue
                finite_mask = np.isfinite(values)
                if not np.any(finite_mask):
                    continue

                x_axis = np.arange(values.size, dtype=np.float32) * float(entry["window"])
                ax.plot(
                    x_axis,
                    values,
                    linewidth=component_linewidth.get(component_name, 1.6),
                    linestyle=style,
                    color=color,
                    alpha=component_alpha.get(component_name, 1.0),
                )
                plotted_any = True

                finite_values = values[finite_mask]
                if finite_values.size > 1 and np.ptp(finite_values) > 0.0:
                    q95 = float(np.quantile(finite_values, 0.95))
                    outlier_mask = finite_mask & (values >= q95)
                    if np.any(outlier_mask):
                        ax.scatter(
                            x_axis[outlier_mask],
                            values[outlier_mask],
                            color=color,
                            s=14,
                            zorder=3,
                            alpha=component_alpha.get(component_name, 1.0),
                        )

                if component_name not in component_handles:
                    component_handles[component_name] = Line2D(
                        [0, 1],
                        [0, 1],
                        color="#333333",
                        linestyle="-",
                        linewidth=component_linewidth.get(component_name, 1.6),
                        alpha=component_alpha.get(component_name, 1.0),
                    )
            if unit_key not in unit_handles and unit_key in compute_labels:
                unit_handles[unit_key] = Line2D(
                    [0, 1],
                    [0, 1],
                    color=compute_colors[unit_key],
                    linestyle=compute_styles.get(unit_key, "-"),
                    linewidth=1.8,
                    label=compute_labels[unit_key],
                )

        if not plotted_any:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0.0)
        if metric_key == "corr":
            ax.set_ylim(-1.05, 1.05)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        else:
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 4), useOffset=False)

        ax.set_ylabel(metric_label)
        ax.set_xlabel("Time (s)")

    fig.suptitle("Chunk-level metrics across Core ML variants", fontsize=14)

    legend_items: list[Line2D] = []
    legend_labels: list[str] = []
    if component_handles:
        legend_items.extend(component_handles.values())
        legend_labels.extend(component_handles.keys())
    if unit_handles:
        legend_items.extend(unit_handles.values())
        legend_labels.extend(compute_labels[key] for key in unit_handles.keys())

    if legend_items:
        fig.legend(
            legend_items,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=max(1, len(legend_items) // 2),
        )
        tight_rect = [0.06, 0.06, 1, 0.92]
    else:
        tight_rect = [0.06, 0.06, 1, 0.96]

    fig.tight_layout(rect=tight_rect)
    annotate_figure(fig, signature, model_metadata)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)



def plot_latency_over_time(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
    output_path: Path,
    signature: str,
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    prepared: list[tuple[str, np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]] = []

    def _collect(prefix: str, entries: dict[str, dict], default_window: float) -> None:
        series: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        for key, result in entries.items():
            timings = result.get("timings") or {}
            baseline = timings.get("baseline", [])
            coreml = timings.get("coreml", [])
            if not baseline or not coreml:
                continue
            length = min(len(baseline), len(coreml))
            if length == 0:
                continue
            window = result.get("window_seconds", default_window)
            label = result.get("compute_unit", key)
            baseline_ms = np.array(baseline[:length], dtype=np.float32) * 1000.0
            coreml_ms = np.array(coreml[:length], dtype=np.float32) * 1000.0
            series.append((label, baseline_ms, coreml_ms, window))

        if not series:
            return

        min_len = min(line[1].shape[0] for line in series)
        ref_baseline = series[0][1]
        ref_window = series[0][3]
        baseline_ms = ref_baseline[:min_len]
        x_axis = np.arange(min_len, dtype=np.float32) * ref_window

        lines: list[tuple[str, np.ndarray]] = []
        for label, _, candidate_ms, _ in series:
            lines.append((label, candidate_ms[:min_len]))

        prepared.append((prefix, x_axis, baseline_ms, lines))

    _collect("Segmentation", seg_results, SEGMENTATION_WINDOW / SAMPLE_RATE)
    _collect("Embedding", emb_results, EMBEDDING_WINDOW / SAMPLE_RATE)

    if not prepared:
        return

    fig, axes = plt.subplots(len(prepared), 1, figsize=(10, 4.5 * len(prepared)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    line_styles = ["-", "--", "-.", ":"]
    for idx, (prefix, x_axis, baseline_ms, lines) in enumerate(prepared):
        ax = axes[idx]
        ax.plot(x_axis, baseline_ms, linewidth=1.8, linestyle="-", label="PyTorch baseline (mps)")
        for line_idx, (label, candidate_ms) in enumerate(lines):
            ax.plot(
                x_axis,
                candidate_ms,
                linewidth=1.5,
                linestyle=line_styles[line_idx % len(line_styles)],
                label=f"Core ML ({label})",
            )
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{prefix} latency over time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        if idx == len(prepared) - 1:
            ax.set_xlabel("Time (s)")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    annotate_figure(fig, signature, model_metadata)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def aggregate_timing_summary(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}

    def _accumulate(results: dict[str, dict], model_key: str, model_label: str) -> None:
        for key, result in results.items():
            if not isinstance(result, dict) or "error" in result:
                continue
            stats = result.get("timings", {}).get("stats", {})
            if not stats:
                continue
            label = result.get("compute_unit", key)
            unit_entry = aggregated.setdefault(
                key,
                {
                    "label": label,
                    "models": {},
                },
            )
            model_entry = unit_entry["models"].setdefault(
                model_key,
                {
                    "label": model_label,
                    "baseline_sum": 0.0,
                    "baseline_count": 0,
                    "inference_sum": 0.0,
                    "inference_count": 0,
                    "compile_sum": 0.0,
                    "warmup_sum": 0.0,
                    "audio_sum": 0.0,
                    "audio_count": 0,
                },
            )

            baseline = float(stats.get("baseline_total", float("nan")))
            if math.isfinite(baseline):
                model_entry["baseline_sum"] += baseline
                model_entry["baseline_count"] += 1

            inference = float(stats.get("coreml_total", float("nan")))
            if math.isfinite(inference):
                model_entry["inference_sum"] += inference
                model_entry["inference_count"] += 1

            compile_time = float(stats.get("compile_time", 0.0))
            if math.isfinite(compile_time):
                model_entry["compile_sum"] += compile_time

            warmup_time = float(stats.get("warmup_time", 0.0))
            if math.isfinite(warmup_time):
                model_entry["warmup_sum"] += warmup_time

            audio_seconds = float(stats.get("audio_seconds", float("nan")))
            if not math.isfinite(audio_seconds):
                processed_seconds = float(stats.get("processed_audio_seconds", float("nan")))
                if math.isfinite(processed_seconds):
                    audio_seconds = processed_seconds
                else:
                    window_seconds = float(result.get("window_seconds", 0.0))
                    chunk_count = len(result.get("timings", {}).get("baseline", []))
                    if window_seconds > 0.0 and chunk_count > 0:
                        audio_seconds = window_seconds * chunk_count

            if math.isfinite(audio_seconds) and audio_seconds > 0.0:
                model_entry["audio_sum"] += audio_seconds
                model_entry["audio_count"] += 1

    _accumulate(seg_results, "segmentation", "Segmentation")
    _accumulate(emb_results, "embedding", "Embedding")

    summary: dict[str, dict[str, Any]] = {}
    for key, entry in aggregated.items():
        if not entry.get("models"):
            continue

        unit_label = str(entry.get("label", key))
        models_summary: dict[str, dict[str, Any]] = {}
        combined_baselines: list[float] = []
        combined_inference: list[float] = []
        combined_compile: list[float] = []
        combined_warmup: list[float] = []
        combined_audio: list[float] = []

        for model_key, model_entry in entry["models"].items():
            baseline_total = (
                model_entry["baseline_sum"]
                if model_entry["baseline_count"] > 0
                else float("nan")
            )
            inference_total = (
                model_entry["inference_sum"]
                if model_entry["inference_count"] > 0
                else float("nan")
            )
            compile_total = model_entry["compile_sum"]
            warmup_total = model_entry["warmup_sum"]
            setup_total = compile_total + warmup_total
            coreml_total_with_setup = (
                inference_total + setup_total
                if math.isfinite(inference_total)
                else float("nan")
            )

            audio_total = (
                model_entry["audio_sum"]
                if model_entry["audio_count"] > 0
                else float("nan")
            )

            speedup = (
                baseline_total / inference_total
                if math.isfinite(baseline_total)
                and math.isfinite(inference_total)
                and inference_total > 0.0
                else float("nan")
            )
            speedup_with_setup = (
                baseline_total / (inference_total + setup_total)
                if math.isfinite(baseline_total)
                and math.isfinite(inference_total)
                and inference_total + setup_total > 0.0
                else float("nan")
            )

            rtf_baseline = (
                baseline_total / audio_total
                if math.isfinite(baseline_total)
                and math.isfinite(audio_total)
                and audio_total > 0.0
                else float("nan")
            )
            rtf_coreml = (
                inference_total / audio_total
                if math.isfinite(inference_total)
                and math.isfinite(audio_total)
                and audio_total > 0.0
                else float("nan")
            )
            rtf_coreml_setup = (
                coreml_total_with_setup / audio_total
                if math.isfinite(coreml_total_with_setup)
                and math.isfinite(audio_total)
                and audio_total > 0.0
                else float("nan")
            )

            model_summary = {
                "label": str(model_entry.get("label", model_key)),
                "baseline_total": baseline_total,
                "coreml_inference_total": inference_total,
                "coreml_compile_total": compile_total,
                "coreml_warmup_total": warmup_total,
                "coreml_total_with_setup": coreml_total_with_setup,
                "speedup": speedup,
                "speedup_with_setup": speedup_with_setup,
                "audio_seconds": audio_total,
                "rtf_baseline": rtf_baseline,
                "rtf_coreml": rtf_coreml,
                "rtf_coreml_with_setup": rtf_coreml_setup,
            }
            models_summary[model_key] = model_summary

            if math.isfinite(baseline_total):
                combined_baselines.append(baseline_total)
            if math.isfinite(inference_total):
                combined_inference.append(inference_total)
            if math.isfinite(compile_total):
                combined_compile.append(compile_total)
            if math.isfinite(warmup_total):
                combined_warmup.append(warmup_total)
            if math.isfinite(audio_total):
                combined_audio.append(audio_total)

        combined_audio_total = max(combined_audio) if combined_audio else float("nan")
        combined_baseline_total = float(sum(combined_baselines)) if combined_baselines else float("nan")
        combined_inference_total = float(sum(combined_inference)) if combined_inference else float("nan")
        combined_compile_total = float(sum(combined_compile)) if combined_compile else 0.0
        combined_warmup_total = float(sum(combined_warmup)) if combined_warmup else 0.0
        combined_setup_total = combined_compile_total + combined_warmup_total
        combined_coreml_with_setup = (
            combined_inference_total + combined_setup_total
            if math.isfinite(combined_inference_total)
            else float("nan")
        )

        combined_speedup = (
            combined_baseline_total / combined_inference_total
            if math.isfinite(combined_baseline_total)
            and math.isfinite(combined_inference_total)
            and combined_inference_total > 0.0
            else float("nan")
        )
        combined_speedup_setup = (
            combined_baseline_total / (combined_inference_total + combined_setup_total)
            if math.isfinite(combined_baseline_total)
            and math.isfinite(combined_inference_total)
            and combined_inference_total + combined_setup_total > 0.0
            else float("nan")
        )

        combined_rtf_baseline = (
            combined_baseline_total / combined_audio_total
            if math.isfinite(combined_baseline_total)
            and math.isfinite(combined_audio_total)
            and combined_audio_total > 0.0
            else float("nan")
        )
        combined_rtf_coreml = (
            combined_inference_total / combined_audio_total
            if math.isfinite(combined_inference_total)
            and math.isfinite(combined_audio_total)
            and combined_audio_total > 0.0
            else float("nan")
        )
        combined_rtf_coreml_setup = (
            combined_coreml_with_setup / combined_audio_total
            if math.isfinite(combined_coreml_with_setup)
            and math.isfinite(combined_audio_total)
            and combined_audio_total > 0.0
            else float("nan")
        )

        summary[key] = {
            "label": unit_label,
            "models": models_summary,
            "combined": {
                "label": "Pipeline total",
                "baseline_total": combined_baseline_total,
                "coreml_inference_total": combined_inference_total,
                "coreml_compile_total": combined_compile_total,
                "coreml_warmup_total": combined_warmup_total,
                "coreml_total_with_setup": combined_coreml_with_setup,
                "speedup": combined_speedup,
                "speedup_with_setup": combined_speedup_setup,
                "audio_seconds": combined_audio_total,
                "rtf_baseline": combined_rtf_baseline,
                "rtf_coreml": combined_rtf_coreml,
                "rtf_coreml_with_setup": combined_rtf_coreml_setup,
                "source": "components",
            },
        }

    return summary


def merge_pipeline_into_summary(
    summary: dict[str, dict[str, Any]],
    pipeline_results: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Overlay end-to-end pipeline timings onto the aggregated summary.

    This keeps component-level statistics intact while ensuring headline numbers
    (run summary, summary plots) reflect the full pipeline measurements.
    """

    if not summary or not pipeline_results:
        return summary

    baseline_entry = pipeline_results.get("baseline")
    if not isinstance(baseline_entry, dict):
        return summary

    baseline_timing = baseline_entry.get("timing", {})
    baseline_runtime = float(baseline_timing.get("runtime_seconds", float("nan")))
    baseline_audio = float(baseline_timing.get("audio_seconds", float("nan")))
    baseline_rtf = float(baseline_timing.get("rtf", float("nan")))

    if (not math.isfinite(baseline_audio) or baseline_audio <= 0.0) and math.isfinite(
        baseline_runtime
    ):
        baseline_audio = float("nan")

    for key, pipeline_entry in pipeline_results.items():
        if key == "baseline":
            continue
        if not isinstance(pipeline_entry, dict):
            continue

        timing = pipeline_entry.get("timing", {})
        runtime = float(timing.get("runtime_seconds", float("nan")))
        if not math.isfinite(runtime):
            continue

        compile_time = float(timing.get("compile_time", 0.0))
        warmup_time = float(timing.get("warmup_time", 0.0))
        total_with_setup = float(
            timing.get("runtime_with_setup", runtime + compile_time + warmup_time)
        )

        audio_seconds = float(timing.get("audio_seconds", baseline_audio))
        if (not math.isfinite(audio_seconds) or audio_seconds <= 0.0) and math.isfinite(
            baseline_audio
        ):
            audio_seconds = baseline_audio

        rtf_coreml = float(
            timing.get(
                "rtf",
                runtime / audio_seconds
                if math.isfinite(runtime)
                and math.isfinite(audio_seconds)
                and audio_seconds > 0.0
                else float("nan"),
            )
        )
        rtf_coreml_setup = float(
            timing.get(
                "rtf_with_setup",
                total_with_setup / audio_seconds
                if math.isfinite(total_with_setup)
                and math.isfinite(audio_seconds)
                and audio_seconds > 0.0
                else float("nan"),
            )
        )

        if not math.isfinite(baseline_rtf) and math.isfinite(baseline_runtime) and math.isfinite(
            audio_seconds
        ) and audio_seconds > 0.0:
            baseline_rtf = baseline_runtime / audio_seconds

        unit_entry = summary.setdefault(
            key,
            {
                "label": pipeline_entry.get("compute_unit", key),
                "models": {},
            },
        )

        combined = {
            "label": "Pipeline total",
            "baseline_total": baseline_runtime,
            "coreml_inference_total": runtime,
            "coreml_compile_total": compile_time,
            "coreml_warmup_total": warmup_time,
            "coreml_total_with_setup": total_with_setup,
            "speedup": (
                baseline_runtime / runtime
                if math.isfinite(baseline_runtime)
                and math.isfinite(runtime)
                and runtime > 0.0
                else float("nan")
            ),
            "speedup_with_setup": (
                baseline_runtime / total_with_setup
                if math.isfinite(baseline_runtime)
                and math.isfinite(total_with_setup)
                and total_with_setup > 0.0
                else float("nan")
            ),
            "audio_seconds": audio_seconds,
            "rtf_baseline": baseline_rtf,
            "rtf_coreml": rtf_coreml,
            "rtf_coreml_with_setup": rtf_coreml_setup,
            "source": "pipeline",
        }

        unit_entry["combined"] = combined

        pipeline_model_summary = {
            "label": "Pipeline",
            "baseline_total": baseline_runtime,
            "coreml_inference_total": runtime,
            "coreml_compile_total": compile_time,
            "coreml_warmup_total": warmup_time,
            "coreml_total_with_setup": total_with_setup,
            "speedup": combined["speedup"],
            "speedup_with_setup": combined["speedup_with_setup"],
            "audio_seconds": audio_seconds,
            "rtf_baseline": baseline_rtf,
            "rtf_coreml": rtf_coreml,
            "rtf_coreml_with_setup": rtf_coreml_setup,
        }
        unit_entry.setdefault("models", {})["pipeline"] = pipeline_model_summary

    return summary


def plot_summary_speed(results: dict[str, dict], output_path: Path, signature: str,
    model_metadata: dict[str, dict[str, str]] | None = None) -> None:
    if not results:
        return

    ordered_keys: list[str] = []
    for candidate in _COMPUTE_UNIT_ORDER:
        if candidate in results:
            ordered_keys.append(candidate)
    for key in results:
        if key not in ordered_keys:
            ordered_keys.append(key)

    model_order: list[str] = []
    for candidate in _MODEL_ORDER:
        for unit_key in ordered_keys:
            unit_models = results.get(unit_key, {}).get("models", {})
            if candidate in unit_models and candidate not in model_order:
                model_order.append(candidate)
    for unit_key in ordered_keys:
        unit_models = results.get(unit_key, {}).get("models", {})
        for model_key in unit_models:
            if model_key not in model_order:
                model_order.append(model_key)

    entries: list[dict[str, Any]] = []
    for unit_key in ordered_keys:
        unit = results[unit_key]
        unit_label = unit.get("label", unit_key)
        model_map = unit.get("models", {})
        for model_key in model_order:
            model_data = model_map.get(model_key)
            if not model_data:
                continue
            entries.append(
                {
                    "unit_key": unit_key,
                    "unit_label": unit_label,
                    "model_key": model_key,
                    "model_label": model_data.get("label", model_key.title()),
                    "baseline_total": float(model_data.get("baseline_total", float("nan"))),
                    "coreml_inference_total": float(model_data.get("coreml_inference_total", float("nan"))),
                    "coreml_compile_total": float(model_data.get("coreml_compile_total", 0.0)),
                    "coreml_warmup_total": float(model_data.get("coreml_warmup_total", 0.0)),
                    "speedup": float(model_data.get("speedup", float("nan"))),
                    "speedup_with_setup": float(model_data.get("speedup_with_setup", float("nan"))),
                }
            )

    if not entries:
        return

    labels = [f"{entry['unit_label']} - {entry['model_label']}" for entry in entries]
    baseline_arr = np.array([entry["baseline_total"] for entry in entries], dtype=np.float32)
    inference_arr = np.array([entry["coreml_inference_total"] for entry in entries], dtype=np.float32)
    compile_arr = np.array([entry["coreml_compile_total"] for entry in entries], dtype=np.float32)
    warmup_arr = np.array([entry["coreml_warmup_total"] for entry in entries], dtype=np.float32)
    plot_baseline = np.nan_to_num(baseline_arr, nan=0.0)
    plot_inference = np.nan_to_num(inference_arr, nan=0.0)
    plot_compile = np.nan_to_num(compile_arr, nan=0.0)
    plot_warmup = np.nan_to_num(warmup_arr, nan=0.0)
    speedups = [entry["speedup"] for entry in entries]
    speedups_with_setup = [entry["speedup_with_setup"] for entry in entries]

    fig_width = max(10.0, 2.4 * len(labels))
    fig, ax_runtime = plt.subplots(1, 1, figsize=(fig_width, 4.8))

    x = np.arange(len(labels), dtype=np.float32)
    width = 0.25
    baseline_offset = -width
    compile_offset = 0.0
    inference_offset = width

    compile_color = "#9edae5"
    warmup_color = "#ffbb78"
    inference_color = "#1f77b4"

    ax_runtime.bar(
        x + baseline_offset,
        plot_baseline,
        width,
        label="PyTorch baseline (mps)",
    )
    ax_runtime.bar(
        x + compile_offset,
        plot_compile,
        width,
        label="Core ML compile",
        color=compile_color,
    )
    ax_runtime.bar(
        x + inference_offset,
        plot_warmup,
        width,
        label="Core ML warmup",
        color=warmup_color,
    )
    ax_runtime.bar(
        x + inference_offset,
        plot_inference,
        width,
        bottom=plot_warmup,
        label="Core ML inference",
        color=inference_color,
    )

    ax_runtime.set_title("Total runtime by model and compute unit")
    ax_runtime.set_ylabel("Total latency (s) — lower is better")
    ax_runtime.set_xticks(x)
    ax_runtime.set_xticklabels(labels, rotation=25, ha="right")
    ax_runtime.grid(True, axis="y", alpha=0.3)

    handles, legend_labels = ax_runtime.get_legend_handles_labels()
    unique_handles: dict[str, Any] = {}
    for handle, label in zip(handles, legend_labels):
        if label not in unique_handles:
            unique_handles[label] = handle
    ax_runtime.legend(unique_handles.values(), unique_handles.keys())

    for idx, (speedup, speedup_setup) in enumerate(zip(speedups, speedups_with_setup)):
        compile_time = plot_compile[idx]
        warmup_time = plot_warmup[idx]
        inference = plot_inference[idx]
        runtime_total = warmup_time + inference
        setup_total = compile_time + runtime_total

        text_x = x[idx] + inference_offset
        if math.isfinite(speedup) and runtime_total > 0.0:
            ax_runtime.text(
                text_x,
                runtime_total * 1.04,
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#1f77b4",
            )
        if (
            math.isfinite(speedup_setup)
            and math.isfinite(setup_total)
            and setup_total > 0.0
        ):
            ax_runtime.text(
                text_x,
                setup_total * 1.16,
                f"{speedup_setup:.2f}x incl",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )

    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    annotate_figure(fig, signature, model_metadata)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pipeline_overview(
    pipeline_results: dict[str, dict[str, Any]] | None,
    output_path: Path,
    signature: str,
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    if not pipeline_results:
        return

    baseline = pipeline_results.get("baseline")
    if not baseline:
        return

    baseline_segments = baseline.get("segments") or []
    baseline_exclusive_segments = baseline.get("exclusive_segments") or []

    candidate_entries: list[dict[str, Any]] = []
    for key, entry in pipeline_results.items():
        if key == "baseline" or not isinstance(entry, dict):
            continue
        if entry.get("error"):
            continue
        segments = entry.get("segments") or []
        exclusive_segments = entry.get("exclusive_segments") or []
        if not segments and not exclusive_segments:
            continue
        candidate_entries.append(
            {
                "key": key,
                "label": entry.get("compute_unit", key),
                "segments": segments,
                "exclusive_segments": exclusive_segments,
                "alignment": entry.get("alignment") or [],
                "exclusive_alignment": entry.get("exclusive_alignment") or [],
                "der": float(entry.get("der_vs_baseline", float("nan"))),
                "exclusive_der": float(
                    entry.get("exclusive_der_vs_baseline", float("nan"))
                ),
                "jer": float(entry.get("jer_vs_baseline", float("nan"))),
                "exclusive_jer": float(
                    entry.get("exclusive_jer_vs_baseline", float("nan"))
                ),
            }
        )

    if not candidate_entries:
        return

    speaker_labels: set[str] = set()
    for _, _, label in baseline_segments:
        speaker_labels.add(label)
    for _, _, label in baseline_exclusive_segments:
        speaker_labels.add(label)
    for entry in candidate_entries:
        for _, _, label in entry["segments"]:  # type: ignore[index]
            speaker_labels.add(label)
        for _, _, label in entry["exclusive_segments"]:  # type: ignore[index]
            speaker_labels.add(label)

    if not speaker_labels:
        return

    cmap = plt.get_cmap("tab20")
    speaker_order = sorted(speaker_labels)
    color_map = {label: cmap(index % cmap.N) for index, label in enumerate(speaker_order)}

    # Preserve compute unit ordering when possible.
    ordered_entries: list[dict[str, Any]] = []
    for candidate in _PIPELINE_DEVICE_ORDER:
        for entry in candidate_entries:
            if entry["key"] == candidate and entry not in ordered_entries:
                ordered_entries.append(entry)
    for entry in candidate_entries:
        if entry not in ordered_entries:
            ordered_entries.append(entry)

    timeline_end = 0.0

    def _update_timeline_from_segments(segments: list[tuple[float, float, str]]) -> None:
        nonlocal timeline_end
        for start, end, _ in segments:
            if math.isfinite(end):
                timeline_end = max(timeline_end, float(end))

    def _update_timeline_from_alignment(alignment: list[dict[str, Any]]) -> None:
        nonlocal timeline_end
        for window in alignment:
            end_val = float(window.get("end", 0.0))
            if math.isfinite(end_val):
                timeline_end = max(timeline_end, end_val)

    _update_timeline_from_segments(baseline_segments)
    _update_timeline_from_segments(baseline_exclusive_segments)
    for entry in ordered_entries:
        _update_timeline_from_segments(entry["segments"])  # type: ignore[arg-type]
        _update_timeline_from_segments(entry["exclusive_segments"])  # type: ignore[arg-type]
        _update_timeline_from_alignment(entry["alignment"])  # type: ignore[arg-type]
        _update_timeline_from_alignment(entry["exclusive_alignment"])  # type: ignore[arg-type]

    baseline_annotation_obj = baseline.get("annotation")
    if timeline_end <= 0.0 and isinstance(baseline_annotation_obj, Annotation):
        timeline = baseline_annotation_obj.get_timeline()
        if timeline and timeline.extent() is not None:
            extent = timeline.extent()
            end_val = float(getattr(extent, "end", 0.0))
            if math.isfinite(end_val):
                timeline_end = max(timeline_end, end_val)

    if timeline_end <= 0.0:
        timeline_end = 1.0

    timeline_rows: list[dict[str, Any]] = []
    if baseline_segments:
        timeline_rows.append(
            {
                "title": "PyTorch baseline (speaker_diarization)",
                "segments": baseline_segments,
                "mismatches": [],
            }
        )
    if baseline_exclusive_segments:
        timeline_rows.append(
            {
                "title": "PyTorch baseline (exclusive_speaker_diarization)",
                "segments": baseline_exclusive_segments,
                "mismatches": [],
            }
        )

    for entry in ordered_entries:
        label = entry.get("label", entry.get("key"))
        der_value = entry.get("der", float("nan"))
        jer_value = entry.get("jer", float("nan"))
        inclusive_title = str(label)
        metrics_bits: list[str] = []
        if isinstance(der_value, float) and math.isfinite(der_value):
            metrics_bits.append(f"DER {der_value * 100:.2f}%")
        if isinstance(jer_value, float) and math.isfinite(jer_value):
            metrics_bits.append(f"JER {jer_value * 100:.2f}%")
        if metrics_bits:
            inclusive_title = f"{label} — {' | '.join(metrics_bits)}"
        if entry.get("segments"):
            alignment = entry.get("alignment") or []
            mismatch_spans = [
                (float(window.get("start", 0.0)), float(window.get("end", 0.0)))
                for window in alignment
                if not window.get("match", False)
            ]
            timeline_rows.append(
                {
                    "title": f"{inclusive_title} (speaker_diarization)",
                    "segments": entry["segments"],
                    "mismatches": mismatch_spans,
                }
            )

        exclusive_segments = entry.get("exclusive_segments") or []
        if exclusive_segments:
            exclusive_der = entry.get("exclusive_der", float("nan"))
            exclusive_jer = entry.get("exclusive_jer", float("nan"))
            exclusive_title = str(label)
            exclusive_bits: list[str] = []
            if isinstance(exclusive_der, float) and math.isfinite(exclusive_der):
                exclusive_bits.append(f"exclusive DER {exclusive_der * 100:.2f}%")
            if isinstance(exclusive_jer, float) and math.isfinite(exclusive_jer):
                exclusive_bits.append(f"exclusive JER {exclusive_jer * 100:.2f}%")
            if exclusive_bits:
                exclusive_title = f"{label} — {' | '.join(exclusive_bits)}"
            exclusive_alignment = entry.get("exclusive_alignment") or []
            exclusive_mismatches = [
                (float(window.get("start", 0.0)), float(window.get("end", 0.0)))
                for window in exclusive_alignment
                if not window.get("match", False)
            ]
            timeline_rows.append(
                {
                    "title": f"{exclusive_title} (exclusive_speaker_diarization)",
                    "segments": exclusive_segments,
                    "mismatches": exclusive_mismatches,
                }
            )

    if not timeline_rows:
        return

    num_rows = len(timeline_rows)
    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(11.5, 2.2 + 2.1 * num_rows),
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    def _plot_row(
        axis: plt.Axes,
        segments: list[tuple[float, float, str]],
        title: str,
        mismatch_spans: list[tuple[float, float]] | None = None,
    ) -> None:
        if mismatch_spans:
            for start, end in mismatch_spans:
                if end <= start:
                    continue
                axis.axvspan(start, end, color="#fddede", alpha=0.45, lw=0.0, zorder=0)

        for start, end, label in segments:
            width = float(end) - float(start)
            if width <= 0.0:
                continue
            color = color_map.get(label, "#b0b0b0")
            axis.barh(
                y=0.5,
                width=width,
                left=float(start),
                height=0.6,
                align="center",
                color=color,
                edgecolor="none",
                zorder=2,
            )

        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xlim(0, timeline_end)
        axis.set_title(title, loc="left", fontsize=11)
        axis.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.8)

    for axis, row in zip(axes, timeline_rows):
        _plot_row(
            axis,
            row.get("segments", []),  # type: ignore[arg-type]
            row.get("title", ""),
            mismatch_spans=row.get("mismatches"),
        )

    axes[-1].set_xlabel("Time (s)")

    legend_patches = [
        Patch(facecolor=color_map[label], edgecolor="none", label=label)
        for label in speaker_order
    ]
    if legend_patches:
        fig.legend(
            legend_patches,
            [patch.get_label() for patch in legend_patches],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=max(1, min(len(legend_patches), 5)),
            frameon=False,
        )
        tight_rect = [0.04, 0.06, 1, 0.94]
    else:
        tight_rect = [0.04, 0.06, 1, 0.98]

    fig.tight_layout(rect=tight_rect)
    annotate_figure(fig, signature, model_metadata)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_plda_parity_checks(
    emb_results: dict[str, dict],
    output_path: Path,
    signature: str,
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    units_with_checks: dict[str, dict[str, ComparisonResult]] = {}
    unit_labels: dict[str, str] = {}

    for key, result in emb_results.items():
        checks = result.get("plda_checks")
        if not checks:
            continue
        units_with_checks[key] = checks
        unit_labels[key] = result.get("compute_unit", key)

    if not units_with_checks:
        return

    # Prefer ALL compute unit for PLDA checks
    selected_key = None
    if "ALL" in units_with_checks:
        selected_key = "ALL"
    else:
        # Fallback to first available
        selected_key = next(iter(units_with_checks))

    selected_checks = units_with_checks[selected_key]
    selected_label = unit_labels.get(selected_key, selected_key)

    check_names = list(selected_checks.keys())
    if not check_names:
        return

    metric_columns: list[tuple[str, str, str]] = [
        ("MSE", "mse", "{:.3e}"),
        ("MAE", "mae", "{:.3e}"),
        ("Max Abs", "max_abs", "{:.3e}"),
        ("Corr", "corr", "{:.6f}"),
    ]

    cosine_present = any(comp.cosine is not None for comp in selected_checks.values())
    if cosine_present:
        metric_columns.append(("Cosine", "cosine", "{:.6f}"))

    cell_text: list[list[str]] = []
    numeric_matrix: list[list[float | None]] = []
    for name in check_names:
        comp = selected_checks[name]
        row_text: list[str] = []
        row_vals: list[float | None] = []
        for _, attr, fmt in metric_columns:
            value = getattr(comp, attr, None)
            if value is None or not math.isfinite(float(value)):
                row_text.append("—")
                row_vals.append(None)
            else:
                numeric = float(value)
                row_text.append(fmt.format(numeric))
                row_vals.append(numeric)
        cell_text.append(row_text)
        numeric_matrix.append(row_vals)

    fig_height = 1.4 + 0.45 * len(check_names)
    fig_width = max(6.5, 1.8 * len(metric_columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    col_labels = [label for label, _, _ in metric_columns]
    table = ax.table(
        cellText=cell_text,
        rowLabels=[name.replace("_", " ") for name in check_names],
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    # Apply a simple shading to highlight lower error magnitudes and high correlations.
    error_cols = {idx for idx, (label, _, _) in enumerate(metric_columns) if label in {"MSE", "MAE", "Max Abs"}}
    corr_cols = {idx for idx, (label, _, _) in enumerate(metric_columns) if label == "Corr"}
    cosine_cols = {idx for idx, (label, _, _) in enumerate(metric_columns) if label == "Cosine"}

    def _shade_error(value: float | None, max_ref: float) -> str:
        if value is None or max_ref <= 0.0:
            return "#f0f0f0"
        # Map smaller errors to deeper green.
        ratio = min(1.0, float(value) / max_ref)
        intensity = 0.85 - 0.6 * (1.0 - ratio)
        return plt.cm.Greens(max(0.0, min(1.0, intensity)))

    def _shade_corr(value: float | None) -> str:
        if value is None:
            return "#f0f0f0"
        # Values near 1 are bright green, near 0 are neutral, negative are reddish.
        norm = (float(value) + 1.0) / 2.0  # [-1,1] -> [0,1]
        return plt.cm.RdYlGn(norm)

    max_error = 0.0
    for row_vals in numeric_matrix:
        for idx in error_cols:
            value = row_vals[idx]
            if value is not None:
                max_error = max(max_error, float(value))

    for row_idx, row_vals in enumerate(numeric_matrix):
        for col_idx, value in enumerate(row_vals):
            cell = table[row_idx + 1, col_idx]
            if col_idx in error_cols:
                cell.set_facecolor(_shade_error(value, max_error))
            elif col_idx in corr_cols or col_idx in cosine_cols:
                cell.set_facecolor(_shade_corr(value))
            else:
                cell.set_facecolor("#ffffff")
            cell.set_edgecolor("#d0d0d0")

    # Style header cells for readability.
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0 or col_idx == -1:
            cell.set_facecolor("#e8e8e8")
            cell.set_edgecolor("#b0b0b0")
            cell.set_text_props(weight="bold")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)
    try:
        table.auto_set_column_width(list(range(len(metric_columns))))
    except AttributeError:  # pragma: no cover - older matplotlib
        pass

    ax.set_title(f"PLDA parity (Core ML vs PyTorch) — {selected_label}")
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    annotate_figure(fig, signature, model_metadata)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_plda_coreml_comparison(
    plda_results: dict[str, dict],
    output_path: Path,
    signature: str,
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    """Plot PLDA CoreML model comparison against reference implementation."""
    
    if not plda_results:
        return
    
    # Select best compute unit for plotting
    selected_key = None
    if "ALL" in plda_results:
        selected_key = "ALL"
    elif "CPU_AND_NE" in plda_results:
        selected_key = "CPU_AND_NE"
    else:
        selected_key = next(iter(plda_results), None)
    
    if selected_key is None or "error" in plda_results.get(selected_key, {}):
        return
    
    result = plda_results[selected_key]
    compute_unit_label = result.get("compute_unit", selected_key)
    
    # Extract comparison results
    features_comp = result.get("features_comparison")
    rho_comp = result.get("rho_comparison")
    scores_comp = result.get("scores_comparison")
    per_dim_corr = result.get("per_dim_corr", [])
    
    if not all([features_comp, rho_comp, scores_comp]):
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. Comparison metrics table
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")
    
    comparisons = {
        "PLDA Features": features_comp,
        "Rho (scaled)": rho_comp,
        "Score Matrix": scores_comp,
    }
    
    metric_columns = [("MSE", "mse", "{:.3e}"), ("MAE", "mae", "{:.3e}"), 
                     ("Max Abs", "max_abs", "{:.3e}"), ("Corr", "corr", "{:.6f}"),
                     ("Cosine", "cosine", "{:.6f}")]
    
    cell_text = []
    for name, comp in comparisons.items():
        row = []
        for _, attr, fmt in metric_columns:
            value = getattr(comp, attr, None)
            if value is None or not math.isfinite(float(value)):
                row.append("—")
            else:
                row.append(fmt.format(float(value)))
        cell_text.append(row)
    
    table = ax_table.table(
        cellText=cell_text,
        rowLabels=list(comparisons.keys()),
        colLabels=[label for label, _, _ in metric_columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    
    # Color cells based on values
    for row_idx in range(len(comparisons)):
        for col_idx in range(len(metric_columns)):
            cell = table[row_idx + 1, col_idx]
            metric_name = metric_columns[col_idx][0]
            if metric_name in ["Corr", "Cosine"]:
                # High correlation is good (green)
                cell.set_facecolor("#d4edda")
            elif metric_name in ["MSE", "MAE", "Max Abs"]:
                # Low error is good (green)
                cell.set_facecolor("#d1ecf1")
            cell.set_edgecolor("#dee2e6")
    
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0 or col_idx == -1:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
    
    ax_table.set_title(f"PLDA CoreML Model Accuracy — {compute_unit_label}", 
                      fontsize=12, weight="bold", pad=15)
    
    # 2. Per-dimension correlation heatmap
    if per_dim_corr:
        ax_corr = fig.add_subplot(gs[1, :])
        corr_array = np.array(per_dim_corr)
        finite_mask = np.isfinite(corr_array)
        
        if np.any(finite_mask):
            im = ax_corr.imshow(
                corr_array.reshape(1, -1),
                aspect="auto",
                cmap="RdYlGn",
                vmin=0.95,
                vmax=1.0,
                interpolation="nearest",
            )
            ax_corr.set_yticks([])
            ax_corr.set_xlabel("PLDA Dimension")
            ax_corr.set_title("Per-Dimension Correlation (Reference vs CoreML)", fontsize=11)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax_corr, orientation="horizontal", pad=0.1, aspect=30)
            cbar.set_label("Correlation")
            
            # Add statistics text
            mean_corr = np.mean(corr_array[finite_mask])
            min_corr = np.min(corr_array[finite_mask])
            stats_text = f"Mean: {mean_corr:.6f} | Min: {min_corr:.6f}"
            ax_corr.text(0.02, 0.98, stats_text, transform=ax_corr.transAxes,
                        va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round",
                        facecolor="white", alpha=0.8))
    
    # 3. Timing comparison
    timings = result.get("timings", {}).get("stats", {})
    if timings:
        ax_time = fig.add_subplot(gs[2, 0])
        
        ref_total = float(timings.get("reference_total", 0))
        coreml_total = float(timings.get("coreml_total", 0))
        
        if math.isfinite(ref_total) and math.isfinite(coreml_total):
            categories = ["Reference\n(NumPy)", "CoreML"]
            times = [ref_total * 1000, coreml_total * 1000]  # Convert to ms
            colors = ["#6c757d", "#1f77b4"]
            
            bars = ax_time.bar(categories, times, color=colors, alpha=0.7)
            ax_time.set_ylabel("Total Latency (ms)")
            ax_time.set_title("Latency Comparison", fontsize=11)
            ax_time.grid(True, axis="y", alpha=0.3)
            
            # Add speedup annotation
            speedup = float(timings.get("speedup", 0))
            if math.isfinite(speedup) and speedup > 0:
                ax_time.text(0.5, 0.95, f"Speedup: {speedup:.2f}×",
                           transform=ax_time.transAxes, ha="center", va="top",
                           fontsize=10, weight="bold",
                           bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))
    
    # 4. Summary statistics
    ax_summary = fig.add_subplot(gs[2, 1])
    ax_summary.axis("off")
    
    num_embeddings = result.get("num_embeddings", 0)
    embedding_dim = result.get("embedding_dim", 0)
    plda_dim = result.get("plda_dim", 0)
    
    summary_text = f"""Model Configuration:
    
Input embeddings: {num_embeddings} × {embedding_dim}
Output PLDA features: {num_embeddings} × {plda_dim}

Accuracy Summary:
• Features MSE: {getattr(features_comp, 'mse', 0):.3e}
• Features Correlation: {getattr(features_comp, 'corr', 0):.6f}
• Features Cosine Sim: {getattr(features_comp, 'cosine', 0):.6f}

• Score Matrix MSE: {getattr(scores_comp, 'mse', 0):.3e}
• Score Matrix Corr: {getattr(scores_comp, 'corr', 0):.6f}
"""
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, family="monospace", va="top",
                   bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.8))
    
    fig.suptitle("PLDA CoreML Model Validation", fontsize=14, weight="bold", y=0.98)
    annotate_figure(fig, signature, model_metadata)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_timing(
    pipeline_results: dict[str, dict[str, Any]] | None,
    output_path: Path,
    signature: str,
    model_metadata: dict[str, dict[str, str]] | None = None,
) -> None:
    if not pipeline_results:
        return

    baseline = pipeline_results.get("baseline")
    if not isinstance(baseline, dict):
        return

    baseline_timing = baseline.get("timing", {})
    baseline_runtime = float(baseline_timing.get("runtime_seconds", float("nan")))
    baseline_rtf = float(baseline_timing.get("rtf", float("nan")))
    baseline_rtfx = float(baseline_timing.get("rtfx", float("nan")))
    if not math.isfinite(baseline_runtime):
        return

    def _build_entry(
        label: str,
        runtime: float,
        rtf: float,
        *,
        color: str,
        error: str | None = None,
        rtfx: float = float("nan"),
        setup_time: float = 0.0,
    ) -> dict[str, Any]:
        runtime_delta = (
            runtime - baseline_runtime
            if math.isfinite(runtime) and math.isfinite(baseline_runtime)
            else float("nan")
        )
        speedup = (
            baseline_runtime / runtime
            if math.isfinite(baseline_runtime)
            and math.isfinite(runtime)
            and runtime > 0.0
            else float("nan")
        )
        rtf_delta = (
            rtf - baseline_rtf
            if math.isfinite(rtf) and math.isfinite(baseline_rtf)
            else float("nan")
        )
        rtf_ratio = (
            baseline_rtf / rtf
            if math.isfinite(baseline_rtf)
            and math.isfinite(rtf)
            and rtf != 0.0
            else float("nan")
        )
        if not math.isfinite(rtfx) and math.isfinite(rtf) and rtf > 0.0:
            rtfx = 1.0 / rtf
        return {
            "label": label,
            "runtime": runtime,
            "rtf": rtf,
            "runtime_delta": runtime_delta,
            "speedup": speedup,
            "rtf_delta": rtf_delta,
            "rtf_ratio": rtf_ratio,
            "rtfx": rtfx,
            "color": color,
            "error": error,
            "setup_time": setup_time,
        }

    categories: list[dict[str, Any]] = []
    categories.append(
        _build_entry(
            "PyTorch (MPS)",
            baseline_runtime,
            baseline_rtf,
            color="#7f7f7f",
            rtfx=baseline_rtfx,
        )
    )

    mps_entry = pipeline_results.get("TORCH_MPS")
    if isinstance(mps_entry, dict):
        timing = mps_entry.get("timing", {})
        runtime = float(timing.get("runtime_seconds", float("nan")))
        rtf = float(timing.get("rtf", float("nan")))
        rtfx = float(timing.get("rtfx", float("nan")))
        label = str(mps_entry.get("compute_unit", "PyTorch MPS"))
        categories.append(
            _build_entry(
                label,
                runtime,
                rtf,
                color="#17becf",
                error=str(mps_entry.get("error")) if mps_entry.get("error") else None,
                rtfx=rtfx,
            )
        )

    # Plot all CoreML compute units (not just the best one)
    coreml_color_map = {
        "ALL": "#1f77b4",  # Blue
        "CPU_AND_NE": "#2ca02c",  # Green
        "CPU_AND_GPU": "#ff7f0e",  # Orange
        "CPU_ONLY": "#d62728",  # Red
    }

    for key, entry in pipeline_results.items():
        if key in ("baseline", "TORCH_MPS"):
            continue
        if not isinstance(entry, dict) or entry.get("error"):
            continue
        timing = entry.get("timing", {})
        runtime = float(timing.get("runtime_seconds", float("nan")))
        if not math.isfinite(runtime):
            continue

        label = entry.get("compute_unit", key)
        color = coreml_color_map.get(key, "#9467bd")  # Purple as default
        rtf = float(timing.get("rtf", float("nan")))
        rtfx = float(timing.get("rtfx", float("nan")))
        setup_time = float(timing.get("setup_time", 0.0))

        categories.append(
            _build_entry(
                f"Core ML ({label})",
                runtime,
                rtf,
                color=color,
                rtfx=rtfx,
                setup_time=setup_time,
            )
        )

    if len(categories) <= 1:
        return

    labels = [entry["label"] for entry in categories]
    runtimes = np.array([entry["runtime"] for entry in categories], dtype=np.float32)
    plot_runtimes = np.nan_to_num(runtimes, nan=0.0)
    colors = [entry["color"] for entry in categories]

    fig_width = max(8.0, 2.4 * len(categories))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 4.8))
    x_vals = np.arange(len(categories), dtype=np.float32)

    bars = ax.bar(x_vals, plot_runtimes, color=colors, width=0.55)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("End-to-end pipeline latency")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, entry in zip(bars, categories):
        runtime = entry["runtime"]
        rtf = entry["rtf"]
        rtfx = float(entry.get("rtfx", float("nan")))
        setup_time = float(entry.get("setup_time", 0.0))

        text_lines = [f"{runtime:.3f}s"] if math.isfinite(runtime) else []

        # Add setup time annotation for CoreML models
        if math.isfinite(setup_time) and setup_time > 0.0:
            text_lines.append(f"(+{setup_time:.3f}s setup)")

        if math.isfinite(rtf):
            suffix = f"RTF {rtf:.3f}"
            if math.isfinite(rtfx):
                suffix += f" | RTFx {rtfx:.2f}"
            ratio = entry.get("rtf_ratio")
            if math.isfinite(ratio):
                suffix += f" | ×{ratio:.2f}"
            text_lines.append(suffix)
        runtime_delta = entry.get("runtime_delta")
        if math.isfinite(runtime_delta) and not math.isclose(runtime_delta, 0.0, abs_tol=1e-6):
            speedup = entry.get("speedup")
            delta_str = f"Δ {runtime_delta:+.3f}s"
            if math.isfinite(speedup):
                delta_str += f" / {speedup:.2f}x"
            text_lines.append(delta_str)
        error_msg = entry.get("error")
        if error_msg:
            text_lines.append("error")

        if text_lines:
            height = bar.get_height()
            label_y = height * 1.02 if height > 0 else 0.05
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                "\n".join(text_lines),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )

    legend_items = [Patch(color="#7f7f7f", label="PyTorch (MPS)")]
    if any(entry["label"].startswith("PyTorch MPS") for entry in categories):
        legend_items.append(Patch(color="#17becf", label="PyTorch MPS"))

    # Add legend for each CoreML compute unit type that appears
    coreml_legend_map = {
        "Core ML (All)": ("#1f77b4", "Core ML ALL"),
        "Core ML (CPU And NE)": ("#2ca02c", "Core ML CPU+ANE"),
        "Core ML (CPU And GPU)": ("#ff7f0e", "Core ML CPU+GPU"),
        "Core ML (CPU Only)": ("#d62728", "Core ML CPU"),
    }
    for label in labels:
        for pattern, (color, legend_label) in coreml_legend_map.items():
            if label.startswith(pattern) and not any(item.get_label() == legend_label for item in legend_items):
                legend_items.append(Patch(color=color, label=legend_label))

    if legend_items:
        ax.legend(handles=legend_items, loc="upper right")

    annotate_figure(fig, signature, model_metadata)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_plots(
    seg_results: dict[str, dict],
    emb_results: dict[str, dict],
    pipeline_results: dict[str, dict[str, Any]] | None,
    base_dir: Path,
    aggregated_summary: dict[str, dict[str, Any]] | None = None,
    plda_results: dict[str, dict] | None = None,
    *,
    specs: dict[str, Any] | None = None,
    audio_seconds: float | None = None,
) -> str:
    plots_dir = ensure_plots_dir(base_dir)
    signature = build_plot_signature(
        base_dir,
        specs=specs,
        audio_seconds=audio_seconds,
    )

    # Generate timestamp and commit hash for filenames
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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

    filename_suffix = f"_{timestamp}_{git_hash}"

    # Extract model metadata
    coreml_dir = base_dir / "coreml_models"
    model_metadata = extract_model_metadata(coreml_dir) if coreml_dir.exists() else None

    plot_combined_metric_grid(
        seg_results,
        emb_results,
        plots_dir / f"metrics_timeseries{filename_suffix}.png",
        signature,
        model_metadata,
    )
    plot_latency_over_time(
        seg_results,
        emb_results,
        plots_dir / f"latency_timeseries{filename_suffix}.png",
        signature,
        model_metadata,
    )
    aggregated = aggregated_summary or aggregate_timing_summary(seg_results, emb_results)
    plot_summary_speed(aggregated, plots_dir / f"summary_speed{filename_suffix}.png", signature, model_metadata)
    plot_plda_parity_checks(emb_results, plots_dir / f"plda_parity{filename_suffix}.png", signature, model_metadata)

    if plda_results:
        plot_plda_coreml_comparison(plda_results, plots_dir / f"plda_coreml{filename_suffix}.png", signature, model_metadata)

    plot_pipeline_overview(
        pipeline_results,
        plots_dir / f"pipeline_overview{filename_suffix}.png",
        signature,
        model_metadata,
    )
    plot_pipeline_timing(
        pipeline_results,
        plots_dir / f"pipeline_timing{filename_suffix}.png",
        signature,
        model_metadata,
    )
    return signature


def human_readable_bytes(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def human_readable_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0.0:
        return "0s"
    remaining = seconds
    hours = int(remaining // 3600)
    remaining -= hours * 3600
    minutes = int(remaining // 60)
    remaining -= minutes * 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m {remaining:05.2f}s"
    if minutes > 0:
        return f"{minutes}m {remaining:05.2f}s"
    return f"{remaining:.2f}s"


def collect_machine_specs() -> dict[str, Any]:
    uname = platform.uname()
    cpu_name = uname.processor or platform.processor() or "Unknown CPU"
    system = uname.system or "Unknown"

    if system.lower() == "darwin":  # macOS: fetch the Apple Silicon marketing name.
        try:
            brand = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
            if brand:
                cpu_name = brand
        except Exception:  # pragma: no cover - platform-specific
            pass

    cpu_count = os.cpu_count() or 1
    machine = uname.machine or "unknown"
    system_label = f"{uname.system} {uname.release}".strip()

    total_memory_gb: float | None = None
    if psutil is not None:
        try:
            total_memory_gb = float(psutil.virtual_memory().total) / (1024 ** 3)
        except (psutil.Error, AttributeError):  # pragma: no cover - psutil edge cases
            total_memory_gb = None

    if total_memory_gb is None:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_memory_gb = float(pages * page_size) / (1024 ** 3)
        except (AttributeError, ValueError, OSError):  # pragma: no cover - platform dependent
            total_memory_gb = None

    return {
        "system": system_label,
        "machine": machine,
        "cpu": cpu_name,
        "cpu_count": cpu_count,
        "memory_gb": total_memory_gb,
    }


def print_run_summary(
    audio_path: Path,
    duration_seconds: float,
    aggregated: dict[str, dict[str, Any]],
    *,
    specs: dict[str, Any] | None = None,
) -> None:
    if specs is None:
        specs = collect_machine_specs()

    try:
        file_size = audio_path.stat().st_size
    except FileNotFoundError:
        file_size = 0

    print("\nRun summary")
    print("-----------")
    system_desc = specs.get("system", "Unknown OS")
    machine_desc = specs.get("machine", "unknown")
    cpu_desc = specs.get("cpu", "Unknown CPU")
    cpu_count = specs.get("cpu_count", 1)
    memory_gb = specs.get("memory_gb")

    info_parts = [f"{system_desc} ({machine_desc})"]
    cpu_detail = cpu_desc
    if isinstance(cpu_count, int) and cpu_count > 0:
        cpu_detail += f" (cores: {cpu_count})"
    info_parts.append(f"CPU: {cpu_detail}")
    if isinstance(memory_gb, (int, float)) and math.isfinite(memory_gb):
        info_parts.append(f"Memory: {memory_gb:.1f} GB")
    print("  Machine -> " + " | ".join(info_parts))

    audio_desc = f"{audio_path}"
    if audio_path.exists():
        audio_desc = audio_path.name
    print(
        "  Audio -> "
        f"{audio_desc} | size {human_readable_bytes(file_size)} | "
        f"duration {human_readable_duration(duration_seconds)} (@ {SAMPLE_RATE} Hz)"
    )

    if not aggregated:
        return

    print("  Pipeline totals ->")
    for unit in aggregated.values():
        label = unit.get("label", "Compute unit")
        combined = unit.get("combined", {})
        if not combined:
            continue
        baseline_total = combined.get("baseline_total")
        inference_total = combined.get("coreml_inference_total")
        speedup = combined.get("speedup")
        total_with_setup = combined.get("coreml_total_with_setup")
        speedup_setup = combined.get("speedup_with_setup")

        def _format(value: Any) -> str:
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                return f"{float(value):.2f}s"
            return "n/a"

        detail = [f"baseline {_format(baseline_total)}", f"coreml {_format(inference_total)}"]
        if isinstance(speedup, (int, float)) and math.isfinite(float(speedup)):
            detail.append(f"speedup {float(speedup):.2f}x")
        if isinstance(total_with_setup, (int, float)) and math.isfinite(float(total_with_setup)):
            detail.append(f"incl setup {_format(total_with_setup)}")
            if isinstance(speedup_setup, (int, float)) and math.isfinite(float(speedup_setup)):
                detail.append(f"setup speedup {float(speedup_setup):.2f}x")
        source = combined.get("source")
        if source == "pipeline":
            label_text = f"{label} (pipeline)"
        elif source == "components":
            label_text = f"{label} (component avg)"
        else:
            label_text = label
        print(f"    {label_text}: " + ", ".join(detail))


def print_pipeline_timing_summary(
    pipeline_results: dict[str, dict[str, Any]] | None,
) -> None:
    if not pipeline_results:
        return

    baseline = pipeline_results.get("baseline")
    if not isinstance(baseline, dict):
        return

    baseline_timing = baseline.get("timing", {})
    baseline_runtime = float(baseline_timing.get("runtime_seconds", float("nan")))
    baseline_rtf = float(baseline_timing.get("rtf", float("nan")))
    baseline_rtfx = float(baseline_timing.get("rtfx", float("nan")))

    if not math.isfinite(baseline_runtime):
        return

    print("\nPipeline runtime")
    print("----------------")
    baseline_line = f"  PyTorch baseline (mps) -> {baseline_runtime:.3f}s"
    baseline_details: list[str] = []
    if math.isfinite(baseline_rtf):
        baseline_details.append(f"RTF {baseline_rtf:.3f}")
    if math.isfinite(baseline_rtfx):
        baseline_details.append(f"RTFx {baseline_rtfx:.2f}")
    if baseline_details:
        baseline_line += " (" + " | ".join(baseline_details) + ")"
    print(baseline_line)

    for key, entry in pipeline_results.items():
        if key == "baseline" or not isinstance(entry, dict):
            continue
        if entry.get("error"):
            label = entry.get("compute_unit", key)
            error_msg = entry.get("error", "Unknown error")
            print(f"  {label} -> ERROR: {error_msg}")
            continue
        timing = entry.get("timing", {})
        runtime = float(timing.get("runtime_seconds", float("nan")))
        if not math.isfinite(runtime):
            continue
        label = entry.get("compute_unit", key)
        rtf = float(timing.get("rtf", float("nan")))
        rtfx = float(timing.get("rtfx", float("nan")))
        delta = float(timing.get("rtf_delta_vs_baseline", float("nan")))
        ratio = float(timing.get("rtf_ratio_vs_baseline", float("nan")))

        line = f"  {label} -> {runtime:.3f}s"
        detail_parts: list[str] = []
        if math.isfinite(rtf):
            detail_parts.append(f"RTF {rtf:.3f}")
        if math.isfinite(rtfx):
            detail_parts.append(f"RTFx {rtfx:.2f}")
        if math.isfinite(delta):
            detail_parts.append(f"Δ {delta:+.3f}")
        if math.isfinite(ratio):
            detail_parts.append(f"× {ratio:.2f} vs baseline")
        if detail_parts:
            line += " (" + " | ".join(detail_parts) + ")"
        print(line)


def format_result(title: str, results: dict[str, dict]) -> None:
    print(f"\n{title}")
    print("=" * len(title))

    if not results:
        print("  No results available")
        return

    for unit_key, result in results.items():
        display_name = result.get(
            "compute_unit",
            unit_key.replace("_", " ").title().replace("Cpu", "CPU").replace("Gpu", "GPU").replace("Ne", "NE"),
        )
        print(f"\n[{display_name}]")

        if "error" in result:
            print(f"  Skipped: {result['error']}")
            continue

        # Handle PLDA comparison results (different structure)
        if "features_comparison" in result:
            features_comp = result["features_comparison"]
            scores_comp = result["scores_comparison"]
            print(f"Embeddings tested: {result.get('num_embeddings', 0)} × {result.get('embedding_dim', 0)}")
            print(f"PLDA features: {result.get('num_embeddings', 0)} × {result.get('plda_dim', 0)}")
            print(
                f"Features -> mse: {features_comp.mse:.6f}, mae: {features_comp.mae:.6f}, "
                f"max_abs: {features_comp.max_abs:.6f}, corr: {features_comp.corr:.6f}, cosine: {features_comp.cosine:.6f}"
            )
            print(
                f"Scores   -> mse: {scores_comp.mse:.6f}, mae: {scores_comp.mae:.6f}, "
                f"max_abs: {scores_comp.max_abs:.6f}, corr: {scores_comp.corr:.6f}"
            )
            
            timings = result.get("timings", {}).get("stats", {})
            if timings:
                ref_total = timings.get("reference_total", 0)
                coreml_total = timings.get("coreml_total", 0)
                speedup = timings.get("speedup", 0)
                if math.isfinite(ref_total) and math.isfinite(coreml_total):
                    print(f"Timing   -> reference: {ref_total*1000:.2f}ms, coreml: {coreml_total*1000:.2f}ms, speedup: {speedup:.2f}×")
            continue
        
        # Handle standard segmentation/embedding results
        aggregate: ComparisonResult = result["aggregate"]
        print(f"CoreML output shape: {result['core_output_shape']}")
        print(
            f"Aggregate -> mse: {aggregate.mse:.6f}, mae: {aggregate.mae:.6f}, "
            f"max_abs: {aggregate.max_abs:.6f}, corr: {aggregate.corr:.6f}"
            + (f", cosine: {aggregate.cosine:.6f}" if aggregate.cosine is not None else "")
        )

        stats = result.get("timings", {}).get("stats", {})
        if stats:
            baseline_total = stats.get("baseline_total", float("nan"))
            coreml_total = stats.get("coreml_total", float("nan"))
            speedup = stats.get("speedup", float("nan"))
            total_with_setup = stats.get("coreml_total_with_setup", float("nan"))
            speedup_with_setup = stats.get("speedup_including_setup", float("nan"))
            compile_time = stats.get("compile_time", float("nan"))
            warmup_time = stats.get("warmup_time", float("nan"))

            print(
                "Timing summary -> "
                f"baseline_total_s: {baseline_total:.4f}, "
                f"coreml_total_s: {coreml_total:.4f}, "
                f"speedup: {speedup:.2f}x"
            )
            if math.isfinite(total_with_setup):
                print(
                    "  Setup-adjusted -> "
                    f"coreml_total_with_setup_s: {total_with_setup:.4f}, "
                    f"speedup_including_setup: {speedup_with_setup:.2f}x"
                )
            if math.isfinite(compile_time) or math.isfinite(warmup_time):
                detail_parts: list[str] = []
                if math.isfinite(compile_time):
                    detail_parts.append(f"compile_s: {compile_time:.4f}")
                if math.isfinite(warmup_time):
                    detail_parts.append(f"warmup_s: {warmup_time:.4f}")
                if detail_parts:
                    print("  Setup details -> " + ", ".join(detail_parts))

        per_class_corr = result.get("per_class_corr")
        if per_class_corr:
            values = np.array(per_class_corr, dtype=np.float32)
            finite = values[np.isfinite(values)]
            if finite.size:
                print(
                    "Class correlation stats -> "
                    f"min: {finite.min():.6f}, median: {np.median(finite):.6f}, "
                    f"max: {finite.max():.6f}"
                )

        per_dim_corr = result.get("per_dim_corr")
        if per_dim_corr:
            values = np.array(per_dim_corr, dtype=np.float32)
            finite = values[np.isfinite(values)]
            if finite.size:
                print(
                    "Embedding dim corr -> "
                    f"min: {finite.min():.6f}, median: {np.median(finite):.6f}, "
                    f"max: {finite.max():.6f}"
                )

        chunk_cosines = result.get("chunk_cosines")
        if chunk_cosines:
            values = np.array(chunk_cosines, dtype=np.float32)
            finite = values[np.isfinite(values)]
            if finite.size:
                print(
                    "Chunk cosine similarity -> "
                    f"mean: {finite.mean():.6f}, min: {finite.min():.6f}, "
                    f"max: {finite.max():.6f}"
                )

        plda_checks = result.get("plda_checks")
        if plda_checks:
            summaries: list[str] = []
            for label, comp in plda_checks.items():
                summaries.append(
                    f"{label}: mse {comp.mse:.2e}, corr {comp.corr:.4f}"
                )
            if summaries:
                print("PLDA parity checks -> " + " | ".join(summaries))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pyannote community-1 PyTorch and Core ML models using a fixed audio clip",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=DEFAULT_AUDIO,
        help="Path to evaluation audio (defaults to yc_first_minute.wav)",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Directory containing the original pyannote speaker diarization checkpoints",
    )
    parser.add_argument(
        "--coreml-dir",
        type=Path,
        default=DEFAULT_COREML_DIR,
        help="Directory containing converted Core ML packages",
    )
    return parser.parse_args()


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()

    if not args.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if not args.model_root.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_root}")
    if not args.coreml_dir.exists():
        raise FileNotFoundError(f"Core ML directory not found: {args.coreml_dir}")

    waveform = load_audio(args.audio_path)
    audio_duration_seconds = float(waveform.shape[-1]) / SAMPLE_RATE
    print(f"Loaded audio: {args.audio_path} -> shape {tuple(waveform.shape)}, sample_rate={SAMPLE_RATE}")
    
    seg_result = compare_segmentation(
        args.model_root,
        args.coreml_dir,
        waveform,
        compute_units=DEFAULT_COMPUTE_UNITS,
    )
    print("Segmentation done")
    emb_result = compare_embedding(
        args.model_root,
        args.coreml_dir,
        waveform,
        compute_units=DEFAULT_COMPUTE_UNITS,
    )
    
    print("Embedding done")
    
    # Extract embeddings for PLDA comparison
    print("Testing PLDA CoreML model...")
    embedding_torch = Model.from_pretrained(str(args.model_root / "embedding")).eval()
    test_embeddings = []
    with torch.inference_mode():
        for chunk in chunk_waveform(waveform, EMBEDDING_WINDOW):
            emb = embedding_torch(chunk).cpu().numpy()
            test_embeddings.append(emb)
    
    test_embeddings_stacked = np.concatenate(test_embeddings, axis=0)
    
    plda_result = compare_plda(
        args.model_root,
        args.coreml_dir,
        test_embeddings_stacked,
        compute_units=DEFAULT_COMPUTE_UNITS,
    )
    print("PLDA comparison done")
    
    print("Running end to end pipeline")
    pipeline_result = compare_pipeline_end_to_end(
        args.model_root,
        args.coreml_dir,
        args.audio_path,
        compute_units=[ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_AND_NE],
        waveform=waveform,
        audio_duration_seconds=audio_duration_seconds,
    )

    print("end to end pipeline done")
    print_pipeline_timing_summary(pipeline_result)
    # format_result("Segmentation", seg_result)
    # format_result("Embedding", emb_result)
    # format_result("PLDA", plda_result)

    specs = collect_machine_specs()
    aggregated_summary = aggregate_timing_summary(seg_result, emb_result)
    aggregated_summary = merge_pipeline_into_summary(aggregated_summary, pipeline_result)
    print_run_summary(
        args.audio_path,
        audio_duration_seconds,
        aggregated_summary,
        specs=specs,
    )

    base_dir = Path(__file__).resolve().parent
    plot_signature = generate_plots(
        seg_result,
        emb_result,
        pipeline_result,
        base_dir,
        aggregated_summary=aggregated_summary,
        plda_results=plda_result,
        specs=specs,
        audio_seconds=audio_duration_seconds,
    )
    print(
        "Plot artifacts written to "
        f"{base_dir / 'plots'} "
        "(metrics_timeseries.png, latency_timeseries.png, summary_speed.png, plda_parity.png, plda_coreml.png, pipeline_overview.png, pipeline_timing.png)"
        f" [{plot_signature}]"
    )


if __name__ == "__main__":
    main()
