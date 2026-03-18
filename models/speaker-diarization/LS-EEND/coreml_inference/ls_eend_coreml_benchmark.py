from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np

from coreml_inference.ls_eend_coreml import build_state_layout, initial_state_tensors
from torch_inference.ls_eend_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    LSEENDInferenceEngine,
    compute_der,
    extract_features,
    load_audio,
    parse_rttm,
    rttm_to_frame_matrix,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LS-EEND CoreML inference across compute-unit modes.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--coreml-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ref-rttm", type=Path, default=None)
    parser.add_argument(
        "--compute-units",
        nargs="+",
        choices=("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"),
        default=("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"),
    )
    parser.add_argument("--warm-runs", type=int, default=3, help="Number of warm inference passes to average per compute-unit mode.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--median", type=int, default=11)
    parser.add_argument("--collar-seconds", type=float, default=0.25)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "coreml" / "benchmark_summary.json",
    )
    return parser


def _compute_units(name: str):
    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def _run_coreml_pass(
    model: ct.models.MLModel,
    features: np.ndarray,
    layout,
) -> np.ndarray:
    state = initial_state_tensors(layout, dtype=np.float32)
    outputs: list[np.ndarray] = []
    zero_frame = np.zeros((1, 1, layout.input_dim), dtype=np.float32)
    for frame_index in range(features.shape[0]):
        frame = features[frame_index : frame_index + 1].reshape(1, 1, -1).astype(np.float32, copy=False)
        prediction = model.predict(
            {
                "frame": frame,
                "enc_ret_kv": state["enc_ret_kv"],
                "enc_ret_scale": state["enc_ret_scale"],
                "enc_conv_cache": state["enc_conv_cache"],
                "dec_ret_kv": state["dec_ret_kv"],
                "dec_ret_scale": state["dec_ret_scale"],
                "top_buffer": state["top_buffer"],
                "ingest": np.array([1.0], dtype=np.float32),
                "decode": np.array([1.0 if frame_index >= layout.conv_delay else 0.0], dtype=np.float32),
            }
        )
        state["enc_ret_kv"] = prediction["enc_ret_kv_out"]
        state["enc_ret_scale"] = prediction["enc_ret_scale_out"]
        state["enc_conv_cache"] = prediction["enc_conv_cache_out"]
        state["dec_ret_kv"] = prediction["dec_ret_kv_out"]
        state["dec_ret_scale"] = prediction["dec_ret_scale_out"]
        state["top_buffer"] = prediction["top_buffer_out"]
        if frame_index >= layout.conv_delay:
            outputs.append(prediction["full_logits"].reshape(1, -1).astype(np.float32, copy=False))

    for _ in range(layout.conv_delay):
        prediction = model.predict(
            {
                "frame": zero_frame,
                "enc_ret_kv": state["enc_ret_kv"],
                "enc_ret_scale": state["enc_ret_scale"],
                "enc_conv_cache": state["enc_conv_cache"],
                "dec_ret_kv": state["dec_ret_kv"],
                "dec_ret_scale": state["dec_ret_scale"],
                "top_buffer": state["top_buffer"],
                "ingest": np.array([0.0], dtype=np.float32),
                "decode": np.array([1.0], dtype=np.float32),
            }
        )
        state["enc_ret_kv"] = prediction["enc_ret_kv_out"]
        state["enc_ret_scale"] = prediction["enc_ret_scale_out"]
        state["enc_conv_cache"] = prediction["enc_conv_cache_out"]
        state["dec_ret_kv"] = prediction["dec_ret_kv_out"]
        state["dec_ret_scale"] = prediction["dec_ret_scale_out"]
        state["top_buffer"] = prediction["top_buffer_out"]
        outputs.append(prediction["full_logits"].reshape(1, -1).astype(np.float32, copy=False))

    if outputs:
        return np.concatenate(outputs, axis=0)
    return np.zeros((0, layout.full_output_dim), dtype=np.float32)


def main() -> None:
    args = build_parser().parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = load_audio(args.audio)
    pytorch_engine = LSEENDInferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device="cpu",
    )
    layout = build_state_layout(pytorch_engine)

    feature_start = time.perf_counter()
    features = extract_features(audio, sample_rate, pytorch_engine.config).numpy()
    feature_seconds = time.perf_counter() - feature_start

    reference_start = time.perf_counter()
    pytorch = pytorch_engine.infer_audio(audio, sample_rate)
    reference_seconds = time.perf_counter() - reference_start

    if args.ref_rttm is not None and args.ref_rttm.exists():
        entries, speaker_labels = parse_rttm(args.ref_rttm)
        reference_binary = rttm_to_frame_matrix(entries, speaker_labels, pytorch.probabilities.shape[0], pytorch.frame_hz)
    else:
        speaker_labels = []
        reference_binary = None

    summary: dict[str, object] = {
        "audio": str(args.audio),
        "coreml_model": str(args.coreml_model),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "num_frames": int(features.shape[0]),
        "audio_seconds": float(len(audio) / max(sample_rate, 1)),
        "feature_seconds": feature_seconds,
        "pytorch_reference_seconds": reference_seconds,
        "warm_runs": int(args.warm_runs),
        "runs": [],
    }

    for compute_name in args.compute_units:
        run_summary: dict[str, object] = {"compute_units": compute_name}
        try:
            load_start = time.perf_counter()
            model = ct.models.MLModel(str(args.coreml_model), compute_units=_compute_units(compute_name))
            load_seconds = time.perf_counter() - load_start

            cold_start = time.perf_counter()
            cold_logits = _run_coreml_pass(model, features, layout)
            cold_seconds = time.perf_counter() - cold_start

            warm_times = []
            warm_logits = cold_logits
            for _ in range(max(1, int(args.warm_runs))):
                warm_start = time.perf_counter()
                warm_logits = _run_coreml_pass(model, features, layout)
                warm_times.append(time.perf_counter() - warm_start)
            warm_seconds = float(np.mean(warm_times))

            cold_probabilities = 1.0 / (1.0 + np.exp(-cold_logits))
            warm_probabilities = 1.0 / (1.0 + np.exp(-warm_logits))
            if not np.isfinite(cold_probabilities).all() or not np.isfinite(warm_probabilities).all():
                raise RuntimeError("CoreML output contains non-finite values.")
            max_abs_error = float(np.max(np.abs(warm_probabilities - pytorch.full_probabilities)))
            mean_abs_error = float(np.mean(np.abs(warm_probabilities - pytorch.full_probabilities)))
            audio_seconds = float(summary["audio_seconds"])
            cold_realtime_factor = cold_seconds / audio_seconds if audio_seconds > 0 else 0.0
            warm_realtime_factor = warm_seconds / audio_seconds if audio_seconds > 0 else 0.0

            run_summary.update(
                {
                    "status": "ok",
                    "load_seconds": load_seconds,
                    "cold_inference_seconds": cold_seconds,
                    "warm_inference_seconds_all": warm_times,
                    "warm_inference_seconds": warm_seconds,
                    "cold_realtime_factor": cold_realtime_factor,
                    "warm_realtime_factor": warm_realtime_factor,
                    "cold_x_realtime": (1.0 / cold_realtime_factor) if cold_realtime_factor > 0 else None,
                    "warm_x_realtime": (1.0 / warm_realtime_factor) if warm_realtime_factor > 0 else None,
                    "max_abs_error": max_abs_error,
                    "mean_abs_error": mean_abs_error,
                }
            )

            if reference_binary is not None:
                diar_metrics = compute_der(
                    probabilities=warm_probabilities[:, 1 : 1 + len(speaker_labels)],
                    reference_binary=reference_binary,
                    threshold=args.threshold,
                    median_width=args.median,
                    collar_seconds=args.collar_seconds,
                    frame_rate=pytorch.frame_hz,
                )
                run_summary["der"] = float(diar_metrics["der"])
        except Exception as exc:
            run_summary.update({"status": "error", "error": str(exc)})
        summary["runs"].append(run_summary)

    save_json(summary, args.output_json)
    print(f"Saved benchmark summary: {args.output_json}")
    for run in summary["runs"]:
        if run["status"] != "ok":
            print(f"{run['compute_units']}: ERROR {run['error']}")
            continue
        der_suffix = f", DER={run['der']:.6f}" if "der" in run else ""
        print(
            f"{run['compute_units']}: load={run['load_seconds']:.3f}s, "
            f"cold={run['cold_inference_seconds']:.3f}s ({run['cold_x_realtime']:.2f}x RT), "
            f"warm={run['warm_inference_seconds']:.3f}s ({run['warm_x_realtime']:.2f}x RT), "
            f"max_abs={run['max_abs_error']:.8f}{der_suffix}"
        )


if __name__ == "__main__":
    main()
