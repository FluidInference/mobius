from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import argparse
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
    save_heatmap,
    save_json,
    write_rttm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LS-EEND online CoreML inference on an audio file.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--coreml-model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="PyTorch checkpoint used for reference validation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ref-rttm", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts" / "coreml_infer")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--median", type=int, default=11)
    parser.add_argument("--collar-seconds", type=float, default=0.25)
    parser.add_argument("--skip-reference-check", action="store_true")
    parser.add_argument(
        "--compute-units",
        choices=("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"),
        default="cpu_only",
    )
    return parser


def _compute_units(name: str):
    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def _coreml_predict(
    model: ct.models.MLModel,
    features: np.ndarray,
    state: dict[str, np.ndarray],
    conv_delay: int,
) -> np.ndarray:
    outputs = []
    zero_frame = np.zeros((1, 1, features.shape[1]), dtype=np.float32)
    for frame_index in range(features.shape[0]):
        frame = features[frame_index : frame_index + 1].reshape(1, 1, -1).astype(np.float32, copy=False)
        decode = np.array([1.0 if frame_index >= conv_delay else 0.0], dtype=np.float32)
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
                "decode": decode,
            }
        )
        state["enc_ret_kv"] = prediction["enc_ret_kv_out"]
        state["enc_ret_scale"] = prediction["enc_ret_scale_out"]
        state["enc_conv_cache"] = prediction["enc_conv_cache_out"]
        state["dec_ret_kv"] = prediction["dec_ret_kv_out"]
        state["dec_ret_scale"] = prediction["dec_ret_scale_out"]
        state["top_buffer"] = prediction["top_buffer_out"]
        if frame_index >= conv_delay:
            outputs.append(prediction["full_logits"].reshape(1, -1))

    for _ in range(conv_delay):
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
        outputs.append(prediction["full_logits"].reshape(1, -1))

    if outputs:
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    return np.zeros((0, state["dec_ret_kv"].shape[1]), dtype=np.float32)


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    engine = LSEENDInferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device="cpu",
    )
    layout = build_state_layout(engine)
    audio, sample_rate = load_audio(args.audio)
    features = extract_features(audio, sample_rate, engine.config).numpy()

    model = ct.models.MLModel(str(args.coreml_model), compute_units=_compute_units(args.compute_units))
    state = initial_state_tensors(layout, dtype=np.float32)
    full_logits = _coreml_predict(model, features, state, layout.conv_delay)
    full_probabilities = 1.0 / (1.0 + np.exp(-full_logits))
    probabilities = full_probabilities[:, 1:-1]

    recording_id = args.audio.stem
    np.save(args.output_dir / f"{recording_id}_coreml_full_logits.npy", full_logits)
    np.save(args.output_dir / f"{recording_id}_coreml_full_probabilities.npy", full_probabilities)
    np.save(args.output_dir / f"{recording_id}_coreml_probabilities.npy", probabilities)

    metrics = {
        "audio": str(args.audio),
        "coreml_model": str(args.coreml_model),
        "num_frames": int(probabilities.shape[0]),
        "real_output_dim": int(probabilities.shape[1]),
    }

    if not args.skip_reference_check:
        pytorch = engine.infer_audio(audio, sample_rate)
        max_abs_error = float(np.max(np.abs(full_probabilities - pytorch.full_probabilities)))
        mean_abs_error = float(np.mean(np.abs(full_probabilities - pytorch.full_probabilities)))
        metrics["reference_check"] = {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
        }
        print(f"Max abs error vs PyTorch: {max_abs_error:.8f}")
        print(f"Mean abs error vs PyTorch: {mean_abs_error:.8f}")

    if args.ref_rttm is not None and args.ref_rttm.exists():
        entries, speaker_labels = parse_rttm(args.ref_rttm)
        reference = rttm_to_frame_matrix(entries, speaker_labels, probabilities.shape[0], engine.model_frame_hz)
        diar_metrics = compute_der(
            probabilities=probabilities[:, : len(speaker_labels)],
            reference_binary=reference,
            threshold=args.threshold,
            median_width=args.median,
            collar_seconds=args.collar_seconds,
            frame_rate=engine.model_frame_hz,
        )
        metrics["der"] = {
            key: value
            for key, value in diar_metrics.items()
            if key not in {"mapped_binary", "mapped_probabilities", "valid_mask"}
        }
        save_heatmap(
            reference_binary=reference,
            mapped_binary=diar_metrics["mapped_binary"],
            mapped_probabilities=diar_metrics["mapped_probabilities"],
            frame_rate=engine.model_frame_hz,
            speaker_labels=speaker_labels,
            output_path=args.output_dir / f"{recording_id}_coreml_heatmap.png",
        )
        write_rttm(
            recording_id=recording_id,
            binary_prediction=diar_metrics["mapped_binary"],
            output_path=args.output_dir / f"{recording_id}_coreml_prediction.rttm",
            frame_rate=engine.model_frame_hz,
            speaker_labels=speaker_labels,
        )
        print(f"DER: {diar_metrics['der']:.6f}")

    metrics_path = args.output_dir / f"{recording_id}_coreml_metrics.json"
    save_json(metrics, metrics_path)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
