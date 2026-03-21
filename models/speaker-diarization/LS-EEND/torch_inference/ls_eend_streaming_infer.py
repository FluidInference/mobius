from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import medfilt

from torch_inference.ls_eend_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    LSEENDInferenceEngine,
    compute_der,
    parse_rttm,
    rttm_to_frame_matrix,
    save_heatmap,
    save_json,
    select_speaker_probabilities,
    write_rttm,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming LS-EEND inference with optional RTTM scoring.")
    parser.add_argument("--audio", type=Path, default=Path(__file__).resolve().parent.parent / "ahnss.wav")
    parser.add_argument("--ref-rttm", type=Path, default=Path(__file__).resolve().parent.parent / "ahnss.rttm")
    parser.add_argument("--coreml-model", type=Path, default=None, help="Optional fixed-shape CoreML package to use instead of the PyTorch checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts" / "ahnss_eval")
    parser.add_argument("--chunk-seconds", type=float, default=60.0, help="Simulated streaming chunk size.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--median", type=int, default=11)
    parser.add_argument("--collar-seconds", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--compute-units",
        choices=("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"),
        default="cpu_only",
        help="CoreML compute units when --coreml-model is used.",
    )
    parser.add_argument("--num-speakers", type=int, default=None, help="Optional known number of speakers.")
    parser.add_argument(
        "--speaker-count-policy",
        choices=("reference", "config"),
        default="reference",
        help="For RTTM-backed scoring, evaluate the first N real-speaker attractors using either the RTTM speaker count or all configured tracks.",
    )
    parser.add_argument(
        "--speaker-backend",
        choices=("nemo", "lseend"),
        default="lseend",
        help="Use raw LS-EEND columns directly or cluster NeMo speaker embeddings on top of LS-EEND speech activity.",
    )
    return parser


def _run_nemo_hybrid(
    audio_path: Path,
    output_dir: Path,
    recording_id: str,
    probabilities: np.ndarray,
    threshold: float,
    median_width: int,
    num_speakers: int | None,
    ref_rttm: Path | None,
) -> tuple[float | None, Path, dict[str, str]]:
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    speech = (probabilities.max(axis=1) > threshold).astype(np.float32)
    if median_width > 1:
        speech = medfilt(speech, kernel_size=median_width).astype(np.float32)

    hybrid_dir = output_dir / "nemo_hybrid"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = hybrid_dir / "manifest.json"
    external_vad_path = hybrid_dir / "external_vad_manifest.json"

    manifest_entry = {
        "audio_filepath": str(audio_path),
        "offset": 0.0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": str(ref_rttm) if ref_rttm else None,
        "num_speakers": num_speakers,
        "uem_filepath": None,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_entry) + "\n")

    with open(external_vad_path, "w", encoding="utf-8") as handle:
        padded = np.pad(speech, (1, 1), constant_values=0)
        changes = np.where(np.diff(padded) != 0)[0]
        for start, stop in zip(changes[::2], changes[1::2]):
            handle.write(
                json.dumps(
                    {
                        "audio_filepath": str(audio_path),
                        "offset": round(start / 10.0, 5),
                        "duration": round((stop - start) / 10.0, 5),
                        "label": "speech",
                        "text": "-",
                        "num_speakers": num_speakers,
                        "rttm_filepath": str(ref_rttm) if ref_rttm else None,
                        "uem_filepath": None,
                        "uniq_id": recording_id,
                    }
                )
                + "\n"
            )

    config = OmegaConf.create(
        {
            "device": None,
            "verbose": False,
            "batch_size": 64,
            "num_workers": 0,
            "sample_rate": 16000,
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(hybrid_dir),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": None,
                    "external_vad_manifest": str(external_vad_path),
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median",
                        "overlap": 0.5,
                        "onset": 0.1,
                        "offset": 0.1,
                        "pad_onset": 0.1,
                        "pad_offset": 0.0,
                        "min_duration_on": 0.0,
                        "min_duration_off": 0.2,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "ecapa_tdnn",
                    "parameters": {
                        "window_length_in_sec": [1.5],
                        "shift_length_in_sec": [0.75],
                        "multiscale_weights": [1],
                        "save_embeddings": True,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": num_speakers is not None,
                        "max_num_speakers": 8,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                    },
                },
            },
        }
    )
    diarizer = ClusteringDiarizer(cfg=config)
    _, speaker_mapping, der_details = diarizer.diarize()
    predicted_rttm = hybrid_dir / "pred_rttms" / f"{recording_id}.rttm"
    return float(der_details[0]) if ref_rttm else None, predicted_rttm, speaker_mapping.get(recording_id, {})


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.coreml_model is not None:
        from ls_eend_coreml_runtime import CoreMLLSEENDInferenceEngine

        engine = CoreMLLSEENDInferenceEngine(
            coreml_model_path=args.coreml_model,
            config_path=args.config,
            compute_units=args.compute_units,
        )
    else:
        engine = LSEENDInferenceEngine(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            actual_num_speakers=args.num_speakers,
        )
    result, updates = engine.simulate_streaming_file(args.audio, args.chunk_seconds)

    recording_id = args.audio.stem
    np.save(args.output_dir / f"{recording_id}_logits.npy", result.logits)
    np.save(args.output_dir / f"{recording_id}_probabilities.npy", result.probabilities)
    np.save(args.output_dir / f"{recording_id}_full_logits.npy", result.full_logits)
    np.save(args.output_dir / f"{recording_id}_full_probabilities.npy", result.full_probabilities)
    save_json({"updates": updates}, args.output_dir / f"{recording_id}_streaming_updates.json")

    if args.ref_rttm and args.ref_rttm.exists():
        entries, speaker_labels = parse_rttm(args.ref_rttm)
        eval_num_speakers = args.num_speakers
        if eval_num_speakers is None and args.speaker_count_policy == "reference":
            eval_num_speakers = len(speaker_labels)
        eval_probabilities = select_speaker_probabilities(result.probabilities, eval_num_speakers)
        eval_logits = select_speaker_probabilities(result.logits, eval_num_speakers)
        np.save(args.output_dir / f"{recording_id}_eval_logits.npy", eval_logits)
        np.save(args.output_dir / f"{recording_id}_eval_probabilities.npy", eval_probabilities)
        reference = rttm_to_frame_matrix(entries, speaker_labels, eval_probabilities.shape[0], result.frame_hz)
        raw_metrics = compute_der(
            probabilities=eval_probabilities,
            reference_binary=reference,
            threshold=args.threshold,
            median_width=args.median,
            collar_seconds=args.collar_seconds,
            frame_rate=result.frame_hz,
        )
        save_heatmap(
            reference_binary=reference,
            mapped_binary=raw_metrics["mapped_binary"],
            mapped_probabilities=raw_metrics["mapped_probabilities"],
            frame_rate=result.frame_hz,
            speaker_labels=speaker_labels,
            output_path=args.output_dir / f"{recording_id}_raw_heatmap.png",
        )

        summary = {
            "raw_lseend": {
                key: value
                for key, value in raw_metrics.items()
                if key not in {"mapped_binary", "mapped_probabilities", "valid_mask"}
            },
            "evaluation": {
                "speaker_count_policy": args.speaker_count_policy,
                "evaluated_num_speakers": int(eval_probabilities.shape[1]),
                "available_model_tracks": int(result.probabilities.shape[1]),
            },
        }

        if args.speaker_backend == "nemo":
            hybrid_der, predicted_rttm, speaker_mapping = _run_nemo_hybrid(
                audio_path=args.audio,
                output_dir=args.output_dir,
                recording_id=recording_id,
                probabilities=eval_probabilities,
                threshold=args.threshold,
                median_width=args.median,
                num_speakers=eval_num_speakers or len(speaker_labels),
                ref_rttm=args.ref_rttm,
            )
            pred_entries, _ = parse_rttm(predicted_rttm)
            for entry in pred_entries:
                entry["speaker"] = speaker_mapping.get(entry["speaker"], entry["speaker"])
            hybrid_binary = rttm_to_frame_matrix(pred_entries, speaker_labels, eval_probabilities.shape[0], result.frame_hz)
            save_heatmap(
                reference_binary=reference,
                mapped_binary=hybrid_binary,
                mapped_probabilities=hybrid_binary,
                frame_rate=result.frame_hz,
                speaker_labels=speaker_labels,
                output_path=args.output_dir / f"{recording_id}_hybrid_heatmap.png",
            )
            write_rttm(
                recording_id=recording_id,
                binary_prediction=hybrid_binary,
                output_path=args.output_dir / f"{recording_id}_hybrid_prediction.rttm",
                frame_rate=result.frame_hz,
                speaker_labels=speaker_labels,
            )
            summary["nemo_hybrid"] = {
                "pyannote_der": hybrid_der,
                "speaker_mapping": speaker_mapping,
                "predicted_rttm": str(predicted_rttm),
            }
            print(f"Hybrid DER: {hybrid_der:.4f}")
            print(f"Hybrid heatmap: {args.output_dir / f'{recording_id}_hybrid_heatmap.png'}")
        else:
            raw_rttm_path = args.output_dir / f"{recording_id}_raw_prediction.rttm"
            write_rttm(
                recording_id=recording_id,
                binary_prediction=raw_metrics["mapped_binary"],
                output_path=raw_rttm_path,
                frame_rate=result.frame_hz,
                speaker_labels=speaker_labels,
            )
            print(f"Raw RTTM: {raw_rttm_path}")

        save_json(summary, args.output_dir / f"{recording_id}_metrics.json")
        print(f"Streaming duration: {result.duration_seconds:.2f}s")
        print(f"Frames: {eval_probabilities.shape[0]}")
        print(f"Evaluated speakers: {eval_probabilities.shape[1]}")
        print(f"Raw LS-EEND DER: {raw_metrics['der']:.4f}")
        print(f"Raw heatmap: {args.output_dir / f'{recording_id}_raw_heatmap.png'}")
    else:
        eval_probabilities = select_speaker_probabilities(result.probabilities, args.num_speakers)
        raw_binary = (eval_probabilities > args.threshold).astype(np.float32)
        write_rttm(
            recording_id=recording_id,
            binary_prediction=raw_binary,
            output_path=args.output_dir / f"{recording_id}_prediction.rttm",
            frame_rate=result.frame_hz,
        )
        print(f"Streaming duration: {result.duration_seconds:.2f}s")
        print(f"Frames: {eval_probabilities.shape[0]}")
        print(f"RTTM: {args.output_dir / f'{recording_id}_prediction.rttm'}")


if __name__ == "__main__":
    main()
