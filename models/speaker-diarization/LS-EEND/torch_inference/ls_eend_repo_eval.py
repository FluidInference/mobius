#!/usr/bin/env python3
"""Compatibility runner for the original LS-EEND repo evaluation path.

This script reproduces the repository's test-time data flow on a local audio/RTTM
pair without depending on the original hardcoded paths or legacy Lightning CLI.
"""

from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import hyperpyyaml
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.signal import medfilt


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.diarization_dataset import KaldiDiarizationDataset as OfflineKaldiDiarizationDataset
from datasets.diarization_dataset_on_the_fly import KaldiDiarizationDataset as StreamingKaldiDiarizationDataset
from torch_inference.ls_eend_runtime import compute_der, save_heatmap, save_json, write_rttm
from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (
    OnlineConformerRetentionDADiarization,
)
from train.utils.loss import pad_labels, pad_preds, pit_loss_multispk, report_diarization_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Input wav file.")
    parser.add_argument("--ref-rttm", type=Path, required=True, help="Reference RTTM.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="LS-EEND checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Repo hyperpyyaml config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Artifact directory.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or mps.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold.")
    parser.add_argument("--median", type=int, default=11, help="Median filter width.")
    parser.add_argument(
        "--collar-seconds",
        type=float,
        default=0.25,
        help="Per-side collar used for paper-style DER.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return hyperpyyaml.load_hyperpyyaml(handle)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def build_kaldi_dir(audio_path: Path, rttm_path: Path, output_dir: Path, target_sample_rate: int) -> Path:
    kaldi_dir = output_dir / "kaldi"
    kaldi_dir.mkdir(parents=True, exist_ok=True)
    recording_id = audio_path.stem
    input_audio, input_sample_rate = sf.read(str(audio_path))
    input_audio = ensure_mono(input_audio)
    kaldi_audio_path = audio_path
    duration_seconds = len(input_audio) / input_sample_rate
    if input_sample_rate != target_sample_rate:
        resampled_audio = librosa.resample(
            input_audio,
            orig_sr=input_sample_rate,
            target_sr=target_sample_rate,
        ).astype(np.float32, copy=False)
        kaldi_audio_path = kaldi_dir / f"{recording_id}_{target_sample_rate}hz.wav"
        sf.write(kaldi_audio_path, resampled_audio, target_sample_rate, subtype="FLOAT")
        duration_seconds = len(resampled_audio) / target_sample_rate
    spk2utts: defaultdict[str, list[str]] = defaultdict(list)
    segments: list[tuple[str, str, float, float, str]] = []
    with open(rttm_path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            parts = line.strip().split()
            if not parts:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            utt = f"{recording_id}_{index:04d}"
            segments.append((utt, recording_id, start, start + duration, speaker))
            spk2utts[speaker].append(utt)
    (kaldi_dir / "wav.scp").write_text(f"{recording_id} {kaldi_audio_path}\n", encoding="utf-8")
    (kaldi_dir / "reco2dur").write_text(f"{recording_id} {duration_seconds:.7f}\n", encoding="utf-8")
    (kaldi_dir / "segments").write_text(
        "".join(f"{utt} {rec} {start:.3f} {end:.3f}\n" for utt, rec, start, end, _ in segments),
        encoding="utf-8",
    )
    (kaldi_dir / "utt2spk").write_text(
        "".join(f"{utt} {speaker}\n" for utt, _, _, _, speaker in segments),
        encoding="utf-8",
    )
    (kaldi_dir / "spk2utt").write_text(
        "".join(f"{speaker} {' '.join(utts)}\n" for speaker, utts in sorted(spk2utts.items())),
        encoding="utf-8",
    )
    (kaldi_dir / "reco2num_spk").write_text(f"{recording_id} {len(spk2utts)}\n", encoding="utf-8")
    return kaldi_dir


def build_streaming_dataset(config: dict, kaldi_dir: Path) -> StreamingKaldiDiarizationDataset:
    return StreamingKaldiDiarizationDataset(
        data_dir=str(kaldi_dir),
        data_type="val",
        chunk_size=config["data"]["val_chunk_size"],
        chunk_step=config["data"]["val_chunk_step"],
        context_size=config["data"]["context_recp"],
        input_transform=config["data"]["feat_type"],
        frame_size=config["data"]["feat"]["win_length"],
        frame_shift=config["data"]["feat"]["hop_length"],
        subsampling=config["data"]["subsampling"],
        rate=config["data"]["feat"]["sample_rate"],
        label_delay=config["data"]["label_delay"],
        n_speakers=config["data"]["num_speakers"],
        use_last_samples=config["data"]["use_last_samples"],
        shuffle=False,
    )


def build_offline_dataset(config: dict, kaldi_dir: Path) -> OfflineKaldiDiarizationDataset:
    return OfflineKaldiDiarizationDataset(
        data_dir=str(kaldi_dir),
        chunk_size=config["data"]["chunk_size"],
        context_size=config["data"]["context_recp"],
        input_transform=config["data"]["feat_type"],
        frame_size=config["data"]["feat"]["win_length"],
        frame_shift=config["data"]["feat"]["hop_length"],
        subsampling=config["data"]["subsampling"],
        rate=config["data"]["feat"]["sample_rate"],
        label_delay=config["data"]["label_delay"],
        n_speakers=config["data"]["num_speakers"],
        use_last_samples=config["data"]["use_last_samples"],
        shuffle=False,
    )


def build_model(config: dict, device: torch.device) -> OnlineConformerRetentionDADiarization:
    model = OnlineConformerRetentionDADiarization(
        n_speakers=config["data"]["num_speakers"],
        in_size=(2 * config["data"]["context_recp"] + 1) * config["data"]["feat"]["n_mels"],
        **config["model"]["params"],
    )
    return model.to(device)


def load_model_state(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)}")
    model_state = {}
    for key, value in state.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value
        else:
            model_state[key] = value
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch:\n"
            f"missing={missing_keys}\n"
            f"unexpected={unexpected_keys}"
        )


def prepare_test_labels(labels: list[torch.Tensor], clip_lengths: list[int]) -> tuple[list[torch.Tensor], list[int]]:
    n_spks = [label.shape[1] for label in labels]
    max_spk = max(n_spks)
    labels = [F.pad(label, (0, max_spk - label.shape[1]), "constant", 0) for label in labels]
    labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=0, batch_first=True)
    batch, num_frames, _ = labels.shape
    frame_index = torch.arange(1, num_frames + 1, device=labels.device).unsqueeze(0).unsqueeze(-1)
    labels_index = frame_index * labels
    labels_index = labels_index.masked_fill_(labels_index == 0, torch.inf)
    sort_index = torch.argsort(torch.min(labels_index, dim=1)[0], dim=1)
    labels = labels[torch.arange(batch, device=labels.device).unsqueeze(1), :, sort_index].transpose(-1, -2)
    labels_silence = torch.ones((labels.shape[0], labels.shape[1]), device=labels.device) - labels.max(-1)[0]
    labels = torch.cat([labels_silence.unsqueeze(-1), labels], dim=-1)
    labels_nonespk = torch.zeros((batch, num_frames, 1), device=labels.device)
    labels = torch.cat([labels, labels_nonespk], dim=-1)
    labels = [label[:clip_length, :n_spk + 2] for label, clip_length, n_spk in zip(labels, clip_lengths, n_spks)]
    return labels, n_spks


def preliminary_metrics(
    preds: list[torch.Tensor],
    labels: list[torch.Tensor],
    n_spks: list[int],
    max_spks: int,
    label_delay: int,
) -> dict | None:
    if max(n_spks) > max_spks:
        return None
    labels_spks = [label[:, 1:-1] for label in labels]
    preds_spks = [pred[:, 1:n_spk + 1] for pred, n_spk in zip(preds, n_spks)]
    max_nspk = max(n_spks)
    labels_spks_pad = pad_labels(labels_spks, max_nspk)
    preds_spks_pad = pad_preds(preds_spks, max_nspk)
    perm_labels_spks = pit_loss_multispk(preds_spks_pad, labels_spks_pad, n_spks)
    perm_labels = [torch.cat([label[:, 0].unsqueeze(-1), perm], dim=-1) for label, perm in zip(labels, perm_labels_spks)]
    perm_labels = [torch.cat([perm, label[:, -1].unsqueeze(-1)], dim=-1) for label, perm in zip(labels, perm_labels)]
    preds_realspk = [pred[:, 1:max_spks + 1] for pred in preds]
    if preds_realspk[0].shape[1] < max_spks:
        preds_realspk = pad_preds(preds_realspk, max_spks)
    labels_realspk = [label[:, 1:-1] for label in perm_labels]
    if labels_realspk[0].shape[1] < max_spks:
        labels_realspk = pad_preds(labels_realspk, max_spks)
    stats = report_diarization_error(preds_realspk, labels_realspk, label_delay=label_delay)
    return {key: float(value[0]) for key, value in stats.items()}


def sort_reference_binary(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape[1] == 0:
        return reference, np.empty((0,), dtype=np.int64)
    active = reference > 0
    inactive = ~active.any(axis=0)
    first_active = np.full(reference.shape[1], reference.shape[0], dtype=np.int64)
    if (~inactive).any():
        first_active[~inactive] = active.argmax(axis=0)[~inactive]
    first_active[inactive] = reference.shape[0] + np.arange(reference.shape[1], dtype=np.int64)[inactive]
    order = np.argsort(first_active, kind="stable")
    return reference[:, order], order


def speaker_labels_from_rttm(rttm_path: Path) -> list[str]:
    first_start: dict[str, float] = {}
    with open(rttm_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            speaker = parts[7]
            start = float(parts[3])
            first_start[speaker] = min(start, first_start.get(speaker, start))
    return [speaker for speaker, _ in sorted(first_start.items(), key=lambda item: (item[1], item[0]))]


def build_annotation(binary: np.ndarray, scale: int, uri: str) -> Annotation:
    annotation = Annotation(uri=uri)
    for speaker_index, frames in enumerate(binary.T):
        padded = F.pad(torch.from_numpy(frames).float(), (1, 1), "constant")
        changes, = torch.where(torch.diff(padded, dim=0) != 0)
        for start, stop in zip(changes[::2], changes[1::2]):
            annotation[Segment(float(start.item() * scale), float(stop.item() * scale))] = str(speaker_index)
    return annotation


def compute_repo_style_der(
    config: dict,
    offline_dataset: OfflineKaldiDiarizationDataset,
    recording_id: str,
    probabilities: np.ndarray,
    threshold: float,
    median_width: int,
    collar_seconds: float,
) -> dict:
    ref_full, _ = offline_dataset.__getfulllabel__(0)
    reference_full, sort_order = sort_reference_binary(ref_full.numpy())
    pred_binary = (probabilities > threshold).astype(np.float32)
    if median_width > 1:
        pred_binary = medfilt(pred_binary, kernel_size=(median_width, 1)).astype(np.float32)
    full_frame_rate = config["data"]["feat"]["sample_rate"] / config["data"]["feat"]["hop_length"]
    collar_frames_total = int(round(collar_seconds * 2 * full_frame_rate))
    metric = DiarizationErrorRate(collar=collar_frames_total)
    reference = build_annotation(reference_full, scale=1, uri=recording_id)
    hypothesis = build_annotation(
        pred_binary,
        scale=config["data"]["subsampling"],
        uri=recording_id,
    )
    result = metric(reference, hypothesis, detailed=True)
    total = float(result["total"])
    return {
        "speaker_scored": total,
        "speaker_miss": float(result["missed detection"]),
        "speaker_false_alarm": float(result["false alarm"]),
        "speaker_error": float(result["confusion"]),
        "der": float((result["missed detection"] + result["false alarm"] + result["confusion"]) / total) if total else 0.0,
        "collar_frames_total": collar_frames_total,
    }


def save_h5(probabilities: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("T_hat", data=probabilities)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    kaldi_dir = build_kaldi_dir(
        args.audio,
        args.ref_rttm,
        args.output_dir,
        target_sample_rate=int(config["data"]["feat"]["sample_rate"]),
    )
    streaming_dataset = build_streaming_dataset(config, kaldi_dir)
    offline_dataset = build_offline_dataset(config, kaldi_dir)
    device = torch.device(args.device)
    model = build_model(config, device)
    load_model_state(model, args.checkpoint)
    model.eval()

    seed = int(config["training"]["seed"] or 0)
    feats, labels, recording_ids = streaming_dataset[(0, seed)]
    recording_id = recording_ids
    feats = [feats.to(device)]
    labels = [labels.to(device)]
    clip_lengths = [feats[0].shape[0]]

    with torch.no_grad():
        prepared_labels, n_spks = prepare_test_labels(labels, clip_lengths)
        preds, _, _ = model.test(feats, clip_lengths, config["data"]["max_speakers"] + 2)

    preliminary = preliminary_metrics(
        preds=preds,
        labels=prepared_labels,
        n_spks=n_spks,
        max_spks=int(config["data"]["max_speakers"]),
        label_delay=int(config["data"]["label_delay"]),
    )

    label_delay = int(config["data"]["label_delay"])
    speaker_logits = preds[0][:, 1:-1]
    if label_delay > 0:
        speaker_logits = torch.cat(
            [speaker_logits[label_delay:], speaker_logits[-1].unsqueeze(0).repeat(label_delay, 1)],
            dim=0,
        )
    speaker_logits_np = speaker_logits.detach().cpu().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-speaker_logits_np))

    raw_dir = args.output_dir / "repo_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / f"{recording_id}.npy", speaker_logits_np)
    save_h5(probabilities, raw_dir / f"{recording_id}.h5")

    reference_binary, sort_order = sort_reference_binary(labels[0].detach().cpu().numpy())
    frame_rate = (
        config["data"]["feat"]["sample_rate"]
        / config["data"]["feat"]["hop_length"]
        / config["data"]["subsampling"]
    )
    corrected = compute_der(
        probabilities=probabilities,
        reference_binary=reference_binary,
        threshold=args.threshold,
        median_width=args.median,
        collar_seconds=args.collar_seconds,
        frame_rate=frame_rate,
    )
    speaker_labels = speaker_labels_from_rttm(args.ref_rttm)
    write_rttm(
        recording_id=recording_id,
        binary_prediction=corrected["mapped_binary"],
        output_path=args.output_dir / f"{recording_id}_repo_eval_prediction.rttm",
        frame_rate=frame_rate,
        speaker_labels=speaker_labels,
    )
    save_heatmap(
        reference_binary=reference_binary,
        mapped_binary=corrected["mapped_binary"],
        mapped_probabilities=corrected["mapped_probabilities"],
        frame_rate=frame_rate,
        speaker_labels=speaker_labels,
        output_path=args.output_dir / f"{recording_id}_repo_eval_heatmap.png",
    )

    repo_style = compute_repo_style_der(
        config=config,
        offline_dataset=offline_dataset,
        recording_id=recording_id,
        probabilities=probabilities,
        threshold=args.threshold,
        median_width=args.median,
        collar_seconds=args.collar_seconds,
    )
    metrics = {
        "recording_id": recording_id,
        "audio": str(args.audio),
        "reference_rttm": str(args.ref_rttm),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": args.device,
        "evaluation": {
            "threshold": args.threshold,
            "median_width": args.median,
            "collar_seconds": args.collar_seconds,
            "frame_rate": frame_rate,
            "paper_style_decode_tracks": int(config["data"]["max_speakers"]),
        },
        "preliminary_repo_test_step": preliminary,
        "repo_style_der": repo_style,
        "corrected_mapped_der": {
            "der": float(corrected["der"]),
            "speaker_scored": float(corrected["speaker_scored"]),
            "speaker_miss": float(corrected["speaker_miss"]),
            "speaker_false_alarm": float(corrected["speaker_false_alarm"]),
            "speaker_error": float(corrected["speaker_error"]),
        },
        "artifacts": {
            "kaldi_dir": str(kaldi_dir),
            "raw_logits_npy": str(raw_dir / f"{recording_id}.npy"),
            "raw_probabilities_h5": str(raw_dir / f"{recording_id}.h5"),
            "prediction_rttm": str(args.output_dir / f"{recording_id}_repo_eval_prediction.rttm"),
            "heatmap": str(args.output_dir / f"{recording_id}_repo_eval_heatmap.png"),
        },
    }
    save_json(metrics, args.output_dir / f"{recording_id}_repo_eval_metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
