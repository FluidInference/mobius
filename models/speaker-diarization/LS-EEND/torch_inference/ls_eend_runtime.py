from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "datasets"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (  # noqa: E402
    OnlineConformerRetentionDADiarization,
)
from torch_inference.ls_eend_stateful import StatefulLSEENDRunner, StreamingFeatureExtractor, StreamingUpdate  # noqa: E402
import feature as ls_feature  # noqa: E402


DEFAULT_CONFIG = ROOT / "conf" / "spk_onl_conformer_retention_enc_dec_nonautoreg_infer.yaml"
DEFAULT_CHECKPOINT = ROOT / "model_checkpoints" / "ls_eend_1-8spk_16_25_avg_model.ckpt"


class _RefLoader(yaml.SafeLoader):
    pass


def _ref_constructor(loader, node):
    return loader.construct_scalar(node)


_RefLoader.add_constructor("!ref", _ref_constructor)


def load_config(config_path: Path) -> dict:
    try:
        import hyperpyyaml

        with open(config_path, "r", encoding="utf-8") as handle:
            config = hyperpyyaml.load_hyperpyyaml(handle)
    except ImportError:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=_RefLoader)
        if isinstance(config["model"]["params"].get("max_seqlen"), str):
            config["model"]["params"]["max_seqlen"] = config["data"]["chunk_size"]
    if config["data"]["num_speakers"] is None:
        config["data"]["num_speakers"] = config["data"]["max_speakers"] + 2
    return config


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1, dtype=np.float32)


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path)
    return ensure_mono(audio), sample_rate


def extract_features(audio: np.ndarray, sample_rate: int, config: dict) -> torch.Tensor:
    target_sr = config["data"]["feat"]["sample_rate"]
    if sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    # The repo dataset path truncates recordings to whole subsampling blocks before
    # STFT extraction. Matching that here keeps the wrapper aligned with the
    # original Kaldi-style evaluation flow.
    frame_shift = config["data"]["feat"]["hop_length"]
    subsampling = config["data"]["subsampling"]
    usable_samples = (len(audio) // (frame_shift * subsampling)) * (frame_shift * subsampling)
    if usable_samples > 0:
        audio = audio[:usable_samples]
    stft = ls_feature.stft(
        audio,
        frame_size=config["data"]["feat"]["win_length"],
        frame_shift=frame_shift,
    )
    feats = ls_feature.transform(stft, config["data"]["feat_type"])
    feats = ls_feature.splice(feats, config["data"]["context_recp"])
    feats, _ = ls_feature.subsample(feats, feats, subsampling)
    return torch.from_numpy(np.array(feats, copy=True)).float()


def frame_hz(config: dict) -> float:
    return config["data"]["feat"]["sample_rate"] / (
        config["data"]["feat"]["hop_length"] * config["data"]["subsampling"]
    )


@dataclass
class InferenceResult:
    logits: np.ndarray
    probabilities: np.ndarray
    full_logits: np.ndarray
    full_probabilities: np.ndarray
    frame_hz: float
    duration_seconds: float


class StreamingSession:
    def __init__(self, engine: "LSEENDInferenceEngine", input_sample_rate: int) -> None:
        self.engine = engine
        self.input_sample_rate = input_sample_rate
        if input_sample_rate != self.engine.target_sample_rate:
            raise ValueError(
                "Stateful LS-EEND streaming expects audio at "
                f"{self.engine.target_sample_rate} Hz, got {input_sample_rate} Hz."
            )
        self.feature_extractor = StreamingFeatureExtractor(self.engine.config)
        self.runner = StatefulLSEENDRunner(self.engine.model, self.engine.decode_max_nspks)
        self.total_input_samples = 0
        self.full_logit_chunks: list[np.ndarray] = []
        self.emitted_frames = 0
        self.finalized = False

    def push_audio(self, chunk: np.ndarray) -> StreamingUpdate | None:
        if self.finalized:
            raise RuntimeError("Streaming session is already finalized.")
        chunk = ensure_mono(chunk)
        if chunk.size == 0:
            return None
        self.total_input_samples += len(chunk)
        features = self.feature_extractor.push_audio(chunk)
        full_logits = self.runner.push_features(features).detach().cpu().numpy()
        return self._store_update(full_logits)

    def finalize(self) -> StreamingUpdate | None:
        if self.finalized:
            return None
        features = self.feature_extractor.finalize()
        flushed_logits = self.runner.push_features(features)
        tail_logits = self.runner.finalize()
        parts = [part for part in (flushed_logits, tail_logits) if part.numel() > 0]
        merged = torch.cat(parts, dim=0).detach().cpu().numpy() if parts else np.zeros((0, self.engine.decode_max_nspks), dtype=np.float32)
        self.finalized = True
        return self._store_update(merged)

    def snapshot(self) -> InferenceResult:
        if self.full_logit_chunks:
            full_logits = np.concatenate(self.full_logit_chunks, axis=0)
        else:
            full_logits = np.zeros((0, self.engine.decode_max_nspks), dtype=np.float32)
        full_probabilities = 1.0 / (1.0 + np.exp(-full_logits))
        logits = full_logits[:, 1:-1]
        probabilities = full_probabilities[:, 1:-1]
        return InferenceResult(
            logits=logits,
            probabilities=probabilities,
            full_logits=full_logits,
            full_probabilities=full_probabilities,
            frame_hz=self.engine.model_frame_hz,
            duration_seconds=float(self.total_input_samples / max(self.input_sample_rate, 1)),
        )

    def _store_update(self, full_logits: np.ndarray) -> StreamingUpdate | None:
        start_frame = self.emitted_frames
        if full_logits.size > 0:
            full_logits = full_logits.astype(np.float32, copy=False)
            self.full_logit_chunks.append(full_logits)
            self.emitted_frames += full_logits.shape[0]
            full_probabilities = 1.0 / (1.0 + np.exp(-full_logits))
            committed_logits = full_logits[:, 1:-1]
            committed_probabilities = full_probabilities[:, 1:-1]
        else:
            committed_logits = np.zeros((0, max(self.engine.decode_max_nspks - 2, 0)), dtype=np.float32)
            committed_probabilities = committed_logits.copy()
        preview_full_logits = self.runner.peek_preview().detach().cpu().numpy()
        if preview_full_logits.size > 0:
            preview_full_logits = preview_full_logits.astype(np.float32, copy=False)
            preview_full_probabilities = 1.0 / (1.0 + np.exp(-preview_full_logits))
            preview_logits = preview_full_logits[:, 1:-1]
            preview_probabilities = preview_full_probabilities[:, 1:-1]
        else:
            preview_logits = np.zeros((0, committed_logits.shape[1]), dtype=np.float32)
            preview_probabilities = preview_logits.copy()
        if committed_logits.size == 0 and preview_logits.size == 0:
            return None
        return StreamingUpdate(
            start_frame=start_frame,
            logits=committed_logits,
            probabilities=committed_probabilities,
            preview_start_frame=self.emitted_frames,
            preview_logits=preview_logits,
            preview_probabilities=preview_probabilities,
            frame_hz=self.engine.model_frame_hz,
            duration_seconds=float(self.total_input_samples / max(self.input_sample_rate, 1)),
            total_emitted_frames=self.emitted_frames,
        )


class LSEENDInferenceEngine:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        config_path: Path = DEFAULT_CONFIG,
        device: str | None = None,
        actual_num_speakers: int | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.config = load_config(self.config_path)
        self.actual_num_speakers = actual_num_speakers
        self.decode_max_nspks = self.config["data"]["max_speakers"] + 2
        self.target_sample_rate = int(self.config["data"]["feat"]["sample_rate"])
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model = self._load_model().to(self.device)
        self.model.eval()
        self.model_frame_hz = frame_hz(self.config)
        stft_fft_size = 1 << (int(self.config["data"]["feat"]["win_length"]) - 1).bit_length()
        self.streaming_latency_seconds = (
            (stft_fft_size // 2)
            + (self.config["data"]["context_recp"] * self.config["data"]["feat"]["hop_length"])
            + (self.config["model"]["params"]["conv_delay"] * self.config["data"]["subsampling"] * self.config["data"]["feat"]["hop_length"])
        ) / self.target_sample_rate

    def _load_model(self) -> OnlineConformerRetentionDADiarization:
        model = OnlineConformerRetentionDADiarization(
            n_speakers=self.config["data"]["num_speakers"],
            in_size=(2 * self.config["data"]["context_recp"] + 1) * self.config["data"]["feat"]["n_mels"],
            **self.config["model"]["params"],
        )
        load_kwargs = {"map_location": "cpu"}
        if "weights_only" in torch.load.__code__.co_varnames:
            load_kwargs["weights_only"] = False
        state_dict = torch.load(self.checkpoint_path, **load_kwargs)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        cleaned = {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }
        model.load_state_dict(cleaned)
        return model

    def create_session(self, input_sample_rate: int) -> StreamingSession:
        return StreamingSession(self, input_sample_rate)

    def infer_audio(self, audio: np.ndarray, sample_rate: int) -> InferenceResult:
        feats = extract_features(audio, sample_rate, self.config).to(self.device)
        with torch.no_grad():
            preds, _, _ = self.model.test([feats], [len(feats)], max_nspks=self.decode_max_nspks)
        full_logits = preds[0].detach().cpu().numpy()
        full_probs = torch.sigmoid(preds[0]).detach().cpu().numpy()
        logits = full_logits[:, 1:-1]
        probs = full_probs[:, 1:-1]
        return InferenceResult(
            logits=logits,
            probabilities=probs,
            full_logits=full_logits,
            full_probabilities=full_probs,
            frame_hz=self.model_frame_hz,
            duration_seconds=float(len(audio) / sample_rate),
        )

    def simulate_streaming_file(
        self,
        audio_path: Path,
        chunk_seconds: float,
    ) -> tuple[InferenceResult, list[dict]]:
        audio, sample_rate = load_audio(audio_path)
        if sample_rate != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.target_sample_rate).astype(np.float32, copy=False)
            sample_rate = self.target_sample_rate
        session = self.create_session(sample_rate)
        chunk_size = max(1, int(round(chunk_seconds * sample_rate)))
        updates = []
        for chunk_index, start in enumerate(range(0, len(audio), chunk_size), start=1):
            stop = min(len(audio), start + chunk_size)
            update = session.push_audio(audio[start:stop])
            updates.append(
                {
                    "chunk_index": chunk_index,
                    "buffer_seconds": round(stop / sample_rate, 3),
                    "num_frames_emitted": int(0 if update is None else update.probabilities.shape[0]),
                    "total_frames_emitted": int(session.emitted_frames),
                }
            )
        final_update = session.finalize()
        if final_update is not None:
            updates.append(
                {
                    "chunk_index": len(updates) + 1,
                    "buffer_seconds": round(len(audio) / sample_rate, 3),
                    "num_frames_emitted": int(final_update.probabilities.shape[0]),
                    "total_frames_emitted": int(session.emitted_frames),
                    "flush": True,
                }
            )
        result = session.snapshot()
        if result.full_probabilities.shape[0] == 0:
            raise ValueError(f"No audio found in {audio_path}")
        return result, updates


def parse_rttm(rttm_path: Path) -> tuple[list[dict], list[str]]:
    entries = []
    speaker_order = []
    with open(rttm_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            speaker = parts[7]
            if speaker not in speaker_order:
                speaker_order.append(speaker)
            entries.append(
                {
                    "recording_id": parts[1],
                    "start": float(parts[3]),
                    "duration": float(parts[4]),
                    "speaker": speaker,
                }
            )
    return entries, speaker_order


def rttm_to_frame_matrix(entries: list[dict], speakers: list[str], num_frames: int, frame_rate: float) -> np.ndarray:
    matrix = np.zeros((num_frames, len(speakers)), dtype=np.float32)
    speaker_to_index = {speaker: index for index, speaker in enumerate(speakers)}
    for entry in entries:
        start = int(round(entry["start"] * frame_rate))
        stop = int(round((entry["start"] + entry["duration"]) * frame_rate))
        matrix[start : min(stop, num_frames), speaker_to_index[entry["speaker"]]] = 1.0
    return matrix


def collar_mask(reference: np.ndarray, collar_frames: int) -> np.ndarray:
    if collar_frames <= 0:
        return np.ones(reference.shape[0], dtype=bool)
    mask = np.ones(reference.shape[0], dtype=bool)
    for column in range(reference.shape[1]):
        padded = np.pad(reference[:, column], (1, 1), constant_values=0)
        changes = np.where(np.diff(padded) != 0)[0]
        for change in changes:
            start = max(0, change - collar_frames)
            stop = min(reference.shape[0], change + collar_frames)
            mask[start:stop] = False
    return mask


def _pair_cost(pred_column: np.ndarray, ref_column: np.ndarray) -> float:
    pred_column = pred_column.astype(bool)
    ref_column = ref_column.astype(bool)
    n_ref = ref_column.sum()
    n_sys = pred_column.sum()
    n_map = np.logical_and(pred_column, ref_column).sum()
    miss = max(n_ref - n_sys, 0)
    false_alarm = max(n_sys - n_ref, 0)
    speaker_error = min(n_ref, n_sys) - n_map
    return float(miss + false_alarm + speaker_error)


def select_speaker_probabilities(probabilities: np.ndarray, num_speakers: int | None) -> np.ndarray:
    if num_speakers is None:
        return probabilities
    clipped = max(0, min(int(num_speakers), probabilities.shape[1]))
    return probabilities[:, :clipped]


def map_predictions(
    prediction_binary: np.ndarray,
    reference_binary: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], list[int]]:
    masked_pred = prediction_binary[valid_mask]
    masked_ref = reference_binary[valid_mask]
    num_pred = prediction_binary.shape[1]
    num_ref = reference_binary.shape[1]
    mapped = np.zeros((prediction_binary.shape[0], num_ref), dtype=np.float32)
    assignment: dict[int, int] = {}
    if num_pred == 0 or num_ref == 0:
        return mapped, assignment, list(range(num_pred))
    cost = np.zeros((num_pred, num_ref), dtype=np.float32)
    for pred_index in range(num_pred):
        for ref_index in range(num_ref):
            cost[pred_index, ref_index] = _pair_cost(masked_pred[:, pred_index], masked_ref[:, ref_index])
    row_index, col_index = linear_sum_assignment(cost)
    matched_pred = set()
    for pred_index, ref_index in zip(row_index, col_index):
        mapped[:, ref_index] = prediction_binary[:, pred_index]
        assignment[int(ref_index)] = int(pred_index)
        matched_pred.add(int(pred_index))
    unmatched_pred = [pred_index for pred_index in range(num_pred) if pred_index not in matched_pred]
    return mapped, assignment, unmatched_pred


def compute_der(
    probabilities: np.ndarray,
    reference_binary: np.ndarray,
    threshold: float,
    median_width: int,
    collar_seconds: float,
    frame_rate: float,
) -> dict:
    prediction_binary = (probabilities > threshold).astype(np.float32)
    if median_width > 1:
        prediction_binary = medfilt(prediction_binary, kernel_size=(median_width, 1)).astype(np.float32)
    valid_mask = collar_mask(reference_binary, int(round(collar_seconds * frame_rate)))
    mapped_binary, assignment, unmatched_pred = map_predictions(prediction_binary, reference_binary, valid_mask)
    mapped_probabilities = np.zeros((probabilities.shape[0], reference_binary.shape[1]), dtype=np.float32)
    for ref_index, pred_index in assignment.items():
        mapped_probabilities[:, ref_index] = probabilities[:, pred_index]
    extra_binary = prediction_binary[:, unmatched_pred] if unmatched_pred else np.zeros((prediction_binary.shape[0], 0), dtype=np.float32)
    scored_reference = np.concatenate(
        [reference_binary, np.zeros((reference_binary.shape[0], extra_binary.shape[1]), dtype=np.float32)],
        axis=1,
    )
    scored_prediction = np.concatenate([mapped_binary, extra_binary], axis=1)
    masked_ref = scored_reference[valid_mask]
    masked_pred = scored_prediction[valid_mask]
    n_ref = masked_ref.sum(axis=1)
    n_sys = masked_pred.sum(axis=1)
    miss = np.maximum(n_ref - n_sys, 0).sum()
    false_alarm = np.maximum(n_sys - n_ref, 0).sum()
    mapped_overlap = np.logical_and(masked_ref == 1, masked_pred == 1).sum(axis=1)
    speaker_error = (np.minimum(n_ref, n_sys) - mapped_overlap).sum()
    speaker_scored = masked_ref.sum()
    der = float((miss + false_alarm + speaker_error) / speaker_scored) if speaker_scored else 0.0
    return {
        "der": der,
        "speaker_scored": float(speaker_scored),
        "speaker_miss": float(miss),
        "speaker_false_alarm": float(false_alarm),
        "speaker_error": float(speaker_error),
        "threshold": threshold,
        "median_width": median_width,
        "collar_seconds": collar_seconds,
        "mapped_binary": mapped_binary,
        "mapped_probabilities": mapped_probabilities,
        "valid_mask": valid_mask,
        "assignment": assignment,
        "unmatched_prediction_indices": unmatched_pred,
    }


def write_rttm(recording_id: str, binary_prediction: np.ndarray, output_path: Path, frame_rate: float, speaker_labels: Iterable[str] | None = None) -> None:
    speaker_labels = list(speaker_labels or [f"spk{index:02d}" for index in range(binary_prediction.shape[1])])
    with open(output_path, "w", encoding="utf-8") as handle:
        for speaker_index, speaker in enumerate(speaker_labels):
            padded = np.pad(binary_prediction[:, speaker_index], (1, 1), constant_values=0)
            changes = np.where(np.diff(padded) != 0)[0]
            for start, stop in zip(changes[::2], changes[1::2]):
                start_seconds = start / frame_rate
                duration_seconds = (stop - start) / frame_rate
                handle.write(
                    f"SPEAKER {recording_id} 1 {start_seconds:.3f} {duration_seconds:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
                )


def save_heatmap(reference_binary: np.ndarray, mapped_binary: np.ndarray, mapped_probabilities: np.ndarray, frame_rate: float, speaker_labels: list[str], output_path: Path) -> None:
    duration_seconds = reference_binary.shape[0] / frame_rate
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
    plots = [
        (reference_binary, "Expected (RTTM)", "Greys"),
        (mapped_binary, "Predicted (Mapped, Binary)", "Greys"),
        (mapped_probabilities, "Predicted (Mapped, Probability)", "viridis"),
    ]
    for axis, (matrix, title, cmap) in zip(axes, plots):
        image = axis.imshow(
            matrix.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0.0, duration_seconds, -0.5, matrix.shape[1] - 0.5],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(title)
        axis.set_yticks(range(len(speaker_labels)))
        axis.set_yticklabels(speaker_labels)
        if cmap != "Greys":
            fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("Time (seconds)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
