from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import json
from pathlib import Path

import coremltools as ct
import librosa
import numpy as np

from coreml_inference.ls_eend_coreml import CoreMLStateLayout, initial_state_tensors
from torch_inference.ls_eend_runtime import (
    InferenceResult,
    StreamingUpdate,
    StreamingFeatureExtractor,
    ensure_mono,
    extract_features,
    frame_hz,
    load_audio,
    load_config,
)


def _compute_units(name: str):
    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def _load_metadata(coreml_model_path: Path) -> dict:
    metadata_path = coreml_model_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing CoreML metadata JSON next to package: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _layout_from_metadata(metadata: dict) -> CoreMLStateLayout:
    return CoreMLStateLayout(
        input_dim=int(metadata["input_dim"]),
        full_output_dim=int(metadata["full_output_dim"]),
        real_output_dim=int(metadata["real_output_dim"]),
        encoder_layers=int(metadata["encoder_layers"]),
        decoder_layers=int(metadata["decoder_layers"]),
        encoder_dim=int(metadata["encoder_dim"]),
        num_heads=int(metadata["num_heads"]),
        key_dim=int(metadata["key_dim"]),
        head_dim=int(metadata["head_dim"]),
        encoder_conv_cache_len=int(metadata["encoder_conv_cache_len"]),
        top_buffer_len=int(metadata["top_buffer_len"]),
        conv_delay=int(metadata["conv_delay"]),
        max_nspks=int(metadata["max_nspks"]),
    )


class CoreMLStreamingSession:
    def __init__(self, engine: "CoreMLLSEENDInferenceEngine", input_sample_rate: int) -> None:
        self.engine = engine
        self.input_sample_rate = int(input_sample_rate)
        if self.input_sample_rate != self.engine.target_sample_rate:
            raise ValueError(
                "Stateful LS-EEND streaming expects audio at "
                f"{self.engine.target_sample_rate} Hz, got {self.input_sample_rate} Hz."
            )
        self.feature_extractor = StreamingFeatureExtractor(self.engine.config)
        self.state = initial_state_tensors(self.engine.layout, dtype=np.float32)
        self.zero_frame = np.zeros((1, 1, self.engine.layout.input_dim), dtype=np.float32)
        self.total_input_samples = 0
        self.total_feature_frames = 0
        self.emitted_frames = 0
        self.full_logit_chunks: list[np.ndarray] = []
        self.finalized = False

    def push_audio(self, chunk: np.ndarray) -> StreamingUpdate | None:
        if self.finalized:
            raise RuntimeError("Streaming session is already finalized.")
        chunk = ensure_mono(chunk)
        if chunk.size == 0:
            return None
        self.total_input_samples += len(chunk)
        features = self.feature_extractor.push_audio(chunk)
        committed = self._ingest_features(features)
        return self._build_update(committed, include_preview=True)

    def finalize(self) -> StreamingUpdate | None:
        if self.finalized:
            return None
        features = self.feature_extractor.finalize()
        committed = self._ingest_features(features)
        pending = self.total_feature_frames - self.emitted_frames
        tail = self._flush_tail(self.state, pending) if pending > 0 else np.zeros((0, self.engine.decode_max_nspks), dtype=np.float32)
        merged = committed if tail.size == 0 else np.concatenate([committed, tail], axis=0)
        self.finalized = True
        self.emitted_frames += tail.shape[0]
        return self._build_update(merged, include_preview=False)

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

    def _ingest_features(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros((0, self.engine.decode_max_nspks), dtype=np.float32)
        outputs: list[np.ndarray] = []
        for frame in features:
            should_decode = 1.0 if self.total_feature_frames >= self.engine.layout.conv_delay else 0.0
            prediction = self.engine._predict_step(
                frame=frame.reshape(1, 1, -1).astype(np.float32, copy=False),
                state=self.state,
                ingest=1.0,
                decode=should_decode,
            )
            self.state = self.engine._next_state(prediction)
            self.total_feature_frames += 1
            if should_decode:
                outputs.append(prediction["full_logits"].reshape(1, -1).astype(np.float32, copy=False))
                self.emitted_frames += 1
        if outputs:
            return np.concatenate(outputs, axis=0)
        return np.zeros((0, self.engine.decode_max_nspks), dtype=np.float32)

    def _flush_tail(self, state: dict[str, np.ndarray], pending: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for _ in range(pending):
            prediction = self.engine._predict_step(
                frame=self.zero_frame,
                state=state,
                ingest=0.0,
                decode=1.0,
            )
            state = self.engine._next_state(prediction)
            outputs.append(prediction["full_logits"].reshape(1, -1).astype(np.float32, copy=False))
        if outputs:
            return np.concatenate(outputs, axis=0)
        return np.zeros((0, self.engine.decode_max_nspks), dtype=np.float32)

    def _build_update(self, committed_full_logits: np.ndarray, include_preview: bool) -> StreamingUpdate | None:
        start_frame = self.emitted_frames - committed_full_logits.shape[0]
        if committed_full_logits.size > 0:
            self.full_logit_chunks.append(committed_full_logits.astype(np.float32, copy=False))
            committed_full_probabilities = 1.0 / (1.0 + np.exp(-committed_full_logits))
            committed_logits = committed_full_logits[:, 1:-1]
            committed_probabilities = committed_full_probabilities[:, 1:-1]
        else:
            committed_logits = np.zeros((0, max(self.engine.decode_max_nspks - 2, 0)), dtype=np.float32)
            committed_probabilities = committed_logits.copy()

        if include_preview:
            pending = self.total_feature_frames - self.emitted_frames
            preview_state = {key: value.copy() for key, value in self.state.items()}
            preview_full_logits = self._flush_tail(preview_state, pending)
        else:
            preview_full_logits = np.zeros((0, self.engine.decode_max_nspks), dtype=np.float32)

        if preview_full_logits.size > 0:
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


class CoreMLLSEENDInferenceEngine:
    def __init__(
        self,
        coreml_model_path: Path,
        config_path: Path | None = None,
        compute_units: str = "cpu_only",
    ) -> None:
        self.coreml_model_path = Path(coreml_model_path)
        self.metadata = _load_metadata(self.coreml_model_path)
        resolved_config = config_path or Path(self.metadata["config"])
        self.config_path = Path(resolved_config)
        self.config = load_config(self.config_path)
        self.layout = _layout_from_metadata(self.metadata)
        self.decode_max_nspks = self.layout.max_nspks
        self.target_sample_rate = int(self.config["data"]["feat"]["sample_rate"])
        self.model_frame_hz = frame_hz(self.config)
        stft_fft_size = 1 << (int(self.config["data"]["feat"]["win_length"]) - 1).bit_length()
        self.streaming_latency_seconds = (
            (stft_fft_size // 2)
            + (self.config["data"]["context_recp"] * self.config["data"]["feat"]["hop_length"])
            + (self.config["model"]["params"]["conv_delay"] * self.config["data"]["subsampling"] * self.config["data"]["feat"]["hop_length"])
        ) / self.target_sample_rate
        self.model = ct.models.MLModel(str(self.coreml_model_path), compute_units=_compute_units(compute_units))

    def create_session(self, input_sample_rate: int) -> CoreMLStreamingSession:
        return CoreMLStreamingSession(self, input_sample_rate)

    def infer_audio(self, audio: np.ndarray, sample_rate: int) -> InferenceResult:
        features = extract_features(audio, sample_rate, self.config).numpy()
        session = self.create_session(self.target_sample_rate)
        session.total_input_samples = len(audio) if sample_rate == self.target_sample_rate else int(
            round(len(audio) * (self.target_sample_rate / max(sample_rate, 1)))
        )
        committed = session._ingest_features(features)
        pending = session.total_feature_frames - session.emitted_frames
        tail = session._flush_tail(session.state, pending) if pending > 0 else np.zeros((0, self.decode_max_nspks), dtype=np.float32)
        full_logits = committed if tail.size == 0 else np.concatenate([committed, tail], axis=0)
        if full_logits.size > 0:
            session.full_logit_chunks = [full_logits.astype(np.float32, copy=False)]
            session.emitted_frames = full_logits.shape[0]
        return session.snapshot()

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

    def _predict_step(
        self,
        frame: np.ndarray,
        state: dict[str, np.ndarray],
        ingest: float,
        decode: float,
    ) -> dict[str, np.ndarray]:
        return self.model.predict(
            {
                "frame": frame,
                "enc_ret_kv": state["enc_ret_kv"],
                "enc_ret_scale": state["enc_ret_scale"],
                "enc_conv_cache": state["enc_conv_cache"],
                "dec_ret_kv": state["dec_ret_kv"],
                "dec_ret_scale": state["dec_ret_scale"],
                "top_buffer": state["top_buffer"],
                "ingest": np.array([ingest], dtype=np.float32),
                "decode": np.array([decode], dtype=np.float32),
            }
        )

    @staticmethod
    def _next_state(prediction: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            "enc_ret_kv": prediction["enc_ret_kv_out"],
            "enc_ret_scale": prediction["enc_ret_scale_out"],
            "enc_conv_cache": prediction["enc_conv_cache_out"],
            "dec_ret_kv": prediction["dec_ret_kv_out"],
            "dec_ret_scale": prediction["dec_ret_scale_out"],
            "top_buffer": prediction["top_buffer_out"],
        }
