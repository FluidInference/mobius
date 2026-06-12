#!/usr/bin/env python3
"""Greedy RNNT decoding over the parakeet-unified CoreML components.

Shared by compare-models.py (parity validation) and benchmark_wer.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np

SAMPLE_RATE = 16000
ENCODER_FRAME_SAMPLES = 1280  # 80 ms
OFFLINE_WINDOW_SAMPLES = 15 * SAMPLE_RATE
MAX_SYMBOLS_PER_FRAME = 10
BLANK_IDX = 1024


class CoreMLRnnt:
    """CoreML component bundle + greedy RNNT decode loop."""

    def __init__(
        self,
        coreml_dir: Path,
        blank_idx: int = BLANK_IDX,
        streaming_suffix: Optional[str] = None,
        encoder_compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE,
    ) -> None:
        self.preprocessor = ct.models.MLModel(
            str(coreml_dir / "parakeet_unified_preprocessor.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        encoder_name = (
            f"parakeet_unified_encoder_streaming_{streaming_suffix}.mlpackage"
            if streaming_suffix
            else "parakeet_unified_encoder.mlpackage"
        )
        self.encoder = ct.models.MLModel(str(coreml_dir / encoder_name), compute_units=encoder_compute_units)
        self.decoder = ct.models.MLModel(
            str(coreml_dir / "parakeet_unified_decoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        self.joint_decision = ct.models.MLModel(
            str(coreml_dir / "parakeet_unified_joint_decision_single_step.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        self.blank_idx = blank_idx
        spec = self.encoder.get_spec()
        self.mel_frames = int(spec.description.input[0].type.multiArrayType.shape[2])

    def encode(self, audio: np.ndarray, num_samples: int) -> Tuple[np.ndarray, int]:
        """audio is zero-padded to the encoder window; num_samples is valid length."""
        mel_out = self.preprocessor.predict(
            {
                "audio_signal": audio[None, :].astype(np.float32),
                "audio_length": np.array([num_samples], dtype=np.int32),
            }
        )
        mel = mel_out["mel"].astype(np.float32)
        # The preprocessor is variable-length; pad mel to the encoder's fixed frame count.
        if mel.shape[2] < self.mel_frames:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, self.mel_frames - mel.shape[2])))
        enc_out = self.encoder.predict(
            {"mel": mel, "mel_length": mel_out["mel_length"].astype(np.int32)}
        )
        return enc_out["encoder"], int(enc_out["encoder_length"][0])

    def init_state(self) -> Tuple[np.ndarray, np.ndarray, int]:
        h = np.zeros((2, 1, 640), dtype=np.float32)
        c = np.zeros((2, 1, 640), dtype=np.float32)
        return h, c, self.blank_idx

    def decoder_step(self, token: int, h: np.ndarray, c: np.ndarray):
        out = self.decoder.predict(
            {
                "targets": np.array([[token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "h_in": h,
                "c_in": c,
            }
        )
        return out["decoder"], out["h_out"], out["c_out"]

    def decode_frames(
        self,
        encoder_out: np.ndarray,
        num_frames: int,
        state: Tuple,
        dec_out: Optional[np.ndarray],
        frame_offset: int = 0,
    ) -> Tuple[List[int], Tuple, np.ndarray]:
        """Greedy RNNT over encoder frames [frame_offset, frame_offset+num_frames)."""
        h, c, last_token = state
        if dec_out is None:
            dec_out, h, c = self.decoder_step(last_token, h, c)
        tokens: List[int] = []
        for t in range(frame_offset, frame_offset + num_frames):
            enc_step = encoder_out[:, :, t : t + 1].astype(np.float32)
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                jd = self.joint_decision.predict(
                    {"encoder_step": enc_step, "decoder_step": dec_out[:, :, :1].astype(np.float32)}
                )
                token = int(jd["token_id"].reshape(-1)[0])
                if token == self.blank_idx:
                    break
                tokens.append(token)
                last_token = token
                dec_out, h, c = self.decoder_step(token, h, c)
        return tokens, (h, c, last_token), dec_out


def offline_transcribe(cm: CoreMLRnnt, audio: np.ndarray) -> List[int]:
    """Greedy decode of audio that fits the fixed 15 s offline window."""
    assert audio.size <= OFFLINE_WINDOW_SAMPLES, f"{audio.size} samples exceed the 15 s offline window"
    window = np.zeros(OFFLINE_WINDOW_SAMPLES, dtype=np.float32)
    window[: audio.size] = audio
    encoder_out, enc_len = cm.encode(window, audio.size)
    tokens, _, _ = cm.decode_frames(encoder_out, enc_len, cm.init_state(), None)
    return tokens


def stream_transcribe(cm: CoreMLRnnt, audio: np.ndarray, context: Tuple[int, int, int]) -> List[int]:
    """Buffered chunked streaming, mirroring NeMo's StreamingBatchedAudioBuffer.

    Feed chunk+right samples first (initial latency), then chunk per step. Run
    the streaming encoder on the (zero-padded) [left|chunk|right] window and
    decode every not-yet-decoded frame while holding back the right context,
    which is re-encoded with more future audio on the next step. The RNNT
    decoder LSTM state and last token persist across chunks.
    """
    left, chunk, right = context
    window_samples = (left + chunk + right) * ENCODER_FRAME_SAMPLES
    chunk_samples = chunk * ENCODER_FRAME_SAMPLES
    right_samples = right * ENCODER_FRAME_SAMPLES

    state = cm.init_state()
    dec_out = None
    all_tokens: List[int] = []
    consumed = 0  # samples fed so far
    decoded_frames = 0  # global encoder frames decoded so far

    while consumed < audio.size:
        feed = chunk_samples + right_samples if consumed == 0 else chunk_samples
        consumed = min(consumed + feed, audio.size)
        is_last = consumed >= audio.size

        buffer_start = max(0, consumed - window_samples)
        # frame-align upward so the buffer never exceeds the window
        buffer_start += (-buffer_start) % ENCODER_FRAME_SAMPLES
        buffer = audio[buffer_start:consumed]
        buffer_start_frame = buffer_start // ENCODER_FRAME_SAMPLES

        window = np.zeros(window_samples, dtype=np.float32)
        window[: buffer.size] = buffer
        encoder_out, enc_len = cm.encode(window, buffer.size)

        right_valid = 0 if is_last else right
        local_start = decoded_frames - buffer_start_frame
        local_end = enc_len - right_valid
        n_frames = local_end - local_start
        if n_frames <= 0:
            continue
        tokens, state, dec_out = cm.decode_frames(encoder_out, n_frames, state, dec_out, frame_offset=local_start)
        all_tokens.extend(tokens)
        decoded_frames += n_frames

    return all_tokens
