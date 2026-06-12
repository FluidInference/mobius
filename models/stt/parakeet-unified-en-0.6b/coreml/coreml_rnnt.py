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
        tokens: List[Tuple[int, int]] = []  # (token, local frame)
        for t in range(frame_offset, frame_offset + num_frames):
            enc_step = encoder_out[:, :, t : t + 1].astype(np.float32)
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                jd = self.joint_decision.predict(
                    {"encoder_step": enc_step, "decoder_step": dec_out[:, :, :1].astype(np.float32)}
                )
                token = int(jd["token_id"].reshape(-1)[0])
                if token == self.blank_idx:
                    break
                tokens.append((token, t))
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
    return [t for t, _ in tokens]


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
        all_tokens.extend(t for t, _ in tokens)
        decoded_frames += n_frames

    return all_tokens


# --- Offline overlapping-batch mode (long audio) -----------------------------
#
# Mirrors FluidAudio's UnifiedAsrManager / ChunkProcessor: frame-aligned
# 14.96 s windows with 2 s overlap, each decoded independently from a fresh
# RNNT state with global frame timestamps, then merged on the overlap with
# time-tolerant token matching (LCS) and SentencePiece word-boundary splicing.

BATCH_CHUNK_SAMPLES = 239_360  # 187 encoder frames
BATCH_OVERLAP_SAMPLES = 32_000  # 25 frames = 2 s
BATCH_STRIDE_SAMPLES = BATCH_CHUNK_SAMPLES - BATCH_OVERLAP_SAMPLES
FRAME_SECONDS = ENCODER_FRAME_SAMPLES / SAMPLE_RATE  # 0.08


def batch_chunk_starts(total_samples: int) -> List[int]:
    if total_samples <= 0:
        return []
    starts = [0]
    start = BATCH_STRIDE_SAMPLES
    while start < total_samples:
        if start + BATCH_OVERLAP_SAMPLES < total_samples:
            starts.append(start)
        start += BATCH_STRIDE_SAMPLES
    return starts


def merge_token_windows(
    left: List[Tuple[int, int]],
    right: List[Tuple[int, int]],
    splice_safe: Optional[set] = None,
) -> List[Tuple[int, int]]:
    """Merge two (token, global_frame) streams whose audio overlapped by 2 s."""
    if not left:
        return right
    if not right:
        return left

    overlap_dur = 2.0
    tol = overlap_dur / 2

    def start_time(tw):
        return tw[1] * FRAME_SECONDS

    left_end = start_time(left[-1]) + FRAME_SECONDS
    right_start = start_time(right[0])
    if left_end <= right_start:
        return left + right

    overlap_left = [
        (i, tw) for i, tw in enumerate(left) if start_time(tw) + FRAME_SECONDS > right_start - overlap_dur
    ]
    overlap_right = [(i, tw) for i, tw in enumerate(right) if start_time(tw) < left_end + overlap_dur]

    def midpoint_merge():
        cutoff = (left_end + right_start) / 2
        left_cut = next((i for i, tw in enumerate(left) if start_time(tw) >= cutoff), len(left))
        right_cut = next((i for i, tw in enumerate(right) if start_time(tw) >= cutoff), len(right))
        if splice_safe is not None:
            # Do not split a word: extend left to finish its word, drop
            # orphaned continuation pieces from right's head.
            if left_cut > 0:
                while left_cut < len(left) and left[left_cut][0] not in splice_safe:
                    left_cut += 1
            while right_cut < len(right) and right[right_cut][0] not in splice_safe:
                right_cut += 1
        return left[:left_cut] + right[right_cut:]

    if len(overlap_left) < 2 or len(overlap_right) < 2:
        return midpoint_merge()

    # LCS over the overlap with time-tolerant token matching.
    n, m = len(overlap_left), len(overlap_right)
    lcs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            li, lt = overlap_left[i]
            ri, rt = overlap_right[j]
            if lt[0] == rt[0] and abs(start_time(lt) - start_time(rt)) < tol:
                lcs[i][j] = lcs[i + 1][j + 1] + 1
            else:
                lcs[i][j] = max(lcs[i + 1][j], lcs[i][j + 1])
    matches = []
    i = j = 0
    while i < n and j < m:
        li, lt = overlap_left[i]
        ri, rt = overlap_right[j]
        if lt[0] == rt[0] and abs(start_time(lt) - start_time(rt)) < tol:
            matches.append((li, ri))
            i += 1
            j += 1
        elif lcs[i + 1][j] >= lcs[i][j + 1]:
            i += 1
        else:
            j += 1

    if not matches:
        return midpoint_merge()

    result = list(left[: matches[0][0]])
    for k, (li, ri) in enumerate(matches):
        result.append(left[li])
        if k + 1 < len(matches):
            nli, nri = matches[k + 1]
            gap_left = left[li + 1 : nli]
            gap_right = right[ri + 1 : nri]
            result.extend(gap_right if len(gap_right) > len(gap_left) else gap_left)

    last_li, last_ri = matches[-1]
    tail = right[last_ri + 1 :]
    if splice_safe is not None and tail and tail[0][0] not in splice_safe:
        # Seam lands mid-word. Prefer the right window's segmentation of the
        # seam word (it usually heard the word from its start): pop the seam
        # word from the result and resume right at the word-initial piece.
        word_start = None
        for idx in range(last_ri, -1, -1):
            if right[idx][0] in splice_safe:
                word_start = idx
                break
        popped = False
        if word_start is not None:
            for back in range(1, min(12, len(result)) + 1):
                if result[-back][0] in splice_safe:
                    del result[-back:]
                    popped = True
                    break
        if word_start is not None and popped:
            result.extend(right[word_start:])
        else:
            # Right began mid-word: left owns the seam word — finish it from
            # left's continuation pieces, resume right at its next word start.
            cursor = last_li + 1
            while cursor < len(left) and left[cursor][0] not in splice_safe:
                result.append(left[cursor])
                cursor += 1
            resume = next((idx for idx, tw in enumerate(tail) if tw[0] in splice_safe), None)
            if resume is not None:
                result.extend(tail[resume:])
    else:
        result.extend(tail)
    return result


def batch_transcribe(
    cm: CoreMLRnnt, audio: np.ndarray, splice_safe: Optional[set] = None
) -> List[int]:
    """Offline overlapping-batch transcription for audio of any length."""
    merged: List[Tuple[int, int]] = []
    for chunk_start in batch_chunk_starts(audio.size):
        chunk = audio[chunk_start : chunk_start + BATCH_CHUNK_SAMPLES]
        window = np.zeros(OFFLINE_WINDOW_SAMPLES, dtype=np.float32)
        window[: chunk.size] = chunk
        encoder_out, enc_len = cm.encode(window, chunk.size)
        tokens, _, _ = cm.decode_frames(encoder_out, enc_len, cm.init_state(), None)
        global_tokens = [(t, f + chunk_start // ENCODER_FRAME_SAMPLES) for t, f in tokens]
        merged = (
            global_tokens if not merged else merge_token_windows(merged, global_tokens, splice_safe)
        )
    merged.sort(key=lambda tw: tw[1])
    return [t for t, _ in merged]


def splice_safe_token_ids(sp) -> set:
    """SentencePiece word-initial (▁-prefixed) or punctuation-only pieces."""
    import unicodedata

    safe = set()
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece.startswith("▁"):
            safe.add(i)
        elif piece and all(unicodedata.category(ch)[0] in ("P", "S") for ch in piece):
            safe.add(i)
    return safe
