#!/usr/bin/env python3
"""Validate CoreML parakeet-unified-en-0.6b against the NeMo reference.

Offline: runs the CoreML preprocessor → encoder → greedy RNNT loop
(decoder + single-step joint decision) on a 15 s window and compares the
transcript and intermediate tensors against NeMo.

Streaming: simulates NeMo's buffered chunked streaming (left/chunk/right
window) with the CoreML streaming encoder and a persistent RNNT decoder
state, and compares against the NeMo offline transcript.

Usage:
    uv run --no-sync python compare-models.py --coreml-dir ./build/parakeet_unified_coreml \
        --audio-file audio/yc_first_minute_16k.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr

SAMPLE_RATE = 16000
ENCODER_FRAME_SAMPLES = 1280  # 80 ms
MAX_SYMBOLS_PER_FRAME = 10


def load_audio(path: Path, max_seconds: Optional[float] = None) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE, f"audio must be {SAMPLE_RATE} Hz, got {sr}"
    if data.ndim > 1:
        data = data[:, 0]
    if max_seconds is not None:
        data = data[: int(max_seconds * SAMPLE_RATE)]
    return data


class CoreMLRnnt:
    """Greedy RNNT decoding over CoreML components."""

    def __init__(self, coreml_dir: Path, blank_idx: int, streaming_suffix: Optional[str] = None) -> None:
        cu = ct.ComputeUnit.CPU_AND_NE
        self.preprocessor = ct.models.MLModel(
            str(coreml_dir / "parakeet_unified_preprocessor.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        encoder_name = (
            f"parakeet_unified_encoder_streaming_{streaming_suffix}.mlpackage"
            if streaming_suffix
            else "parakeet_unified_encoder.mlpackage"
        )
        self.encoder = ct.models.MLModel(str(coreml_dir / encoder_name), compute_units=cu)
        self.decoder = ct.models.MLModel(
            str(coreml_dir / "parakeet_unified_decoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY
        )
        self.joint_decision = ct.models.MLModel(
            str(coreml_dir / "parakeet_unified_joint_decision_single_step.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        self.blank_idx = blank_idx
        self.decoder_state_shape = None  # discovered on first call

    def encode(self, audio: np.ndarray, num_samples: int) -> Tuple[np.ndarray, int]:
        """audio is zero-padded to the encoder window; num_samples is valid length."""
        mel_out = self.preprocessor.predict(
            {
                "audio_signal": audio[None, :].astype(np.float32),
                "audio_length": np.array([num_samples], dtype=np.int32),
            }
        )
        enc_out = self.encoder.predict(
            {"mel": mel_out["mel"].astype(np.float32), "mel_length": mel_out["mel_length"].astype(np.int32)}
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


def run_offline(args, asr_model, blank_idx: int) -> None:
    print("\n=== OFFLINE (15 s window) ===")
    audio = load_audio(args.audio_file, max_seconds=15.0)
    num_samples = audio.size
    window = np.zeros(15 * SAMPLE_RATE, dtype=np.float32)
    window[:num_samples] = audio

    # NeMo reference
    with torch.inference_mode():
        ref = asr_model.transcribe([str(args.audio_file)])
    ref_text = ref[0].text
    print(f"NeMo   : {ref_text}")

    # CoreML chain
    cm = CoreMLRnnt(args.coreml_dir, blank_idx)
    encoder_out, enc_len = cm.encode(window, num_samples)

    # Component parity: torch encoder on same input
    with torch.inference_mode():
        audio_t = torch.from_numpy(window).unsqueeze(0)
        len_t = torch.tensor([num_samples], dtype=torch.long)
        mel_t, mel_len_t = asr_model.preprocessor(input_signal=audio_t, length=len_t)
        enc_t, enc_len_t = asr_model.encoder(audio_signal=mel_t, length=mel_len_t)
    diff = np.abs(enc_t.numpy() - encoder_out)
    print(f"encoder parity: max_abs={diff.max():.5f} mean_abs={diff.mean():.6f} "
          f"len torch={int(enc_len_t[0])} coreml={enc_len}")

    tokens, _, _ = cm.decode_frames(encoder_out, enc_len, cm.init_state(), None)
    cm_text = asr_model.tokenizer.ids_to_text(tokens)
    print(f"CoreML : {cm_text}")
    print(f"MATCH: {cm_text.strip() == ref_text.strip()}")


def run_streaming(args, asr_model, blank_idx: int) -> None:
    left, chunk, right = (int(x) for x in args.streaming_context.split(","))
    suffix = f"{left}_{chunk}_{right}"
    print(f"\n=== STREAMING (context [{left},{chunk},{right}], "
          f"latency {(chunk + right) * 0.08:.2f} s) ===")

    audio = load_audio(args.audio_file, max_seconds=args.streaming_seconds)
    cm = CoreMLRnnt(args.coreml_dir, blank_idx, streaming_suffix=suffix)

    window_samples = (left + chunk + right) * ENCODER_FRAME_SAMPLES
    chunk_samples = chunk * ENCODER_FRAME_SAMPLES
    right_samples = right * ENCODER_FRAME_SAMPLES

    # NeMo offline reference on the same audio span
    with torch.inference_mode():
        sf.write("/tmp/_stream_ref.wav", audio, SAMPLE_RATE)
        ref = asr_model.transcribe(["/tmp/_stream_ref.wav"])
    print(f"NeMo offline : {ref[0].text}")

    # Buffered streaming: mirror NeMo's StreamingBatchedAudioBuffer — the
    # buffer grows until it holds left+chunk+right samples, then slides by
    # chunk. Frame bookkeeping is done in absolute (global) encoder frames.
    state = cm.init_state()
    dec_out = None
    all_tokens: List[int] = []
    consumed = 0  # samples fed so far
    decoded_frames = 0  # global encoder frames decoded so far

    while consumed < audio.size:
        # feed chunk+right on the first step (initial latency), then chunk
        feed = chunk_samples + right_samples if consumed == 0 else chunk_samples
        consumed = min(consumed + feed, audio.size)
        is_last = consumed >= audio.size

        buffer_start = max(0, consumed - window_samples)
        # keep the buffer start aligned to encoder frames
        buffer_start -= buffer_start % ENCODER_FRAME_SAMPLES
        buffer = audio[buffer_start:consumed]
        buffer_start_frame = buffer_start // ENCODER_FRAME_SAMPLES

        window = np.zeros(window_samples, dtype=np.float32)
        window[: buffer.size] = buffer
        encoder_out, enc_len = cm.encode(window, buffer.size)

        # decode every frame not yet decoded, holding back the right context
        # (which will be re-encoded with more future audio next step)
        right_valid = 0 if is_last else right
        local_start = decoded_frames - buffer_start_frame
        local_end = enc_len - right_valid
        n_frames = local_end - local_start
        if n_frames <= 0:
            continue
        tokens, state, dec_out = cm.decode_frames(encoder_out, n_frames, state, dec_out, frame_offset=local_start)
        all_tokens.extend(tokens)
        decoded_frames += n_frames

    cm_text = asr_model.tokenizer.ids_to_text(all_tokens)
    print(f"CoreML stream: {cm_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-dir", type=Path, default=Path("build/parakeet_unified_coreml"))
    parser.add_argument("--audio-file", type=Path, default=Path("audio/yc_first_minute_16k_15s.wav"))
    parser.add_argument("--nemo-path", type=Path, default=Path("parakeet-unified-en-0.6b.nemo"))
    parser.add_argument("--streaming-context", type=str, default="70,13,13")
    parser.add_argument("--streaming-seconds", type=float, default=30.0)
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-streaming", action="store_true")
    args = parser.parse_args()

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(args.nemo_path), map_location="cpu")
    asr_model.eval()
    # The released .nemo has no validation_ds section; transcribe() dereferences it.
    from omegaconf import OmegaConf, open_dict

    with open_dict(asr_model.cfg):
        if asr_model.cfg.get("validation_ds") is None:
            asr_model.cfg.validation_ds = OmegaConf.create({})

    blank_idx = int(asr_model.decoder.blank_idx)

    if not args.skip_offline:
        run_offline(args, asr_model, blank_idx)
    if not args.skip_streaming:
        run_streaming(args, asr_model, blank_idx)


if __name__ == "__main__":
    main()
