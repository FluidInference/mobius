#!/usr/bin/env python3
"""Validate CoreML parakeet-unified-en-0.6b against the NeMo reference.

Offline: runs the CoreML preprocessor → encoder → greedy RNNT loop
(decoder + single-step joint decision) on a 15 s window and compares the
transcript and encoder tensors against NeMo.

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
from typing import Optional

import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr

from coreml_rnnt import SAMPLE_RATE, CoreMLRnnt, offline_transcribe, stream_transcribe


def load_audio(path: Path, max_seconds: Optional[float] = None) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE, f"audio must be {SAMPLE_RATE} Hz, got {sr}"
    if data.ndim > 1:
        data = data[:, 0]
    if max_seconds is not None:
        data = data[: int(max_seconds * SAMPLE_RATE)]
    return data


def run_offline(args, asr_model, blank_idx: int) -> None:
    print("\n=== OFFLINE (15 s window) ===")
    audio = load_audio(args.audio_file, max_seconds=15.0)

    with torch.inference_mode():
        ref = asr_model.transcribe([str(args.audio_file)])
    ref_text = ref[0].text
    print(f"NeMo   : {ref_text}")

    cm = CoreMLRnnt(args.coreml_dir, blank_idx)
    window = np.zeros(15 * SAMPLE_RATE, dtype=np.float32)
    window[: audio.size] = audio
    encoder_out, enc_len = cm.encode(window, audio.size)

    # Component parity: torch encoder on the same input
    with torch.inference_mode():
        audio_t = torch.from_numpy(window).unsqueeze(0)
        len_t = torch.tensor([audio.size], dtype=torch.long)
        mel_t, mel_len_t = asr_model.preprocessor(input_signal=audio_t, length=len_t)
        enc_t, enc_len_t = asr_model.encoder(audio_signal=mel_t, length=mel_len_t)
    diff = np.abs(enc_t.numpy() - encoder_out)
    print(f"encoder parity: max_abs={diff.max():.5f} mean_abs={diff.mean():.6f} "
          f"len torch={int(enc_len_t[0])} coreml={enc_len}")

    tokens = offline_transcribe(cm, audio)
    cm_text = asr_model.tokenizer.ids_to_text(tokens)
    print(f"CoreML : {cm_text}")
    print(f"MATCH: {cm_text.strip() == ref_text.strip()}")


def run_streaming(args, asr_model, blank_idx: int) -> None:
    left, chunk, right = (int(x) for x in args.streaming_context.split(","))
    print(f"\n=== STREAMING (context [{left},{chunk},{right}], "
          f"latency {(chunk + right) * 0.08:.2f} s) ===")

    audio = load_audio(args.audio_file, max_seconds=args.streaming_seconds)
    cm = CoreMLRnnt(args.coreml_dir, blank_idx, streaming_suffix=f"{left}_{chunk}_{right}")

    # NeMo offline reference on the same audio span
    with torch.inference_mode():
        sf.write("/tmp/_stream_ref.wav", audio, SAMPLE_RATE)
        ref = asr_model.transcribe(["/tmp/_stream_ref.wav"])
    print(f"NeMo offline : {ref[0].text}")

    tokens = stream_transcribe(cm, audio, (left, chunk, right))
    cm_text = asr_model.tokenizer.ids_to_text(tokens)
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
