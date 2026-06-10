#!/usr/bin/env python3
"""Parity: fused decoder_joint_decision vs the shipped two-model reference.

Drives the REAL streaming pipeline on real audio (LibriSpeech flac):
  numpy port of Swift AudioMelSpectrogram (the shipped preprocessor.mlmodelc
  is stripped of its MIL; production computes mel natively in Swift anyway)
  -> streaming_encoder.mlmodelc (cache-aware, 160ms
  chunks, valid_out_len=2) -> RNNT greedy loop replicating
  Sources/FluidAudio/ASR/Parakeet/Streaming/RnntDecoder.swift semantics
  (maxSymbolsPerStep=2, blank=1026, eou=1024, state committed only on
  non-blank emission, state reset after EOU).

Both pipelines run their own independent autoregressive state machines; we
compare every inner step:
  token_id   exact match
  token_prob max |ref - fused|
  h_out      max |ref - fused|
  c_out      max |ref - fused|
and the full emitted token sequence.

Usage:
    python parity_fused_decode.py \
        --model-dir "$HOME/Library/Application Support/FluidAudio/Models/parakeet-eou-streaming/160ms" \
        --fused /tmp/eou_fused/decoder_joint_decision_fused.mlpackage \
        --audio /path/to/audio.flac --seconds 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

BLANK_ID = 1026
EOU_ID = 1024
MAX_SYMBOLS = 2
CHUNK_SAMPLES = 2560  # 160ms variant


def load_compiled(path: Path) -> ct.models.CompiledMLModel:
    return ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)


def _slaney_mel_filterbank(n_fft=512, n_mels=128, sr=16000) -> np.ndarray:
    """Slaney-scale, Slaney-normalized mel filterbank (librosa default),
    matching Sources/FluidAudio/Shared/AudioMelSpectrogram.swift."""
    f_sp = 200.0 / 3.0
    min_log_hz, log_step = 1000.0, np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp

    def hz_to_mel(hz):
        hz = np.asarray(hz, dtype=np.float64)
        return np.where(hz >= min_log_hz, min_log_mel + np.log(hz / min_log_hz) / log_step, hz / f_sp)

    def mel_to_hz(mel):
        mel = np.asarray(mel, dtype=np.float64)
        return np.where(mel >= min_log_mel, min_log_hz * np.exp(log_step * (mel - min_log_mel)), f_sp * mel)

    mel_pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2))
    fft_freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        f_l, f_c, f_r = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
        norm = 2.0 / (f_r - f_l)
        rising = (fft_freqs >= f_l) & (fft_freqs < f_c)
        falling = (fft_freqs >= f_c) & (fft_freqs <= f_r)
        fb[m, rising] = norm * (fft_freqs[rising] - f_l) / (f_c - f_l)
        fb[m, falling] = norm * (f_r - fft_freqs[falling]) / (f_r - f_c)
    return fb


_MEL_FB = _slaney_mel_filterbank()
_HANN = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(400) / 399.0))).astype(np.float32)


def mel_chunk(samples: np.ndarray) -> np.ndarray:
    """Numpy port of AudioMelSpectrogram.computeFlat for one chunk: [1, 128, T]."""
    n_fft, hop, win, preemph = 512, 160, 400, 0.97
    pad = n_fft // 2
    x = np.concatenate([[samples[0]], samples[1:] - preemph * samples[:-1]]).astype(np.float32)
    x = np.pad(x, (pad, pad))
    n_frames = 1 + (len(x) - win) // hop
    off = (n_fft - win) // 2  # window centered in the FFT buffer
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop + off
        seg = x[start: start + win]
        frames[i, off: off + len(seg)] = seg * _HANN[: len(seg)]
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2  # power
    mel = spec.astype(np.float32) @ _MEL_FB.T  # [T, n_mels]
    mel = np.log(mel + 2.0 ** -24)
    return mel.T[None, :, :].astype(np.float32)  # [1, 128, T]


def encoder_steps(model_dir: Path, audio: np.ndarray) -> np.ndarray:
    """Run the real streaming pipeline, return encoder frames [N, 512]."""
    enc = load_compiled(model_dir / "streaming_encoder.mlmodelc")

    pre_cache = np.zeros((1, 128, 16), dtype=np.float32)
    cache_ch = np.zeros((17, 1, 70, 512), dtype=np.float32)
    cache_tm = np.zeros((17, 1, 512, 8), dtype=np.float32)
    cache_len = np.zeros((1,), dtype=np.int32)

    frames = []
    n_chunks = len(audio) // CHUNK_SAMPLES
    for i in range(n_chunks):
        chunk = audio[i * CHUNK_SAMPLES: (i + 1) * CHUNK_SAMPLES]
        mel = mel_chunk(chunk)[:, :, :17]  # [1, 128, 17]
        out = enc.predict({
            "audio_signal": mel,
            "audio_length": np.array([mel.shape[2]], dtype=np.int32),
            "pre_cache": pre_cache,
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_tm,
            "cache_last_channel_len": cache_len,
        })
        pre_cache = out["new_pre_cache"].astype(np.float32)
        cache_ch = out["new_cache_last_channel"].astype(np.float32)
        cache_tm = out["new_cache_last_time"].astype(np.float32)
        cache_len = out["new_cache_last_channel_len"].astype(np.int32)
        encoded = out["encoded_output"]  # [1, 512, 2]
        for t in range(encoded.shape[2]):
            frames.append(encoded[0, :, t].astype(np.float32))
    return np.stack(frames)  # [N, 512]


class RefDecoder:
    """Two-model reference, mirroring RnntDecoder.swift."""

    def __init__(self, model_dir: Path):
        self.decoder = load_compiled(model_dir / "decoder.mlmodelc")
        self.joint = load_compiled(model_dir / "joint_decision.mlmodelc")
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 640), dtype=np.float32)
        self.c = np.zeros((1, 1, 640), dtype=np.float32)
        self.last_token = BLANK_ID

    def step(self, enc_step: np.ndarray):
        d = self.decoder.predict({
            "targets": np.array([[self.last_token]], dtype=np.int32),
            "target_length": np.array([1], dtype=np.int32),
            "h_in": self.h,
            "c_in": self.c,
        })
        j = self.joint.predict({
            "encoder_step": enc_step,
            "decoder_step": d["decoder"].astype(np.float32),
        })
        top1 = float(j["top_k_logits"].reshape(-1)[0])
        return (
            int(j["token_id"].reshape(-1)[0]),
            float(j["token_prob"].reshape(-1)[0]),
            d["h_out"].astype(np.float32),
            d["c_out"].astype(np.float32),
            top1,
        )


class FusedDecoder:
    def __init__(self, fused_path: Path):
        if str(fused_path).endswith(".mlmodelc"):
            self.model = load_compiled(fused_path)
        else:
            self.model = ct.models.MLModel(
                str(fused_path), compute_units=ct.ComputeUnit.CPU_ONLY
            )
        try:
            input_names = {i.name for i in self.model.get_spec().description.input}
        except Exception:
            input_names = {"targets", "h_in", "c_in", "encoder_step"}
        self.needs_target_length = "target_length" in input_names
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 640), dtype=np.float32)
        self.c = np.zeros((1, 1, 640), dtype=np.float32)
        self.last_token = BLANK_ID

    def step(self, enc_step: np.ndarray):
        inputs = {
            "targets": np.array([[self.last_token]], dtype=np.int32),
            "h_in": self.h,
            "c_in": self.c,
            "encoder_step": enc_step,
        }
        if self.needs_target_length:
            inputs["target_length"] = np.array([1], dtype=np.int32)
        o = self.model.predict(inputs)
        top1 = float(o["top_k_logits"].reshape(-1)[0]) if "top_k_logits" in o else float("nan")
        return (
            int(o["token_id"].reshape(-1)[0]),
            float(o["token_prob"].reshape(-1)[0]),
            o["h_out"].astype(np.float32),
            o["c_out"].astype(np.float32),
            top1,
        )


def run_loop(dec, frames: np.ndarray):
    """Greedy RNNT loop; returns (tokens, per-step records)."""
    tokens, records = [], []
    for t in range(frames.shape[0]):
        enc_step = frames[t][None, :, None].astype(np.float32)  # [1, 512, 1]
        symbols = 0
        while symbols < MAX_SYMBOLS:
            tok, prob, h_new, c_new, top1 = dec.step(enc_step)
            records.append((t, symbols, tok, prob, h_new, c_new, top1))
            if tok == BLANK_ID:
                break
            if tok == EOU_ID:
                tokens.append(tok)
                dec.reset()  # new utterance (Swift breaks the outer loop here)
                break
            tokens.append(tok)
            dec.last_token = tok
            dec.h, dec.c = h_new, c_new
            symbols += 1
    return tokens, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--fused", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True, nargs="+")
    ap.add_argument("--seconds", type=float, default=10.0, help="max seconds per file")
    args = ap.parse_args()

    total_steps = total_mismatch = 0
    n_files = n_seq_diff = n_text_diff = 0
    all_seq_match = True
    max_dp = max_dh = max_dc = max_dl = 0.0

    vocab = {}
    vocab_path = args.model_dir / "vocab.json"
    if vocab_path.exists():
        import json
        vocab = {int(k): v for k, v in json.loads(vocab_path.read_text()).items()}

    def detok(ids):
        return "".join(vocab.get(i, f"<{i}>") for i in ids if i != EOU_ID).replace("▁", " ").strip()

    ref = RefDecoder(args.model_dir)
    fused = FusedDecoder(args.fused)

    for audio_path in args.audio:
        audio, sr = sf.read(str(audio_path), dtype="float32")
        assert sr == 16000, f"expected 16 kHz, got {sr}"
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio[: int(args.seconds * sr)]

        print(f"{Path(audio_path).name}: streaming encoder on {len(audio)/sr:.1f}s...")
        frames = encoder_steps(args.model_dir, audio)

        ref.reset()
        fused.reset()
        ref_tokens, ref_rec = run_loop(ref, frames)
        fused_tokens, fused_rec = run_loop(fused, frames)

        seq_match = ref_tokens == fused_tokens
        all_seq_match &= seq_match
        n = min(len(ref_rec), len(fused_rec))
        tok_mismatch = 0
        for i in range(n):
            rt, rs, rtok, rprob, rh, rc, rl = ref_rec[i]
            ft, fs, ftok, fprob, fh, fc, fl = fused_rec[i]
            if (rt, rs, rtok) != (ft, fs, ftok):
                tok_mismatch += 1
                if tok_mismatch <= 5:
                    print(f"  MISMATCH step {i}: ref(t={rt},s={rs},tok={rtok}) fused(t={ft},s={fs},tok={ftok})")
                continue
            max_dp = max(max_dp, abs(rprob - fprob))
            max_dh = max(max_dh, float(np.abs(rh - fh).max()))
            max_dc = max(max_dc, float(np.abs(rc - fc).max()))
            if not (np.isnan(rl) or np.isnan(fl)):
                max_dl = max(max_dl, abs(rl - fl))
        total_steps += n
        total_mismatch += tok_mismatch
        n_files += 1
        text_diff = False
        if not seq_match:
            n_seq_diff += 1
            if vocab:
                rt_text, ft_text = detok(ref_tokens), detok(fused_tokens)
                text_diff = rt_text != ft_text
                n_text_diff += int(text_diff)
                if text_diff:
                    print(f"  TEXT ref  : {rt_text}")
                    print(f"  TEXT fused: {ft_text}")
        print(
            f"  {frames.shape[0]} enc frames, {n} decode steps, "
            f"{len(ref_tokens)} tokens, seq match: {seq_match}, step mismatches: {tok_mismatch}"
            + (", TEXT DIFFERS" if text_diff else "")
        )

    print()
    print(f"files: {n_files}, sequences differing: {n_seq_diff}, transcripts differing: {n_text_diff}")
    print(f"token sequences identical : {all_seq_match}")
    print(f"decode steps compared     : {total_steps} (mismatched: {total_mismatch})")
    print(f"max |h_out|  diff         : {max_dh:.3e}")
    print(f"max |c_out|  diff         : {max_dc:.3e}")
    print(f"max |token_prob| diff     : {max_dp:.3e}  (fp16 softmax; unused by Swift host)")
    print(f"max |top1 logit| diff     : {max_dl:.3e}  (fp16 GEMM accumulation, model-boundary effect)")
    strict = all_seq_match and total_mismatch == 0 and max_dp < 1e-5 and max_dh < 1e-5 and max_dc < 1e-5
    functional = all_seq_match and total_mismatch == 0 and max_dh < 1e-5 and max_dc < 1e-5
    print(f"STRICT parity (<1e-5 incl. probs) : {'PASS' if strict else 'FAIL'}")
    print(f"FUNCTIONAL parity (tokens exact + state <1e-5): {'PASS' if functional else 'FAIL'}")
    raise SystemExit(0 if functional else 1)


if __name__ == "__main__":
    main()
