#!/usr/bin/env python3
"""Offline head-to-head latency benchmark: FluidAudio fused Sortformer vs Argmax.

Both process the same 30.72 s window (mel [1,128,3072] = 3072 x 10 ms hops). Argmax's
Sortformer is an OFFLINE batch model (3-model chain, no streaming state); FluidAudio's
offline export is a single fused graph (mel -> speaker_preds). This measures the offline
encoder/throughput gap only — it is NOT a streaming comparison.

Models (override paths via the constants below):
  - FluidAudio fused offline:  exported via the NeMo offline path (streaming_mode=False);
    inputs mel[1,128,3072] + mel_length[1] -> speaker_preds.
  - Argmax:  argmaxinc/speakerkit-pro  sortformer/v2-1/384_94MB/{MelSpectrogram,
    AudioConformerPreEncoder,SortformerFullEncoder}.mlmodelc  (proprietary; download via
    the Argmax v2 Playground app or the speakerkit-pro HF repo).

Random inputs of the correct shape/dtype — valid for latency of these static-shape graphs.
Interleaved per-stage timing (median of N after warmup), ComputeUnit.ALL.

Usage:  python offline_argmax_bench.py
Result (M5 Pro, 2026-06-23): FluidAudio 1.3-1.4x faster offline; see Documentation/
Diarization/Sortformer.md#benchmarks.
"""
import os
import time

import numpy as np
import coremltools as ct

# --- model paths (override with env vars) ---
ARGMAX_DIR = os.environ.get("ARGMAX_DIR", "/tmp/argmax_sf/sortformer/v2-1/384_94MB")
OURS_FP16 = os.environ.get("OURS_FP16", "/tmp/sf_offline_fp16.mlmodelc")
OURS_PALETTE6 = os.environ.get("OURS_PALETTE6", "/tmp/sf_offline_palette6.mlmodelc")

CU = ct.ComputeUnit.ALL
WINDOW_S = 491520 / 16000.0  # 30.72 s
WARMUP, RUNS = 12, 120


def load(path):
    return ct.models.CompiledMLModel(path, compute_units=CU)


def bench(model, feed, warm=WARMUP, n=RUNS):
    for _ in range(warm):
        model.predict(feed)
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        model.predict(feed)
        ts.append((time.perf_counter() - s) * 1e3)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def rtfx(ms):
    return WINDOW_S / (ms / 1e3)


def main():
    print("loading models...", flush=True)
    mel = load(f"{ARGMAX_DIR}/MelSpectrogram.mlmodelc")
    pre = load(f"{ARGMAX_DIR}/AudioConformerPreEncoder.mlmodelc")
    full = load(f"{ARGMAX_DIR}/SortformerFullEncoder.mlmodelc")
    ours_fp16 = load(OURS_FP16)
    ours_p6 = load(OURS_PALETTE6)

    f16 = np.float16
    audio = {"audio": np.random.randn(491520).astype(f16)}
    melfeat = {"melspectrogram_features": np.random.randn(1, 1, 3073, 128).astype(f16)}
    fullin = {
        "downsampled_melspectrogram_features": np.random.randn(1, 512, 1, 384).astype(f16),
        "conformer_encoder_padding_mask": np.zeros((1, 384), f16),
        "conformer_encoder_qk_mask": np.zeros((1, 1, 384, 384), f16),
        "transformer_encoder_mask": np.zeros((1, 384), f16),
        "input_1": np.ones((1, 1, 1, 1), f16),
    }
    ours_in = {"mel": np.random.randn(1, 128, 3072).astype(np.float32), "mel_length": np.array([3072], np.int32)}

    mel_ms, _ = bench(mel, audio)
    pre_ms, _ = bench(pre, melfeat)
    full_ms, _ = bench(full, fullin)
    ours_ms, ours_p95 = bench(ours_fp16, ours_in)
    p6_ms, p6_p95 = bench(ours_p6, ours_in)

    argmax_enc = pre_ms + full_ms
    argmax_e2e = mel_ms + pre_ms + full_ms
    ours_e2e = ours_ms + mel_ms
    p6_e2e = p6_ms + mel_ms

    print(f"\nWindow = {WINDOW_S:.2f}s, ComputeUnit.ALL, median of {RUNS} ({WARMUP} warmup)\n")
    print(f"{'stage':38} {'ms':>8} {'RTFx':>9}")
    print(f"{'Argmax MelSpectrogram':38} {mel_ms:8.2f} {rtfx(mel_ms):9.0f}")
    print(f"{'Argmax AudioConformerPreEncoder':38} {pre_ms:8.2f} {rtfx(pre_ms):9.0f}")
    print(f"{'Argmax SortformerFullEncoder':38} {full_ms:8.2f} {rtfx(full_ms):9.0f}")
    print(f"{'OURS fused offline (fp16)':38} {ours_ms:8.2f} {rtfx(ours_ms):9.0f}   p95 {ours_p95:.2f}")
    print(f"{'OURS fused offline (palette6)':38} {p6_ms:8.2f} {rtfx(p6_ms):9.0f}   p95 {p6_p95:.2f}")
    print("\n=== ENCODER only (mel->preds, fair apples-to-apples) ===")
    print(f"  Argmax (PreEnc+FullEnc)  {argmax_enc:8.2f} ms  {rtfx(argmax_enc):6.0f}x  (2 calls, ANE->GPU handoff)")
    print(f"  OURS fp16 fused          {ours_ms:8.2f} ms  {rtfx(ours_ms):6.0f}x  (1 call)  -> {argmax_enc/ours_ms:.2f}x faster")
    print(f"  OURS palette6 fused      {p6_ms:8.2f} ms  {rtfx(p6_ms):6.0f}x            -> {argmax_enc/p6_ms:.2f}x faster")
    print("\n=== END-TO-END incl mel (same MelSpectrogram tax both sides) ===")
    print(f"  Argmax full (3 calls)    {argmax_e2e:8.2f} ms  {rtfx(argmax_e2e):6.0f}x")
    print(f"  OURS fp16 + mel          {ours_e2e:8.2f} ms  {rtfx(ours_e2e):6.0f}x  -> {argmax_e2e/ours_e2e:.2f}x faster")
    print(f"  OURS palette6 + mel      {p6_e2e:8.2f} ms  {rtfx(p6_e2e):6.0f}x  -> {argmax_e2e/p6_e2e:.2f}x faster")


if __name__ == "__main__":
    main()
