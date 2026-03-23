#!/usr/bin/env python3
"""Diagnose robotic CoreML audio by isolating code generation vs speech decoding.

Cross-tests:
  1. PyTorch codes → CoreML SpeechDecoder  (isolates SpeechDecoder)
  2. CoreML codes → PyTorch speech_tokenizer.decode()  (isolates code generation)
  3. PyTorch codes → PyTorch speech_tokenizer.decode()  (reference)
  4. CoreML codes → CoreML SpeechDecoder  (current pipeline)

If (1) sounds good, the SpeechDecoder is fine and the issue is code generation.
If (1) sounds robotic, the SpeechDecoder itself is the problem.
If (2) sounds good, the codes are fine and the issue is in SpeechDecoder.

Usage:
    python diagnose_robotic.py
    python diagnose_robotic.py --text "Hello world, this is a test."
"""

import numpy as np
import struct
import time
import os
import sys
import argparse

SAMPLE_RATE = 24000

# Reuse constants from inference.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (
    write_wav, load_models, build_prefill_embeddings, synthesize,
    sample_top_k, apply_repetition_penalty,
    MODEL_PATH, EOS_TOKEN, CODEC_VOCAB_SIZE, MAX_CODEC_TOKENS,
    _text_proj, _code_emb, TTS_PAD_TOKEN_ID
)


def generate_coreml_codes(text, models, seed=42, speaker_embedding=None):
    """Run CoreML pipeline and return (audio, codes)."""
    audio, frames = synthesize(text, models, greedy=False, seed=seed,
                               speaker_embedding=speaker_embedding)
    codes = np.array(frames, dtype=np.int32)
    return audio, codes


def decode_codes_coreml(codes, speech_dec):
    """Decode codebook frames through CoreML SpeechDecoder."""
    audio_chunks = []
    for i in range(codes.shape[0]):
        frame = codes[i].reshape(1, 16, 1).astype(np.int32)
        out = speech_dec.predict({"audio_codes": frame})
        audio_chunks.append(out["audio"].flatten())
    return np.concatenate(audio_chunks).astype(np.float32)


def decode_codes_pytorch(codes):
    """Decode codebook frames through PyTorch speech_tokenizer."""
    import torch
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from qwen_tts import Qwen3TTSModel

    print("  Loading PyTorch model for speech_tokenizer...")
    m = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map="cpu", dtype=torch.float32)

    codes_tensor = torch.tensor(codes, dtype=torch.long)
    # speech_tokenizer.decode expects [{"audio_codes": tensor}] where tensor is [N, 16]
    if codes_tensor.ndim == 2 and codes_tensor.shape[1] == 16:
        pass  # Already [N, 16]
    elif codes_tensor.ndim == 2 and codes_tensor.shape[0] == 16:
        codes_tensor = codes_tensor.T  # [16, N] → [N, 16]

    print(f"  Decoding {codes_tensor.shape[0]} frames with PyTorch speech_tokenizer...")
    with torch.inference_mode():
        wavs, fs = m.model.speech_tokenizer.decode([{"audio_codes": codes_tensor}])
    audio = wavs[0]
    if hasattr(audio, 'cpu'):
        audio = audio.cpu().numpy()
    return np.array(audio, dtype=np.float32).flatten()


def save_normalized(audio, path):
    """Normalize and save audio."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    write_wav(path, audio)
    dur = len(audio) / SAMPLE_RATE
    print(f"  Saved: {path} ({dur:.2f}s)")
    return audio


def compare_audio_stats(audios, names):
    """Print comparative stats for audio arrays."""
    print("\n" + "=" * 70)
    print("AUDIO COMPARISON")
    print("=" * 70)

    header = f"{'Name':<35} {'Duration':>8} {'RMS':>8} {'Peak':>8} {'ZeroCross':>10}"
    print(header)
    print("-" * len(header))

    for name, audio in zip(names, audios):
        dur = len(audio) / SAMPLE_RATE
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max()
        # Zero crossing rate (indicator of noisiness/buzziness)
        zc = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        print(f"{name:<35} {dur:>7.2f}s {rms:>8.4f} {peak:>8.4f} {zc:>10.4f}")

    # Pairwise correlation for overlapping duration
    print("\nPairwise correlation (first N samples where both exist):")
    for i in range(len(audios)):
        for j in range(i + 1, len(audios)):
            n = min(len(audios[i]), len(audios[j]))
            a, b = audios[i][:n], audios[j][:n]
            corr = np.corrcoef(a, b)[0, 1]
            mse = np.mean((a - b) ** 2)
            print(f"  {names[i]} vs {names[j]}: corr={corr:.4f}, MSE={mse:.6f}")


def compare_code_stats(codes_dict):
    """Compare codebook token distributions."""
    print("\n" + "=" * 70)
    print("CODEBOOK COMPARISON")
    print("=" * 70)

    for name, codes in codes_dict.items():
        print(f"\n{name}: {codes.shape[0]} frames x {codes.shape[1]} codebooks")
        for cb in [0, 1, 7, 15]:
            col = codes[:, cb]
            unique = len(np.unique(col))
            print(f"  CB{cb:2d}: {unique:3d} unique, range [{col.min():4d}, {col.max():4d}], "
                  f"mean={col.mean():.0f}")

    # Compare CB0 overlap between pairs
    names = list(codes_dict.keys())
    if len(names) >= 2:
        print("\nCB0 token overlap:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = set(codes_dict[names[i]][:, 0].tolist())
                b = set(codes_dict[names[j]][:, 0].tolist())
                overlap = len(a & b)
                total = len(a | b)
                print(f"  {names[i]} vs {names[j]}: {overlap}/{total} shared tokens "
                      f"({100*overlap/total:.0f}% Jaccard)")


def main():
    parser = argparse.ArgumentParser(description="Diagnose robotic CoreML audio")
    parser.add_argument("--text", default="Hello world, this is a test of the text to speech system.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speaker", type=str, default=None,
                        help="Path to speaker embedding .npy file")
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Skip PyTorch decoding (faster, only test CoreML SpeechDecoder)")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    out_dir = "/tmp/diagnose_robotic"
    os.makedirs(out_dir, exist_ok=True)

    speaker_embedding = None
    if args.speaker:
        speaker_embedding = np.load(args.speaker)
        print(f"Speaker embedding: {args.speaker}")

    # Load CoreML models
    print("Loading CoreML models...")
    models = load_models(cpu_only=args.cpu_only)

    # Load existing PyTorch reference codes
    pytorch_codes_path = "/tmp/pytorch_ref_codes.npy"
    if not os.path.exists(pytorch_codes_path):
        print(f"ERROR: {pytorch_codes_path} not found. Run gen_pytorch_ref.py first.")
        return
    pytorch_codes = np.load(pytorch_codes_path).astype(np.int32)
    print(f"\nPyTorch reference codes: {pytorch_codes.shape}")

    # ─── Test 1: Generate CoreML codes ───
    print("\n" + "=" * 70)
    print("TEST 1: Generate CoreML codes")
    print("=" * 70)
    coreml_audio, coreml_codes = generate_coreml_codes(
        args.text, models, seed=args.seed, speaker_embedding=speaker_embedding
    )
    save_normalized(coreml_audio, f"{out_dir}/coreml_full_pipeline.wav")
    np.save(f"{out_dir}/coreml_codes.npy", coreml_codes)

    # ─── Test 2: PyTorch codes → CoreML SpeechDecoder ───
    print("\n" + "=" * 70)
    print("TEST 2: PyTorch codes → CoreML SpeechDecoder")
    print("=" * 70)
    pytorch_via_coreml_audio = decode_codes_coreml(pytorch_codes, models["SpeechDecoder"])
    save_normalized(pytorch_via_coreml_audio, f"{out_dir}/pytorch_codes_coreml_decoder.wav")

    # ─── Test 3: CoreML codes → CoreML SpeechDecoder (same as Test 1, but re-decoded) ───
    print("\n" + "=" * 70)
    print("TEST 3: CoreML codes → CoreML SpeechDecoder (re-decoded)")
    print("=" * 70)
    coreml_redecoded_audio = decode_codes_coreml(coreml_codes, models["SpeechDecoder"])
    save_normalized(coreml_redecoded_audio, f"{out_dir}/coreml_codes_coreml_decoder.wav")

    audios = [coreml_audio, pytorch_via_coreml_audio, coreml_redecoded_audio]
    audio_names = ["CoreML full pipeline", "PyTorch codes→CoreML decoder", "CoreML codes→CoreML decoder"]

    if not args.skip_pytorch:
        # ─── Test 4: PyTorch codes → PyTorch speech_tokenizer ───
        print("\n" + "=" * 70)
        print("TEST 4: PyTorch codes → PyTorch speech_tokenizer (reference)")
        print("=" * 70)
        pytorch_ref_audio = decode_codes_pytorch(pytorch_codes)
        save_normalized(pytorch_ref_audio, f"{out_dir}/pytorch_codes_pytorch_decoder.wav")

        # ─── Test 5: CoreML codes → PyTorch speech_tokenizer ───
        print("\n" + "=" * 70)
        print("TEST 5: CoreML codes → PyTorch speech_tokenizer")
        print("=" * 70)
        coreml_via_pytorch_audio = decode_codes_pytorch(coreml_codes)
        save_normalized(coreml_via_pytorch_audio, f"{out_dir}/coreml_codes_pytorch_decoder.wav")

        audios.extend([pytorch_ref_audio, coreml_via_pytorch_audio])
        audio_names.extend(["PyTorch codes→PyTorch decoder", "CoreML codes→PyTorch decoder"])

    # ─── Compare ───
    compare_code_stats({
        "PyTorch": pytorch_codes,
        "CoreML": coreml_codes,
    })

    compare_audio_stats(audios, audio_names)

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("DIAGNOSIS GUIDE")
    print("=" * 70)
    print(f"\nAll files saved in: {out_dir}/")
    print("\nListen and compare:")
    print("  1. pytorch_codes_coreml_decoder.wav  — PyTorch codes through CoreML SpeechDecoder")
    print("     → If robotic: SpeechDecoder is the problem")
    print("     → If natural: SpeechDecoder is fine, issue is in code generation")
    print()
    if not args.skip_pytorch:
        print("  2. coreml_codes_pytorch_decoder.wav  — CoreML codes through PyTorch decoder")
        print("     → If robotic: Code generation (CodeDecoder/MultiCodeDecoder) is the problem")
        print("     → If natural: Codes are fine, issue is in SpeechDecoder")
        print()
        print("  3. pytorch_codes_pytorch_decoder.wav — Reference (both PyTorch)")
        print("     → This should sound the best. If it's also robotic, the codes themselves are bad.")
    print()
    print("  4. coreml_full_pipeline.wav           — Current CoreML output")
    print("     → The one that sounds robotic")
    print()

    # Open files for listening
    print("Opening audio files for comparison...")
    import subprocess
    for name in audio_names:
        path = f"{out_dir}/{name.replace(' ', '_').replace('→', '_to_').replace('(', '').replace(')', '')}.wav"
        # Find actual file
    wavs = [f for f in os.listdir(out_dir) if f.endswith('.wav')]
    wavs.sort()
    for w in wavs:
        print(f"  {w}")


if __name__ == "__main__":
    main()
