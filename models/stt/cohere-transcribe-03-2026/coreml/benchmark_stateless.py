#!/usr/bin/env python3
"""Benchmark stateful vs stateless decoder performance."""

import time
from pathlib import Path
import sys
import json

import coremltools as ct
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent / "f16"))
from cohere_mel_spectrogram import CohereMelSpectrogram

# Language prompt and special tokens
ENGLISH_PROMPT = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3


def load_encoder():
    """Load encoder (FP16 from f16/ directory)."""
    encoder_path = Path("f16/cohere_encoder.mlpackage")
    if not encoder_path.exists():
        # Try build-35s
        encoder_path = Path("build-35s/cohere_encoder.mlpackage")

    print(f"Loading encoder from {encoder_path}...")
    encoder = ct.models.MLModel(str(encoder_path))
    print("✓ Encoder loaded")
    return encoder


def load_decoders():
    """Load both stateful and stateless decoders."""
    stateful_path = Path("f16/cohere_decoder_stateful.mlpackage")
    if not stateful_path.exists():
        stateful_path = Path("build-35s/cohere_decoder_stateful.mlpackage")

    stateless_path = Path("build-stateless-nn/cohere_decoder_stateless.mlpackage")

    print(f"Loading stateful decoder from {stateful_path}...")
    stateful = ct.models.MLModel(str(stateful_path))
    print("✓ Stateful loaded")

    print(f"Loading stateless decoder from {stateless_path}...")
    stateless = ct.models.MLModel(str(stateless_path))
    print("✓ Stateless loaded")

    return stateful, stateless


def encode_audio(encoder, audio_path):
    """Encode audio to hidden states."""
    # Load audio
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Mel spectrogram
    mel_processor = CohereMelSpectrogram()
    mel = mel_processor(audio)

    # Pad/truncate to 3500 frames
    if mel.shape[2] > 3500:
        mel_padded = mel[:, :, :3500]
        actual_length = 3500
    else:
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])))
        actual_length = mel.shape[2]

    # Encode
    start = time.perf_counter()
    encoder_out = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([actual_length], dtype=np.int32),
    })
    encode_time = time.perf_counter() - start

    return encoder_out["hidden_states"], encode_time, len(audio) / 16000


def decode_stateful(decoder, encoder_hidden, max_tokens=108):
    """Decode with stateful decoder."""
    state = decoder.make_state()

    enc_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, enc_seq_len), dtype=np.float16)

    tokens = []
    last_token = None
    times = []

    for step in range(max_tokens):
        current_token = ENGLISH_PROMPT[step] if step < len(ENGLISH_PROMPT) else last_token

        input_id = np.array([[current_token]], dtype=np.int32)
        attention_mask = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
        position_ids = np.array([[step]], dtype=np.int32)

        start = time.perf_counter()
        decoder_out = decoder.predict(
            {
                "input_id": input_id,
                "encoder_hidden_states": encoder_hidden.astype(np.float16),
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_mask,
                "position_ids": position_ids,
            },
            state=state,
        )
        step_time = time.perf_counter() - start
        times.append(step_time)

        next_token = int(np.argmax(decoder_out["logits"][0]))
        last_token = next_token

        if step >= len(ENGLISH_PROMPT) - 1:
            tokens.append(next_token)
            if next_token == EOS_TOKEN_ID:
                break

    return tokens, times


def decode_stateless(decoder, encoder_hidden, max_tokens=108):
    """Decode with stateless decoder."""
    enc_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, enc_seq_len), dtype=np.float32)

    tokens = []
    all_tokens = list(ENGLISH_PROMPT[:1])  # Start with BOS
    times = []

    for step in range(max_tokens):
        # Prepare all tokens so far
        input_ids = np.array([all_tokens], dtype=np.int32)

        start = time.perf_counter()
        decoder_out = decoder.predict({
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden.astype(np.float32),
            "cross_attention_mask": cross_mask,
        })
        step_time = time.perf_counter() - start
        times.append(step_time)

        next_token = int(np.argmax(decoder_out["logits"][0]))

        # Feed prompt for first steps
        if step < len(ENGLISH_PROMPT) - 1:
            all_tokens.append(ENGLISH_PROMPT[step + 1])
        else:
            all_tokens.append(next_token)
            tokens.append(next_token)

            if next_token == EOS_TOKEN_ID:
                break

    return tokens, times


def tokens_to_text(tokens, vocab):
    """Convert tokens to text."""
    text_tokens = []
    for token_id in tokens:
        if token_id <= 4 or token_id == EOS_TOKEN_ID:
            continue
        token_str = vocab.get(token_id, "")
        if token_str.startswith("<|"):
            continue
        text_tokens.append(token_str)

    text = "".join(text_tokens).replace("▁", " ").strip()
    return text


def benchmark_audio(encoder, stateful, stateless, vocab, audio_path):
    """Benchmark single audio file."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {Path(audio_path).name}")
    print(f"{'='*70}")

    # Encode (same for both)
    encoder_hidden, encode_time, audio_duration = encode_audio(encoder, audio_path)
    print(f"\nAudio duration: {audio_duration:.2f}s")
    print(f"Encode time: {encode_time*1000:.1f}ms")

    # Decode with stateful
    print("\n[Stateful Decoder]")
    tokens_stateful, times_stateful = decode_stateful(stateful, encoder_hidden)
    total_stateful = sum(times_stateful)
    avg_stateful = np.mean(times_stateful) * 1000
    text_stateful = tokens_to_text(tokens_stateful, vocab)

    print(f"  Tokens: {len(tokens_stateful)}")
    print(f"  Total decode: {total_stateful*1000:.1f}ms")
    print(f"  Per token: {avg_stateful:.1f}ms (avg)")
    print(f"  Text: {text_stateful}")

    # Decode with stateless
    print("\n[Stateless Decoder]")
    tokens_stateless, times_stateless = decode_stateless(stateless, encoder_hidden)
    total_stateless = sum(times_stateless)
    avg_stateless = np.mean(times_stateless) * 1000
    text_stateless = tokens_to_text(tokens_stateless, vocab)

    print(f"  Tokens: {len(tokens_stateless)}")
    print(f"  Total decode: {total_stateless*1000:.1f}ms")
    print(f"  Per token: {avg_stateless:.1f}ms (avg)")
    print(f"  Text: {text_stateless}")

    # Comparison
    print("\n[Comparison]")
    slowdown = total_stateless / total_stateful
    print(f"  Stateless slowdown: {slowdown:.2f}x")
    print(f"  Per-token slowdown: {avg_stateless/avg_stateful:.2f}x")

    # Total times
    total_time_stateful = encode_time + total_stateful
    total_time_stateless = encode_time + total_stateless
    rtf_stateful = total_time_stateful / audio_duration
    rtf_stateless = total_time_stateless / audio_duration

    print(f"\n[Total Pipeline]")
    print(f"  Stateful: {total_time_stateful*1000:.1f}ms (RTFx: {rtf_stateful:.3f})")
    print(f"  Stateless: {total_time_stateless*1000:.1f}ms (RTFx: {rtf_stateless:.3f})")

    # Text match
    match = "✅" if text_stateful == text_stateless else "❌"
    print(f"\n[Text Match] {match}")
    if text_stateful != text_stateless:
        print(f"  Stateful:  {text_stateful}")
        print(f"  Stateless: {text_stateless}")

    return {
        "file": Path(audio_path).name,
        "duration": audio_duration,
        "encode_ms": encode_time * 1000,
        "stateful_tokens": len(tokens_stateful),
        "stateful_total_ms": total_stateful * 1000,
        "stateful_per_token_ms": avg_stateful,
        "stateless_tokens": len(tokens_stateless),
        "stateless_total_ms": total_stateless * 1000,
        "stateless_per_token_ms": avg_stateless,
        "slowdown": slowdown,
        "rtf_stateful": rtf_stateful,
        "rtf_stateless": rtf_stateless,
        "text_match": text_stateful == text_stateless,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_files", nargs="+", help="Audio files to benchmark")
    args = parser.parse_args()

    # Load models
    print("Loading models...")
    encoder = load_encoder()
    stateful, stateless = load_decoders()

    # Load vocab
    vocab_path = Path("f16/vocab.json")
    if not vocab_path.exists():
        vocab_path = Path("build-35s/vocab.json")
    with open(vocab_path) as f:
        vocab = {int(k): v for k, v in json.load(f).items()}

    print(f"\n{'='*70}")
    print("Models loaded, starting benchmark...")
    print(f"{'='*70}")

    # Benchmark each file
    results = []
    for audio_file in args.audio_files:
        result = benchmark_audio(encoder, stateful, stateless, vocab, audio_file)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    avg_slowdown = np.mean([r["slowdown"] for r in results])
    avg_per_token_stateful = np.mean([r["stateful_per_token_ms"] for r in results])
    avg_per_token_stateless = np.mean([r["stateless_per_token_ms"] for r in results])

    print(f"Files tested: {len(results)}")
    print(f"Average stateful per-token: {avg_per_token_stateful:.1f}ms")
    print(f"Average stateless per-token: {avg_per_token_stateless:.1f}ms")
    print(f"Average slowdown: {avg_slowdown:.2f}x")
    print(f"Text matches: {sum(r['text_match'] for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
