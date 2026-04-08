#!/usr/bin/env python3
"""
Example usage of Cohere Transcribe Cache-External CoreML models.

Requirements:
    pip install coremltools numpy librosa soundfile sentencepiece
"""

import argparse
from pathlib import Path
import numpy as np
import coremltools as ct
import soundfile as sf
import librosa
import sentencepiece as spm

# Cohere config
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 400
MAX_FRAMES = 3500
MAX_SEQ_LEN = 108

# Special tokens - CRITICAL: Use correct EOS token!
START_TOKEN = 4
EOS_TOKEN = 3  # <|endoftext|> - verified from model.generation_config.eos_token_id


def compute_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """Compute mel spectrogram matching Cohere's preprocessing."""
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000,
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80) / 80
    mel = np.clip(mel, -1, 1)

    return mel


def pad_mel(mel, target_frames=MAX_FRAMES):
    """Pad mel spectrogram to target frames."""
    n_mels, n_frames = mel.shape

    if n_frames >= target_frames:
        return mel[:, :target_frames], n_frames

    padded = np.zeros((n_mels, target_frames), dtype=np.float32)
    padded[:, :n_frames] = mel

    return padded, n_frames


def create_attention_mask(seq_len):
    """Create causal attention mask for given sequence length."""
    return np.zeros((1, 1, 1, seq_len), dtype=np.float32)


def transcribe(audio_path, encoder, decoder, vocabulary):
    """Transcribe audio using cache-external decoder."""
    print(f"Transcribing: {audio_path}")

    # 1. Load audio
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s")

    # 2. Compute mel spectrogram
    mel = compute_mel_spectrogram(audio, sr)
    padded_mel, actual_frames = pad_mel(mel)
    print(f"  Mel frames: {actual_frames} (padded to {MAX_FRAMES})")

    # 3. Encode
    encoder_input = {
        "input_features": np.expand_dims(padded_mel, axis=0).astype(np.float32),
        "feature_length": np.array([actual_frames], dtype=np.int32),
    }
    encoder_output = encoder.predict(encoder_input)
    encoder_hidden = encoder_output["hidden_states"]
    print(f"  Encoder output shape: {encoder_hidden.shape}")

    # 4. Initialize caches (8 layers × K/V)
    k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

    # Cross-attention mask (all ones - attend to all encoder positions)
    encoder_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

    # 5. Decode with cache-external pattern
    tokens = []
    current_token = START_TOKEN

    for step in range(MAX_SEQ_LEN):
        # Build decoder input
        input_dict = {
            "input_id": np.array([[current_token]], dtype=np.int32),
            "position_id": np.array([[step]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float32),
            "cross_attention_mask": cross_mask,
            "attention_mask": create_attention_mask(step + 1),  # Grows each step!
        }

        # Add all K/V caches to input
        for i in range(8):
            input_dict[f"k_cache_{i}"] = k_caches[i]
            input_dict[f"v_cache_{i}"] = v_caches[i]

        # Run decoder (single step)
        output = decoder.predict(input_dict)

        # Sample next token (greedy)
        logits = output["logits"]
        next_token = int(np.argmax(logits[0]))

        # Update caches with outputs from model
        for i in range(8):
            k_caches[i] = output[f"k_cache_{i}_out"]
            v_caches[i] = output[f"v_cache_{i}_out"]

        # Check for EOS (end of sequence)
        if next_token == EOS_TOKEN:
            print(f"  EOS detected at step {step}")
            break

        tokens.append(next_token)
        current_token = next_token

    print(f"  Generated {len(tokens)} tokens")

    # 6. Detokenize
    text_tokens = []
    for token_id in tokens:
        if token_id <= 4 or token_id == EOS_TOKEN or token_id >= len(vocabulary):
            continue
        token = vocabulary[token_id]
        if token.startswith("<|"):
            continue
        text_tokens.append(token)

    text = "".join(text_tokens).replace("▁", " ").strip()

    return text


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Cohere Cache-External")
    parser.add_argument("audio", help="Path to audio file (.wav, .flac, etc.)")
    parser.add_argument("--encoder", default="cohere_encoder.mlpackage", help="Path to encoder")
    parser.add_argument("--decoder", default="cohere_decoder_cache_external.mlpackage", help="Path to decoder")
    parser.add_argument("--tokenizer", default="tokenizer.model", help="Path to tokenizer")
    args = parser.parse_args()

    print("=" * 70)
    print("Cohere Transcribe - Cache-External Decoder")
    print("=" * 70)
    print()

    # Load models
    print("Loading models...")
    encoder = ct.models.MLModel(args.encoder)
    decoder = ct.models.MLModel(args.decoder)
    print("  ✓ Models loaded")

    # Load vocabulary
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocabulary = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    print(f"  ✓ Loaded {len(vocabulary)} tokens")
    print()

    # Transcribe
    text = transcribe(args.audio, encoder, decoder, vocabulary)

    print()
    print("=" * 70)
    print("TRANSCRIPTION")
    print("=" * 70)
    print(text)
    print()


if __name__ == "__main__":
    main()
