#!/usr/bin/env python3
"""Debug token generation for cache-external decoder."""

import argparse
from pathlib import Path
import numpy as np
import coremltools as ct
import soundfile as sf
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq

# Cohere config
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 400
MAX_FRAMES = 3500
MAX_SEQ_LEN = 108

# Special tokens
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


def encode_with_pytorch(mel, actual_frames, pytorch_model):
    """Encode using PyTorch model."""
    with torch.no_grad():
        input_features = torch.from_numpy(mel[np.newaxis, :, :]).float()
        feature_length = torch.tensor([actual_frames], dtype=torch.int32)

        encoder_outputs = pytorch_model.encoder(
            input_features=input_features,
            length=feature_length,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state

        if pytorch_model.encoder_decoder_proj is not None:
            hidden_states = pytorch_model.encoder_decoder_proj(hidden_states)

        return hidden_states.numpy()


def create_attention_mask(seq_len):
    """Create causal attention mask for given sequence length."""
    return np.zeros((1, 1, 1, seq_len), dtype=np.float32)


def decode_with_cache_external_debug(encoder_hidden, decoder_model, vocabulary):
    """Decode using cache-external decoder with debug output."""

    # Initialize caches
    k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

    # Cross-attention mask
    encoder_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

    tokens = []
    current_token = START_TOKEN

    print(f"\nStarting decode with START_TOKEN={START_TOKEN}, EOS_TOKEN={EOS_TOKEN}")
    print(f"Encoder hidden shape: {encoder_hidden.shape}")
    print()

    for step in range(MAX_SEQ_LEN):
        # Build input
        input_dict = {
            "input_id": np.array([[current_token]], dtype=np.int32),
            "position_id": np.array([[step]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float32),
            "cross_attention_mask": cross_mask,
            "attention_mask": create_attention_mask(step + 1),
        }

        # Add caches
        for i in range(8):
            input_dict[f"k_cache_{i}"] = k_caches[i]
            input_dict[f"v_cache_{i}"] = v_caches[i]

        # Run decoder
        output = decoder_model.predict(input_dict)

        # Sample next token
        logits = output["logits"]
        next_token = int(np.argmax(logits[0]))

        # Get top 5 tokens for debugging
        top5_indices = np.argsort(logits[0])[-5:][::-1]
        top5_probs = logits[0][top5_indices]

        # Show token info
        token_str = vocabulary[next_token] if next_token < len(vocabulary) else f"<OUT_OF_RANGE_{next_token}>"

        print(f"Step {step:3d}: current={current_token:5d}, next={next_token:5d} '{token_str}' (logit={logits[0][next_token]:.2f})")
        print(f"         Top 5: ", end="")
        for idx in top5_indices:
            tok = vocabulary[idx] if idx < len(vocabulary) else f"<OOR_{idx}>"
            print(f"{idx}({tok})={logits[0][idx]:.1f} ", end="")
        print()

        # Update caches
        for i in range(8):
            k_caches[i] = output[f"k_cache_{i}_out"]
            v_caches[i] = output[f"v_cache_{i}_out"]

        # Check EOS
        if next_token == EOS_TOKEN:
            print(f"\n✓ EOS token detected at step {step}")
            break

        tokens.append(next_token)
        current_token = next_token

        # Stop early for debugging
        if step >= 20:
            print(f"\n⚠️ Stopping at step {step} for debugging")
            break

    print(f"\nGenerated {len(tokens)} tokens")
    print(f"Token IDs: {tokens[:20]}...")  # First 20

    # Detokenize
    text_tokens = []
    for token_id in tokens:
        if token_id <= 4 or token_id == EOS_TOKEN or token_id >= len(vocabulary):
            continue
        token = vocabulary[token_id]
        if token.startswith("<|"):
            continue
        text_tokens.append(token)

    text = "".join(text_tokens).replace("▁", " ").strip()

    print(f"\nDecoded text: '{text}'")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", default="build-test/cohere_decoder_cache_external.mlpackage")
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--audio", default="librispeech_test_samples/sample_07.wav")
    args = parser.parse_args()

    print("="*70)
    print("Debug Token Generation - Cache-External Decoder")
    print("="*70)
    print()

    # Load audio
    print(f"[1/5] Loading audio: {args.audio}")
    audio, sr = sf.read(args.audio)
    print(f"   Duration: {len(audio)/sr:.2f}s")

    # Compute mel
    print("\n[2/5] Computing mel spectrogram...")
    mel = compute_mel_spectrogram(audio, sr)
    padded_mel, actual_frames = pad_mel(mel)
    print(f"   Mel shape: {mel.shape}, padded: {padded_mel.shape}")

    # Load PyTorch model
    print(f"\n[3/5] Loading PyTorch model...")
    pytorch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    pytorch_model.eval()
    print("   ✓ Loaded")

    # Encode
    print("\n[4/5] Encoding with PyTorch...")
    encoder_hidden = encode_with_pytorch(padded_mel, actual_frames, pytorch_model)
    print(f"   Encoder output shape: {encoder_hidden.shape}")

    # Load CoreML decoder
    print(f"\n[5/5] Loading CoreML decoder...")
    decoder = ct.models.MLModel(args.decoder)
    print("   ✓ Loaded")

    # Load vocabulary
    print("\nLoading vocabulary...")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load("../cohere-pytorch/tokenizer.model")
        vocabulary = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"   ✓ Loaded {len(vocabulary)} tokens")
        print(f"   Token 4 (START): '{vocabulary[4]}'")
        print(f"   Token 151643 (EOS): '{vocabulary[151643] if 151643 < len(vocabulary) else 'OUT OF RANGE'}'")
    except Exception as e:
        print(f"   ⚠️ Error loading vocab: {e}")
        vocabulary = ["<unk>"] * 200000  # Large enough for token 151643

    # Decode with debug
    print("\n" + "="*70)
    print("DECODING WITH DEBUG")
    print("="*70)

    text = decode_with_cache_external_debug(encoder_hidden, decoder, vocabulary)

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"\nDecoded: '{text}'")


if __name__ == "__main__":
    main()
