#!/usr/bin/env python3
"""Test WER for cache-external decoder using compiled .mlmodelc (via .mlpackage fallback).

Since CoreMLTools can't load .mlmodelc directly, this test uses the .mlpackage
but verifies the .mlmodelc exists and documents that Swift would use it.
"""

import argparse
from pathlib import Path
import numpy as np
import coremltools as ct
import soundfile as sf
import librosa
import jiwer
from tqdm import tqdm
import json
import torch
from transformers import AutoModelForSpeechSeq2Seq

# Cohere config
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 400
MAX_FRAMES = 3500
MAX_SEQ_LEN = 108

# Special tokens (FIXED!)
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


def decode_with_cache_external(encoder_hidden, decoder_model, vocabulary):
    """Decode using cache-external decoder (Parakeet pattern)."""

    # Initialize caches
    k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

    # Cross-attention mask
    encoder_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

    tokens = []
    current_token = START_TOKEN

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

        # Update caches
        for i in range(8):
            k_caches[i] = output[f"k_cache_{i}_out"]
            v_caches[i] = output[f"v_cache_{i}_out"]

        # Check EOS
        if next_token == EOS_TOKEN:
            break

        tokens.append(next_token)
        current_token = next_token

    return detokenize(tokens, vocabulary)


def detokenize(token_ids, vocabulary):
    """Convert token IDs to text."""
    tokens = []
    for token_id in token_ids:
        if token_id <= 4 or token_id == EOS_TOKEN or token_id >= len(vocabulary):
            continue
        token = vocabulary[token_id]
        if token.startswith("<|"):
            continue
        tokens.append(token)

    text = "".join(tokens).replace("▁", " ").strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlmodelc", default="build-test/cohere_decoder_cache_external.mlmodelc")
    parser.add_argument("--mlpackage", default="build-test/cohere_decoder_cache_external.mlpackage")
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--test-dir", default="librispeech_test_samples")
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    print("="*70)
    print("Cohere Cache-External Decoder WER Test (.mlmodelc)")
    print("="*70)
    print("\nNote:")
    print("  • CoreMLTools can't load .mlmodelc directly")
    print("  • Using .mlpackage for Python testing")
    print("  • Swift would use the .mlmodelc (faster loading)")
    print()

    # Check .mlmodelc exists
    mlmodelc_path = Path(args.mlmodelc)
    if mlmodelc_path.exists():
        print(f"✓ Compiled model exists: {args.mlmodelc}")
    else:
        print(f"✗ Compiled model not found: {args.mlmodelc}")
        print("  Run: xcrun coremlcompiler compile <mlpackage> <output_dir>")
        return

    # Load test samples
    manifest_file = Path(args.test_dir) / "manifest.json"
    if not manifest_file.exists():
        print(f"No samples found at {manifest_file}")
        return

    with open(manifest_file) as f:
        samples = json.load(f)[:args.num_samples]

    print(f"\n[1/4] Loading PyTorch model...")
    pytorch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    pytorch_model.eval()
    print("  ✓ PyTorch model loaded")

    print(f"\n[2/4] Loading CoreML decoder (.mlpackage for Python)...")
    print(f"  {args.mlpackage}")
    decoder = ct.models.MLModel(args.mlpackage)
    print("  ✓ CoreML decoder loaded")
    print("  Note: Swift would load the .mlmodelc instead")

    print(f"\n[3/4] Loading vocabulary...")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load("../cohere-pytorch/tokenizer.model")
        vocabulary = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  ✓ Loaded {len(vocabulary)} tokens")
    except Exception as e:
        print(f"  ⚠️ Using placeholder vocab: {e}")
        vocabulary = ["<unk>"] * 16384

    print(f"\n[4/4] Running WER test on {len(samples)} samples...")

    results = []
    hypotheses = []
    references = []

    for sample in tqdm(samples):
        # Load audio
        audio, sr = sf.read(sample["audio"])

        # Compute mel
        mel = compute_mel_spectrogram(audio, sr)
        padded_mel, actual_frames = pad_mel(mel)

        # Encode with PyTorch
        encoder_hidden = encode_with_pytorch(padded_mel, actual_frames, pytorch_model)

        # Decode with CoreML cache-external
        hypothesis = decode_with_cache_external(encoder_hidden, decoder, vocabulary)

        reference = sample["text"].lower()
        hypothesis = hypothesis.lower()

        hypotheses.append(hypothesis)
        references.append(reference)

        wer = jiwer.wer(reference, hypothesis)

        results.append({
            "id": sample["id"],
            "duration": sample["duration"],
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer
        })

    # Compute overall WER
    overall_wer = jiwer.wer(references, hypotheses)

    print("\n" + "="*70)
    print("RESULTS - .mlmodelc Verification")
    print("="*70)

    print(f"\nOverall WER: {overall_wer*100:.2f}%")
    print(f"Expected: 11.95% (from .mlpackage test)")
    print(f"\nPer-sample WER:")
    for r in results:
        print(f"  Sample {r['id']:2d} ({r['duration']:5.1f}s): {r['wer']*100:6.2f}%")

    print("\n" + "="*70)
    print("✅ WER Test Complete!")
    print("="*70)
    print(f"\nNote: Swift would use {args.mlmodelc} (compiled) for faster loading.")
    print(f"Results should match .mlpackage test (11.95% WER on 10 samples).")


if __name__ == "__main__":
    main()
