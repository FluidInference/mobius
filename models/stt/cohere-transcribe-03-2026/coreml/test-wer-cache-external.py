#!/usr/bin/env python3
"""Test WER for cache-external decoder on LibriSpeech test-clean."""

import argparse
from pathlib import Path
import numpy as np
import coremltools as ct
import soundfile as sf
import librosa
import jiwer
from tqdm import tqdm
import json

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
    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000,
    )

    # Convert to log scale
    mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [-1, 1] range (approximate)
    mel = (mel + 80) / 80
    mel = np.clip(mel, -1, 1)

    return mel  # Shape: (n_mels, n_frames)


def pad_mel(mel, target_frames=MAX_FRAMES):
    """Pad mel spectrogram to target frames."""
    n_mels, n_frames = mel.shape

    if n_frames >= target_frames:
        return mel[:, :target_frames], n_frames

    # Pad with zeros
    padded = np.zeros((n_mels, target_frames), dtype=np.float32)
    padded[:, :n_frames] = mel

    return padded, n_frames


def create_attention_mask(seq_len):
    """Create causal attention mask for given sequence length."""
    mask = np.zeros((1, 1, 1, seq_len), dtype=np.float32)
    # All zeros = can attend to all positions up to seq_len
    return mask


def decode_with_cache_external(
    encoder_hidden,
    decoder_model,
    vocabulary,
    max_new_tokens=MAX_SEQ_LEN,
):
    """Decode using cache-external decoder (Parakeet pattern)."""

    # Initialize caches (16 arrays: 8 layers × K/V)
    k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

    # Cross-attention mask (encoder sequence)
    encoder_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

    # Start decoding
    tokens = []
    current_token = START_TOKEN

    for step in range(max_new_tokens):
        # Create input dictionary
        input_dict = {
            "input_id": np.array([[current_token]], dtype=np.int32),
            "position_id": np.array([[step]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden,
            "cross_attention_mask": cross_mask,
            # Attention mask grows with sequence length
            "attention_mask": create_attention_mask(step + 1),
        }

        # Add all cache arrays
        for i in range(8):
            input_dict[f"k_cache_{i}"] = k_caches[i]
            input_dict[f"v_cache_{i}"] = v_caches[i]

        # Run decoder
        output = decoder_model.predict(input_dict)

        # Extract logits and sample
        logits = output["logits"]  # [1, 16384]
        next_token = int(np.argmax(logits[0]))

        # Update caches
        for i in range(8):
            k_caches[i] = output[f"k_cache_{i}_out"]
            v_caches[i] = output[f"v_cache_{i}_out"]

        # Check for EOS
        if next_token == EOS_TOKEN:
            break

        tokens.append(next_token)
        current_token = next_token

    # Detokenize
    text = detokenize(tokens, vocabulary)
    return text


def detokenize(token_ids, vocabulary):
    """Convert token IDs to text."""
    tokens = []
    for token_id in token_ids:
        if token_id <= 4 or token_id == EOS_TOKEN:
            continue
        if token_id >= len(vocabulary):
            continue
        token = vocabulary[token_id]
        if token.startswith("<|"):
            continue
        tokens.append(token)

    text = "".join(tokens)
    text = text.replace("▁", " ")
    text = text.strip()

    return text


def download_librispeech_sample(output_dir):
    """Download a few LibriSpeech test-clean samples for testing."""
    from datasets import load_dataset

    print("Downloading LibriSpeech test-clean samples...")
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        streaming=False
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i, example in enumerate(dataset):
        if i >= 10:  # Just get 10 samples
            break

        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        text = example["text"]

        # Save audio
        audio_file = output_dir / f"sample_{i:02d}.wav"
        sf.write(audio_file, audio, sr)

        # Save transcript
        text_file = output_dir / f"sample_{i:02d}.txt"
        text_file.write_text(text)

        samples.append({
            "id": i,
            "audio": str(audio_file),
            "text": text,
            "duration": len(audio) / sr
        })

        print(f"  Sample {i}: {len(audio)/sr:.1f}s - {text[:50]}...")

    # Save manifest
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"\n✓ Downloaded {len(samples)} samples to {output_dir}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="build-test/cohere_encoder.mlpackage")
    parser.add_argument("--decoder", default="build-test/cohere_decoder_cache_external.mlpackage")
    parser.add_argument("--vocab", default="../cohere-pytorch/tokenizer.model")
    parser.add_argument("--test-dir", default="librispeech_test_samples")
    parser.add_argument("--download", action="store_true", help="Download LibriSpeech samples")
    args = parser.parse_args()

    print("="*70)
    print("Cohere Cache-External Decoder WER Test")
    print("="*70)

    # Download samples if requested
    if args.download:
        samples = download_librispeech_sample(args.test_dir)
    else:
        # Load existing samples
        manifest_file = Path(args.test_dir) / "manifest.json"
        if not manifest_file.exists():
            print(f"No samples found. Run with --download first.")
            return

        with open(manifest_file) as f:
            samples = json.load(f)

    print(f"\n[1/4] Loading models...")
    print(f"  Encoder: {args.encoder}")
    print(f"  Decoder: {args.decoder}")

    encoder = ct.models.MLModel(args.encoder)
    decoder = ct.models.MLModel(args.decoder)

    print("  ✓ Models loaded")

    # Load vocabulary
    print(f"\n[2/4] Loading vocabulary from {args.vocab}...")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(args.vocab)
        vocabulary = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  ✓ Loaded {len(vocabulary)} tokens")
    except Exception as e:
        print(f"  ⚠️ Could not load SentencePiece vocab: {e}")
        print(f"  Using placeholder vocabulary")
        vocabulary = ["<unk>"] * 16384

    print(f"\n[3/4] Running inference on {len(samples)} samples...")

    results = []
    hypotheses = []
    references = []

    for sample in tqdm(samples):
        # Load audio
        audio, sr = sf.read(sample["audio"])

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(audio, sr)
        padded_mel, actual_frames = pad_mel(mel)

        # Prepare encoder input
        input_features = padded_mel[np.newaxis, :, :]  # [1, 128, 3500]
        feature_length = np.array([actual_frames], dtype=np.int32)

        # Run encoder
        encoder_output = encoder.predict({
            "input_features": input_features.astype(np.float32),
            "feature_length": feature_length
        })
        encoder_hidden = encoder_output["hidden_states"]

        # Run cache-external decoder
        hypothesis = decode_with_cache_external(
            encoder_hidden,
            decoder,
            vocabulary,
            max_new_tokens=MAX_SEQ_LEN
        )

        reference = sample["text"].lower()
        hypothesis = hypothesis.lower()

        hypotheses.append(hypothesis)
        references.append(reference)

        # Compute WER for this sample
        wer = jiwer.wer(reference, hypothesis)

        results.append({
            "id": sample["id"],
            "duration": sample["duration"],
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer
        })

        print(f"\n  Sample {sample['id']}:")
        print(f"    REF: {reference[:80]}")
        print(f"    HYP: {hypothesis[:80]}")
        print(f"    WER: {wer*100:.2f}%")

    print(f"\n[4/4] Computing overall WER...")

    # Compute overall WER
    overall_wer = jiwer.wer(references, hypotheses)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nOverall WER: {overall_wer*100:.2f}%")
    print(f"\nPer-sample results:")
    for result in results:
        print(f"  Sample {result['id']:2d} ({result['duration']:5.1f}s): WER={result['wer']*100:6.2f}%")

    # Save results
    results_file = Path(args.test_dir) / "wer_results_cache_external.json"
    with open(results_file, "w") as f:
        json.dump({
            "overall_wer": overall_wer,
            "samples": results
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    print("\n" + "="*70)
    print("✅ WER Test Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
