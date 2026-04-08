#!/usr/bin/env python3
"""Test WER for cache-external decoder (PyTorch encoder, CoreML decoder).

This hybrid approach:
- Uses PyTorch for encoder (fast, no export needed)
- Uses CoreML for cache-external decoder (what we want to test!)
- Computes WER on LibriSpeech test-clean
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
        # Prepare input
        input_features = torch.from_numpy(mel[np.newaxis, :, :]).float()  # [1, 128, 3500]
        feature_length = torch.tensor([actual_frames], dtype=torch.int32)

        # Run encoder
        encoder_outputs = pytorch_model.encoder(
            input_features=input_features,
            length=feature_length,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Apply projection
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


def download_librispeech_samples(output_dir, num_samples=10):
    """Download LibriSpeech test-clean samples."""
    from datasets import load_dataset

    print(f"Downloading {num_samples} LibriSpeech test-clean samples...")
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break

        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        text = example["text"]

        audio_file = output_dir / f"sample_{i:02d}.wav"
        sf.write(audio_file, audio, sr)

        text_file = output_dir / f"sample_{i:02d}.txt"
        text_file.write_text(text)

        samples.append({
            "id": i,
            "audio": str(audio_file),
            "text": text,
            "duration": len(audio) / sr
        })

        print(f"  {i+1}/{num_samples}: {len(audio)/sr:.1f}s - {text[:50]}...")

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"✓ Downloaded to {output_dir}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", default="build-test/cohere_decoder_cache_external.mlpackage")
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--test-dir", default="librispeech_test_samples")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    print("="*70)
    print("Cohere Cache-External Decoder WER Test (Hybrid)")
    print("="*70)
    print("\nApproach:")
    print("  • PyTorch encoder (fast, no export needed)")
    print("  • CoreML cache-external decoder (what we're testing!)")
    print("  • LibriSpeech test-clean WER evaluation")
    print()

    # Download samples
    if args.download:
        samples = download_librispeech_samples(args.test_dir, args.num_samples)
    else:
        manifest_file = Path(args.test_dir) / "manifest.json"
        if not manifest_file.exists():
            print("No samples found. Run with --download first.")
            return
        with open(manifest_file) as f:
            samples = json.load(f)

    print(f"\n[1/4] Loading PyTorch model...")
    pytorch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    pytorch_model.eval()
    print("  ✓ PyTorch model loaded")

    print(f"\n[2/4] Loading CoreML decoder...")
    print(f"  {args.decoder}")
    decoder = ct.models.MLModel(args.decoder)
    print("  ✓ CoreML decoder loaded")

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

        print(f"\n  Sample {sample['id']} ({sample['duration']:.1f}s):")
        print(f"    REF: {reference[:70]}")
        print(f"    HYP: {hypothesis[:70]}")
        print(f"    WER: {wer*100:.2f}%")

    # Compute overall WER
    overall_wer = jiwer.wer(references, hypotheses)

    print("\n" + "="*70)
    print("RESULTS - Cache-External Decoder")
    print("="*70)

    print(f"\nOverall WER: {overall_wer*100:.2f}%")
    print(f"\nPer-sample WER:")
    for r in results:
        print(f"  Sample {r['id']:2d} ({r['duration']:5.1f}s): {r['wer']*100:6.2f}%")

    # Save results
    results_file = Path(args.test_dir) / "wer_results_cache_external.json"
    with open(results_file, "w") as f:
        json.dump({"overall_wer": overall_wer, "samples": results}, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    print("\n" + "="*70)
    print("✅ WER Test Complete!")
    print("="*70)
    print(f"\nCache-External Decoder WER: {overall_wer*100:.2f}%")


if __name__ == "__main__":
    main()
