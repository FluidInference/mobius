#!/usr/bin/env python3
"""Test per-language cache-external decoders on FLEURS dataset.

Each language uses its dedicated decoder with language bias baked in.
"""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import load_dataset
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq

NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
MAX_SEQ_LEN = 108

# Cohere mel spectrogram config
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 400
MAX_FRAMES = 3500

# Language mapping: FLEURS code -> (language name, decoder filename)
LANGUAGE_MAP = {
    "en_us": ("english", "cohere_decoder_english.mlpackage"),
    "fr_fr": ("french", "cohere_decoder_french.mlpackage"),
    "es_419": ("spanish", "cohere_decoder_spanish.mlpackage"),
    "cmn_hans_cn": ("chinese", "cohere_decoder_chinese.mlpackage"),
}

# Special tokens
START_TOKEN = 4
END_TOKEN = 5


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


def load_encoder():
    """Load PyTorch encoder for baseline."""
    print("[1/3] Loading PyTorch encoder...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


def load_decoder_for_language(language_name: str, decoder_dir: Path):
    """Load CoreML decoder for specific language."""
    decoder_filename = None
    for lang_code, (lang_name, filename) in LANGUAGE_MAP.items():
        if lang_name == language_name:
            decoder_filename = filename
            break

    if not decoder_filename:
        raise ValueError(f"Unknown language: {language_name}")

    decoder_path = decoder_dir / decoder_filename
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder not found: {decoder_path}")

    print(f"   Loading {language_name} decoder from {decoder_path.name}")
    decoder = ct.models.MLModel(str(decoder_path))
    return decoder


def encode_pytorch(model, mel, actual_frames):
    """Encode mel spectrogram using PyTorch encoder."""
    with torch.no_grad():
        input_features = torch.from_numpy(mel[np.newaxis, :, :]).float()
        feature_length = torch.tensor([actual_frames], dtype=torch.int32)

        encoder_outputs = model.encoder(
            input_features=input_features,
            length=feature_length,
            return_dict=True,
        )

        hidden_states = encoder_outputs.last_hidden_state

        # Apply encoder-decoder projection if present
        if model.encoder_decoder_proj is not None:
            hidden_states = model.encoder_decoder_proj(hidden_states)

    return hidden_states.numpy()


def decode_coreml_per_language(
    decoder, encoder_hidden, vocabulary, max_new_tokens=96
):
    """Decode using language-specific CoreML decoder (cache-external)."""
    batch_size = 1
    encoder_hidden_np = encoder_hidden
    encoder_seq_len = encoder_hidden_np.shape[1]

    # Initialize KV caches
    k_caches = [
        np.zeros((batch_size, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32)
        for _ in range(NUM_LAYERS)
    ]
    v_caches = [
        np.zeros((batch_size, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32)
        for _ in range(NUM_LAYERS)
    ]

    generated_ids = [START_TOKEN]
    current_token = START_TOKEN

    for step in range(max_new_tokens):
        # Prepare inputs
        input_id = np.array([[current_token]], dtype=np.int32)
        position_id = np.array([[step]], dtype=np.int32)
        cross_attn_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)
        attn_mask = np.zeros((1, 1, 1, step + 1), dtype=np.float32)

        # Build input dict
        inputs = {
            "input_id": input_id,
            "position_id": position_id,
            "encoder_hidden_states": encoder_hidden_np,
            "cross_attention_mask": cross_attn_mask,
            "attention_mask": attn_mask,
        }

        for i in range(NUM_LAYERS):
            inputs[f"k_cache_{i}"] = k_caches[i]
            inputs[f"v_cache_{i}"] = v_caches[i]

        # Run decoder
        outputs = decoder.predict(inputs)

        # Get logits and next token
        logits = outputs["logits"]
        next_token = int(np.argmax(logits[0]))

        # Update caches
        for i in range(NUM_LAYERS):
            k_caches[i] = outputs[f"k_cache_{i}_out"]
            v_caches[i] = outputs[f"v_cache_{i}_out"]

        # Check for end token
        if next_token == END_TOKEN:
            break

        generated_ids.append(next_token)
        current_token = next_token

    # Decode tokens
    tokens_to_decode = [t for t in generated_ids if t not in [START_TOKEN, END_TOKEN]]
    hypothesis = "".join([vocabulary.get(t, f"<unk_{t}>") for t in tokens_to_decode])
    hypothesis = hypothesis.replace("▁", " ").strip()

    return hypothesis


def test_language(
    language_code: str,
    num_samples: int,
    encoder_model,
    decoder_dir: Path,
    vocabulary: dict,
):
    """Test a single language with its dedicated decoder."""
    language_name, decoder_filename = LANGUAGE_MAP[language_code]

    print(f"\n{'='*70}")
    print(f"Testing {language_name.upper()} (FLEURS: {language_code})")
    print(f"{'='*70}")
    print(f"Decoder: {decoder_filename}")
    print(f"Samples: {num_samples}")

    # Load language-specific decoder
    decoder = load_decoder_for_language(language_name, decoder_dir)

    # Load FLEURS dataset
    print(f"\nLoading FLEURS {language_code} dataset...")
    dataset = load_dataset(
        "google/fleurs", language_code, split="test", trust_remote_code=True
    )

    results = []
    total_wer = 0.0

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        reference = sample["transcription"]

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(audio, sr)
        mel_padded, actual_frames = pad_mel(mel)

        # Encode
        encoder_hidden = encode_pytorch(encoder_model, mel_padded, actual_frames)

        # Decode with language-specific decoder
        hypothesis = decode_coreml_per_language(decoder, encoder_hidden, vocabulary)

        # Compute WER
        sample_wer = wer(reference, hypothesis) * 100

        total_wer += sample_wer

        result = {
            "sample_id": i,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(sample_wer, 2),
        }
        results.append(result)

        print(f"\nSample {i}:")
        print(f"  REF: {reference[:100]}...")
        print(f"  HYP: {hypothesis[:100]}...")
        print(f"  WER: {sample_wer:.1f}%")

    avg_wer = total_wer / len(results) if results else 0.0

    print(f"\n{'='*70}")
    print(f"{language_name.upper()} Results")
    print(f"{'='*70}")
    print(f"Average WER: {avg_wer:.1f}%")

    return {
        "language": language_name,
        "fleurs_code": language_code,
        "decoder": decoder_filename,
        "num_samples": len(results),
        "average_wer": round(avg_wer, 2),
        "samples": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder-dir",
        default="build-per-language",
        help="Directory containing per-language decoders",
    )
    parser.add_argument(
        "--languages",
        default="en_us,fr_fr,es_419,cmn_hans_cn",
        help="Comma-separated FLEURS language codes",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Samples per language")
    parser.add_argument("--output", default="per_language_results.json", help="Output JSON file")
    args = parser.parse_args()

    decoder_dir = Path(args.decoder_dir)
    if not decoder_dir.exists():
        print(f"❌ Decoder directory not found: {decoder_dir}")
        return

    languages = [lang.strip() for lang in args.languages.split(",")]

    print("="*70)
    print("Per-Language Decoder Test on FLEURS")
    print("="*70)
    print(f"Languages: {', '.join(languages)}")
    print(f"Samples per language: {args.num_samples}")
    print(f"Decoder directory: {decoder_dir}")

    # Load encoder
    encoder_model = load_encoder()

    # Load vocabulary
    print("\n[2/3] Loading vocabulary...")
    vocab_path = Path("f16/vocab.json")
    with open(vocab_path) as f:
        vocab_data = json.load(f)
    vocabulary = {int(k): v for k, v in vocab_data.items()}
    print(f"   Loaded {len(vocabulary)} tokens")

    # Test each language
    print("\n[3/3] Testing languages...")
    all_results = []

    for language_code in languages:
        if language_code not in LANGUAGE_MAP:
            print(f"⚠️ Unknown language code: {language_code}")
            continue

        try:
            result = test_language(
                language_code,
                args.num_samples,
                encoder_model,
                decoder_dir,
                vocabulary,
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to test {language_code}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    summary_table = []
    for result in all_results:
        summary_table.append({
            "Language": result["language"].capitalize(),
            "FLEURS Code": result["fleurs_code"],
            "Samples": result["num_samples"],
            "Avg WER": f"{result['average_wer']:.1f}%",
        })

    # Print table
    print(f"\n{'Language':<12} {'FLEURS Code':<15} {'Samples':<10} {'Avg WER':<10}")
    print("-" * 70)
    for row in summary_table:
        print(f"{row['Language']:<12} {row['FLEURS Code']:<15} {row['Samples']:<10} {row['Avg WER']:<10}")

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(
            {
                "test_config": {
                    "decoder_dir": str(decoder_dir),
                    "languages": languages,
                    "num_samples": args.num_samples,
                },
                "results": all_results,
                "summary": summary_table,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
