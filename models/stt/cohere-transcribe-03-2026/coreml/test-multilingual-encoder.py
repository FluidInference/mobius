#!/usr/bin/env python3
"""Test multilingual encoder + cache-external decoder."""

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
EOS_TOKEN = 3

FLEURS_LANGUAGES = {
    "en_us": "English",
    "fr_fr": "French",
    "es_419": "Spanish",
    "cmn_hans_cn": "Mandarin Chinese",
}


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


def encode_with_coreml(mel, actual_frames, encoder):
    """Encode using CoreML encoder."""
    # Add batch dimension
    mel_batch = mel[np.newaxis, :, :]

    output = encoder.predict({
        "input_features": mel_batch.astype(np.float32),
        "feature_length": np.array([actual_frames], dtype=np.int32)
    })

    return output["hidden_states"]


def create_attention_mask(seq_len):
    """Create causal attention mask for given sequence length."""
    return np.zeros((1, 1, 1, seq_len), dtype=np.float32)


def decode_with_cache_external(encoder_hidden, decoder_model, vocabulary):
    """Decode using cache-external decoder (no language conditioning)."""

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
    parser.add_argument("--encoder", default="build-test/cohere_encoder_multilingual.mlpackage")
    parser.add_argument("--decoder", default="build-test/cohere_decoder_cache_external.mlpackage")
    parser.add_argument("--languages", default="en_us,fr_fr,es_419,cmn_hans_cn")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output", default="multilingual_encoder_test_results.json")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",")]

    print("="*70)
    print("Multilingual Encoder Test")
    print("="*70)
    print(f"\nEncoder: {args.encoder}")
    print(f"Decoder: {args.decoder}")
    print(f"\nLanguages: {', '.join([FLEURS_LANGUAGES.get(l, l) for l in languages])}")
    print(f"Samples per language: {args.num_samples}")
    print()

    # Load CoreML encoder
    print(f"[1/3] Loading CoreML encoder...")
    encoder = ct.models.MLModel(args.encoder)
    print("  ✓ CoreML encoder loaded")

    # Load CoreML decoder
    print(f"\n[2/3] Loading CoreML decoder...")
    decoder = ct.models.MLModel(args.decoder)
    print("  ✓ CoreML decoder loaded")

    # Load vocabulary
    print(f"\n[3/3] Loading vocabulary...")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load("../cohere-pytorch/tokenizer.model")
        vocabulary = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        print(f"  ✓ Loaded {len(vocabulary)} tokens")
    except Exception as e:
        print(f"  ⚠️ Using placeholder vocab: {e}")
        vocabulary = ["<unk>"] * 16384

    print()

    # Process each language
    all_results = []
    language_stats = {}

    for lang_code in languages:
        lang_name = FLEURS_LANGUAGES.get(lang_code, lang_code)
        print("="*70)
        print(f"Processing: {lang_name} ({lang_code})")
        print("="*70)

        # Load samples
        manifest_file = Path(f"fleurs_samples/{lang_code}/manifest.json")
        if not manifest_file.exists():
            print(f"No samples found. Samples should exist from previous test.")
            continue
        with open(manifest_file) as f:
            samples = json.load(f)[:args.num_samples]

        print(f"\nTranscribing {len(samples)} samples...")

        results = []
        hypotheses = []
        references = []

        for sample in tqdm(samples, desc=f"{lang_code}"):
            try:
                # Load audio
                audio, sr = sf.read(sample["audio"])

                # Compute mel
                mel = compute_mel_spectrogram(audio, sr)
                padded_mel, actual_frames = pad_mel(mel)

                # Encode with CoreML multilingual encoder
                encoder_hidden = encode_with_coreml(padded_mel, actual_frames, encoder)

                # Decode with cache-external decoder
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
                    "wer": wer,
                    "language": lang_code
                })

            except Exception as e:
                print(f"\n  Error on sample {sample['id']}: {e}")
                import traceback
                traceback.print_exc()

        # Compute language stats
        if len(hypotheses) > 0:
            overall_wer = jiwer.wer(references, hypotheses)
            language_stats[lang_code] = {
                "language_name": lang_name,
                "num_samples": len(results),
                "overall_wer": overall_wer,
                "samples": results
            }

            print(f"\n{lang_name} Results:")
            print(f"  Samples: {len(results)}")
            print(f"  Overall WER: {overall_wer*100:.2f}%")

            # Show first 3 examples
            print(f"\n  Sample outputs:")
            for i, r in enumerate(results[:3]):
                print(f"    [{i}] REF: {r['reference'][:80]}")
                print(f"        HYP: {r['hypothesis'][:80]}")
                print(f"        WER: {r['wer']*100:.1f}%")

            all_results.extend(results)

    # Save results
    output = {
        "languages": language_stats,
        "overall": {
            "total_samples": len(all_results),
            "languages_tested": len(languages)
        }
    }

    output_file = Path(args.output)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*70)
    print("OVERALL RESULTS - Multilingual Encoder")
    print("="*70)
    for lang_code in languages:
        if lang_code in language_stats:
            stats = language_stats[lang_code]
            print(f"{stats['language_name']:20s}: {stats['overall_wer']*100:6.2f}% WER ({stats['num_samples']} samples)")

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
