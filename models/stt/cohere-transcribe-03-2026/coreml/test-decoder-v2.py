#!/usr/bin/env python3
"""Test V2 decoder with language_id input."""

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
EOS_TOKEN = 3

# Language ID mapping (matches export script)
LANGUAGE_IDS = {
    "en_us": 0,       # English
    "fr_fr": 1,       # French
    "es_419": 2,      # Spanish
    "cmn_hans_cn": 3, # Chinese
}

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


def decode_with_v2(encoder_hidden, decoder_model, vocabulary, language_code):
    """Decode using V2 decoder with language_id."""

    # Get language ID
    language_id = LANGUAGE_IDS.get(language_code, 0)

    # Initialize caches
    k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

    # Cross-attention mask
    encoder_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

    tokens = []
    current_token = START_TOKEN

    for step in range(MAX_SEQ_LEN):
        # Build input with language_id
        input_dict = {
            "language_id": np.array([language_id], dtype=np.int32),
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
    parser.add_argument("--decoder", default="build-v2/cohere_decoder_cache_external_v2.mlpackage")
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--languages", default="en_us,fr_fr,es_419,cmn_hans_cn")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output", default="decoder_v2_results.json")
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",")]

    print("="*70)
    print("Decoder V2 Test - Language Conditioning via language_id")
    print("="*70)
    print(f"\nLanguages: {', '.join([FLEURS_LANGUAGES.get(l, l) for l in languages])}")
    print(f"Samples per language: {args.num_samples}")
    print(f"\nUsing language_id input (no token prompts needed)")
    print()

    # Load PyTorch model
    print(f"[1/3] Loading PyTorch model...")
    pytorch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    pytorch_model.eval()
    print("  ✓ PyTorch model loaded")

    # Load CoreML decoder V2
    print(f"\n[2/3] Loading CoreML decoder V2...")
    print(f"  {args.decoder}")
    decoder = ct.models.MLModel(args.decoder)
    print("  ✓ CoreML decoder V2 loaded")

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
        lang_id = LANGUAGE_IDS.get(lang_code, 0)
        print("="*70)
        print(f"Processing: {lang_name} ({lang_code})")
        print(f"Language ID: {lang_id}")
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

                # Encode with PyTorch
                encoder_hidden = encode_with_pytorch(padded_mel, actual_frames, pytorch_model)

                # Decode with CoreML V2 decoder
                hypothesis = decode_with_v2(encoder_hidden, decoder, vocabulary, lang_code)

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

        # Compute language stats
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
    print("OVERALL RESULTS - DECODER V2")
    print("="*70)
    for lang_code in languages:
        if lang_code in language_stats:
            stats = language_stats[lang_code]
            print(f"{stats['language_name']:20s}: {stats['overall_wer']*100:6.2f}% WER ({stats['num_samples']} samples)")

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
