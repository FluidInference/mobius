#!/usr/bin/env python3
"""Benchmark Cohere Transcribe CoreML models on LibriSpeech or FLEURS.

Examples:
    # Test FP16 models on 10 LibriSpeech samples
    python benchmark.py --precision fp16 --samples 10

    # Test Q8 models on 100 FLEURS samples (Japanese)
    python benchmark.py --precision q8 --samples 100 --dataset fleurs --language ja_jp

    # Test with normalized WER (removes punctuation)
    python benchmark.py --precision fp16 --samples 10 --normalize

    # Output to custom file
    python benchmark.py --precision q8 --samples 50 --output results.json
"""

import sys
from pathlib import Path
import argparse

# Add tools/ to path for the numpy feature extractor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import numpy as np
import coremltools as ct
from cohere_features_v2 import CohereMelSpectrogram
from datasets import load_dataset
from jiwer import wer
from jiwer.transforms import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
import json
import time


# Create text normalization pipeline using jiwer
# Works for all languages: English, CJK, European, Cyrillic, Arabic, etc.
normalize_text = Compose([
    ToLowerCase(),           # Convert to lowercase (all case-bearing scripts)
    RemovePunctuation(),     # Remove punctuation (Latin, CJK, Cyrillic, Arabic, etc.)
    RemoveMultipleSpaces(),  # Normalize whitespace
    Strip(),                 # Strip leading/trailing whitespace
])


def benchmark(precision="fp16", num_samples=10, normalize=False, output_file=None,
              dataset="librispeech", language="en_us", models_dir=None):
    """Run benchmark on specified precision and number of samples."""

    # models_dir defaults to ./<precision> (caller should pass an absolute path
    # populated via `huggingface-cli download FluidInference/cohere-transcribe-03-2026-coreml`).
    model_dir = models_dir if models_dir is not None else precision

    print("="*70)
    print(f"Cohere Transcribe Benchmark ({precision.upper()}, {num_samples} samples)")
    print(f"Dataset: {dataset.upper()}" + (f" ({language})" if dataset == "fleurs" else ""))
    print(f"Models:  {model_dir}")
    if normalize:
        print("WER: Punctuation-normalized")
    print("="*70)

    # Configuration
    PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
    EOS_TOKEN_ID = 3
    MAX_NEW_TOKENS = 200

    # Load models
    print(f"\n[1/4] Loading {precision.upper()} CoreML models...")
    encoder = ct.models.MLModel(f"{model_dir}/cohere_encoder.mlpackage")
    decoder = ct.models.MLModel(f"{model_dir}/cohere_decoder_stateful.mlpackage")
    print(f"   ✓ {precision.upper()} models loaded")

    # Load vocab
    print("\n[2/4] Loading vocabulary...")
    with open(f"{model_dir}/vocab.json") as f:
        vocab = {int(k): v for k, v in json.load(f).items()}
    print("   ✓ Vocabulary loaded")

    # Load dataset
    if dataset == "librispeech":
        print(f"\n[3/4] Loading {num_samples} samples from LibriSpeech test-clean...")
        ds = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    elif dataset == "fleurs":
        print(f"\n[3/4] Loading {num_samples} samples from FLEURS ({language})...")
        ds = load_dataset("google/fleurs", language, split="train", streaming=True, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    samples = []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        samples.append(sample)
    print(f"   ✓ Loaded {len(samples)} samples")

    # Process samples
    print(f"\n[4/4] Transcribing {num_samples} samples...")
    mel_processor = CohereMelSpectrogram()
    results = []
    start_time = time.time()

    for sample_idx, sample in enumerate(samples):
        sample_start = time.time()

        audio = sample['audio']['array'].astype(np.float32)
        # LibriSpeech uses 'text', FLEURS uses 'transcription'
        text_field = 'transcription' if dataset == 'fleurs' else 'text'
        ground_truth = sample[text_field].lower()
        duration = len(audio) / 16000.0

        # Compute mel spectrogram
        mel = mel_processor(audio)
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])))

        # Encode
        encoder_output = encoder.predict({
            "input_features": mel_padded.astype(np.float32),
            "feature_length": np.array([mel.shape[2]], dtype=np.int32)
        })
        encoder_hidden = encoder_output["hidden_states"]

        # Decode with stateful decoder
        state = decoder.make_state()
        tokens = []

        for step in range(MAX_NEW_TOKENS):
            current_token = PROMPT_IDS[step] if step < len(PROMPT_IDS) else tokens[-1]

            decoder_output = decoder.predict({
                "input_id": np.array([[current_token]], dtype=np.int32),
                "encoder_hidden_states": encoder_hidden.astype(np.float16),
                "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
                "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
                "position_ids": np.array([[step]], dtype=np.int32),
            }, state=state)

            next_token = int(np.argmax(decoder_output["logits"][0]))
            tokens.append(next_token)

            if next_token == EOS_TOKEN_ID:
                break

        # Decode tokens to text
        text_tokens = []
        for token_id in tokens:
            if token_id <= 4 or token_id == EOS_TOKEN_ID:
                continue
            token_str = vocab.get(token_id, "")
            if token_str.startswith("<|"):
                continue
            text_tokens.append(token_str)

        hypothesis = "".join(text_tokens).replace("▁", " ").strip()

        # Calculate WER
        if normalize:
            ground_truth_norm = normalize_text(ground_truth)
            hypothesis_norm = normalize_text(hypothesis)
            sample_wer = wer(ground_truth_norm, hypothesis_norm) * 100
        else:
            sample_wer = wer(ground_truth, hypothesis) * 100

        sample_time = time.time() - sample_start

        if (sample_idx + 1) % 10 == 0:
            print(f"   Processed {sample_idx + 1}/{num_samples} samples...")

        results.append({
            "duration": duration,
            "ground_truth": ground_truth,
            "hypothesis": hypothesis,
            "wer": sample_wer,
            "processing_time": sample_time,
        })

    total_time = time.time() - start_time

    # Calculate statistics
    print("\n" + "="*70)
    print(f"RESULTS ({num_samples} Samples, {precision.upper()}, {dataset.upper()}" +
          (f" {language}" if dataset == "fleurs" else "") + ")")
    if normalize:
        print("WER: Punctuation-normalized")
    else:
        print("WER: Raw with punctuation")
    print("="*70)

    avg_wer = np.mean([r["wer"] for r in results])
    median_wer = np.median([r["wer"] for r in results])
    perfect_matches = sum(1 for r in results if r["wer"] < 5.0)
    good_matches = sum(1 for r in results if r["wer"] < 20.0)
    perfect_pct = (perfect_matches / len(results)) * 100
    good_pct = (good_matches / len(results)) * 100

    print(f"\n📊 Quality Metrics:")
    print(f"   Average WER:         {avg_wer:.2f}%")
    print(f"   Median WER:          {median_wer:.2f}%")
    print(f"   Perfect (WER < 5%):  {perfect_matches}/{len(results)} ({perfect_pct:.1f}%)")
    print(f"   Good (WER < 20%):    {good_matches}/{len(results)} ({good_pct:.1f}%)")

    print(f"\n⚡ Performance Metrics:")
    avg_proc_time = np.mean([r["processing_time"] for r in results])
    avg_audio_duration = np.mean([r["duration"] for r in results])
    avg_rtfx = avg_proc_time / avg_audio_duration if avg_audio_duration > 0 else 0
    print(f"   Avg processing time: {avg_proc_time:.2f}s")
    print(f"   Avg audio duration:  {avg_audio_duration:.2f}s")
    print(f"   Avg RTFx:            {avg_rtfx:.2f}x")
    print(f"   Total time:          {total_time:.1f}s")

    print(f"\n📈 WER Distribution:")
    wer_ranges = [
        ("Perfect (0-5%)", 0, 5),
        ("Excellent (5-10%)", 5, 10),
        ("Good (10-20%)", 10, 20),
        ("Fair (20-50%)", 20, 50),
        ("Poor (50-100%)", 50, 100),
        ("Failed (>100%)", 100, float('inf')),
    ]

    for label, min_wer, max_wer in wer_ranges:
        count = sum(1 for r in results if min_wer <= r["wer"] < max_wer)
        pct = (count / len(results)) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:20s} {count:3d} ({pct:5.1f}%) {bar}")

    # Show worst samples
    if num_samples >= 5:
        print(f"\n❌ Worst 5 samples:")
        worst_samples = sorted(results, key=lambda x: x["wer"], reverse=True)[:5]
        for i, r in enumerate(worst_samples):
            print(f"\n   {i+1}. WER: {r['wer']:.2f}% ({r['duration']:.1f}s)")
            print(f"      GT:  {r['ground_truth'][:80]}...")
            print(f"      Hyp: {r['hypothesis'][:80]}...")

    # Save results to JSON
    if output_file is None:
        dataset_suffix = f"{dataset}_{language}" if dataset == "fleurs" else dataset
        output_file = f"benchmark_{precision}_{dataset_suffix}_{num_samples}_{'normalized' if normalize else 'raw'}.json"

    with open(output_file, "w") as f:
        json.dump({
            "precision": precision,
            "dataset": dataset,
            "language": language if dataset == "fleurs" else None,
            "num_samples": len(results),
            "normalized": normalize,
            "avg_wer": avg_wer,
            "median_wer": median_wer,
            "perfect_matches": perfect_matches,
            "perfect_pct": perfect_pct,
            "good_matches": good_matches,
            "good_pct": good_pct,
            "avg_rtfx": avg_rtfx,
            "total_time": total_time,
            "results": results,
        }, f, indent=2)
    print(f"\n💾 Saved detailed results to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Cohere Transcribe CoreML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--precision", "-p",
        choices=["fp16", "q8"],
        default="fp16",
        help="Model precision to test (default: fp16)"
    )

    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of samples to test (default: 10)"
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use punctuation-normalized WER (removes punctuation/capitalization)"
    )

    parser.add_argument(
        "--dataset", "-d",
        choices=["librispeech", "fleurs"],
        default="librispeech",
        help="Dataset to test on (default: librispeech)"
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en_us",
        help="Language code for FLEURS dataset (e.g., ja_jp, fr_fr, es_419). Default: en_us"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file (default: benchmark_<precision>_<dataset>_<samples>_<normalized|raw>.json)"
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing cohere_encoder.mlpackage, cohere_decoder_stateful.mlpackage, "
             "vocab.json (default: ./<precision>). Populate via "
             "`huggingface-cli download FluidInference/cohere-transcribe-03-2026-coreml`."
    )

    args = parser.parse_args()

    try:
        benchmark(
            precision=args.precision,
            num_samples=args.samples,
            normalize=args.normalize,
            output_file=args.output,
            dataset=args.dataset,
            language=args.language,
            models_dir=args.models_dir,
        )
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
