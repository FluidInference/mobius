#!/usr/bin/env python3
"""Benchmark CJK languages using Character Error Rate (CER) instead of WER.

CJK languages (Chinese, Japanese, Korean) don't have clear word boundaries,
so CER is the appropriate metric instead of WER.

Examples:
    # Test FP16 models on 100 samples for all CJK languages
    python benchmark_cjk_cer.py --precision fp16 --samples 100

    # Test Q8 models on 100 samples
    python benchmark_cjk_cer.py --precision q8 --samples 100

    # Test specific CJK language
    python benchmark_cjk_cer.py --precision q8 --samples 100 --language ja_jp
"""

import sys
from pathlib import Path
import argparse

# Add model directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "f16"))

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram
from datasets import load_dataset
from jiwer import cer
from jiwer.transforms import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
import json
import time


# Create text normalization pipeline using jiwer
normalize_text = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])


CJK_LANGUAGES = [
    ("ja_jp", "Japanese"),
    ("cmn_hans_cn", "Chinese (Mandarin)"),
    ("ko_kr", "Korean"),
]


def benchmark_single(precision="fp16", num_samples=10, normalize=False, output_file=None, language="ja_jp"):
    """Run CER benchmark on a single CJK language."""

    model_dir = precision

    print("="*70)
    print(f"Cohere Transcribe CER Benchmark ({precision.upper()}, {num_samples} samples)")
    print(f"Language: {language}")
    if normalize:
        print("CER: Punctuation-normalized")
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
    with open("f16/vocab.json") as f:
        vocab = {int(k): v for k, v in json.load(f).items()}
    print("   ✓ Vocabulary loaded")

    # Load dataset
    print(f"\n[3/4] Loading {num_samples} samples from FLEURS ({language})...")
    ds = load_dataset("google/fleurs", language, split="train", streaming=True, trust_remote_code=True)

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
        # FLEURS uses 'transcription' field
        ground_truth = sample['transcription'].lower()
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

        # Calculate CER (not WER!)
        if normalize:
            ground_truth_norm = normalize_text(ground_truth)
            hypothesis_norm = normalize_text(hypothesis)
            sample_cer = cer(ground_truth_norm, hypothesis_norm) * 100
        else:
            sample_cer = cer(ground_truth, hypothesis) * 100

        sample_time = time.time() - sample_start

        if (sample_idx + 1) % 10 == 0:
            print(f"   Processed {sample_idx + 1}/{num_samples} samples...")

        results.append({
            "duration": duration,
            "ground_truth": ground_truth,
            "hypothesis": hypothesis,
            "cer": sample_cer,
            "processing_time": sample_time,
        })

    total_time = time.time() - start_time

    # Calculate statistics
    print("\n" + "="*70)
    print(f"RESULTS ({num_samples} Samples, {precision.upper()}, {language})")
    if normalize:
        print("CER: Punctuation-normalized")
    else:
        print("CER: Raw with punctuation")
    print("="*70)

    avg_cer = np.mean([r["cer"] for r in results])
    median_cer = np.median([r["cer"] for r in results])
    perfect_matches = sum(1 for r in results if r["cer"] < 5.0)
    good_matches = sum(1 for r in results if r["cer"] < 20.0)
    perfect_pct = (perfect_matches / len(results)) * 100
    good_pct = (good_matches / len(results)) * 100

    print(f"\n📊 Quality Metrics:")
    print(f"   Average CER:         {avg_cer:.2f}%")
    print(f"   Median CER:          {median_cer:.2f}%")
    print(f"   Perfect (CER < 5%):  {perfect_matches}/{len(results)} ({perfect_pct:.1f}%)")
    print(f"   Good (CER < 20%):    {good_matches}/{len(results)} ({good_pct:.1f}%)")

    print(f"\n⚡ Performance Metrics:")
    avg_proc_time = np.mean([r["processing_time"] for r in results])
    avg_audio_duration = np.mean([r["duration"] for r in results])
    avg_rtfx = avg_proc_time / avg_audio_duration if avg_audio_duration > 0 else 0
    print(f"   Avg processing time: {avg_proc_time:.2f}s")
    print(f"   Avg audio duration:  {avg_audio_duration:.2f}s")
    print(f"   Avg RTFx:            {avg_rtfx:.2f}x")
    print(f"   Total time:          {total_time:.1f}s")

    print(f"\n📈 CER Distribution:")
    cer_ranges = [
        ("Perfect (0-5%)", 0, 5),
        ("Excellent (5-10%)", 5, 10),
        ("Good (10-20%)", 10, 20),
        ("Fair (20-50%)", 20, 50),
        ("Poor (50-100%)", 50, 100),
        ("Failed (>100%)", 100, float('inf')),
    ]

    for label, min_cer, max_cer in cer_ranges:
        count = sum(1 for r in results if min_cer <= r["cer"] < max_cer)
        pct = (count / len(results)) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:20s} {count:3d} ({pct:5.1f}%) {bar}")

    # Show worst samples
    if num_samples >= 5:
        print(f"\n❌ Worst 5 samples:")
        worst_samples = sorted(results, key=lambda x: x["cer"], reverse=True)[:5]
        for i, r in enumerate(worst_samples):
            print(f"\n   {i+1}. CER: {r['cer']:.2f}% ({r['duration']:.1f}s)")
            print(f"      GT:  {r['ground_truth'][:80]}...")
            print(f"      Hyp: {r['hypothesis'][:80]}...")

    # Save results to JSON
    if output_file is None:
        output_file = f"benchmark_{precision}_fleurs_{language}_{num_samples}_{'normalized' if normalize else 'raw'}_cer.json"

    with open(output_file, "w") as f:
        json.dump({
            "precision": precision,
            "language": language,
            "num_samples": len(results),
            "normalized": normalize,
            "metric": "cer",
            "avg_cer": avg_cer,
            "median_cer": median_cer,
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


def benchmark_all_cjk(precision="fp16", num_samples=100, normalize=False):
    """Run CER benchmark on all CJK languages."""

    print("="*70)
    print(f"Running CER benchmark on all {len(CJK_LANGUAGES)} CJK languages")
    print(f"Precision: {precision.upper()}, Samples: {num_samples} per language")
    print("="*70)

    results_summary = []

    for i, (lang_code, lang_name) in enumerate(CJK_LANGUAGES, 1):
        print(f"\n[{i}/{len(CJK_LANGUAGES)}] Testing {lang_name} ({lang_code})...")
        print("-"*70)

        try:
            # Run benchmark for this language
            import subprocess
            result = subprocess.run([
                "uv", "run", "python", "benchmark_cjk_cer.py",
                "--precision", precision,
                "--samples", str(num_samples),
                "--language", lang_code,
                "--normalize" if normalize else "--no-normalize"
            ], check=True, capture_output=True, text=True)

            # Extract CER from output
            for line in result.stdout.split('\n'):
                if "Average CER:" in line:
                    metric_value = line.split("Average CER:")[1].strip().split("%")[0].strip()
                    results_summary.append({
                        "language": lang_name,
                        "code": lang_code,
                        "cer": float(metric_value)
                    })
                    print(f"   ✓ {lang_name}: {metric_value}% CER")
                    break

        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed: {e}")
            results_summary.append({
                "language": lang_name,
                "code": lang_code,
                "cer": None,
                "error": str(e)
            })

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - All CJK Languages (CER)")
    print("="*70)
    print(f"\nPrecision: {precision.upper()}, Samples: {num_samples} per language\n")

    successful = [r for r in results_summary if r.get("cer") is not None]
    failed = [r for r in results_summary if r.get("cer") is None]

    if successful:
        # Sort by CER
        successful.sort(key=lambda x: x["cer"])

        print("Results (sorted by CER):")
        print(f"{'Language':<25} {'Code':<15} {'CER':<10}")
        print("-"*50)
        for r in successful:
            print(f"{r['language']:<25} {r['code']:<15} {r['cer']:>6.2f}%")

        avg_cer = sum(r["cer"] for r in successful) / len(successful)
        print(f"\n{'Average across all CJK languages':<40} {avg_cer:>6.2f}%")

    if failed:
        print(f"\n\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['language']} ({r['code']})")

    print("\n" + "="*70)
    print(f"Individual results saved to: benchmark_{precision}_fleurs_*_normalized_cer.json")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CJK languages using CER metric",
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
        help="Use punctuation-normalized CER (removes punctuation/capitalization)"
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize (for use in subprocess calls)"
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        choices=["ja_jp", "cmn_hans_cn", "ko_kr", "all"],
        default="all",
        help="Language to test (default: all CJK languages)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file (default: benchmark_<precision>_fleurs_<lang>_<samples>_<normalized|raw>_cer.json)"
    )

    args = parser.parse_args()

    try:
        if args.language == "all":
            benchmark_all_cjk(
                precision=args.precision,
                num_samples=args.samples,
                normalize=args.normalize
            )
        else:
            benchmark_single(
                precision=args.precision,
                num_samples=args.samples,
                normalize=args.normalize,
                output_file=args.output,
                language=args.language
            )
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
