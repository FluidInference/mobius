#!/usr/bin/env python3
"""Run benchmark on all Cohere-supported languages from FLEURS dataset."""

import subprocess
import sys
from pathlib import Path

LANGUAGES = [
    ("en_us", "English"),
    ("ja_jp", "Japanese"),
    ("fr_fr", "French"),
    ("es_419", "Spanish"),
    ("de_de", "German"),
    ("cmn_hans_cn", "Chinese (Mandarin)"),
    ("ko_kr", "Korean"),
    ("it_it", "Italian"),
    ("pt_br", "Portuguese"),
    ("ru_ru", "Russian"),
    ("tr_tr", "Turkish"),
    ("nl_nl", "Dutch"),
    ("pl_pl", "Polish"),
    ("sv_se", "Swedish"),
]

def main():
    precision = sys.argv[1] if len(sys.argv) > 1 else "q8"
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print("="*70)
    print(f"Running benchmark on all {len(LANGUAGES)} languages")
    print(f"Precision: {precision.upper()}, Samples: {samples} per language")
    print("="*70)

    results_summary = []

    for i, (lang_code, lang_name) in enumerate(LANGUAGES, 1):
        print(f"\n[{i}/{len(LANGUAGES)}] Testing {lang_name} ({lang_code})...")
        print("-"*70)

        try:
            result = subprocess.run([
                "uv", "run", "python", "benchmark.py",
                "--precision", precision,
                "--samples", str(samples),
                "--dataset", "fleurs",
                "--language", lang_code,
                "--normalize"
            ], check=True, capture_output=True, text=True)

            # Extract WER from output
            for line in result.stdout.split('\n'):
                if "Average WER:" in line:
                    wer = line.split("Average WER:")[1].strip().split("%")[0].strip()
                    results_summary.append({
                        "language": lang_name,
                        "code": lang_code,
                        "wer": float(wer)
                    })
                    print(f"   ✓ {lang_name}: {wer}% WER")
                    break

        except subprocess.CalledProcessError as e:
            print(f"   ✗ Failed: {e}")
            results_summary.append({
                "language": lang_name,
                "code": lang_code,
                "wer": None,
                "error": str(e)
            })

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - All Languages")
    print("="*70)
    print(f"\nPrecision: {precision.upper()}, Samples: {samples} per language\n")

    successful = [r for r in results_summary if r.get("wer") is not None]
    failed = [r for r in results_summary if r.get("wer") is None]

    if successful:
        # Sort by WER
        successful.sort(key=lambda x: x["wer"])

        print("Results (sorted by WER):")
        print(f"{'Language':<25} {'Code':<15} {'WER':<10}")
        print("-"*50)
        for r in successful:
            print(f"{r['language']:<25} {r['code']:<15} {r['wer']:>6.2f}%")

        avg_wer = sum(r["wer"] for r in successful) / len(successful)
        print(f"\n{'Average across all languages':<40} {avg_wer:>6.2f}%")

    if failed:
        print(f"\n\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['language']} ({r['code']})")

    print("\n" + "="*70)
    print(f"Individual results saved to: benchmark_{precision}_fleurs_*_normalized.json")
    print("="*70)

if __name__ == "__main__":
    main()
