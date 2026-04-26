#!/usr/bin/env python3
"""Download FLEURS datasets for Swift benchmark."""

import argparse
from pathlib import Path
import soundfile as sf
from datasets import load_dataset


def download_fleurs_for_swift(language: str, num_samples: int, base_dir: Path):
    """Download FLEURS and organize for Swift CLI."""
    print(f"\n{'='*70}")
    print(f"Downloading FLEURS: {language} ({num_samples} samples)")
    print(f"{'='*70}")

    # Create output directory
    lang_dir = base_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Load FLEURS dataset
    print(f"Loading dataset from HuggingFace...")
    dataset = load_dataset("google/fleurs", language, split="test", streaming=False)

    # Create transcript file
    transcript_file = lang_dir / f"{language}.trans.txt"
    with open(transcript_file, "w") as f:
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break

            # Save audio
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            text = example["transcription"]

            file_id = f"sample_{i:04d}"
            audio_file = lang_dir / f"{file_id}.wav"
            sf.write(audio_file, audio, sr)

            # Write transcript line
            f.write(f"{file_id} {text}\n")

            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i + 1}/{num_samples}")

    print(f"✓ Downloaded to {lang_dir}")
    print(f"  Audio files: {num_samples}")
    print(f"  Transcripts: {transcript_file.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="en_us,fr_fr,es_419,cmn_hans_cn")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default=Path.home() / "Library/Application Support/FluidAudio/Datasets/fleurs"
    )
    args = parser.parse_args()

    languages = [lang.strip() for lang in args.languages.split(",")]
    base_dir = Path(args.output_dir)

    print("FLEURS Download for Swift Benchmark")
    print(f"Languages: {', '.join(languages)}")
    print(f"Samples per language: {args.num_samples}")
    print(f"Output directory: {base_dir}")

    for lang in languages:
        download_fleurs_for_swift(lang, args.num_samples, base_dir)

    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"Total languages: {len(languages)}")
    print(f"Samples per language: {args.num_samples}")
    print(f"\nReady for Swift benchmark:")
    print(f"  .build/release/fluidaudiocli cohere-benchmark \\")
    print(f"    --dataset fleurs \\")
    print(f"    --languages {','.join(languages)} \\")
    print(f"    --max-files {args.num_samples} \\")
    print(f"    --output fleurs_swift_results.json")


if __name__ == "__main__":
    main()
