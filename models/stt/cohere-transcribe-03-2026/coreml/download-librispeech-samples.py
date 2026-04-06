#!/usr/bin/env python3
"""Download LibriSpeech test-clean samples for benchmarking."""

import tarfile
import urllib.request
from pathlib import Path

# Small subset of LibriSpeech test-clean with ground truth
SAMPLES = [
    {
        "speaker": "1089",
        "chapter": "134686",
        "utterance": "0000",
        "text": "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce"
    },
    {
        "speaker": "1089",
        "chapter": "134686",
        "utterance": "0001",
        "text": "stuff it into you his belly counselled him"
    },
    {
        "speaker": "1089",
        "chapter": "134686",
        "utterance": "0002",
        "text": "after early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels"
    },
    {
        "speaker": "1089",
        "chapter": "134686",
        "utterance": "0003",
        "text": "he moaned to himself like some baffled prowling beast"
    },
    {
        "speaker": "1089",
        "chapter": "134686",
        "utterance": "0004",
        "text": "he wanted to feel again his own sin and to see again the vision of his sin"
    },
]

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def download_librispeech_samples(output_dir: Path, num_samples: int = 5):
    """Download LibriSpeech test-clean samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Downloading LibriSpeech test-clean Samples")
    print("="*70)

    # Download full tar.gz (it's ~350MB, but we extract only what we need)
    tar_path = output_dir / "test-clean.tar.gz"

    if not tar_path.exists():
        print(f"\nDownloading {LIBRISPEECH_URL}...")
        print(f"Size: ~350 MB (this may take a few minutes)")

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024**2)
            mb_total = total_size / (1024**2)
            print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end="")

        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path, reporthook=progress)
        print("\n  ✓ Downloaded")
    else:
        print(f"\n  ✓ Already downloaded: {tar_path}")

    # Extract only the specific files we need
    print(f"\nExtracting {num_samples} sample(s)...")

    extracted_files = []

    with tarfile.open(tar_path, 'r:gz') as tar:
        for sample in SAMPLES[:num_samples]:
            # File path in archive
            file_name = f"{sample['speaker']}-{sample['chapter']}-{sample['utterance']}.flac"
            archive_path = f"LibriSpeech/test-clean/{sample['speaker']}/{sample['chapter']}/{file_name}"

            output_file = output_dir / file_name

            if output_file.exists():
                print(f"  ✓ Already extracted: {file_name}")
            else:
                # Extract this specific file
                try:
                    member = tar.getmember(archive_path)
                    member.name = file_name  # Flatten the path
                    tar.extract(member, output_dir)

                    # Move to flattened location
                    extracted = output_dir / archive_path
                    if extracted.exists():
                        extracted.rename(output_file)
                        # Clean up directory structure
                        (output_dir / "LibriSpeech").rmdir() if (output_dir / "LibriSpeech/test-clean").exists() else None

                    print(f"  ✓ Extracted: {file_name}")
                except KeyError:
                    print(f"  ⚠️  Not found in archive: {file_name}")
                    continue

            extracted_files.append({
                "path": output_file,
                "text": sample["text"],
                "speaker": sample["speaker"],
                "chapter": sample["chapter"],
                "utterance": sample["utterance"],
            })

    # Clean up extracted LibriSpeech directory structure if it exists
    librispeech_dir = output_dir / "LibriSpeech"
    if librispeech_dir.exists():
        import shutil
        shutil.rmtree(librispeech_dir)

    print(f"\n{'='*70}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"  Extracted {len(extracted_files)} samples to {output_dir}")

    # Save ground truth to text file
    ground_truth_file = output_dir / "ground_truth.txt"
    with open(ground_truth_file, 'w') as f:
        for sample in extracted_files:
            f.write(f"{sample['path'].name}\t{sample['text']}\n")

    print(f"  Ground truth saved to {ground_truth_file}")

    return extracted_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download LibriSpeech test-clean samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-audio"),
        help="Output directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to download (max 5)"
    )

    args = parser.parse_args()

    samples = download_librispeech_samples(args.output_dir, args.num_samples)

    print(f"\nSamples:")
    for sample in samples:
        print(f"  {sample['path'].name}")
        print(f"    Text: {sample['text']}")


if __name__ == "__main__":
    main()
