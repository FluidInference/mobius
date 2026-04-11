"""
Download pre-trained MB-MelGAN model from Google Drive.

Uses gdown to handle Google Drive download properly.
"""

import gdown
import tarfile
import os
from pathlib import Path


def download_mbmelgan():
    """Download MB-MelGAN from Google Drive"""

    print("=" * 80)
    print("Downloading MB-MelGAN Pre-trained Model")
    print("=" * 80)

    # Model info
    model_tag = "vctk_multi_band_melgan.v2"
    google_drive_id = "10PRQpHMFPE7RjF-MHYqvupK9S0xwBlJ_"  # From PRETRAINED_MODEL_LIST

    print(f"\nModel: {model_tag}")
    print(f"Sample rate: 24kHz (matches CosyVoice3!)")
    print(f"Type: Multi-Band MelGAN")
    print(f"Language: English (multi-speaker)")

    # Create output directory
    output_dir = Path("mbmelgan_pretrained")
    output_dir.mkdir(exist_ok=True)

    # Download paths
    download_url = f"https://drive.google.com/uc?id={google_drive_id}"
    output_tar = output_dir / f"{model_tag}.tar.gz"
    extract_dir = output_dir / model_tag

    # Check if already downloaded
    if extract_dir.exists() and list(extract_dir.glob("*.pkl")):
        print(f"\n✓ Model already downloaded: {extract_dir}")
        checkpoint = list(extract_dir.glob("checkpoint*.pkl"))[0]
        print(f"  Checkpoint: {checkpoint}")
        return True

    print(f"\nDownloading from Google Drive...")
    print(f"  URL: https://drive.google.com/file/d/{google_drive_id}")
    print(f"  Output: {output_tar}")
    print(f"(This may take a few minutes)")

    try:
        # Download tar.gz using gdown (handles Google Drive properly)
        print(f"\nDownloading...")
        gdown.download(download_url, str(output_tar), quiet=False)
        print(f"✓ Downloaded: {output_tar.stat().st_size / 1024 / 1024:.2f} MB")

        # Extract tar.gz
        print(f"\nExtracting...")
        extract_dir.mkdir(exist_ok=True)

        with tarfile.open(output_tar, "r:*") as tar:
            # Extract all members, flattening directory structure
            for member in tar.getmembers():
                if member.isreg():  # Regular file
                    member.name = os.path.basename(member.name)
                    tar.extract(member, extract_dir)
                    print(f"  ✓ {member.name}")

        # Clean up tar file
        output_tar.unlink()

        print(f"\n" + "=" * 80)
        print(f"✅ SUCCESS! Downloaded MB-MelGAN")
        print("=" * 80)

        # Find checkpoint
        checkpoints = list(extract_dir.glob("checkpoint*.pkl"))
        if checkpoints:
            print(f"\nCheckpoint: {checkpoints[0]}")
        else:
            print(f"\n⚠️  No checkpoint.pkl found")

        # List all files
        print(f"\nFiles in {extract_dir}:")
        for f in sorted(extract_dir.iterdir()):
            if f.is_file():
                size = f.stat().st_size / 1024 / 1024
                print(f"  - {f.name}: {size:.2f} MB")

        print(f"\n✅ Ready for CoreML conversion!")
        print(f"\nNext step: Load these weights into MB-MelGAN and test CoreML conversion")

        return True

    except Exception as e:
        print(f"\n❌ Download failed:")
        print(f"   Error: {e}")

        import traceback

        traceback.print_exc()

        print(f"\nTroubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Manual download link:")
        print(f"   https://drive.google.com/file/d/{google_drive_id}/view")
        print(f"   Save as: {output_tar}")
        print(f"3. Try alternative model: ljspeech_multi_band_melgan.v2 (22.05kHz)")

        return False


if __name__ == "__main__":
    import sys

    success = download_mbmelgan()
    sys.exit(0 if success else 1)
