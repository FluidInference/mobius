#!/usr/bin/env python3
"""Export voice conditioning for PocketTTS using CoreML encoder.

Pure CoreML voice cloning - no PyTorch required for inference.

Converts an audio file to the binary format expected by FluidAudio's Swift PocketTTS.

Usage:
    python export_voice_coreml.py input.wav --output custom_audio_prompt.bin
    python export_voice_coreml.py voices_dir/ --output-dir ./constants_bin/

Requirements:
    pip install coremltools numpy scipy

First run convert_mimi_encoder.py to create the CoreML encoder model.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy import signal

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.absolute()

# PocketTTS constants
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms frames
VOICE_PROMPT_LENGTH = 125  # Standard voice prompt length
EMBEDDING_DIM = 1024


def load_audio(path: Path) -> np.ndarray:
    """Load and preprocess audio for the encoder."""
    # Try scipy first (no extra deps)
    try:
        sr, audio = wavfile.read(str(path))
    except Exception:
        # Try librosa as fallback
        try:
            import librosa
            audio, sr = librosa.load(str(path), sr=None, mono=True)
        except ImportError:
            raise RuntimeError(f"Cannot load {path}. Install scipy or librosa.")

    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 24kHz if needed
    if sr != SAMPLE_RATE:
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = signal.resample(audio, num_samples)
        logger.info(f"  Resampled from {sr}Hz to {SAMPLE_RATE}Hz")

    return audio.astype(np.float32)


def pad_audio(audio: np.ndarray) -> np.ndarray:
    """Pad audio to multiple of frame size."""
    length = len(audio)
    pad_length = (FRAME_SIZE - (length % FRAME_SIZE)) % FRAME_SIZE
    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length))
    return audio


def find_encoder_model() -> Path:
    """Find the CoreML encoder model."""
    candidates = [
        SCRIPT_DIR / "mimi_encoder.mlpackage",
        SCRIPT_DIR / "mimi_encoder.mlmodelc",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "CoreML encoder not found. Run:\n"
        "  python coreml/convert_models/convert/convert_mimi_encoder.py"
    )


def encode_audio_coreml(audio: np.ndarray, encoder_path: Path) -> np.ndarray:
    """Encode audio to voice conditioning using CoreML."""
    import coremltools as ct

    # Load encoder
    logger.info(f"Loading CoreML encoder: {encoder_path}")
    encoder = ct.models.MLModel(str(encoder_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)

    # Prepare input: [1, 1, T]
    audio = pad_audio(audio)
    audio_input = audio.reshape(1, 1, -1).astype(np.float32)

    logger.info(f"  Input shape: {audio_input.shape}")
    logger.info(f"  Duration: {len(audio) / SAMPLE_RATE:.2f}s")

    # Run encoder
    result = encoder.predict({"audio": audio_input})
    conditioning = result["conditioning"]

    logger.info(f"  Output shape: {conditioning.shape}")

    # conditioning: [1, num_frames, 1024] -> [num_frames, 1024]
    return conditioning.squeeze(0)


def pad_or_truncate(conditioning: np.ndarray, target_length: int = VOICE_PROMPT_LENGTH) -> np.ndarray:
    """Pad or truncate conditioning to standard length."""
    current_length = conditioning.shape[0]

    if current_length == target_length:
        return conditioning
    elif current_length > target_length:
        logger.info(f"  Truncating from {current_length} to {target_length} frames")
        return conditioning[:target_length]
    else:
        logger.info(f"  Padding from {current_length} to {target_length} frames")
        padding = np.zeros((target_length - current_length, EMBEDDING_DIM), dtype=np.float32)
        return np.concatenate([conditioning, padding], axis=0)


def save_conditioning_bin(conditioning: np.ndarray, output_path: Path):
    """Save conditioning as raw Float32 binary file."""
    conditioning = conditioning.astype(np.float32)
    flattened = conditioning.flatten()

    with open(output_path, 'wb') as f:
        f.write(flattened.tobytes())

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")


def export_voice(
    audio_path: Path,
    output_path: Path,
    encoder_path: Path,
    voice_name: str | None = None,
    target_length: int = VOICE_PROMPT_LENGTH,
):
    """Export a single voice to CoreML-compatible format."""
    # Determine output file path
    if output_path.is_dir():
        if voice_name is None:
            voice_name = audio_path.stem
        output_file = output_path / f"{voice_name}_audio_prompt.bin"
    else:
        output_file = output_path

    logger.info(f"Processing: {audio_path}")

    # Load and encode audio
    audio = load_audio(audio_path)
    conditioning = encode_audio_coreml(audio, encoder_path)

    # Pad/truncate to standard length
    conditioning = pad_or_truncate(conditioning, target_length)

    # Save
    save_conditioning_bin(conditioning, output_file)

    return output_file


def export_directory(
    input_dir: Path,
    output_dir: Path,
    encoder_path: Path,
    target_length: int = VOICE_PROMPT_LENGTH,
):
    """Export all audio files in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a'}
    audio_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in audio_extensions]

    if not audio_files:
        logger.error(f"No audio files found in {input_dir}")
        return []

    logger.info(f"Found {len(audio_files)} audio files")

    exported = []
    for audio_path in sorted(audio_files):
        try:
            output_file = export_voice(
                audio_path, output_dir, encoder_path,
                target_length=target_length
            )
            exported.append(output_file)
        except Exception as e:
            logger.error(f"Failed to export {audio_path}: {e}")

    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Export voice conditioning using CoreML encoder (no PyTorch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Requirements:
  1. Convert encoder first:
     python coreml/convert_models/convert/convert_mimi_encoder.py

  2. Export voices:
     python coreml/export_voice_coreml.py voice.wav -o custom_audio_prompt.bin

Examples:
  # Export single voice
  python export_voice_coreml.py voice.wav -o custom_audio_prompt.bin

  # Export to directory
  python export_voice_coreml.py voice.wav --output-dir ./constants_bin/

  # Batch export
  python export_voice_coreml.py ./voices/ --output-dir ./constants_bin/
"""
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .bin file path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Voice name for filename"
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="Path to mimi_encoder.mlpackage"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=VOICE_PROMPT_LENGTH,
        help=f"Target conditioning length (default: {VOICE_PROMPT_LENGTH})"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    # Find encoder
    if args.encoder:
        encoder_path = args.encoder
    else:
        try:
            encoder_path = find_encoder_model()
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    # Determine output
    if args.output_dir:
        output_path = args.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
    elif args.output:
        output_path = args.output
    else:
        output_path = Path(".")

    # Export
    if args.input.is_dir():
        exported = export_directory(
            args.input, output_path, encoder_path,
            target_length=args.frames
        )
        logger.info(f"\n✅ Exported {len(exported)} voices")
    else:
        output_file = export_voice(
            args.input, output_path, encoder_path,
            voice_name=args.name,
            target_length=args.frames
        )
        logger.info(f"\n✅ Exported voice to: {output_file}")


if __name__ == "__main__":
    main()
