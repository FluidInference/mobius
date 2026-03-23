#!/usr/bin/env python3
"""Extract speaker embedding (x-vector) from reference audio.

Uses the ECAPA-TDNN speaker encoder from the Qwen3-TTS model to extract a
1024-dimensional speaker embedding from a reference audio clip.

Usage:
    python extract_speaker_embedding.py reference.wav -o speaker.npy
    python extract_speaker_embedding.py reference.wav  # saves to /tmp/speaker_embedding.npy

The output .npy file can then be passed to inference.py via --speaker:
    python inference.py "Hello world" --speaker speaker.npy
"""

import argparse
import os
import sys
import struct
import numpy as np
import torch

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_0.6b")


def read_wav(path):
    """Read a WAV file and return (samples_float32, sample_rate)."""
    with open(path, "rb") as f:
        riff = f.read(4)
        assert riff == b"RIFF", "Not a WAV file"
        f.read(4)  # file size
        wave = f.read(4)
        assert wave == b"WAVE"

        sr = None
        samples = None
        bits_per_sample = 16

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                channels = struct.unpack("<H", fmt_data[2:4])[0]
                sr = struct.unpack("<I", fmt_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                raw = f.read(chunk_size)
                if bits_per_sample == 16:
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif bits_per_sample == 32:
                    samples = np.frombuffer(raw, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported bits_per_sample: {bits_per_sample}")
                if channels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
            else:
                f.read(chunk_size)

    assert sr is not None and samples is not None, "Failed to parse WAV"
    return samples, sr


def load_model(model_path):
    """Load the Qwen3-TTS model (needed for speaker encoder + mel spectrogram)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen_tts import Qwen3TTSModel

    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32)
    return model.model  # Qwen3TTSForConditionalGeneration


@torch.inference_mode()
def extract_embedding(model, audio, sr=24000):
    """Extract 1024-dim speaker embedding from audio.

    Uses the model's built-in extract_speaker_embedding() method to ensure
    exact mel spectrogram and encoder behavior matches the training pipeline.

    Args:
        model: Qwen3TTSForConditionalGeneration model
        audio: numpy float32 array of audio samples
        sr: Sample rate (must be 24000)

    Returns:
        numpy float32 array of shape (1024,)
    """
    embedding = model.extract_speaker_embedding(audio, sr)
    return embedding.cpu().numpy()


def resample_if_needed(audio, sr, target_sr=24000):
    """Simple linear interpolation resampling."""
    if sr == target_sr:
        return audio
    ratio = target_sr / sr
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(audio) - 1)
    left = indices.astype(int)
    right = np.minimum(left + 1, len(audio) - 1)
    frac = indices - left
    return (audio[left] * (1 - frac) + audio[right] * frac).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract speaker embedding from reference audio")
    parser.add_argument("audio", help="Path to reference audio WAV file")
    parser.add_argument("-o", "--output", default="/tmp/speaker_embedding.npy",
                        help="Output path for .npy embedding file")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="Path to Qwen3-TTS model directory")
    args = parser.parse_args()

    # Read audio
    print(f"Reading: {args.audio}")
    audio, sr = read_wav(args.audio)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s, sample rate: {sr}Hz, samples: {len(audio)}")

    # Resample to 24kHz if needed
    if sr != 24000:
        print(f"  Resampling {sr}Hz → 24000Hz")
        audio = resample_if_needed(audio, sr, 24000)
        sr = 24000

    # Load model
    model = load_model(args.model)

    # Extract embedding
    print("Extracting speaker embedding...")
    embedding = extract_embedding(model, audio, sr)

    # Print stats
    print(f"  Shape: {embedding.shape}")
    print(f"  Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
    print(f"  L2 norm: {np.linalg.norm(embedding):.4f}")

    # Save
    np.save(args.output, embedding)
    print(f"\nSaved: {args.output}")
    print(f"\nUsage with inference.py:")
    print(f'  python inference.py "Your text here" --speaker {args.output}')


if __name__ == "__main__":
    main()
