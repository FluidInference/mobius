#!/usr/bin/env python3
"""Export Cohere encoder traced with multilingual audio samples.

Strategy: Instead of tracing with random noise, trace with actual FLEURS samples
from multiple languages. This might help the encoder preserve language-specific
acoustic features better.
"""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq
import librosa
import soundfile as sf
from datasets import load_dataset


class EncoderWrapper(nn.Module):
    """Wrapper that combines encoder + projection layer."""

    def __init__(self, encoder, encoder_decoder_proj):
        super().__init__()
        self.encoder = encoder
        self.encoder_decoder_proj = encoder_decoder_proj

    def forward(self, input_features, feature_length):
        """
        Args:
            input_features: (batch, n_mels, n_frames) mel spectrogram
            feature_length: (batch,) int32 - actual length before padding

        Returns:
            hidden_states: (batch, encoded_frames, decoder_hidden_size)
        """
        encoder_outputs = self.encoder(
            input_features=input_features,
            length=feature_length,
            return_dict=True
        )

        hidden_states = encoder_outputs.last_hidden_state

        if self.encoder_decoder_proj is not None:
            hidden_states = self.encoder_decoder_proj(hidden_states)

        return hidden_states


def compute_mel_spectrogram(audio, sr=16000):
    """Compute Cohere mel spectrogram."""
    SAMPLE_RATE = 16000
    N_MELS = 128
    HOP_LENGTH = 160
    N_FFT = 400

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


def load_multilingual_sample():
    """Load one sample from each of 4 languages and average their mel specs.

    This creates a "neutral" multilingual reference for tracing.
    """
    print("\n[MULTILINGUAL] Loading FLEURS samples...")

    languages = ["en_us", "fr_fr", "es_419", "cmn_hans_cn"]
    mel_specs = []

    for lang in languages:
        print(f"  Loading {lang}...")

        # Check if we already have samples
        sample_path = Path(f"fleurs_samples/{lang}/sample_0000.wav")
        if sample_path.exists():
            # Use existing sample
            audio, sr = sf.read(sample_path)
            print(f"    ✓ Using existing sample ({len(audio)/sr:.2f}s)")
        else:
            # Download from HuggingFace
            print(f"    Downloading from HuggingFace...")
            dataset = load_dataset("google/fleurs", lang, split="test", streaming=True)
            example = next(iter(dataset))
            audio = example["audio"]["array"]
            sr = example["audio"]["sampling_rate"]
            print(f"    ✓ Downloaded ({len(audio)/sr:.2f}s)")

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(audio, sr)

        # Pad/trim to 3500 frames (35 seconds)
        if mel.shape[1] < 3500:
            mel = np.pad(mel, ((0, 0), (0, 3500 - mel.shape[1])), mode='constant')
        else:
            mel = mel[:, :3500]

        mel_specs.append(mel)

    # Average all mel spectrograms
    avg_mel = np.mean(mel_specs, axis=0)

    print(f"\n  ✓ Created averaged multilingual mel spectrogram")
    print(f"    Shape: {avg_mel.shape}")
    print(f"    Languages: {', '.join(languages)}")

    return avg_mel


def export_encoder_multilingual(output_dir: Path, precision: str = "float16", use_random: bool = False):
    """Export the Cohere encoder traced with multilingual data."""
    print("="*70)
    print("Cohere Encoder Export - Multilingual Tracing")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\n[1/6] Loading model from HuggingFace...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print("   ✓ Model loaded")

    # Wrap encoder
    print("\n[2/6] Wrapping encoder...")
    wrapped_encoder = EncoderWrapper(model.encoder, model.encoder_decoder_proj)
    wrapped_encoder.eval()
    print("   ✓ Encoder wrapped")

    # Create example inputs
    print("\n[3/6] Creating example inputs...")
    batch_size = 1
    n_mels = 128
    max_frames = 3500

    if use_random:
        print("   Using random noise (baseline)")
        example_input_features = torch.randn(batch_size, n_mels, max_frames)
    else:
        print("   Using multilingual averaged mel spectrogram")
        avg_mel = load_multilingual_sample()
        example_input_features = torch.from_numpy(avg_mel[np.newaxis, :, :]).float()

    example_feature_length = torch.tensor([max_frames], dtype=torch.int32)

    print(f"   Input features: {example_input_features.shape}")
    print(f"   Feature length: {example_feature_length.shape}")
    print(f"   Value range: [{example_input_features.min():.3f}, {example_input_features.max():.3f}]")

    # Test forward pass first
    print("\n[4/6] Testing forward pass...")
    with torch.no_grad():
        test_output = wrapped_encoder(example_input_features, example_feature_length)
        print(f"   Output shape: {test_output.shape}")
        print(f"   Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")

    # Trace the model
    print("\n[5/6] Tracing encoder...")
    with torch.no_grad():
        traced_encoder = torch.jit.trace(
            wrapped_encoder,
            (example_input_features, example_feature_length),
            check_trace=False,
        )

    # Convert to CoreML
    print(f"\n[6/6] Converting to CoreML ({precision})...")

    inputs = [
        ct.TensorType(name="input_features", shape=example_input_features.shape, dtype=np.float32),
        ct.TensorType(name="feature_length", shape=example_feature_length.shape, dtype=np.int32),
    ]

    compute_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced_encoder,
        inputs=inputs,
        outputs=[ct.TensorType(name="hidden_states")],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=compute_precision,
    )

    # Save
    suffix = "_multilingual" if not use_random else "_random"
    output_path = output_dir / f"cohere_encoder{suffix}.mlpackage"
    mlmodel.save(str(output_path))

    print(f"   ✓ Saved to: {output_path}")

    import subprocess
    try:
        size_mb = subprocess.check_output(["du", "-sh", str(output_path)]).decode().split()[0]
        print(f"   Size: {size_mb}")
    except:
        pass

    print("\n" + "="*70)
    print("ENCODER EXPORT COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print(f"\nTracing method: {'Multilingual averaged mel' if not use_random else 'Random noise'}")
    print(f"\nModel inputs:")
    print(f"  - input_features: (1, 128, 3500) float32 - mel spectrogram")
    print(f"  - feature_length: (1,) int32 - actual length")
    print(f"\nModel output:")
    print(f"  - hidden_states: (1, 438, 1024) {precision} - encoder output")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("build-multilingual"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--random", action="store_true", help="Use random noise (baseline)")
    args = parser.parse_args()

    try:
        export_encoder_multilingual(args.output_dir, args.precision, args.random)
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
