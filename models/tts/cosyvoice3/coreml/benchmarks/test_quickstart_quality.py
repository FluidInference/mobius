"""
Test the quality of the quickstart MB-MelGAN model.

This script:
1. Loads a real mel spectrogram from CosyVoice generation
2. Runs it through the CoreML MB-MelGAN model
3. Saves the output and compares to original
"""

import torch
import numpy as np
import coremltools as ct
import soundfile as sf
from pathlib import Path

print("=" * 80)
print("Testing MB-MelGAN Quickstart Model Quality")
print("=" * 80)

# Load CoreML model
print("\n1. Loading CoreML model...")
model_path = Path("mbmelgan_quickstart/mbmelgan_quickstart_coreml.mlpackage")
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("Run quick_finetune.py first")
    exit(1)

mlmodel = ct.models.MLModel(str(model_path))
print(f"   ✓ Loaded: {model_path}")

# Load a real mel spectrogram from training data
print("\n2. Loading real mel spectrogram...")
mel_files = list(Path("mbmelgan_training_data/mels").glob("*.pt"))
if not mel_files:
    print("   ❌ No mel spectrograms found in mbmelgan_training_data/mels/")
    print("   Generation may still be in progress...")
    exit(1)

mel_path = mel_files[0]
mel = torch.load(mel_path)
print(f"   ✓ Loaded: {mel_path.name}")
print(f"   Shape: {mel.shape}")

# Also load the corresponding original audio for comparison
audio_path = Path("mbmelgan_training_data/audio") / f"{mel_path.stem}.wav"
if audio_path.exists():
    orig_audio, orig_sr = sf.read(str(audio_path))
    print(f"   ✓ Original audio: {audio_path.name} ({len(orig_audio)} samples, {orig_sr} Hz)")
else:
    print(f"   ⚠️  Original audio not found: {audio_path.name}")
    orig_audio = None

# Prepare mel for CoreML inference
print("\n3. Running CoreML inference...")
mel_np = mel.numpy()
print(f"   Input shape: {mel_np.shape}")

# Model expects fixed size (1, 80, 125) - crop or pad to match
expected_frames = 125
actual_frames = mel_np.shape[2]

if actual_frames > expected_frames:
    print(f"   Cropping from {actual_frames} to {expected_frames} frames")
    mel_np = mel_np[:, :, :expected_frames]
elif actual_frames < expected_frames:
    print(f"   Padding from {actual_frames} to {expected_frames} frames")
    padding = np.zeros((mel_np.shape[0], mel_np.shape[1], expected_frames - actual_frames))
    mel_np = np.concatenate([mel_np, padding], axis=2)

print(f"   Adjusted shape: {mel_np.shape}")

try:
    # Run inference
    output = mlmodel.predict({"mel_spectrogram": mel_np})
    audio_bands = output["audio_bands"]

    print(f"   ✓ Inference complete")
    print(f"   Output shape: {audio_bands.shape}")

    # MB-MelGAN outputs 4 sub-bands, need to combine them
    # For now, just take the mean across bands
    if len(audio_bands.shape) == 3:
        # [1, 4, samples] -> [samples]
        audio_out = audio_bands[0].mean(axis=0)
    else:
        audio_out = audio_bands.squeeze()

    print(f"   Combined audio shape: {audio_out.shape}")

except Exception as e:
    print(f"   ❌ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Save output
print("\n4. Saving output...")
output_dir = Path("mbmelgan_quality_test")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "quickstart_output.wav"
sf.write(str(output_path), audio_out, 22050)
print(f"   ✓ Saved: {output_path}")

# Save original for comparison
if orig_audio is not None:
    orig_output_path = output_dir / "original_cosyvoice.wav"
    sf.write(str(orig_output_path), orig_audio, orig_sr)
    print(f"   ✓ Saved original: {orig_output_path}")

# Statistics
print("\n" + "=" * 80)
print("Quality Assessment")
print("=" * 80)

print(f"\nQuickstart Model Output:")
print(f"  - Duration: {len(audio_out) / 22050:.2f}s")
print(f"  - Sample rate: 22050 Hz")
print(f"  - Min/Max: {audio_out.min():.4f} / {audio_out.max():.4f}")
print(f"  - Mean: {audio_out.mean():.4f}")
print(f"  - Std: {audio_out.std():.4f}")

if orig_audio is not None:
    print(f"\nOriginal CosyVoice Audio:")
    print(f"  - Duration: {len(orig_audio) / orig_sr:.2f}s")
    print(f"  - Sample rate: {orig_sr} Hz")
    print(f"  - Min/Max: {orig_audio.min():.4f} / {orig_audio.max():.4f}")
    print(f"  - Mean: {orig_audio.mean():.4f}")
    print(f"  - Std: {orig_audio.std():.4f}")

    # Length comparison
    duration_diff = abs(len(audio_out) / 22050 - len(orig_audio) / orig_sr)
    print(f"\nDuration difference: {duration_diff:.2f}s")

print("\n" + "=" * 80)
print("✅ Quality test complete!")
print("=" * 80)

print(f"\nListen to the outputs:")
print(f"  - Quickstart model: {output_path}")
if orig_audio is not None:
    print(f"  - Original CosyVoice: {orig_output_path}")

print(f"\n📝 Note:")
print(f"  The quickstart model was trained on synthetic data (10 epochs, 100 samples)")
print(f"  Quality should improve significantly with real CosyVoice data")
print(f"  Current training data generation: 10/1000 samples (1%)")
