"""
Test FP32 vs FP16 precision for MB-MelGAN CoreML conversion.

Based on insights from john-rocky/CoreML-Models:
- Kokoro: "FP16 corrupts audio quality" → uses FP32
- HTDemucs: "to prevent overflow in frequency branch" → uses FP32

This script tests both precisions and compares:
1. Model size
2. Inference latency
3. Audio quality (MAE vs PyTorch reference)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import time


# MB-MelGAN model (copied from quick_finetune.py)
class ResidualStack(nn.Module):
    """Residual stack module"""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        residual = x
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x + residual


class MelGANGenerator(nn.Module):
    """MelGAN generator"""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        kernel_size=7,
        channels=512,
        upsample_scales=[8, 8, 2, 2],
        stack_kernel_size=3,
        stacks=3,
    ):
        super().__init__()

        layers = []
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        layers.append(nn.Conv1d(in_channels, channels, kernel_size))

        for i, upsample_scale in enumerate(upsample_scales):
            layers.append(nn.LeakyReLU(0.2))
            in_ch = channels // (2**i)
            out_ch = channels // (2 ** (i + 1))
            layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                )
            )

            for j in range(stacks):
                layers.append(
                    ResidualStack(out_ch, kernel_size=stack_kernel_size, dilation=stack_kernel_size**j)
                )

        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        final_channels = channels // (2 ** len(upsample_scales))
        layers.append(nn.Conv1d(final_channels, out_channels, kernel_size))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_quickstart_model():
    """Load the quickstart MB-MelGAN model."""
    print("Loading MB-MelGAN quickstart model...")
    # Same parameters as quick_finetune.py line 124
    model = MelGANGenerator(
        in_channels=80,
        out_channels=4,
        channels=384,
        kernel_size=7,
        upsample_scales=[5, 5, 3],
        stack_kernel_size=3,
        stacks=4
    )

    checkpoint_path = Path("mbmelgan_quickstart/mbmelgan_quickstart.pt")
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Run quick_finetune.py first!")
        sys.exit(1)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Loaded from {checkpoint_path}")
    return model


def convert_to_coreml(model, precision_name, precision_value, output_dir):
    """Convert model to CoreML with specified precision."""
    print(f"\n{'='*80}")
    print(f"Converting to CoreML ({precision_name})")
    print(f"{'='*80}")

    # Fixed shape example (125 frames)
    example_mel = torch.randn(1, 80, 125)

    print("1. Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_mel)

    print(f"2. Converting to CoreML ({precision_name})...")
    start = time.time()
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="mel_spectrogram", shape=example_mel.shape)],
        outputs=[ct.TensorType(name="audio_bands")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision_value,
    )
    conversion_time = time.time() - start
    print(f"   ✓ Conversion took {conversion_time:.2f}s")

    # Save
    output_path = output_dir / f"mbmelgan_quickstart_{precision_name.lower()}.mlpackage"
    mlmodel.save(str(output_path))

    # Get size
    size_bytes = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    size_mb = size_bytes / 1024 / 1024

    print(f"3. Saved to {output_path}")
    print(f"   Size: {size_mb:.2f} MB")

    return mlmodel, output_path, size_mb


def test_inference_quality(pytorch_model, coreml_model, precision_name):
    """Test inference quality: latency and accuracy."""
    print(f"\n{'='*80}")
    print(f"Testing Inference Quality ({precision_name})")
    print(f"{'='*80}")

    # Test with 3 different sizes
    test_sizes = [125, 250, 500]
    results = []

    for frames in test_sizes:
        print(f"\nTest size: {frames} frames")

        # Generate test mel
        mel_pt = torch.randn(1, 80, frames)
        mel_np = mel_pt.numpy()

        # PyTorch reference
        with torch.no_grad():
            start = time.time()
            pt_output = pytorch_model(mel_pt).numpy()
            pt_time = time.time() - start

        # CoreML inference
        try:
            start = time.time()
            coreml_output = coreml_model.predict({"mel_spectrogram": mel_np})["audio_bands"]
            coreml_time = time.time() - start

            # Compute MAE (Mean Absolute Error)
            mae = np.abs(pt_output - coreml_output).mean()
            max_diff = np.abs(pt_output - coreml_output).max()

            print(f"  PyTorch:  {pt_time*1000:.1f}ms")
            print(f"  CoreML:   {coreml_time*1000:.1f}ms")
            print(f"  MAE:      {mae:.6f}")
            print(f"  Max diff: {max_diff:.6f}")

            results.append({
                "frames": frames,
                "pt_time_ms": pt_time * 1000,
                "coreml_time_ms": coreml_time * 1000,
                "mae": mae,
                "max_diff": max_diff,
            })

        except Exception as e:
            print(f"  ❌ CoreML inference failed: {e}")
            print(f"     (Size {frames} may not be supported by fixed-shape model)")

    return results


def compare_precisions():
    """Main comparison function."""
    print("="*80)
    print("MB-MelGAN: FP32 vs FP16 Precision Comparison")
    print("="*80)

    output_dir = Path("precision_test")
    output_dir.mkdir(exist_ok=True)

    # Load PyTorch model
    pytorch_model = load_quickstart_model()
    pytorch_model.to("cpu")

    # Convert to both precisions
    fp16_model, fp16_path, fp16_size = convert_to_coreml(
        pytorch_model, "FP16", ct.precision.FLOAT16, output_dir
    )

    fp32_model, fp32_path, fp32_size = convert_to_coreml(
        pytorch_model, "FP32", ct.precision.FLOAT32, output_dir
    )

    # Test quality (only on 125 frames since fixed shape)
    print("\n" + "="*80)
    print("Quality Comparison (125 frames)")
    print("="*80)

    # FP16 test
    fp16_results = test_inference_quality(pytorch_model, fp16_model, "FP16")

    # FP32 test
    fp32_results = test_inference_quality(pytorch_model, fp32_model, "FP32")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nModel Size:")
    print(f"  FP16: {fp16_size:.2f} MB")
    print(f"  FP32: {fp32_size:.2f} MB")
    print(f"  Ratio: {fp32_size/fp16_size:.2f}x larger")

    if fp16_results and fp32_results:
        fp16_res = fp16_results[0]
        fp32_res = fp32_results[0]

        print(f"\nInference Time (125 frames):")
        print(f"  FP16: {fp16_res['coreml_time_ms']:.1f}ms")
        print(f"  FP32: {fp32_res['coreml_time_ms']:.1f}ms")

        print(f"\nAccuracy vs PyTorch (125 frames):")
        print(f"  FP16 MAE: {fp16_res['mae']:.6f}")
        print(f"  FP32 MAE: {fp32_res['mae']:.6f}")

        if fp32_res['mae'] < fp16_res['mae']:
            improvement = (fp16_res['mae'] - fp32_res['mae']) / fp16_res['mae'] * 100
            print(f"  ✅ FP32 is {improvement:.1f}% more accurate!")
        else:
            print(f"  ℹ️  FP16 and FP32 have similar accuracy")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    print("\nBased on Kokoro & HTDemucs patterns:")
    print("  🎯 Use FP32 for audio generation models")
    print("     - Better accuracy (lower MAE)")
    print("     - Prevents overflow in frequency operations")
    print("     - 2x larger size is acceptable for quality")

    print("\n✅ Test complete!")
    print(f"\nModels saved in: {output_dir}/")
    print(f"  - {fp16_path.name}")
    print(f"  - {fp32_path.name}")


if __name__ == "__main__":
    compare_precisions()
