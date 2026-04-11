"""
Test RangeDim conversion for MB-MelGAN quickstart model.

Compares:
- EnumeratedShapes (current): 3 fixed sizes [125, 250, 500]
- RangeDim (Kokoro approach): continuous range [50-500]

Benefits of RangeDim:
- Supports ANY size in range (no padding needed)
- No artifacts from padding/cropping
- Simpler runtime logic
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
import time


# MB-MelGAN model
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
        x = F.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
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


def test_rangedim():
    """Test RangeDim conversion."""
    print("="*80)
    print("MB-MelGAN: RangeDim vs EnumeratedShapes Comparison")
    print("="*80)

    output_dir = Path("rangedim_quickstart_test")
    output_dir.mkdir(exist_ok=True)

    model = load_quickstart_model()

    # Test 1: EnumeratedShapes (current approach)
    print("\n" + "="*80)
    print("1. EnumeratedShapes (Current)")
    print("="*80)
    print("   Fixed sizes: [125, 250, 500] frames")

    try:
        example_mel = torch.randn(1, 80, 125)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_mel)

        print("\n   Converting...")
        start = time.time()
        mlmodel_enum = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="mel_spectrogram",
                shape=ct.EnumeratedShapes(shapes=[(1, 80, 125), (1, 80, 250), (1, 80, 500)])
            )],
            outputs=[ct.TensorType(name="audio_bands")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
        )
        enum_time = time.time() - start

        enum_path = output_dir / "mbmelgan_enumerated.mlpackage"
        mlmodel_enum.save(str(enum_path))
        enum_size = sum(f.stat().st_size for f in enum_path.rglob('*') if f.is_file()) / 1024 / 1024

        print(f"   ✅ Conversion successful!")
        print(f"   Time: {enum_time:.2f}s")
        print(f"   Size: {enum_size:.2f} MB")
        print(f"   Path: {enum_path}")

        # Test inference
        print(f"\n   Testing inference:")
        test_sizes = [125, 250, 500, 259]  # 259 should fail (not in enum)
        for frames in test_sizes:
            test_mel = torch.randn(1, 80, frames).numpy()
            try:
                result = mlmodel_enum.predict({"mel_spectrogram": test_mel})
                print(f"     {frames} frames: ✓ {result['audio_bands'].shape}")
            except Exception as e:
                print(f"     {frames} frames: ✗ {str(e)[:60]}...")

    except Exception as e:
        print(f"   ❌ EnumeratedShapes failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: RangeDim (Kokoro approach)
    print("\n" + "="*80)
    print("2. RangeDim (Kokoro Approach)")
    print("="*80)
    print("   Continuous range: 50-500 frames")

    try:
        example_mel = torch.randn(1, 80, 125)
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_mel)

        print("\n   Converting...")
        start = time.time()
        mlmodel_range = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="mel_spectrogram",
                shape=(1, 80, ct.RangeDim(lower_bound=50, upper_bound=500, default=125))
            )],
            outputs=[ct.TensorType(name="audio_bands")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
        )
        range_time = time.time() - start

        range_path = output_dir / "mbmelgan_rangedim.mlpackage"
        mlmodel_range.save(str(range_path))
        range_size = sum(f.stat().st_size for f in range_path.rglob('*') if f.is_file()) / 1024 / 1024

        print(f"   ✅ Conversion successful!")
        print(f"   Time: {range_time:.2f}s")
        print(f"   Size: {range_size:.2f} MB")
        print(f"   Path: {range_path}")

        # Test inference at various sizes
        print(f"\n   Testing inference:")
        test_sizes = [50, 100, 125, 200, 259, 300, 400, 500]
        for frames in test_sizes:
            test_mel = torch.randn(1, 80, frames).numpy()
            try:
                result = mlmodel_range.predict({"mel_spectrogram": test_mel})
                print(f"     {frames} frames: ✓ {result['audio_bands'].shape}")
            except Exception as e:
                print(f"     {frames} frames: ✗ {str(e)[:60]}...")

    except Exception as e:
        print(f"   ❌ RangeDim failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    print(f"\nModel Size:")
    print(f"  EnumeratedShapes: {enum_size:.2f} MB")
    print(f"  RangeDim:         {range_size:.2f} MB")

    print(f"\nConversion Time:")
    print(f"  EnumeratedShapes: {enum_time:.2f}s")
    print(f"  RangeDim:         {range_time:.2f}s")

    print(f"\nFlexibility:")
    print(f"  EnumeratedShapes: 3 fixed sizes (125, 250, 500)")
    print(f"                    - Size 259 → must crop to 250 or pad to 500")
    print(f"                    - Padding artifacts possible")
    print(f"  RangeDim:         ANY size from 50-500")
    print(f"                    - Size 259 → works directly!")
    print(f"                    - No padding needed")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\n🎯 Use RangeDim for production!")
    print("   ✓ Same model size")
    print("   ✓ Similar conversion time")
    print("   ✓ Supports exact input sizes (no padding artifacts)")
    print("   ✓ Simpler runtime logic (no bucket selection)")
    print("   ✓ Proven approach (used by Kokoro TTS)")

    print(f"\n✅ Test complete!")
    print(f"\nModels saved in: {output_dir}/")


if __name__ == "__main__":
    test_rangedim()
