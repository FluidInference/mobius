"""
Quick fine-tuning demo for MB-MelGAN (no CosyVoice3 needed).

This script generates synthetic training data and demonstrates the fine-tuning process.
For production, use generate_training_data.py with real CosyVoice3 outputs.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
from tqdm import tqdm
import numpy as np


# MB-MelGAN model
class ResidualStack(nn.Module):
    """Residual stack module"""

    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation)

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


def generate_synthetic_data(num_samples=100):
    """Generate synthetic (mel, audio) pairs for demo"""
    print("Generating synthetic training data...")

    data = []
    for i in range(num_samples):
        # Random mel spectrogram [1, 80, 125]
        mel = torch.randn(1, 80, 125)

        # Random audio [1, 9375] (125 * 75 = 9375)
        audio = torch.randn(1, 9375)

        data.append((mel, audio))

    print(f"✓ Generated {num_samples} synthetic samples")
    return data


def quick_finetune(num_epochs=10, num_samples=100):
    """Quick fine-tuning demo"""

    print("=" * 80)
    print("MB-MelGAN Quick Fine-tuning Demo")
    print("=" * 80)

    output_dir = Path("mbmelgan_quickstart")
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Setup")
    print(f"   Device: {device}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Samples: {num_samples}")

    # Create model
    print(f"\n2. Creating model...")
    model = MelGANGenerator(
        in_channels=80, out_channels=4, channels=384, kernel_size=7, upsample_scales=[5, 5, 3], stack_kernel_size=3, stacks=4
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model: {total_params:,} parameters")

    # Load pre-trained weights if available
    checkpoint_path = Path("mbmelgan_pretrained/vctk_multi_band_melgan.v2/checkpoint-1000000steps.pkl")
    if checkpoint_path.exists():
        print(f"\n3. Loading pre-trained weights...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if "model" in checkpoint and "generator" in checkpoint["model"]:
                state_dict = checkpoint["model"]["generator"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            print(f"   ✓ Pre-trained weights loaded")
        except Exception as e:
            print(f"   ⚠️  Failed: {e}")
    else:
        print(f"\n3. No pre-trained weights found")
        print(f"   Training from scratch...")

    # Generate synthetic data
    print(f"\n4. Preparing data...")
    train_data = generate_synthetic_data(num_samples)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    l1_loss = nn.L1Loss()

    # Training
    print(f"\n5. Training...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}")
        for mel, audio in pbar:
            mel = mel.to(device)
            audio = audio.to(device)

            optimizer.zero_grad()
            pred_bands = model(mel)

            # Simple loss (just for demo)
            pred_audio = pred_bands.mean(dim=1)
            if pred_audio.shape != audio.shape:
                audio = F.interpolate(audio.unsqueeze(1), size=pred_audio.shape[1], mode="linear").squeeze(1)

            loss = l1_loss(pred_audio, audio)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_data)
        print(f"   Epoch {epoch+1} - Loss: {avg_loss:.4f}")

    # Save
    print(f"\n6. Saving model...")
    torch.save(model.state_dict(), output_dir / "mbmelgan_quickstart.pt")
    print(f"   ✓ Saved: {output_dir}/mbmelgan_quickstart.pt")

    # Test CoreML conversion
    print(f"\n7. Testing CoreML conversion...")
    model.eval()
    model.to("cpu")

    example_mel = torch.randn(1, 80, 125)

    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_mel)

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="mel_spectrogram", shape=example_mel.shape)],
            outputs=[ct.TensorType(name="audio_bands")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
        )

        coreml_path = output_dir / "mbmelgan_quickstart_coreml.mlpackage"
        mlmodel.save(str(coreml_path))

        print(f"   ✅ CoreML conversion successful!")
        print(f"   ✓ Saved: {coreml_path}")

        # Test inference
        import numpy as np

        mel_np = example_mel.numpy()
        prediction = mlmodel.predict({"mel_spectrogram": mel_np})
        print(f"   ✓ Inference test: {prediction['audio_bands'].shape}")

    except Exception as e:
        print(f"   ❌ CoreML conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"\n" + "=" * 80)
    print(f"✅ Quick fine-tuning demo complete!")
    print("=" * 80)

    print(f"\nResults:")
    print(f"  - PyTorch model: {output_dir}/mbmelgan_quickstart.pt")
    print(f"  - CoreML model: {output_dir}/mbmelgan_quickstart_coreml.mlpackage")

    print(f"\n📝 Note: This used synthetic data for demo purposes.")
    print(f"For production, use real CosyVoice3 data:")
    print(f"  1. Download CosyVoice3 model")
    print(f"  2. Run: python generate_training_data.py")
    print(f"  3. Run: python train_mbmelgan.py")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    success = quick_finetune(num_epochs=args.epochs, num_samples=args.samples)
    sys.exit(0 if success else 1)
