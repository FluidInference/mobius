"""
Fine-tune MB-MelGAN on CosyVoice3 mel spectrograms.

This script:
1. Loads pre-trained MB-MelGAN weights
2. Sets up training pipeline
3. Fine-tunes on CosyVoice3 (mel, audio) pairs
4. Tests CoreML conversion periodically
5. Saves checkpoints
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import coremltools as ct
from pathlib import Path
from tqdm import tqdm
import numpy as np


# MB-MelGAN model (same as test script)
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

        # Initial conv
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        layers.append(nn.Conv1d(in_channels, channels, kernel_size))

        # Upsampling layers
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

            # Residual stacks
            for j in range(stacks):
                layers.append(
                    ResidualStack(out_ch, kernel_size=stack_kernel_size, dilation=stack_kernel_size**j)
                )

        # Final layers
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        final_channels = channels // (2 ** len(upsample_scales))
        layers.append(nn.Conv1d(final_channels, out_channels, kernel_size))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MBMelGANDataset(torch.utils.data.Dataset):
    """Dataset for MB-MelGAN training"""

    def __init__(self, data_dir, max_length=9600):
        self.data_dir = Path(data_dir)
        self.mel_files = sorted(list((self.data_dir / "mels").glob("*.pt")))
        self.max_length = max_length

        print(f"Found {len(self.mel_files)} training samples")

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        # Load mel and audio
        mel_path = self.mel_files[idx]
        audio_path = self.data_dir / "audio" / f"{mel_path.stem}.wav"

        mel = torch.load(mel_path)  # [1, 80, frames]
        audio, sr = torchaudio.load(audio_path)  # [1, samples]

        # Remove batch dimension
        mel = mel.squeeze(0)  # [80, frames]
        audio = audio.squeeze(0)  # [samples]

        # Truncate to max_length
        if audio.shape[0] > self.max_length:
            start = np.random.randint(0, audio.shape[0] - self.max_length)
            audio = audio[start : start + self.max_length]

            # Calculate corresponding mel frames
            hop_length = 300
            mel_start = start // hop_length
            mel_end = (start + self.max_length) // hop_length
            mel = mel[:, mel_start:mel_end]

        return mel, audio


def test_coreml_conversion(model, device="cpu"):
    """Test if model still converts to CoreML with flexible input shapes"""
    model.eval()
    model.to(device)

    example_mel = torch.randn(1, 80, 125).to(device)

    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_mel)

        # Use EnumeratedShapes for flexible input length
        # Support mel spectrograms from 50 to 500 frames
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="mel_spectrogram",
                shape=ct.EnumeratedShapes(shapes=[(1, 80, 125), (1, 80, 250), (1, 80, 500)])
            )],
            outputs=[ct.TensorType(name="audio_bands")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
        )

        return True, None
    except Exception as e:
        return False, str(e)


def train_mbmelgan(
    data_dir="mbmelgan_training_data",
    checkpoint_path="mbmelgan_pretrained/vctk_multi_band_melgan.v2/checkpoint-1000000steps.pkl",
    output_dir="mbmelgan_finetuned",
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    test_coreml_every=5,
):
    """Fine-tune MB-MelGAN"""

    print("=" * 80)
    print("Fine-tuning MB-MelGAN on CosyVoice3")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Setup")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Epochs: {num_epochs}")

    # Create model
    print(f"\n2. Creating model...")
    generator_params = {
        "in_channels": 80,
        "out_channels": 4,  # 4 bands
        "channels": 384,
        "kernel_size": 7,
        "upsample_scales": [5, 5, 3],  # 75x upsampling
        "stack_kernel_size": 3,
        "stacks": 4,
    }

    model = MelGANGenerator(**generator_params)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created: {total_params:,} parameters")

    # Load pre-trained weights
    print(f"\n3. Loading pre-trained weights...")
    print(f"   Checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model" in checkpoint and "generator" in checkpoint["model"]:
            state_dict = checkpoint["model"]["generator"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print(f"   ✓ Weights loaded (strict=False)")
    except Exception as e:
        print(f"   ⚠️  Failed to load weights: {e}")
        print(f"   Training from random initialization...")

    # Create dataset
    print(f"\n4. Loading dataset...")
    dataset = MBMelGANDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"   ✓ Dataset: {len(dataset)} samples")
    print(f"   ✓ Batches per epoch: {len(dataloader)}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    l1_loss = nn.L1Loss()

    # Training loop
    print(f"\n5. Training...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (mel, audio) in enumerate(pbar):
            mel = mel.to(device)  # [B, 80, frames]
            audio = audio.to(device)  # [B, samples]

            # Forward pass
            optimizer.zero_grad()
            pred_bands = model(mel)  # [B, 4, samples*75]

            # Target: we need to match the output shape
            # For now, use L1 loss on averaged bands vs audio
            # (In production, would use proper PQMF synthesis)
            pred_audio = pred_bands.mean(dim=1)  # [B, samples*75]

            # Resize audio to match prediction
            if pred_audio.shape[1] != audio.shape[1]:
                audio = F.interpolate(audio.unsqueeze(1), size=pred_audio.shape[1], mode="linear").squeeze(1)

            # Loss
            loss = l1_loss(pred_audio, audio)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        print(f"   Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"   ✓ Saved checkpoint: {checkpoint_path}")

        # Test CoreML conversion
        if (epoch + 1) % test_coreml_every == 0:
            print(f"   Testing CoreML conversion...")
            success, error = test_coreml_conversion(model, device)
            if success:
                print(f"   ✅ CoreML conversion: OK")
            else:
                print(f"   ❌ CoreML conversion failed: {error}")

            model.train()  # Back to training mode

    # Final save
    final_path = output_dir / "mbmelgan_finetuned_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete!")
    print(f"   Final model: {final_path}")

    # Final CoreML conversion
    print(f"\n6. Final CoreML conversion...")
    success, error = test_coreml_conversion(model, device)
    if success:
        print(f"   ✅ CoreML conversion successful!")

        # Save CoreML model with flexible input shapes
        model.eval()
        model.to("cpu")
        example_mel = torch.randn(1, 80, 125)

        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_mel)

        # Use EnumeratedShapes for flexible input length
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="mel_spectrogram",
                shape=ct.EnumeratedShapes(shapes=[(1, 80, 125), (1, 80, 250), (1, 80, 500)])
            )],
            outputs=[ct.TensorType(name="audio_bands")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
        )

        coreml_path = output_dir / "mbmelgan_finetuned_coreml.mlpackage"
        mlmodel.save(str(coreml_path))
        print(f"   ✓ Saved: {coreml_path}")
        print(f"   ✓ Supports mel frames: 125, 250, 500")
    else:
        print(f"   ❌ Final CoreML conversion failed: {error}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="mbmelgan_training_data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="mbmelgan_pretrained/vctk_multi_band_melgan.v2/checkpoint-1000000steps.pkl",
    )
    parser.add_argument("--output-dir", type=str, default="mbmelgan_finetuned")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-coreml-every", type=int, default=5)
    args = parser.parse_args()

    success = train_mbmelgan(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        test_coreml_every=args.test_coreml_every,
    )

    sys.exit(0 if success else 1)
