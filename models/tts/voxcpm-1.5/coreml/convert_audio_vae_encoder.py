"""Convert the VoxCPM 1.5 AudioVAE encoder to CoreML.

The encoder takes raw 44.1kHz audio and produces latent features.
Input: [1, 1, N] audio waveform (N must be multiple of hop_length=1764)
Output: [1, 64, T] latent features (T = N / 1764)

Architecture: 5 CausalEncoderBlocks with Snake activations, depthwise convolutions,
and weight-normalized causal convolutions. Encoder rates [2,3,6,7,7] give
total stride = 1764 (= hop_length).
"""

import time

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


def remove_weight_norm_recursive(module: nn.Module):
    """Remove weight_norm from all sub-modules (required for tracing)."""
    for name, child in module.named_children():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            pass
        remove_weight_norm_recursive(child)


class TraceableSnake1d(nn.Module):
    """Snake activation that avoids shape tuple indexing (unsupported by CoreML).

    Original: x + (1/alpha) * sin(alpha * x)^2
    The original uses @torch.jit.script with shape = x.shape; x.reshape(shape[0], shape[1], -1)
    which triggers a CoreML __getitem__ error. Since our input is already 3D [B, C, T],
    the reshape is a no-op.
    """

    def __init__(self, alpha: torch.Tensor):
        super().__init__()
        self.alpha = nn.Parameter(alpha.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


def replace_snake_activations(module: nn.Module):
    """Replace all Snake1d modules with traceable versions."""
    for name, child in module.named_children():
        if child.__class__.__name__ == "Snake1d":
            traceable = TraceableSnake1d(child.alpha.data)
            setattr(module, name, traceable)
        else:
            replace_snake_activations(child)


class TraceableEncoder(nn.Module):
    """Wraps AudioVAE.encoder for tracing.

    The original encode() does:
      1. preprocess (pad to multiple of hop_length)
      2. encoder.block(audio) -> hidden
      3. encoder.fc_mu(hidden) -> mu (latent)
    We flatten this into a single forward pass.
    """

    def __init__(self, encoder):
        super().__init__()
        self.block = encoder.block
        self.fc_mu = encoder.fc_mu

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [1, 1, N] raw waveform (already padded to hop_length multiple)
        Returns:
            latent: [1, 64, T] latent features
        """
        hidden = self.block(audio)
        mu = self.fc_mu(hidden)
        return mu


def main():
    print("=== Converting AudioVAE Encoder to CoreML ===\n")

    # Load model
    print("[1/5] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    vae = model.tts_model.audio_vae
    vae = vae.float().cpu().eval()

    # Remove weight_norm and replace Snake activations (required for torch.jit.trace + CoreML)
    print("[2/5] Removing weight_norm, replacing Snake activations...")
    remove_weight_norm_recursive(vae.encoder)
    replace_snake_activations(vae.encoder)

    encoder = TraceableEncoder(vae.encoder)
    encoder.eval()

    # Get PyTorch reference output
    # Use 5 seconds of audio (5 * 44100 = 220500 samples)
    audio_len = 5 * 44100  # 220500, must be multiple of 1764
    assert audio_len % 1764 == 0, f"Audio length {audio_len} not multiple of 1764"
    test_audio = torch.randn(1, 1, audio_len)

    with torch.no_grad():
        pt_output = encoder(test_audio)
    print(f"  PyTorch: input {test_audio.shape} -> output {pt_output.shape}")
    print(f"  Latent stats: min={pt_output.min():.4f}, max={pt_output.max():.4f}")

    # Trace
    print("\n[3/5] Tracing with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(encoder, test_audio)

    # Verify trace
    with torch.no_grad():
        traced_output = traced(test_audio)
    max_diff = (pt_output - traced_output).abs().max().item()
    print(f"  Trace parity: max diff = {max_diff:.2e}")

    # Convert to CoreML
    # Note: Fixed shape only - RangeDim/EnumeratedShapes don't work because the
    # original AudioVAE has data-dependent assertions that get baked into the trace.
    # The pipeline pads/truncates prompt audio to this fixed size.
    print("\n[4/5] Converting to CoreML...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="audio", shape=(1, 1, audio_len)),
        ],
        outputs=[
            ct.TensorType(name="latent"),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    # Save
    out_path = "audio_vae_encoder.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Validate CoreML output
    print("\n[5/5] Validating CoreML output...")
    coreml_pred = mlmodel.predict({"audio": test_audio.numpy()})
    coreml_output = coreml_pred["latent"]
    max_diff = np.abs(pt_output.numpy() - coreml_output).max()
    mean_diff = np.abs(pt_output.numpy() - coreml_output).mean()
    print(f"  CoreML parity: max diff = {max_diff:.2e}, mean diff = {mean_diff:.2e}")

    if max_diff < 1e-3:
        print("  PASS: CoreML output matches PyTorch within tolerance")
    elif max_diff < 1e-2:
        print("  WARN: Slight numerical drift, but likely acceptable")
    else:
        print("  FAIL: Large numerical discrepancy, needs investigation")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
