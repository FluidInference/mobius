"""Convert the VoxCPM 1.5 AudioVAE decoder to CoreML.

The decoder takes latent features and produces 44.1kHz audio.
Input: [1, 64, T] latent features
Output: [1, 1, T*1764] audio waveform

Architecture: 5 CausalDecoderBlocks with Snake activations, depthwise convolutions,
transposed convolutions for upsampling, and weight-normalized causal convolutions.
Decoder rates [7,7,6,3,2] give total stride = 1764 (= hop_length).
"""

import time

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


def remove_weight_norm_recursive(module: nn.Module):
    """Remove weight_norm from all sub-modules."""
    for name, child in module.named_children():
        try:
            torch.nn.utils.remove_weight_norm(child)
        except ValueError:
            pass
        remove_weight_norm_recursive(child)


class TraceableSnake1d(nn.Module):
    """Snake activation without shape tuple indexing."""

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


class TraceableDecoder(nn.Module):
    """Wraps AudioVAE.decoder for tracing.

    The original decode() just calls self.decoder(latent).
    """

    def __init__(self, decoder):
        super().__init__()
        self.model = decoder.model

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [1, 64, T] latent features
        Returns:
            audio: [1, 1, T*1764] reconstructed waveform
        """
        return self.model(latent)


def main():
    print("=== Converting AudioVAE Decoder to CoreML ===\n")

    # Load model
    print("[1/5] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    vae = model.tts_model.audio_vae
    vae = vae.float().cpu().eval()

    # Remove weight_norm and replace Snake activations
    print("[2/5] Removing weight_norm, replacing Snake activations...")
    remove_weight_norm_recursive(vae.decoder)
    replace_snake_activations(vae.decoder)

    decoder = TraceableDecoder(vae.decoder)
    decoder.eval()

    # Test with latent of 25 timesteps (= 1 second of audio)
    # For generation, typical output is ~100-500 timesteps
    latent_len = 50  # ~2 seconds
    test_latent = torch.randn(1, 64, latent_len)

    with torch.no_grad():
        pt_output = decoder(test_latent)
    expected_audio_len = latent_len * 1764
    print(f"  PyTorch: input {test_latent.shape} -> output {pt_output.shape}")
    print(f"  Expected audio length: {expected_audio_len}, actual: {pt_output.shape[-1]}")
    print(f"  Audio stats: min={pt_output.min():.4f}, max={pt_output.max():.4f}")

    # Trace
    print("\n[3/5] Tracing with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, test_latent)

    # Verify trace
    with torch.no_grad():
        traced_output = traced(test_latent)
    max_diff = (pt_output - traced_output).abs().max().item()
    print(f"  Trace parity: max diff = {max_diff:.2e}")

    # Convert to CoreML with flexible latent length
    print("\n[4/5] Converting to CoreML...")
    t0 = time.time()
    # Use RangeDim for variable-length latent (4 to 2000 frames)
    latent_dim = ct.RangeDim(lower_bound=4, upper_bound=2000, default=latent_len)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="latent", shape=(1, 64, latent_dim)),
        ],
        outputs=[
            ct.TensorType(name="audio"),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    # Save
    out_path = "audio_vae_decoder.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Validate CoreML output
    print("\n[5/5] Validating CoreML output...")
    coreml_pred = mlmodel.predict({"latent": test_latent.numpy()})
    coreml_output = coreml_pred["audio"]
    diff = np.abs(pt_output.numpy() - coreml_output)
    corr = np.corrcoef(pt_output.numpy().flatten(), coreml_output.flatten())[0, 1]
    print(f"  Absolute: max={diff.max():.4e}, mean={diff.mean():.4e}")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Audio range: [{pt_output.min():.4f}, {pt_output.max():.4f}]")

    if corr > 0.999:
        print("  PASS: Excellent correlation")
    elif corr > 0.99:
        print("  WARN: Good correlation, minor drift")
    else:
        print("  FAIL: Poor correlation, needs investigation")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
