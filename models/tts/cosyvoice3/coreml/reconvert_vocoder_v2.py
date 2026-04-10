#!/usr/bin/env python3
"""
Re-convert vocoder with different CoreML settings to fix loading hang.

The original conversion used:
- minimum_deployment_target=ct.target.iOS17
- compute_units=ct.ComputeUnit.CPU_ONLY
- Default neuralnetwork format
- FP32 precision

This version tries:
- minimum_deployment_target=ct.target.macOS14 (same as flow)
- compute_units=ct.ComputeUnit.ALL (same as flow)
- convert_to='mlprogram' (same as flow)
- FP16 precision (same as flow)
"""

import sys
from pathlib import Path
import torch
import coremltools as ct
from huggingface_hub import hf_hub_download
import numpy as np

# Add cosyvoice repo to path
REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
sys.path.insert(0, str(REPO_PATH))

from generator_coreml import CausalHiFTGeneratorCoreML
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CACHE_DIR = Path.home() / ".cache" / "cosyvoice3_analysis"


class VocoderWrapper(torch.nn.Module):
    """Wrapper for CoreML tracing."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, mel):
        audio, _ = self.generator.inference(mel, finalize=True)
        return audio


def load_vocoder():
    """Load the vocoder model."""
    print("=" * 80)
    print("Loading CosyVoice3 Vocoder")
    print("=" * 80)

    # Download checkpoint
    print("\nDownloading hift.pt...")
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="hift.pt",
        cache_dir=str(CACHE_DIR)
    )

    # Create F0 predictor
    f0_predictor = CausalConvRNNF0Predictor(
        num_class=1,
        in_channels=80,
        cond_channels=512,
    )

    # Create generator
    print("Creating generator with custom ISTFT...")
    generator = CausalHiFTGeneratorCoreML(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        conv_pre_look_right=4,
        f0_predictor=f0_predictor,
    )

    # Load weights
    print("Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(checkpoint, strict=False)
    generator.eval()
    print("✓ Weights loaded")

    return generator


def convert_with_settings(generator, example_input, output_path, config_name, **kwargs):
    """Try converting with specific settings."""
    print("\n" + "=" * 80)
    print(f"Converting: {config_name}")
    print("=" * 80)
    print(f"Settings: {kwargs}")

    wrapper = VocoderWrapper(generator)
    wrapper.eval()

    # Trace
    print("\nTracing...")
    with torch.inference_mode():
        traced_model = torch.jit.trace(wrapper, example_input)
    print("✓ Traced")

    # Convert
    try:
        print("\nConverting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="mel",
                    shape=example_input.shape,
                    dtype=np.float32,
                )
            ],
            outputs=[
                ct.TensorType(
                    name="audio",
                    dtype=np.float32,
                )
            ],
            **kwargs
        )
        print(f"✓ Conversion successful")

        # Save
        mlmodel.save(str(output_path))
        print(f"✓ Saved to: {output_path}")

        return True

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return False


def main():
    """Try multiple conversion configurations."""
    output_dir = Path(__file__).parent / "converted"
    output_dir.mkdir(exist_ok=True)

    # Load model
    generator = load_vocoder()

    # Create example input
    example_mel = torch.randn(1, 80, 50)  # Small input for fast testing

    # Configuration 1: macOS14 + ALL + mlprogram + FP16 (like Flow)
    print("\n" + "=" * 80)
    print("ATTEMPT 1: Match Flow Decoder Settings")
    print("=" * 80)
    success = convert_with_settings(
        generator,
        example_mel,
        output_dir / "hift_vocoder_v2.mlpackage",
        "macOS14_ALL_mlprogram_FP16",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
    )

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS! Config 1 worked")
        print("=" * 80)
        print(f"\nNew model: converted/hift_vocoder_v2.mlpackage")
        print("\nNext: Test loading this model in Swift")
        return

    # Configuration 2: macOS14 + CPU_ONLY + mlprogram + FP32
    print("\n" + "=" * 80)
    print("ATTEMPT 2: macOS14 + CPU_ONLY + mlprogram + FP32")
    print("=" * 80)
    success = convert_with_settings(
        generator,
        example_mel,
        output_dir / "hift_vocoder_v3.mlpackage",
        "macOS14_CPU_mlprogram_FP32",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        convert_to='mlprogram',
    )

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS! Config 2 worked")
        print("=" * 80)
        print(f"\nNew model: converted/hift_vocoder_v3.mlpackage")
        return

    # Configuration 3: iOS16 + ALL + neuralnetwork (older spec)
    print("\n" + "=" * 80)
    print("ATTEMPT 3: iOS16 + ALL + neuralnetwork (older)")
    print("=" * 80)
    success = convert_with_settings(
        generator,
        example_mel,
        output_dir / "hift_vocoder_v4.mlpackage",
        "iOS16_ALL_neuralnetwork",
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS! Config 3 worked")
        print("=" * 80)
        print(f"\nNew model: converted/hift_vocoder_v4.mlpackage")
        return

    print("\n" + "=" * 80)
    print("All conversions failed")
    print("=" * 80)
    print("\nThe model architecture may not be compatible with CoreML.")


if __name__ == "__main__":
    main()
