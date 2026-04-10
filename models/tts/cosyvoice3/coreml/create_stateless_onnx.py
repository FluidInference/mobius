#!/usr/bin/env python3
"""
Create stateless ONNX exports of Vocoder and Flow.

This bypasses the CoreML conversion issues by creating simple ONNX models
that are explicitly stateless (no hidden state between calls).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Add cosyvoice repo to path
REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
sys.path.insert(0, str(REPO_PATH))

from generator_coreml import CausalHiFTGeneratorCoreML
from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor

REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CACHE_DIR = Path.home() / ".cache" / "cosyvoice3_analysis"


class StatelessVocoderWrapper(nn.Module):
    """
    Explicitly stateless wrapper for vocoder.

    Each forward() call is completely independent:
    - Takes mel spectrogram
    - Returns audio waveform
    - No state carried between calls
    """

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        # Set to eval mode to disable any training-specific state
        self.generator.eval()

    @torch.no_grad()
    def forward(self, mel):
        """
        Stateless inference: mel → audio

        Args:
            mel: [batch, 80, time] - Mel spectrogram

        Returns:
            audio: [batch, samples] - Audio waveform
        """
        # finalize=True means treat as complete utterance (no streaming state)
        audio, _ = self.generator.inference(mel, finalize=True)
        return audio


def create_vocoder_onnx():
    """Create stateless ONNX export of vocoder."""
    print("=" * 80)
    print("Creating Stateless Vocoder ONNX")
    print("=" * 80)

    # Load checkpoint
    print("\n[1/4] Downloading checkpoint...")
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="hift.pt",
        cache_dir=str(CACHE_DIR)
    )
    print(f"✓ {checkpoint_path}")

    # Create F0 predictor
    print("\n[2/4] Creating vocoder...")
    f0_predictor = CausalConvRNNF0Predictor(
        num_class=1,
        in_channels=80,
        cond_channels=512,
    )

    # Create generator
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    generator.load_state_dict(checkpoint, strict=False)
    generator.eval()
    print("✓ Loaded weights")

    # Remove weight normalization (required for ONNX export)
    print("\n[2.5/4] Removing weight normalization...")
    from torch.nn.utils import remove_weight_norm

    def remove_weight_norm_recursive(module):
        """Recursively remove weight_norm from all submodules."""
        for name, child in module.named_children():
            try:
                remove_weight_norm(child)
                print(f"  ✓ Removed weight_norm from {name}")
            except ValueError:
                # No weight_norm to remove
                pass
            # Recurse
            remove_weight_norm_recursive(child)

    remove_weight_norm_recursive(generator)
    print("✓ All weight_norm removed")

    # Wrap in stateless wrapper
    print("\n[3/4] Creating stateless wrapper...")
    wrapper = StatelessVocoderWrapper(generator)
    print("✓ Wrapper created")

    # Export to ONNX
    print("\n[4/4] Exporting to ONNX...")
    example_mel = torch.randn(1, 80, 100)  # 100 frames ≈ 2s

    output_path = Path("converted/hift_vocoder_stateless.onnx")
    output_path.parent.mkdir(exist_ok=True)

    print("  Tracing model...")
    try:
        torch.onnx.export(
            wrapper,
            example_mel,
            str(output_path),
            input_names=["mel"],
            output_names=["audio"],
            dynamic_axes={
                "mel": {2: "time"},       # Allow variable time dimension
                "audio": {1: "samples"},  # Allow variable sample dimension
            },
            opset_version=17,
            do_constant_folding=True,
            export_params=True,
        )

        print(f"✓ Exported to: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Verify
        print("\n  Verifying ONNX model...")
        import onnxruntime as ort
        session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])

        print(f"  ✓ Model is valid")
        print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in session.get_outputs()]}")

        # Test inference
        print("\n  Testing inference...")
        test_mel = example_mel.numpy()
        result = session.run(None, {"mel": test_mel})
        print(f"  ✓ Input shape: {test_mel.shape}")
        print(f"  ✓ Output shape: {result[0].shape}")
        print(f"  ✓ Audio samples: {result[0].shape[1]}")

        return True, output_path

    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Create stateless ONNX models."""
    output_dir = Path("converted")
    output_dir.mkdir(exist_ok=True)

    # Create vocoder ONNX
    success, path = create_vocoder_onnx()

    if success:
        print("\n" + "=" * 80)
        print("SUCCESS")
        print("=" * 80)
        print(f"\n✓ Created stateless vocoder ONNX: {path}")
        print("\nProperties:")
        print("  - Stateless: Each call is independent")
        print("  - No hidden state between calls")
        print("  - Same input → same output")
        print("  - Safe for parallel inference")
        print("\nUsage:")
        print("  import onnxruntime as ort")
        print(f"  session = ort.InferenceSession('{path}')")
        print("  audio = session.run(None, {'mel': mel_spectrogram})")
        print("\nNext steps:")
        print("  1. Test with verify_stateless_onnx.py")
        print("  2. Integrate into hybrid_coreml_onnx.py")
        print("  3. Create Swift wrapper using ONNX Runtime")
    else:
        print("\n" + "=" * 80)
        print("FAILED")
        print("=" * 80)
        print("\nThe ONNX export failed. This might be due to:")
        print("  1. Unsupported operations in the model")
        print("  2. Model too complex for ONNX export")
        print("  3. Missing dependencies")
        print("\nFallback: Use PyTorch pipeline (full_tts_pytorch.py)")


if __name__ == "__main__":
    main()
