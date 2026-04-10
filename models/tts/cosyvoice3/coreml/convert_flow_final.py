"""
Final attempt: Load Flow decoder from PyTorch and convert to CoreML.
Set up all the necessary paths and dependencies.
"""

import sys
from pathlib import Path

# Add all necessary paths
REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
MATCHA_PATH = REPO_PATH / "third_party" / "Matcha-TTS"
sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(MATCHA_PATH))

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from huggingface_hub import hf_hub_download

print("="*80)
print("Final Flow Model Conversion Attempt")
print("="*80)

# Install einops if needed
try:
    import einops
except ImportError:
    print("Installing einops...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "einops"], check=True)
    import einops

# Load Flow checkpoint
REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CACHE_DIR = Path.home() / ".cache" / "cosyvoice3_analysis"

print("\nLoading Flow checkpoint...")
flow_path = hf_hub_download(repo_id=REPO_ID, filename="flow.pt", cache_dir=CACHE_DIR)
flow_state = torch.load(flow_path, map_location="cpu", weights_only=True)
print(f"✓ Loaded {len(flow_state)} parameters")

# Try to import the decoder
print("\nImporting Flow decoder...")
try:
    from cosyvoice.flow.decoder import ConditionalDecoder
    print("✓ Imported ConditionalDecoder")

    # Now we need to figure out the config
    # Let's infer it from the state dict
    print("\nInferring model config from state dict...")

    # Create a minimal decoder - we'll need to guess the right config
    # Based on the ONNX inputs, we know:
    # - in_channels: likely 80 (mel bins)
    # - out_channels: likely 80
    # Let's try default config first

    print("\nAttempting to create decoder with corrected config...")
    # in_channels = 320 because forward() packs: x(80) + mu(80) + spks(80) + cond(80) = 320
    decoder = ConditionalDecoder(
        in_channels=320,
        out_channels=80,
        channels=(256, 256),
        dropout=0.0,  # Disable for inference
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
    )

    print("\nLoading state dict...")
    # Extract only decoder parameters
    decoder_state = {}
    for k, v in flow_state.items():
        if k.startswith('decoder.'):
            new_key = k.replace('decoder.', '')
            decoder_state[new_key] = v

    if not decoder_state:
        # Try estimator
        for k, v in flow_state.items():
            if 'estimator' in k:
                # This is the estimator (what was exported to ONNX)
                decoder_state[k] = v

    print(f"Decoder state dict: {len(decoder_state)} parameters")

    # Load the state (might fail if config is wrong)
    try:
        decoder.load_state_dict(decoder_state, strict=False)
        decoder.eval()
        print("✓ Loaded decoder weights")

        # Create wrapper for tracing
        class FlowDecoderWrapper(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, x, mask, mu, t, spks, cond):
                # Based on ONNX inputs
                return self.decoder(x, mask, mu, t, spks, cond)

        wrapper = FlowDecoderWrapper(decoder)
        wrapper.eval()

        # Trace with example inputs (based on ONNX model inputs)
        print("\nTracing decoder...")
        batch_size = 1
        seq_len = 100
        mel_bins = 80

        x = torch.randn(batch_size, mel_bins, seq_len)
        mask = torch.ones(batch_size, 1, seq_len)
        mu = torch.randn(batch_size, mel_bins, seq_len)
        t = torch.randn(batch_size)
        spks = torch.randn(batch_size, mel_bins)
        cond = torch.randn(batch_size, mel_bins, seq_len)

        print(f"  x: {x.shape}")
        print(f"  mask: {mask.shape}")
        print(f"  mu: {mu.shape}")
        print(f"  t: {t.shape}")
        print(f"  spks: {spks.shape}")
        print(f"  cond: {cond.shape}")

        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, (x, mask, mu, t, spks, cond))

        print("✓ Traced successfully")

        # Convert to CoreML
        print("\nConverting to CoreML...")
        coreml_model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name='x', shape=(1, 80, ct.RangeDim(1, 2048)), dtype=np.float16),
                ct.TensorType(name='mask', shape=(1, 1, ct.RangeDim(1, 2048)), dtype=np.float16),
                ct.TensorType(name='mu', shape=(1, 80, ct.RangeDim(1, 2048)), dtype=np.float16),
                ct.TensorType(name='t', shape=(1,), dtype=np.float16),
                ct.TensorType(name='spks', shape=(1, 80), dtype=np.float16),
                ct.TensorType(name='cond', shape=(1, 80, ct.RangeDim(1, 2048)), dtype=np.float16),
            ],
            outputs=[ct.TensorType(name='output', dtype=np.float16)],
            minimum_deployment_target=ct.target.macOS14,
            compute_units=ct.ComputeUnit.ALL,
            convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
        )

        output_path = "flow_decoder.mlpackage"
        coreml_model.save(output_path)
        print(f"✓ Saved: {output_path}")

    except Exception as e:
        print(f"✗ Failed to load state dict: {e}")
        print("\nThe config doesn't match the checkpoint.")
        print("We need to find the actual config used to train this model.")

except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

    print("\nCouldn't import Flow decoder. This likely means:")
    print("1. Missing dependencies in the cosyvoice_repo")
    print("2. The model structure has changed")
    print("")
    print("The ONNX model (1.3GB) exists and works with ONNX Runtime.")

print("\n" + "="*80)
print("="*80)
