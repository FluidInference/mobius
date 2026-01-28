"""Test CoreML model on CPU only."""
import torch
import numpy as np
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder


def test_mimi_decoder():
    print("Loading PyTorch model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    
    pytorch_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    pytorch_decoder.eval()
    
    # Load CoreML with CPU only
    print("Loading CoreML model (CPU only)...")
    coreml_decoder = ct.models.MLModel(
        'mimi_decoder_v2.mlpackage',
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    
    # Initialize state
    state = pytorch_decoder.init_state(batch_size=1)
    
    # Test input
    torch.manual_seed(42)
    latent = torch.randn(1, 512, 1)
    
    # Prepare inputs
    print("Preparing inputs...")
    coreml_inputs = {
        'latent': latent.numpy().astype(np.float32),
        'upsample_partial': state['upsample_partial'].numpy().astype(np.float32),
        'attn0_cache': state['attn0_cache'].numpy().astype(np.float32),
        'attn0_offset': np.array([0.0], dtype=np.float32),
        'attn0_end_offset': np.array([0.0], dtype=np.float32),
        'attn1_cache': state['attn1_cache'].numpy().astype(np.float32),
        'attn1_offset': np.array([0.0], dtype=np.float32),
        'attn1_end_offset': np.array([0.0], dtype=np.float32),
        'conv0_prev': state['conv0_prev'].numpy().astype(np.float32),
        'conv0_first': state['conv0_first'].numpy().astype(np.float32),
        'convtr0_partial': state['convtr0_partial'].numpy().astype(np.float32),
        'res0_conv0_prev': state['res0_conv0_prev'].numpy().astype(np.float32),
        'res0_conv0_first': state['res0_conv0_first'].numpy().astype(np.float32),
        'res0_conv1_prev': state['res0_conv1_prev'].numpy().astype(np.float32),
        'res0_conv1_first': state['res0_conv1_first'].numpy().astype(np.float32),
        'convtr1_partial': state['convtr1_partial'].numpy().astype(np.float32),
        'res1_conv0_prev': state['res1_conv0_prev'].numpy().astype(np.float32),
        'res1_conv0_first': state['res1_conv0_first'].numpy().astype(np.float32),
        'res1_conv1_prev': state['res1_conv1_prev'].numpy().astype(np.float32),
        'res1_conv1_first': state['res1_conv1_first'].numpy().astype(np.float32),
        'convtr2_partial': state['convtr2_partial'].numpy().astype(np.float32),
        'res2_conv0_prev': state['res2_conv0_prev'].numpy().astype(np.float32),
        'res2_conv0_first': state['res2_conv0_first'].numpy().astype(np.float32),
        'res2_conv1_prev': state['res2_conv1_prev'].numpy().astype(np.float32),
        'res2_conv1_first': state['res2_conv1_first'].numpy().astype(np.float32),
        'conv_final_prev': state['conv_final_prev'].numpy().astype(np.float32),
        'conv_final_first': state['conv_final_first'].numpy().astype(np.float32),
    }
    
    print("Running CoreML inference...")
    coreml_outputs = coreml_decoder.predict(coreml_inputs)
    
    print(f"Success! Output keys: {list(coreml_outputs.keys())[:3]}...")
    coreml_audio = list(coreml_outputs.values())[0]
    print(f"Audio shape: {coreml_audio.shape}")
    print(f"Audio range: [{coreml_audio.min():.4f}, {coreml_audio.max():.4f}]")


if __name__ == "__main__":
    test_mimi_decoder()
