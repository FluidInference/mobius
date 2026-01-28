"""Test CoreML model against PyTorch."""
import torch
import numpy as np
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder


def test_mimi_decoder():
    print("Loading models...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    
    # PyTorch
    pytorch_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    pytorch_decoder.eval()
    
    # CoreML
    print("Loading CoreML model...")
    coreml_decoder = ct.models.MLModel('mimi_decoder_v2.mlpackage')
    
    # Initialize state
    state = pytorch_decoder.init_state(batch_size=1)
    
    # Test input
    torch.manual_seed(42)
    latent = torch.randn(1, 512, 1)
    
    # PyTorch inference
    print("Running PyTorch inference...")
    args = (
        latent,
        state['upsample_partial'],
        state['attn0_cache'], state['attn0_offset'].float(), state['attn0_end_offset'].float(),
        state['attn1_cache'], state['attn1_offset'].float(), state['attn1_end_offset'].float(),
        state['conv0_prev'], state['conv0_first'],
        state['convtr0_partial'],
        state['res0_conv0_prev'], state['res0_conv0_first'],
        state['res0_conv1_prev'], state['res0_conv1_first'],
        state['convtr1_partial'],
        state['res1_conv0_prev'], state['res1_conv0_first'],
        state['res1_conv1_prev'], state['res1_conv1_first'],
        state['convtr2_partial'],
        state['res2_conv0_prev'], state['res2_conv0_first'],
        state['res2_conv1_prev'], state['res2_conv1_first'],
        state['conv_final_prev'], state['conv_final_first'],
    )
    
    with torch.no_grad():
        pytorch_outputs = pytorch_decoder(*args)
    pytorch_audio = pytorch_outputs[0].numpy()
    
    # CoreML inference
    print("Running CoreML inference...")
    coreml_inputs = {
        'latent': latent.numpy(),
        'upsample_partial': state['upsample_partial'].numpy(),
        'attn0_cache': state['attn0_cache'].numpy(),
        'attn0_offset': np.array([0.0], dtype=np.float32),
        'attn0_end_offset': np.array([0.0], dtype=np.float32),
        'attn1_cache': state['attn1_cache'].numpy(),
        'attn1_offset': np.array([0.0], dtype=np.float32),
        'attn1_end_offset': np.array([0.0], dtype=np.float32),
        'conv0_prev': state['conv0_prev'].numpy(),
        'conv0_first': state['conv0_first'].numpy(),
        'convtr0_partial': state['convtr0_partial'].numpy(),
        'res0_conv0_prev': state['res0_conv0_prev'].numpy(),
        'res0_conv0_first': state['res0_conv0_first'].numpy(),
        'res0_conv1_prev': state['res0_conv1_prev'].numpy(),
        'res0_conv1_first': state['res0_conv1_first'].numpy(),
        'convtr1_partial': state['convtr1_partial'].numpy(),
        'res1_conv0_prev': state['res1_conv0_prev'].numpy(),
        'res1_conv0_first': state['res1_conv0_first'].numpy(),
        'res1_conv1_prev': state['res1_conv1_prev'].numpy(),
        'res1_conv1_first': state['res1_conv1_first'].numpy(),
        'convtr2_partial': state['convtr2_partial'].numpy(),
        'res2_conv0_prev': state['res2_conv0_prev'].numpy(),
        'res2_conv0_first': state['res2_conv0_first'].numpy(),
        'res2_conv1_prev': state['res2_conv1_prev'].numpy(),
        'res2_conv1_first': state['res2_conv1_first'].numpy(),
        'conv_final_prev': state['conv_final_prev'].numpy(),
        'conv_final_first': state['conv_final_first'].numpy(),
    }
    
    coreml_outputs = coreml_decoder.predict(coreml_inputs)
    
    # Find audio output
    output_keys = list(coreml_outputs.keys())
    print(f"CoreML output keys: {output_keys[:5]}...")
    
    # First output should be audio
    coreml_audio = list(coreml_outputs.values())[0]
    
    print(f"\nPyTorch audio shape: {pytorch_audio.shape}")
    print(f"CoreML audio shape: {coreml_audio.shape}")
    
    # Compare
    max_diff = np.max(np.abs(pytorch_audio - coreml_audio))
    mean_diff = np.mean(np.abs(pytorch_audio - coreml_audio))
    
    print(f"\nMax difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("\n✓ CoreML output matches PyTorch!")
        return True
    else:
        print(f"\n⚠ Outputs differ by {max_diff:.6f}")
        return False


if __name__ == "__main__":
    test_mimi_decoder()
