"""Debug traceable decoder by comparing layer outputs."""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder
from pocket_tts.modules.stateful_module import init_states, increment_steps


def compare_single_frame():
    """Compare a single frame decode between original and traceable."""
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    mimi = model.mimi
    
    # Create test input - random 512-dim latent
    torch.manual_seed(42)
    test_latent = torch.randn(1, 512, 1)
    
    print(f"Test latent shape: {test_latent.shape}")
    print(f"Test latent range: [{test_latent.min():.3f}, {test_latent.max():.3f}]")
    
    # Initialize original mimi state
    print("\n--- ORIGINAL MIMI ---")
    mimi_state = init_states(mimi, batch_size=1, sequence_length=100)
    
    # Decode with original
    with torch.no_grad():
        orig_audio = mimi.decode_from_latent(test_latent, mimi_state)
    
    print(f"Original audio shape: {orig_audio.shape}")
    print(f"Original audio range: [{orig_audio.min():.4f}, {orig_audio.max():.4f}]")
    
    # Create traceable decoder
    print("\n--- TRACEABLE DECODER ---")
    traceable = TraceableMimiDecoder.from_mimi(mimi)
    traceable.eval()
    trace_state = traceable.init_state(batch_size=1)
    
    # Decode with traceable
    with torch.no_grad():
        outputs = traceable(
            test_latent,
            trace_state['upsample_partial'],
            trace_state['attn0_cache'], trace_state['attn0_offset'], trace_state['attn0_end_offset'],
            trace_state['attn1_cache'], trace_state['attn1_offset'], trace_state['attn1_end_offset'],
            trace_state['conv0_prev'], trace_state['conv0_first'],
            trace_state['convtr0_partial'],
            trace_state['res0_conv0_prev'], trace_state['res0_conv0_first'],
            trace_state['res0_conv1_prev'], trace_state['res0_conv1_first'],
            trace_state['convtr1_partial'],
            trace_state['res1_conv0_prev'], trace_state['res1_conv0_first'],
            trace_state['res1_conv1_prev'], trace_state['res1_conv1_first'],
            trace_state['convtr2_partial'],
            trace_state['res2_conv0_prev'], trace_state['res2_conv0_first'],
            trace_state['res2_conv1_prev'], trace_state['res2_conv1_first'],
            trace_state['conv_final_prev'], trace_state['conv_final_first'],
        )
    
    trace_audio = outputs[0]
    print(f"Traceable audio shape: {trace_audio.shape}")
    print(f"Traceable audio range: [{trace_audio.min():.4f}, {trace_audio.max():.4f}]")
    
    # Compare
    print("\n--- COMPARISON ---")
    orig_np = orig_audio.numpy().flatten()
    trace_np = trace_audio.numpy().flatten()
    
    min_len = min(len(orig_np), len(trace_np))
    orig_np = orig_np[:min_len]
    trace_np = trace_np[:min_len]
    
    diff = orig_np - trace_np
    print(f"Max diff: {np.max(np.abs(diff)):.6f}")
    print(f"Mean diff: {np.mean(np.abs(diff)):.6f}")
    print(f"Correlation: {np.corrcoef(orig_np, trace_np)[0,1]:.6f}")
    
    # Check first few samples
    print("\nFirst 10 samples:")
    print(f"Original:  {orig_np[:10]}")
    print(f"Traceable: {trace_np[:10]}")
    
    return orig_audio, trace_audio


def debug_upsample():
    """Debug just the upsample layer."""
    print("\n" + "="*60)
    print("DEBUGGING UPSAMPLE LAYER")
    print("="*60)
    
    from pocket_tts import TTSModel
    from coreml_modules import CoreMLStreamingConvTranspose1d
    
    model = TTSModel.load_model()
    mimi = model.mimi
    
    # Get original upsample
    orig_upsample = mimi.upsample
    
    # Create CoreML version
    coreml_convtr = CoreMLStreamingConvTranspose1d(orig_upsample.convtr.convtr)
    
    # Test input
    torch.manual_seed(42)
    x = torch.randn(1, 512, 1)
    
    # Original upsample
    mimi_state = init_states(mimi, batch_size=1, sequence_length=100)
    with torch.no_grad():
        orig_out = orig_upsample(x, mimi_state)
    
    print(f"Original upsample output: {orig_out.shape}, range [{orig_out.min():.4f}, {orig_out.max():.4f}]")
    
    # CoreML upsample
    partial = coreml_convtr.init_state(batch_size=1)
    with torch.no_grad():
        coreml_out, new_partial = coreml_convtr(x, partial)
    
    print(f"CoreML upsample output: {coreml_out.shape}, range [{coreml_out.min():.4f}, {coreml_out.max():.4f}]")
    
    # Compare
    min_len = min(orig_out.shape[-1], coreml_out.shape[-1])
    diff = (orig_out[..., :min_len] - coreml_out[..., :min_len]).abs()
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    
    return orig_out, coreml_out


def debug_attention():
    """Debug attention layer."""
    print("\n" + "="*60)
    print("DEBUGGING ATTENTION LAYER")
    print("="*60)
    
    from pocket_tts import TTSModel
    from coreml_modules import CoreMLStreamingAttention
    
    model = TTSModel.load_model()
    mimi = model.mimi
    
    # Get original attention (first layer of decoder_transformer)
    orig_transformer = mimi.decoder_transformer.transformer
    orig_layer = orig_transformer.layers[0]
    
    # Create CoreML attention
    coreml_attn = CoreMLStreamingAttention(
        embed_dim=512,
        num_heads=8,
        capacity=256,
        context=256,
        rope_max_period=orig_transformer.max_period
    )
    # Copy weights
    coreml_attn.in_proj.weight.data.copy_(orig_layer.self_attn.in_proj.weight.data)
    coreml_attn.out_proj.weight.data.copy_(orig_layer.self_attn.out_proj.weight.data)
    
    # Test input
    torch.manual_seed(42)
    x = torch.randn(1, 8, 512)  # [B, T, D]
    
    # Original attention
    mimi_state = init_states(mimi, batch_size=1, sequence_length=100)
    # Get the transformer layer's state
    layer_state = mimi_state.get('decoder_transformer.transformer.layers.0.self_attn', {})
    
    with torch.no_grad():
        orig_out = orig_layer.self_attn(x, layer_state)
    
    print(f"Original attn output: {orig_out.shape}, range [{orig_out.min():.4f}, {orig_out.max():.4f}]")
    
    # CoreML attention
    cache, offset, end_offset = coreml_attn.init_state(batch_size=1)
    with torch.no_grad():
        coreml_out, new_cache, new_offset, new_end_offset = coreml_attn(x, cache, offset, end_offset)
    
    print(f"CoreML attn output: {coreml_out.shape}, range [{coreml_out.min():.4f}, {coreml_out.max():.4f}]")
    
    # Compare
    diff = (orig_out - coreml_out).abs()
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    
    return orig_out, coreml_out


if __name__ == "__main__":
    debug_upsample()
    debug_attention()
    compare_single_frame()
