"""Debug attention layer specifically."""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states


def debug_attention():
    print("Loading models...")
    model = TTSModel.load_model()
    mimi = model.mimi
    
    # Create traceable decoder
    traceable = TraceableMimiDecoder.from_mimi(mimi)
    traceable.eval()
    
    # Get the original transformer layer
    orig_layer = mimi.decoder_transformer.transformer.layers[0]
    
    # Test input - same as what upsample would produce
    torch.manual_seed(42)
    x = torch.randn(1, 16, 512)  # [B, T, C] after upsample transpose
    
    # Init states
    mimi_state = init_states(mimi, batch_size=1, sequence_length=100)
    trace_state = traceable.init_state(batch_size=1)
    
    print(f"Input x: {x.shape}")
    
    # ===================== CHECK WEIGHTS =====================
    print("\n--- WEIGHT CHECK ---")
    w1 = orig_layer.self_attn.in_proj.weight
    w2 = traceable.attn0_in_proj.weight
    print(f"in_proj weight match: {torch.allclose(w1, w2)}")
    
    w1 = orig_layer.self_attn.out_proj.weight
    w2 = traceable.attn0_out_proj.weight
    print(f"out_proj weight match: {torch.allclose(w1, w2)}")
    
    # ===================== ORIGINAL ATTENTION =====================
    print("\n--- ORIGINAL ATTENTION ---")
    
    # Get layer state from mimi_state
    # The state key is based on module path
    layer_prefix = 'decoder_transformer.transformer.layers.0.self_attn'
    orig_attn_state = mimi_state.get(layer_prefix, {})
    
    # Original self-attention
    with torch.no_grad():
        orig_attn_out = orig_layer.self_attn(x, mimi_state)
    
    print(f"Original attention output: {orig_attn_out.shape}")
    print(f"Output range: [{orig_attn_out.min():.4f}, {orig_attn_out.max():.4f}]")
    
    # ===================== TRACEABLE ATTENTION =====================
    print("\n--- TRACEABLE ATTENTION ---")
    
    with torch.no_grad():
        trace_attn_out, _, _, _ = traceable._streaming_attention(
            x, 
            traceable.attn0_in_proj, 
            traceable.attn0_out_proj,
            trace_state['attn0_cache'], 
            trace_state['attn0_offset'], 
            trace_state['attn0_end_offset']
        )
    
    print(f"Traceable attention output: {trace_attn_out.shape}")
    print(f"Output range: [{trace_attn_out.min():.4f}, {trace_attn_out.max():.4f}]")
    
    # ===================== COMPARE =====================
    print("\n--- COMPARISON ---")
    diff = (orig_attn_out - trace_attn_out).abs()
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    
    # Check QKV projections
    print("\n--- QKV CHECK ---")
    qkv_orig = orig_layer.self_attn.in_proj(x)
    qkv_trace = traceable.attn0_in_proj(x)
    print(f"QKV diff: {(qkv_orig - qkv_trace).abs().max():.6f}")
    
    # Check if the issue is in RoPE or attention masking
    print("\n--- DETAILED DEBUG ---")
    B, T, _ = x.shape
    H = 8
    D = 64
    
    qkv = traceable.attn0_in_proj(x).reshape(B, T, 3, H, D)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    print(f"Q shape: {q.shape}")
    print(f"Q range: [{q.min():.4f}, {q.max():.4f}]")
    
    # Check original attention's internal state
    print("\nOriginal attention state keys:", list(mimi_state.keys())[:5])


if __name__ == "__main__":
    debug_attention()
