"""Debug traceable decoder step by step."""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states


def debug_step_by_step():
    print("Loading models...")
    model = TTSModel.load_model()
    mimi = model.mimi
    
    # Create traceable decoder
    traceable = TraceableMimiDecoder.from_mimi(mimi)
    traceable.eval()
    
    # Test input
    torch.manual_seed(42)
    latent = torch.randn(1, 512, 1)
    
    # Init states
    mimi_state = init_states(mimi, batch_size=1, sequence_length=100)
    trace_state = traceable.init_state(batch_size=1)
    
    print(f"\nInput latent: {latent.shape}")
    
    # ===================== UPSAMPLE =====================
    print("\n" + "="*60)
    print("STEP 1: UPSAMPLE")
    print("="*60)
    
    with torch.no_grad():
        orig_upsample = mimi.upsample(latent, mimi_state)
    print(f"Original upsample: {orig_upsample.shape}, range [{orig_upsample.min():.4f}, {orig_upsample.max():.4f}]")
    
    # Traceable upsample
    x, new_upsample_partial = traceable._streaming_convtr(
        latent, traceable.upsample_convtr, trace_state['upsample_partial'], 16
    )
    print(f"Traceable upsample: {x.shape}, range [{x.min():.4f}, {x.max():.4f}]")
    
    diff = (orig_upsample - x).abs()
    print(f"Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # ===================== TRANSFORMER =====================
    print("\n" + "="*60)
    print("STEP 2: TRANSFORMER")
    print("="*60)
    
    # Original transformer
    with torch.no_grad():
        (orig_trans_out,) = mimi.decoder_transformer(orig_upsample, mimi_state)
    print(f"Original transformer: {orig_trans_out.shape}, range [{orig_trans_out.min():.4f}, {orig_trans_out.max():.4f}]")
    
    # Traceable transformer
    x_trans = x.transpose(1, 2)  # [B, T, C]
    
    # Layer 0
    residual = x_trans
    x_norm = traceable.norm0_1(x_trans)
    attn_out, new_attn0_cache, new_attn0_offset, new_attn0_end_offset = traceable._streaming_attention(
        x_norm, traceable.attn0_in_proj, traceable.attn0_out_proj,
        trace_state['attn0_cache'], trace_state['attn0_offset'], trace_state['attn0_end_offset']
    )
    x_trans = residual + attn_out * traceable.gamma0_1
    
    residual = x_trans
    x_norm = traceable.norm0_2(x_trans)
    ffn_out = traceable.linear0_2(torch.nn.functional.gelu(traceable.linear0_1(x_norm)))
    x_trans = residual + ffn_out * traceable.gamma0_2
    
    # Layer 1
    residual = x_trans
    x_norm = traceable.norm1_1(x_trans)
    attn_out, new_attn1_cache, new_attn1_offset, new_attn1_end_offset = traceable._streaming_attention(
        x_norm, traceable.attn1_in_proj, traceable.attn1_out_proj,
        trace_state['attn1_cache'], trace_state['attn1_offset'], trace_state['attn1_end_offset']
    )
    x_trans = residual + attn_out * traceable.gamma1_1
    
    residual = x_trans
    x_norm = traceable.norm1_2(x_trans)
    ffn_out = traceable.linear1_2(torch.nn.functional.gelu(traceable.linear1_1(x_norm)))
    x_trans = residual + ffn_out * traceable.gamma1_2
    
    trace_trans_out = x_trans.transpose(1, 2)  # [B, C, T]
    
    print(f"Traceable transformer: {trace_trans_out.shape}, range [{trace_trans_out.min():.4f}, {trace_trans_out.max():.4f}]")
    
    diff = (orig_trans_out - trace_trans_out).abs()
    print(f"Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    if diff.max() > 0.01:
        print(">>> TRANSFORMER OUTPUT DIFFERS <<<")
    
    # ===================== SEANet DECODER =====================
    print("\n" + "="*60)
    print("STEP 3: SEANet DECODER")
    print("="*60)
    
    # Original decoder
    with torch.no_grad():
        orig_audio = mimi.decoder(orig_trans_out, mimi_state)
    print(f"Original decoder: {orig_audio.shape}, range [{orig_audio.min():.4f}, {orig_audio.max():.4f}]")
    
    # Traceable decoder (continue from transformer output)
    x = trace_trans_out
    
    # Conv0
    x, new_conv0_prev, new_conv0_first = traceable._streaming_conv(
        x, traceable.conv0, trace_state['conv0_prev'], trace_state['conv0_first'], 6
    )
    x = torch.nn.functional.elu(x, alpha=1.0)
    
    # ConvTr0
    x, new_convtr0_partial = traceable._streaming_convtr(x, traceable.convtr0, trace_state['convtr0_partial'], 6)
    x = torch.nn.functional.elu(x, alpha=1.0)
    
    # ResBlock0
    residual = x
    x_res = torch.nn.functional.elu(x, alpha=1.0)
    x_res, new_res0_conv0_prev, new_res0_conv0_first = traceable._streaming_conv(
        x_res, traceable.res0_conv0, trace_state['res0_conv0_prev'], trace_state['res0_conv0_first'], 2
    )
    x_res = torch.nn.functional.elu(x_res, alpha=1.0)
    x_res, new_res0_conv1_prev, new_res0_conv1_first = traceable._streaming_conv(
        x_res, traceable.res0_conv1, trace_state['res0_conv1_prev'], trace_state['res0_conv1_first'], 0
    )
    x = x + x_res
    
    # ConvTr1
    x, new_convtr1_partial = traceable._streaming_convtr(x, traceable.convtr1, trace_state['convtr1_partial'], 5)
    x = torch.nn.functional.elu(x, alpha=1.0)
    
    # ResBlock1
    residual = x
    x_res = torch.nn.functional.elu(x, alpha=1.0)
    x_res, new_res1_conv0_prev, new_res1_conv0_first = traceable._streaming_conv(
        x_res, traceable.res1_conv0, trace_state['res1_conv0_prev'], trace_state['res1_conv0_first'], 2
    )
    x_res = torch.nn.functional.elu(x_res, alpha=1.0)
    x_res, new_res1_conv1_prev, new_res1_conv1_first = traceable._streaming_conv(
        x_res, traceable.res1_conv1, trace_state['res1_conv1_prev'], trace_state['res1_conv1_first'], 0
    )
    x = x + x_res
    
    # ConvTr2
    x, new_convtr2_partial = traceable._streaming_convtr(x, traceable.convtr2, trace_state['convtr2_partial'], 4)
    x = torch.nn.functional.elu(x, alpha=1.0)
    
    # ResBlock2
    residual = x
    x_res = torch.nn.functional.elu(x, alpha=1.0)
    x_res, new_res2_conv0_prev, new_res2_conv0_first = traceable._streaming_conv(
        x_res, traceable.res2_conv0, trace_state['res2_conv0_prev'], trace_state['res2_conv0_first'], 2
    )
    x_res = torch.nn.functional.elu(x_res, alpha=1.0)
    x_res, new_res2_conv1_prev, new_res2_conv1_first = traceable._streaming_conv(
        x_res, traceable.res2_conv1, trace_state['res2_conv1_prev'], trace_state['res2_conv1_first'], 0
    )
    x = x + x_res
    
    # Final conv
    x, new_conv_final_prev, new_conv_final_first = traceable._streaming_conv(
        x, traceable.conv_final, trace_state['conv_final_prev'], trace_state['conv_final_first'], 2
    )
    
    trace_audio = x
    print(f"Traceable decoder: {trace_audio.shape}, range [{trace_audio.min():.4f}, {trace_audio.max():.4f}]")
    
    # Compare shapes
    min_len = min(orig_audio.shape[-1], trace_audio.shape[-1])
    orig_trimmed = orig_audio[..., :min_len]
    trace_trimmed = trace_audio[..., :min_len]
    
    diff = (orig_trimmed - trace_trimmed).abs()
    print(f"Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    corr = torch.corrcoef(torch.stack([orig_trimmed.flatten(), trace_trimmed.flatten()]))[0, 1]
    print(f"Correlation: {corr:.6f}")


if __name__ == "__main__":
    debug_step_by_step()
