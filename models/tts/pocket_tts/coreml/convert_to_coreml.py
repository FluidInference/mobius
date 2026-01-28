"""Convert traceable decoder to CoreML with fixed RoPE."""
import torch
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder


def convert_mimi_decoder():
    """Convert Mimi decoder to CoreML."""
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    
    print("Creating traceable decoder...")
    decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    decoder.eval()
    
    # Create example inputs
    print("Creating example inputs...")
    state = decoder.init_state(batch_size=1)
    
    example_inputs = (
        torch.randn(1, 512, 1),  # latent
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
    
    # Trace
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, example_inputs)
    
    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="latent", shape=(1, 512, 1)),
            ct.TensorType(name="upsample_partial", shape=state['upsample_partial'].shape),
            ct.TensorType(name="attn0_cache", shape=state['attn0_cache'].shape),
            ct.TensorType(name="attn0_offset", shape=(1,)),
            ct.TensorType(name="attn0_end_offset", shape=(1,)),
            ct.TensorType(name="attn1_cache", shape=state['attn1_cache'].shape),
            ct.TensorType(name="attn1_offset", shape=(1,)),
            ct.TensorType(name="attn1_end_offset", shape=(1,)),
            ct.TensorType(name="conv0_prev", shape=state['conv0_prev'].shape),
            ct.TensorType(name="conv0_first", shape=(1,)),
            ct.TensorType(name="convtr0_partial", shape=state['convtr0_partial'].shape),
            ct.TensorType(name="res0_conv0_prev", shape=state['res0_conv0_prev'].shape),
            ct.TensorType(name="res0_conv0_first", shape=(1,)),
            ct.TensorType(name="res0_conv1_prev", shape=state['res0_conv1_prev'].shape),
            ct.TensorType(name="res0_conv1_first", shape=(1,)),
            ct.TensorType(name="convtr1_partial", shape=state['convtr1_partial'].shape),
            ct.TensorType(name="res1_conv0_prev", shape=state['res1_conv0_prev'].shape),
            ct.TensorType(name="res1_conv0_first", shape=(1,)),
            ct.TensorType(name="res1_conv1_prev", shape=state['res1_conv1_prev'].shape),
            ct.TensorType(name="res1_conv1_first", shape=(1,)),
            ct.TensorType(name="convtr2_partial", shape=state['convtr2_partial'].shape),
            ct.TensorType(name="res2_conv0_prev", shape=state['res2_conv0_prev'].shape),
            ct.TensorType(name="res2_conv0_first", shape=(1,)),
            ct.TensorType(name="res2_conv1_prev", shape=state['res2_conv1_prev'].shape),
            ct.TensorType(name="res2_conv1_first", shape=(1,)),
            ct.TensorType(name="conv_final_prev", shape=state['conv_final_prev'].shape),
            ct.TensorType(name="conv_final_first", shape=(1,)),
        ],
        minimum_deployment_target=ct.target.macOS13,
    )
    
    # Save
    output_path = "mimi_decoder_v2.mlpackage"
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")
    
    return mlmodel


if __name__ == "__main__":
    convert_mimi_decoder()
