"""Test CoreML models against PyTorch reference."""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_mimi_decoder():
    """Test CoreML Mimi decoder against PyTorch."""
    import coremltools as ct
    from pocket_tts import TTSModel
    from traceable_decoder import TraceableMimiDecoder

    print("=" * 60)
    print("Testing Mimi Decoder")
    print("=" * 60)

    # Load models
    print("Loading models...")
    model = TTSModel.load_model()
    mimi = model.mimi

    pytorch_decoder = TraceableMimiDecoder.from_mimi(mimi)
    pytorch_decoder.eval()

    coreml_decoder = ct.models.MLModel('mimi_decoder.mlpackage')

    # Initialize state
    state = pytorch_decoder.init_state(batch_size=1)

    # Test input
    torch.manual_seed(42)
    latent = torch.randn(1, 512, 1)

    # Prepare inputs
    args = (
        latent,
        state['upsample_partial'],
        state['attn0_cache'], state['attn0_offset'], state['attn0_end_offset'],
        state['attn1_cache'], state['attn1_offset'], state['attn1_end_offset'],
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

    # PyTorch inference
    print("Running PyTorch inference...")
    with torch.no_grad():
        pytorch_outputs = pytorch_decoder(*args)
    pytorch_audio = pytorch_outputs[0].numpy()

    # CoreML inference
    print("Running CoreML inference...")
    coreml_inputs = {
        'latent': latent.numpy(),
        'upsample_partial': state['upsample_partial'].numpy(),
        'attn0_cache': state['attn0_cache'].numpy(),
        'attn0_offset': state['attn0_offset'].numpy(),
        'attn0_end_offset': state['attn0_end_offset'].numpy(),
        'attn1_cache': state['attn1_cache'].numpy(),
        'attn1_offset': state['attn1_offset'].numpy(),
        'attn1_end_offset': state['attn1_end_offset'].numpy(),
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

    # Find audio output (first output)
    coreml_audio = list(coreml_outputs.values())[0]

    # Compare
    print(f"\nPyTorch audio shape: {pytorch_audio.shape}")
    print(f"CoreML audio shape: {coreml_audio.shape}")

    max_diff = np.max(np.abs(pytorch_audio - coreml_audio))
    mean_diff = np.mean(np.abs(pytorch_audio - coreml_audio))

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Mimi Decoder: CoreML output matches PyTorch!")
        return True
    else:
        print(f"⚠ Mimi Decoder: Outputs differ by {max_diff}")
        return False


def test_flowlm_backbone():
    """Test CoreML FlowLM backbone against PyTorch."""
    import coremltools as ct
    from pocket_tts import TTSModel
    from traceable_flowlm import TraceableFlowLMBackbone

    print("\n" + "=" * 60)
    print("Testing FlowLM Backbone")
    print("=" * 60)

    # Load models
    print("Loading models...")
    model = TTSModel.load_model()
    flow_lm = model.flow_lm

    max_seq_len = 100
    pytorch_backbone = TraceableFlowLMBackbone.from_flowlm(flow_lm, max_seq_len=max_seq_len)
    pytorch_backbone.eval()

    coreml_backbone = ct.models.MLModel('flowlm_backbone.mlpackage')

    # Initialize state
    state = pytorch_backbone.init_state(batch_size=1)

    # Test inputs
    torch.manual_seed(42)
    sequence = torch.randn(1, 1, 32)
    text_embeddings = torch.randn(1, 5, 1024)
    bos_emb = flow_lm.bos_emb.data

    # PyTorch inference
    print("Running PyTorch inference...")
    args = (
        sequence,
        text_embeddings,
        bos_emb,
        state['cache0'], state['position0'],
        state['cache1'], state['position1'],
        state['cache2'], state['position2'],
        state['cache3'], state['position3'],
        state['cache4'], state['position4'],
        state['cache5'], state['position5'],
    )

    with torch.no_grad():
        pytorch_outputs = pytorch_backbone(*args)
    pytorch_out = pytorch_outputs[0].numpy()
    pytorch_eos = pytorch_outputs[1].numpy()

    # CoreML inference
    print("Running CoreML inference...")
    coreml_inputs = {
        'sequence': sequence.numpy(),
        'text_embeddings': text_embeddings.numpy(),
        'bos_emb': bos_emb.numpy(),
        'cache0': state['cache0'].numpy(),
        'position0': state['position0'].numpy(),
        'cache1': state['cache1'].numpy(),
        'position1': state['position1'].numpy(),
        'cache2': state['cache2'].numpy(),
        'position2': state['position2'].numpy(),
        'cache3': state['cache3'].numpy(),
        'position3': state['position3'].numpy(),
        'cache4': state['cache4'].numpy(),
        'position4': state['position4'].numpy(),
        'cache5': state['cache5'].numpy(),
        'position5': state['position5'].numpy(),
    }

    coreml_outputs = coreml_backbone.predict(coreml_inputs)

    # Find transformer output and eos (first two outputs)
    output_keys = list(coreml_outputs.keys())
    coreml_out = coreml_outputs[output_keys[0]]
    coreml_eos = coreml_outputs[output_keys[1]]

    # Compare
    print(f"\nPyTorch output shape: {pytorch_out.shape}")
    print(f"CoreML output shape: {coreml_out.shape}")

    max_diff = np.max(np.abs(pytorch_out - coreml_out))
    mean_diff = np.mean(np.abs(pytorch_out - coreml_out))

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ FlowLM Backbone: CoreML output matches PyTorch!")
        return True
    else:
        print(f"⚠ FlowLM Backbone: Outputs differ by {max_diff}")
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    decoder_ok = test_mimi_decoder()
    backbone_ok = test_flowlm_backbone()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Mimi Decoder: {'PASS' if decoder_ok else 'FAIL'}")
    print(f"FlowLM Backbone: {'PASS' if backbone_ok else 'FAIL'}")
