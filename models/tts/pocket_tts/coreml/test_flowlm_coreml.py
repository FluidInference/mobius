"""Test CoreML FlowLM against PyTorch."""
import torch
import numpy as np
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_flowlm import TraceableFlowLMBackbone


def test_flowlm_coreml():
    print("Loading PyTorch model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    flow_lm = model.flow_lm

    max_seq_len = 200
    pytorch_backbone = TraceableFlowLMBackbone.from_flowlm(flow_lm, max_seq_len=max_seq_len)
    pytorch_backbone.eval()

    print("Loading CoreML model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    coreml_backbone = ct.models.MLModel(
        os.path.join(script_dir, 'flowlm_backbone_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )

    # Initialize state with zeros (NaN would propagate through matmul before masking)
    state = {}
    for i in range(6):
        state[f'cache{i}'] = torch.zeros(2, 1, max_seq_len, 16, 64)
        state[f'position{i}'] = torch.zeros(1)

    # Create test inputs
    torch.manual_seed(42)
    sequence = torch.randn(1, 1, 32)
    text_embeddings = torch.randn(1, 10, 1024)
    bos_emb = flow_lm.bos_emb.data

    # PyTorch forward
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        pytorch_outputs = pytorch_backbone(
            sequence, text_embeddings, bos_emb,
            state['cache0'], state['position0'],
            state['cache1'], state['position1'],
            state['cache2'], state['position2'],
            state['cache3'], state['position3'],
            state['cache4'], state['position4'],
            state['cache5'], state['position5'],
        )

    pytorch_transformer = pytorch_outputs[0].numpy()
    pytorch_eos = pytorch_outputs[1].numpy()

    # CoreML forward
    print("Running CoreML inference...")
    coreml_inputs = {
        'sequence': sequence.numpy().astype(np.float32),
        'text_embeddings': text_embeddings.numpy().astype(np.float32),
        'bos_emb': bos_emb.numpy().astype(np.float32),
        'cache0': state['cache0'].numpy().astype(np.float32),
        'position0': state['position0'].numpy().astype(np.float32),
        'cache1': state['cache1'].numpy().astype(np.float32),
        'position1': state['position1'].numpy().astype(np.float32),
        'cache2': state['cache2'].numpy().astype(np.float32),
        'position2': state['position2'].numpy().astype(np.float32),
        'cache3': state['cache3'].numpy().astype(np.float32),
        'position3': state['position3'].numpy().astype(np.float32),
        'cache4': state['cache4'].numpy().astype(np.float32),
        'position4': state['position4'].numpy().astype(np.float32),
        'cache5': state['cache5'].numpy().astype(np.float32),
        'position5': state['position5'].numpy().astype(np.float32),
    }

    coreml_outputs = coreml_backbone.predict(coreml_inputs)

    # Find transformer output (shape [1, 1, 1024])
    coreml_transformer = coreml_outputs['input']  # renamed from 'transformer_out'
    coreml_eos = coreml_outputs['var_2414']  # EOS output

    print(f"\nPyTorch transformer shape: {pytorch_transformer.shape}")
    print(f"CoreML transformer shape: {coreml_transformer.shape}")
    print(f"PyTorch EOS shape: {pytorch_eos.shape}")
    print(f"CoreML EOS shape: {coreml_eos.shape}")

    # Compare transformer output
    max_diff_trans = np.max(np.abs(pytorch_transformer - coreml_transformer))
    mean_diff_trans = np.mean(np.abs(pytorch_transformer - coreml_transformer))
    corr_trans = np.corrcoef(pytorch_transformer.flatten(), coreml_transformer.flatten())[0, 1]

    print(f"\nTransformer output:")
    print(f"  Max difference: {max_diff_trans:.6f}")
    print(f"  Mean difference: {mean_diff_trans:.6f}")
    print(f"  Correlation: {corr_trans:.6f}")

    # Compare EOS output
    max_diff_eos = np.max(np.abs(pytorch_eos - coreml_eos))
    print(f"\nEOS output:")
    print(f"  Max difference: {max_diff_eos:.6f}")
    print(f"  PyTorch: {pytorch_eos.flatten()[0]:.4f}")
    print(f"  CoreML: {coreml_eos.flatten()[0]:.4f}")

    if corr_trans > 0.99:
        print("\n✓ CoreML FlowLM output matches PyTorch!")
        return True
    else:
        print(f"\n⚠ Outputs differ (corr={corr_trans:.4f})")
        return False


if __name__ == "__main__":
    test_flowlm_coreml()
