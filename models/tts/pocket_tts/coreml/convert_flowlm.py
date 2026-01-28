"""Convert FlowLM backbone to CoreML."""
import torch
import numpy as np
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_flowlm import TraceableFlowLMBackbone


def convert_flowlm_to_coreml():
    print("Loading original PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    flow_lm = model.flow_lm

    print("Creating traceable FlowLM backbone...")
    max_seq_len = 200  # Smaller for testing
    backbone = TraceableFlowLMBackbone.from_flowlm(flow_lm, max_seq_len=max_seq_len)
    backbone.eval()

    print("Initializing state...")
    state = backbone.init_state(batch_size=1)

    # Create example inputs
    print("Creating example inputs...")
    T_text = 150  # Fixed: voice (~125) + text (~25) padded
    sequence = torch.randn(1, 1, 32)  # Single latent frame
    text_embeddings = torch.randn(1, T_text, 1024)
    bos_emb = flow_lm.bos_emb.data

    example_inputs = (
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

    # Test PyTorch forward
    print("Testing PyTorch forward pass...")
    with torch.no_grad():
        pytorch_outputs = backbone(*example_inputs)
    print(f"  Transformer output: {pytorch_outputs[0].shape}")
    print(f"  EOS output: {pytorch_outputs[1].shape}")

    # Trace the model
    print("\nTracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(backbone, example_inputs)

    print("Converting to CoreML...")

    B = 1
    T = 1
    H = 16
    D = 64

    inputs = [
        ct.TensorType(name="sequence", shape=(B, T, 32)),
        ct.TensorType(name="text_embeddings", shape=(B, T_text, 1024)),
        ct.TensorType(name="bos_emb", shape=(32,)),
    ]

    # Add cache and position inputs for each layer
    for i in range(6):
        inputs.append(ct.TensorType(name=f"cache{i}", shape=(2, B, max_seq_len, H, D)))
        inputs.append(ct.TensorType(name=f"position{i}", shape=(B,)))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "flowlm_backbone_v3.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    print(f"Saved to {output_path}")

    # Print input/output info
    spec = mlmodel.get_spec()
    print("\n=== INPUTS ===")
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = list(inp.type.multiArrayType.shape)
            print(f"  {inp.name}: {shape}")
    print("\n=== OUTPUTS ===")
    for out in spec.description.output:
        if out.type.HasField('multiArrayType'):
            shape = list(out.type.multiArrayType.shape)
            print(f"  {out.name}: {shape}")

    return output_path


if __name__ == "__main__":
    convert_flowlm_to_coreml()
