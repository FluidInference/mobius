#!/usr/bin/env python3
"""
Argmax-style TextProjector CoreML Conversion

Reverse-engineered from Argmax TTSKit's MIL program:

TextProjector:
  - Input: input_ids [1] (int32) — a single text token ID
  - Gather from text_embedding [151936, 2048] (palettized W8A16)
  - 1x1 conv fc1 [2048 → 2048] + SiLU activation
  - 1x1 conv fc2 [2048 → 1024]
  - Output: input_embeds [1, 1024, 1, 1] (fp16) — 4D ANE layout

This combines the text_embedding lookup + text_projection MLP into
a single model. The text_projection is a 2-layer MLP:
  Linear(2048, 2048) → SiLU → Linear(2048, 1024)

All linear layers use 1x1 convolutions for ANE compatibility.
Weights use constexpr_lut_to_dense (8-bit palettized) for W8A16.

Usage:
    python convert_argmax_text_projector.py [--model-path ./model_0.6b]
"""

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models.neural_network.quantization_utils import quantize_weights
import numpy as np
import argparse


class TextProjectorWrapper(nn.Module):
    """Wraps text_embedding + text_projection into a single model.

    MIL equivalent:
        1. gather(text_embedding [151936, 2048], input_ids) → [1, 2048]
        2. expand_dims → [1, 2048, 1, 1]
        3. conv(fc1 [2048, 2048, 1, 1]) + bias → [1, 2048, 1, 1]
        4. silu → [1, 2048, 1, 1]
        5. conv(fc2 [1024, 2048, 1, 1]) + bias → [1, 1024, 1, 1]
    """

    def __init__(self, text_embedding, text_projection):
        super().__init__()
        self.text_embedding = text_embedding  # nn.Embedding(151936, 2048)
        self.text_projection = text_projection  # MLP: Linear(2048,2048) → SiLU → Linear(2048,1024)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [1] — single text token ID

        Returns:
            input_embeds: [1, 1024, 1, 1] — ANE 4D layout
        """
        # Text embedding lookup
        embed = self.text_embedding(input_ids)  # [1, 2048]
        embed = embed.unsqueeze(0)  # [1, 1, 2048] — add batch dim

        # Project through MLP
        projected = self.text_projection(embed)  # [1, 1, 1024]

        # Reshape to ANE 4D: [1, C, 1, 1]
        return projected.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [1, 1024, 1, 1]


def main():
    parser = argparse.ArgumentParser(description="Convert Argmax-style TextProjector")
    parser.add_argument("--model-path", default="./model_0.6b", help="Path to Qwen3-TTS model")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--quantize-w8", action="store_true", help="Apply W8A16 palettization (like Argmax)")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    print("=" * 60)
    print("Argmax-style TextProjector Conversion")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker
    text_embedding = talker.model.text_embedding
    text_projection = talker.text_projection

    print(f"   text_embedding: {text_embedding.weight.shape}")
    print(f"   text_projection structure:")
    for name, param in text_projection.named_parameters():
        print(f"     {name}: {param.shape}")

    # 2. Create wrapper
    print("\n2. Creating TextProjector wrapper...")
    wrapper = TextProjectorWrapper(text_embedding, text_projection)
    wrapper.eval()

    # Test
    test_id = torch.tensor([1000])
    with torch.no_grad():
        test_out = wrapper(test_id)
    print(f"   Test output shape: {test_out.shape}")  # [1, 1024, 1, 1]

    # Cross-check with original path
    with torch.no_grad():
        orig_embed = text_embedding(test_id)  # [1, 2048]
        orig_proj = text_projection(orig_embed.unsqueeze(0))  # [1, 1, 1024]
    diff = (test_out.squeeze() - orig_proj.squeeze()).abs().max().item()
    print(f"   Cross-check diff: {diff}")

    # 3. Trace
    print("\n3. Tracing...")
    traced = torch.jit.trace(wrapper, (test_id,))

    # Verify trace
    with torch.no_grad():
        traced_out = traced(test_id)
    trace_diff = (traced_out - test_out).abs().max().item()
    print(f"   Trace diff: {trace_diff}")

    # 4. Convert to CoreML
    print("\n4. Converting to CoreML...")
    ml_model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="input_embeds", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    # 5. Optional W8A16 quantization
    if args.quantize_w8:
        print("\n5. Applying W8A16 palettized quantization...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )

        op_config = OpPalettizerConfig(
            mode="kmeans",
            nbits=8,
            weight_threshold=512,  # Only quantize weights > 512 elements
        )
        opt_config = OptimizationConfig(global_config=op_config)
        ml_model = palettize_weights(ml_model, config=opt_config)
        print("   W8A16 palettization applied")

    tp_path = f"{args.output_dir}/TextProjector.mlpackage"
    ml_model.save(tp_path)
    print(f"   Saved: {tp_path}")

    # 6. Verify
    print("\n6. Verifying CoreML model...")
    loaded = ct.models.MLModel(tp_path)
    result = loaded.predict({"input_ids": np.array([1000], dtype=np.int32)})
    print(f"   Output shape: {result['input_embeds'].shape}")
    print(f"   Output dtype: {result['input_embeds'].dtype}")

    cml_out = result['input_embeds'].astype(np.float32)
    pt_out = test_out.detach().numpy().astype(np.float32)
    diff = np.abs(cml_out - pt_out).max()
    print(f"   Max diff PyTorch vs CoreML: {diff}")

    # Test with a few more token IDs
    for token_id in [0, 151644, 77091, 198, 151671, 151672]:
        with torch.no_grad():
            pt_result = wrapper(torch.tensor([token_id]))
        cml_result = loaded.predict({"input_ids": np.array([token_id], dtype=np.int32)})
        d = np.abs(pt_result.numpy().astype(np.float32) - cml_result['input_embeds'].astype(np.float32)).max()
        print(f"   Token {token_id:>6d}: diff={d:.6f}")

    print("\n" + "=" * 60)
    print(f"Done! Model saved: {tp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
