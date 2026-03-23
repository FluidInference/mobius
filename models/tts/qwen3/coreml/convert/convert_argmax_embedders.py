#!/usr/bin/env python3
"""
Argmax-style CodeEmbedder + MultiCodeEmbedder CoreML Conversion

Reverse-engineered from Argmax TTSKit's MIL programs:

CodeEmbedder:
  - Input: input_ids [1] (int32) — a single CB0 codec token ID
  - Gather from codec_embedding [3072, 1024]
  - Output: input_embeds [1, 1024, 1, 1] (fp16) — 4D ANE layout

MultiCodeEmbedder:
  - Input: input_ids [1] (int32) — a linearized index into concatenated CB1-15 embeddings
  - Gather from combined_embedding [30720, 1024]  (15 codebooks × 2048 entries)
  - Output: input_embeds [1, 1024, 1, 1] (fp16) — 4D ANE layout

The key insight: Argmax concatenates all 15 code predictor embedding tables
(each [2048, 1024]) into one flat [30720, 1024] table. The caller computes
the linearized index: index = codebook_idx * 2048 + token_id.

Usage:
    python convert_argmax_embedders.py [--model-path ./model_0.6b]
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import argparse


class CodeEmbedderWrapper(nn.Module):
    """Wraps the main codec_embedding for CB0 token lookup.

    MIL equivalent:
        gather(codec_embedding_weight, input_ids) → [1, 1024]
        expand_dims → [1, 1024, 1]
        expand_dims → [1, 1024, 1, 1]
    """

    def __init__(self, codec_embedding):
        super().__init__()
        self.codec_embedding = codec_embedding  # nn.Embedding(3072, 1024)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [1] — single CB0 token ID

        Returns:
            input_embeds: [1, 1024, 1, 1] — ANE 4D layout
        """
        embed = self.codec_embedding(input_ids)  # [1, 1024]
        # Reshape to ANE 4D: [1, C, 1, 1]
        return embed.unsqueeze(-1).unsqueeze(-1)  # [1, 1024, 1, 1]


class MultiCodeEmbedderWrapper(nn.Module):
    """Wraps the 15 code predictor embedding tables as one concatenated table.

    Argmax concatenates all 15 CP embedding tables into [30720, 1024].
    The caller linearizes: idx = codebook_idx * 2048 + token_id.

    MIL equivalent:
        gather(codec_embedding_weight [30720, 1024], input_ids) → [1, 1024]
        expand_dims → [1, 1024, 1]
        expand_dims → [1, 1024, 1, 1]
    """

    def __init__(self, cp_embeddings):
        super().__init__()
        # Concatenate all 15 embedding tables into one
        weights = []
        for emb in cp_embeddings:
            weights.append(emb.weight.detach())
        combined = torch.cat(weights, dim=0)  # [30720, 1024]
        self.combined_embedding = nn.Embedding(
            combined.shape[0], combined.shape[1]
        )
        self.combined_embedding.weight = nn.Parameter(combined, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [1] — linearized index (codebook_idx * 2048 + token_id)

        Returns:
            input_embeds: [1, 1024, 1, 1] — ANE 4D layout
        """
        embed = self.combined_embedding(input_ids)  # [1, 1024]
        return embed.unsqueeze(-1).unsqueeze(-1)  # [1, 1024, 1, 1]


def main():
    parser = argparse.ArgumentParser(description="Convert Argmax-style embedders")
    parser.add_argument("--model-path", default="./model_0.6b", help="Path to Qwen3-TTS model")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Apply W16A16 palettization")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    print("=" * 60)
    print("Argmax-style CodeEmbedder + MultiCodeEmbedder Conversion")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker
    codec_embedding = talker.model.codec_embedding
    cp_embeddings = talker.code_predictor.get_input_embeddings()

    print(f"   codec_embedding: {codec_embedding.weight.shape}")
    print(f"   CP embeddings: {len(cp_embeddings)} tables, each {cp_embeddings[0].weight.shape}")

    # Extract weights before releasing the model
    codec_weight = codec_embedding.weight.detach().clone()
    cp_weights = [emb.weight.detach().clone() for emb in cp_embeddings]

    # Release model to free memory
    del model, talker, codec_embedding, cp_embeddings
    import gc
    gc.collect()

    # 2. CodeEmbedder
    print("\n2. Converting CodeEmbedder...")
    ce_embedding = nn.Embedding(codec_weight.shape[0], codec_weight.shape[1])
    ce_embedding.weight = nn.Parameter(codec_weight, requires_grad=False)
    code_embedder = CodeEmbedderWrapper(ce_embedding)
    code_embedder.eval()

    # Test
    test_id = torch.tensor([1000])
    with torch.no_grad():
        test_out = code_embedder(test_id)
    print(f"   Test output shape: {test_out.shape}")  # [1, 1024, 1, 1]

    # Trace
    traced_ce = torch.jit.trace(code_embedder, (test_id,))

    # Convert to CoreML
    ml_ce = ct.convert(
        traced_ce,
        inputs=[ct.TensorType(name="input_ids", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="input_embeds", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    ce_path = f"{args.output_dir}/CodeEmbedder.mlpackage"
    ml_ce.save(ce_path)
    print(f"   Saved: {ce_path}")

    # Verify
    loaded_ce = ct.models.MLModel(ce_path)
    result = loaded_ce.predict({"input_ids": np.array([1000], dtype=np.int32)})
    print(f"   Verify shape: {result['input_embeds'].shape}")
    print(f"   Verify dtype: {result['input_embeds'].dtype}")

    # Free CE resources
    del code_embedder, traced_ce, ml_ce, loaded_ce, ce_embedding
    gc.collect()

    # 3. MultiCodeEmbedder
    print("\n3. Converting MultiCodeEmbedder...")
    combined_weight = torch.cat(cp_weights, dim=0)  # [30720, 1024]
    mce_embedding = nn.Embedding(combined_weight.shape[0], combined_weight.shape[1])
    mce_embedding.weight = nn.Parameter(combined_weight, requires_grad=False)

    # Use simpler wrapper that takes pre-built embedding
    class SimpleMultiCodeEmbedder(nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
        def forward(self, input_ids):
            embed = self.embedding(input_ids)
            return embed.unsqueeze(-1).unsqueeze(-1)

    multi_embedder = SimpleMultiCodeEmbedder(mce_embedding)
    multi_embedder.eval()

    # Test: codebook 0, token 500 → linearized index = 0 * 2048 + 500 = 500
    test_id_multi = torch.tensor([500])
    with torch.no_grad():
        test_out_multi = multi_embedder(test_id_multi)
    print(f"   Test output shape: {test_out_multi.shape}")  # [1, 1024, 1, 1]

    # Verify linearization: codebook 5, token 100 → 5*2048 + 100 = 10340
    test_id_linear = torch.tensor([10340])
    with torch.no_grad():
        embed_linear = multi_embedder(test_id_linear)
        embed_direct_w = cp_weights[5][100]  # direct weight lookup
    print(f"   Linearization check: {torch.allclose(embed_linear.squeeze(), embed_direct_w, atol=1e-5)}")

    # Trace
    traced_mce = torch.jit.trace(multi_embedder, (test_id_multi,))

    # Convert to CoreML
    ml_mce = ct.convert(
        traced_mce,
        inputs=[ct.TensorType(name="input_ids", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="input_embeds", dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    mce_path = f"{args.output_dir}/MultiCodeEmbedder.mlpackage"
    ml_mce.save(mce_path)
    print(f"   Saved: {mce_path}")

    # Verify
    loaded_mce = ct.models.MLModel(mce_path)
    result_mce = loaded_mce.predict({"input_ids": np.array([500], dtype=np.int32)})
    print(f"   Verify shape: {result_mce['input_embeds'].shape}")
    print(f"   Verify dtype: {result_mce['input_embeds'].dtype}")

    # Compare with PyTorch
    pt_embed = test_out_multi.detach().numpy()
    cml_embed = result_mce['input_embeds']
    diff = np.abs(pt_embed.astype(np.float32) - cml_embed.astype(np.float32)).max()
    print(f"   Max diff PyTorch vs CoreML: {diff}")

    print("\n" + "=" * 60)
    print("Done! Models saved:")
    print(f"  {ce_path}")
    print(f"  {mce_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
