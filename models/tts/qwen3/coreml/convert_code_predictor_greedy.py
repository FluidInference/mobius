#!/usr/bin/env python3
"""
Qwen3-TTS Code Predictor - Standalone Greedy CoreML Model

Converts the code predictor as a standalone model that:
- Takes: past_hidden [1, 1, 1024] + cb0_token [1, 1]
- Returns: codebooks [1, 15] (CB1-15 tokens)

The code predictor runs 15 autoregressive steps internally using greedy decoding.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np


class StandaloneGreedyCodePredictor(nn.Module):
    """Standalone greedy code predictor for CoreML conversion."""

    def __init__(self, code_predictor, codec_embedding, num_groups=16):
        super().__init__()
        self.code_predictor = code_predictor
        self.codec_embedding = codec_embedding
        self.num_groups = num_groups
        self.embeddings = code_predictor.get_input_embeddings()
        self.lm_head = code_predictor.lm_head
        self.model = code_predictor.model

    def forward(
        self,
        past_hidden: torch.Tensor,
        cb0_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate CB1-15 tokens using greedy decoding.

        Args:
            past_hidden: [B, 1, 1024] - hidden state from LM decoder
            cb0_token: [B, 1] - CB0 token ID

        Returns:
            codebooks: [B, 15] - CB1-15 tokens
        """
        # Get CB0 embedding
        cb0_embed = self.codec_embedding(cb0_token)  # [B, 1, 1024]

        # Initial hidden: [past_hidden, cb0_embed]
        hidden = torch.cat([past_hidden, cb0_embed], dim=1)  # [B, 2, 1024]

        output_tokens = []

        for i in range(self.num_groups - 1):  # 15 steps
            # Run through transformer
            outputs = self.model(inputs_embeds=hidden, use_cache=False)
            hidden_states = outputs.last_hidden_state

            # Get logits for this codebook
            logits = self.lm_head[i](hidden_states[:, -1:, :])  # [B, 1, 2048]

            # Greedy selection
            next_token = torch.argmax(logits, dim=-1)  # [B, 1]
            output_tokens.append(next_token)

            # Get embedding for next step
            next_embed = self.embeddings[i](next_token)  # [B, 1, 1024]

            # Append to hidden for next iteration
            hidden = torch.cat([hidden, next_embed], dim=1)

        return torch.cat(output_tokens, dim=1)  # [B, 15]


def main():
    print("=" * 60)
    print("Qwen3-TTS Code Predictor - Standalone Greedy")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b", device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker

    print("\n2. Creating standalone wrapper...")
    wrapper = StandaloneGreedyCodePredictor(
        talker.code_predictor,
        talker.model.codec_embedding,
        num_groups=talker.config.num_code_groups,
    )
    wrapper.eval()

    print("\n3. Testing wrapper...")
    past_hidden = torch.randn(1, 1, 1024)
    cb0_token = torch.tensor([[1995]])

    with torch.no_grad():
        codebooks = wrapper(past_hidden, cb0_token)
    print(f"   Codebooks shape: {codebooks.shape}")
    print(f"   Codebooks: {codebooks[0].tolist()}")

    print("\n4. Tracing for CoreML...")
    example_inputs = (
        torch.randn(1, 1, 1024),  # past_hidden
        torch.tensor([[1000]]),     # cb0_token
    )

    try:
        traced = torch.jit.trace(wrapper, example_inputs)
        print("   Tracing succeeded!")

        # Verify traced model
        with torch.no_grad():
            traced_out = traced(past_hidden, cb0_token)
        print(f"   Traced output: {traced_out[0].tolist()}")
        print(f"   Match: {torch.equal(codebooks, traced_out)}")

        print("\n5. Converting to CoreML...")
        inputs = [
            ct.TensorType(name="past_hidden", shape=(1, 1, 1024), dtype=np.float32),
            ct.TensorType(name="cb0_token", shape=(1, 1), dtype=np.int32),
        ]

        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=[
                ct.TensorType(name="codebooks", dtype=np.int32),
            ],
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT32,
        )

        output_path = "qwen3_tts_code_predictor_greedy.mlpackage"
        mlmodel.save(output_path)
        print(f"\n6. Saved to {output_path}")

        # Verify CoreML
        print("\n7. Verifying CoreML model...")
        loaded = ct.models.MLModel(output_path)
        coreml_out = loaded.predict({
            "past_hidden": past_hidden.numpy().astype(np.float32),
            "cb0_token": cb0_token.numpy().astype(np.int32),
        })
        print(f"   CoreML output: {coreml_out['codebooks']}")
        print(f"   Match with PyTorch: {np.array_equal(coreml_out['codebooks'].flatten(), codebooks[0].numpy())}")

    except Exception as e:
        print(f"   Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
