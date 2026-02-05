#!/usr/bin/env python3
"""
Qwen3-TTS LM Decode V9 Greedy - CoreML-compatible with greedy code predictor

This replaces code_predictor.generate() with manual greedy decoding,
making it traceable for CoreML conversion.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_KV_LEN = 300


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_simple(q, k, cos, sin):
    if cos.dim() == 4:
        cos = cos[0]
        sin = sin[0]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GreedyCodePredictor(nn.Module):
    """Manual greedy implementation of code_predictor for CoreML tracing."""

    def __init__(self, code_predictor, num_groups=16):
        super().__init__()
        self.code_predictor = code_predictor
        self.num_groups = num_groups
        # Pre-extract the embedding layers and lm_head
        self.embeddings = code_predictor.get_input_embeddings()
        self.lm_head = code_predictor.lm_head
        self.model = code_predictor.model

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Greedy decode CB1-15 tokens.

        Args:
            inputs_embeds: [B, 2, hidden] - [past_hidden, cb0_embed]

        Returns:
            codebooks: [B, 15] - CB1-15 tokens
        """
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device

        # Initialize output tokens
        output_tokens = []

        # Current hidden state from input
        hidden = inputs_embeds

        for i in range(self.num_groups - 1):  # Generate CB1-15 (15 tokens)
            # Run through transformer
            outputs = self.model(inputs_embeds=hidden, use_cache=False)
            hidden_states = outputs.last_hidden_state

            # Get logits for this codebook
            logits = self.lm_head[i](hidden_states[:, -1:, :])  # [B, 1, vocab]

            # Greedy selection
            next_token = torch.argmax(logits, dim=-1)  # [B, 1]
            output_tokens.append(next_token)

            # Get embedding for next step
            next_embed = self.embeddings[i](next_token)  # [B, 1, hidden]

            # Append to hidden for next iteration
            hidden = torch.cat([hidden, next_embed], dim=1)

        # Stack all tokens
        return torch.cat(output_tokens, dim=1)  # [B, 15]


class TracableDecodeV9Greedy(nn.Module):
    """Decode with greedy code predictor - CoreML traceable."""

    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head

        # Use greedy code predictor
        self.greedy_predictor = GreedyCodePredictor(
            talker.code_predictor, num_groups=talker.config.num_code_groups
        )

        self.config = talker.config
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_code_groups = self.config.num_code_groups

    def forward(
        self,
        token_id: torch.Tensor,
        past_hidden: torch.Tensor,
        trailing_text_embed: torch.Tensor,
        kv_cache: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple:
        """
        Generate next token with greedy code predictor.

        Returns:
            logits: [B, vocab_size]
            new_kv_cache: [56, B, num_kv_heads, seq_len+1, head_dim]
            new_past_hidden: [B, 1, hidden_size]
            all_codebooks: [B, 16]
        """
        # Get CB0 embedding
        last_id_hidden = self.codec_embedding(token_id)  # [B, 1, hidden]

        # Run greedy code predictor
        predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)  # [B, 2, hidden]
        cb1_15 = self.greedy_predictor(predictor_input)  # [B, 15]

        # Sum all codebook embeddings
        codec_hiddens = [last_id_hidden]
        for i in range(self.num_code_groups - 1):
            cb_embed = self.greedy_predictor.embeddings[i](cb1_15[:, i : i + 1])
            codec_hiddens.append(cb_embed)

        codec_hiddens = torch.cat(codec_hiddens, dim=1)  # [B, 16, hidden]
        inputs_embeds = codec_hiddens.sum(dim=1, keepdim=True)  # [B, 1, hidden]

        # Add trailing text embedding
        hidden_states = inputs_embeds + trailing_text_embed

        # Position embeddings
        pos_1d = position.unsqueeze(0).expand(3, -1)
        position_ids = pos_1d.unsqueeze(-1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Process through layers
        new_keys = []
        new_values = []
        cache_idx = 0

        for layer in self.layers:
            layer_key_cache = kv_cache[cache_idx]
            layer_value_cache = kv_cache[cache_idx + 1]

            hidden_states, new_key, new_value = self._run_layer_with_cache(
                layer, hidden_states, layer_key_cache, layer_value_cache, cos, sin
            )

            new_keys.append(new_key)
            new_values.append(new_value)
            cache_idx += 2

        # Final norm
        hidden_states = self.norm(hidden_states)
        new_past_hidden = hidden_states

        # Get logits
        logits = self.codec_head(hidden_states).squeeze(1)

        # Build new KV cache
        new_kv_list = []
        for i in range(self.num_layers):
            old_key = kv_cache[i * 2]
            old_value = kv_cache[i * 2 + 1]
            new_kv_list.append(torch.cat([old_key, new_keys[i]], dim=2))
            new_kv_list.append(torch.cat([old_value, new_values[i]], dim=2))
        new_kv_cache = torch.stack(new_kv_list, dim=0)

        # All codebooks
        all_codebooks = torch.cat([token_id, cb1_15], dim=1)

        return logits, new_kv_cache, new_past_hidden, all_codebooks

    def _run_layer_with_cache(
        self, layer, hidden_states, key_cache, value_cache, cos, sin
    ):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, new_key, new_value = self._run_attention_with_cache(
            layer.self_attn, hidden_states, key_cache, value_cache, cos, sin
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_key, new_value

    def _run_attention_with_cache(
        self, attn, hidden_states, key_cache, value_cache, cos, sin
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = attn.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        key_states = attn.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )
        value_states = attn.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )

        query_states = attn.q_norm(query_states).transpose(1, 2)
        key_states = attn.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb_simple(
            query_states, key_states, cos, sin
        )

        new_key = key_states
        new_value = value_states

        full_key = torch.cat([key_cache, key_states], dim=2)
        full_value = torch.cat([value_cache, value_states], dim=2)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            full_key = full_key.repeat_interleave(n_rep, dim=1)
            full_value = full_value.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(query_states, full_key.transpose(-1, -2)) / (
            self.head_dim**0.5
        )
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, full_value)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)

        return attn_output, new_key, new_value


def main():
    print("=" * 60)
    print("Qwen3-TTS LM Decode V9 Greedy")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b", device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker

    print("\n2. Creating wrapper...")
    wrapper = TracableDecodeV9Greedy(talker)
    wrapper.eval()

    print("\n3. Testing wrapper...")
    token_id = torch.tensor([[1995]])
    past_hidden = torch.randn(1, 1, 1024)
    kv_cache = torch.randn(56, 1, 8, 139, 128)
    position = torch.tensor([139])

    TTS_PAD_TOKEN_ID = 151671
    with torch.no_grad():
        tts_pad_ids = torch.tensor([[TTS_PAD_TOKEN_ID]])
        tts_pad_embed = talker.text_projection(talker.model.text_embedding(tts_pad_ids))

    with torch.no_grad():
        logits, new_kv, new_hidden, all_cb = wrapper(
            token_id, past_hidden, tts_pad_embed, kv_cache, position
        )

    print(f"   Logits shape: {logits.shape}")
    print(f"   New KV shape: {new_kv.shape}")
    print(f"   New hidden shape: {new_hidden.shape}")
    print(f"   All codebooks shape: {all_cb.shape}")
    print(f"   All codebooks: {all_cb[0].tolist()}")

    print("\n4. Tracing for CoreML...")
    kv_len = 139
    example_inputs = (
        torch.tensor([[1000]]),
        torch.randn(1, 1, 1024),  # past_hidden
        tts_pad_embed,
        torch.randn(56, 1, 8, kv_len, 128),
        torch.tensor([kv_len]),
    )

    try:
        traced = torch.jit.trace(wrapper, example_inputs)
        print("   Tracing succeeded!")

        print("\n5. Converting to CoreML...")
        inputs = [
            ct.TensorType(name="token_id", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="past_hidden", shape=(1, 1, 1024), dtype=np.float32),
            ct.TensorType(
                name="trailing_text_embed", shape=(1, 1, 1024), dtype=np.float32
            ),
            ct.TensorType(
                name="kv_cache",
                shape=(56, 1, 8, ct.RangeDim(lower_bound=1, upper_bound=MAX_KV_LEN), 128),
                dtype=np.float32,
            ),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ]

        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=[
                ct.TensorType(name="logits", dtype=np.float32),
                ct.TensorType(name="new_kv_cache", dtype=np.float32),
                ct.TensorType(name="new_past_hidden", dtype=np.float32),
                ct.TensorType(name="all_codebooks", dtype=np.int32),
            ],
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT32,
        )

        output_path = "qwen3_tts_lm_decode_v9_greedy.mlpackage"
        mlmodel.save(output_path)
        print(f"\n6. Saved to {output_path}")

    except Exception as e:
        print(f"   Tracing/conversion failed: {e}")
        print("\n   The greedy code predictor contains operations that")
        print("   cannot be traced. Need a different approach.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
