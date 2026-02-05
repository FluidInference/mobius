#!/usr/bin/env python3
"""
Qwen3-TTS LM Decode V10 - Full codebook embedding sum with past_hidden

Takes all 16 codebook token IDs, sums their embeddings (using the correct
embedding tables for CB0 vs CB1-15), runs through the transformer, and outputs:
- logits for next CB0 token
- new KV cache
- past_hidden for code predictor

CB0 uses the main codec_embedding table.
CB1-15 use the code predictor's embedding tables.
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_KV_LEN = 300


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
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


class TracableDecodeV10(nn.Module):
    """
    Decode with full codebook embedding sum.

    Takes all 16 codebook IDs, sums their embeddings using the correct
    embedding tables, runs the LM transformer, and outputs logits + past_hidden.
    """

    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding
        self.cp_embeddings = talker.code_predictor.get_input_embeddings()
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head

        self.config = talker.config
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_code_groups = self.config.num_code_groups

    def forward(
        self,
        cb0_id: torch.Tensor,
        cb1_15_ids: torch.Tensor,
        trailing_text_embed: torch.Tensor,
        kv_cache: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple:
        """
        Generate next token logits with full codebook embedding.

        Args:
            cb0_id: [B, 1] - CB0 token ID
            cb1_15_ids: [B, 15] - CB1-15 token IDs
            trailing_text_embed: [B, 1, hidden_size] - text embedding (tts_pad)
            kv_cache: [56, B, num_kv_heads, seq_len, head_dim]
            position: [B] - current position

        Returns:
            logits: [B, vocab_size]
            new_kv_cache: [56, B, num_kv_heads, seq_len+1, head_dim]
            past_hidden: [B, 1, hidden_size]
        """
        # CB0 embedding from main codec_embedding
        cb0_embed = self.codec_embedding(cb0_id)  # [B, 1, hidden]

        # CB1-15 embeddings from code predictor embedding tables
        embeds_sum = cb0_embed.squeeze(1)  # [B, hidden]
        for i in range(self.num_code_groups - 1):
            cb_embed = self.cp_embeddings[i](cb1_15_ids[:, i:i+1])  # [B, 1, hidden]
            embeds_sum = embeds_sum + cb_embed.squeeze(1)  # [B, hidden]

        inputs_embeds = embeds_sum.unsqueeze(1)  # [B, 1, hidden]

        # Add trailing text embedding
        hidden_states = inputs_embeds + trailing_text_embed  # [B, 1, hidden]

        # Position embeddings
        pos_1d = position.unsqueeze(0).expand(3, -1)
        position_ids = pos_1d.unsqueeze(-1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Process through layers with KV cache
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
        past_hidden = hidden_states

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

        return logits, new_kv_cache, past_hidden

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
    print("Qwen3-TTS LM Decode V10 - Full Codebook Sum + past_hidden")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b", device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker

    print("\n2. Creating wrapper...")
    wrapper = TracableDecodeV10(talker)
    wrapper.eval()

    print("\n3. Testing wrapper...")
    cb0_id = torch.tensor([[1995]])
    cb1_15_ids = torch.tensor([[1159, 355, 22, 1174, 1093, 625, 1814, 1058, 905, 1846, 248, 1677, 889, 812, 901]])
    kv_cache = torch.randn(56, 1, 8, 139, 128)
    position = torch.tensor([139])

    TTS_PAD_TOKEN_ID = 151671
    with torch.no_grad():
        tts_pad_ids = torch.tensor([[TTS_PAD_TOKEN_ID]])
        tts_pad_embed = talker.text_projection(talker.model.text_embedding(tts_pad_ids))

    with torch.no_grad():
        logits, new_kv, past_hidden = wrapper(cb0_id, cb1_15_ids, tts_pad_embed, kv_cache, position)

    print(f"   Logits shape: {logits.shape}")
    print(f"   New KV shape: {new_kv.shape}")
    print(f"   Past hidden shape: {past_hidden.shape}")
    print(f"   Top token: {torch.argmax(logits, dim=-1).item()}")

    print("\n4. Tracing for CoreML...")
    kv_len = 139
    example_inputs = (
        torch.tensor([[1000]]),
        torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]]),
        tts_pad_embed,
        torch.randn(56, 1, 8, kv_len, 128),
        torch.tensor([kv_len]),
    )

    traced = torch.jit.trace(wrapper, example_inputs)

    print("\n5. Converting to CoreML...")
    inputs = [
        ct.TensorType(name="cb0_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="cb1_15_ids", shape=(1, 15), dtype=np.int32),
        ct.TensorType(name="trailing_text_embed", shape=(1, 1, 1024), dtype=np.float32),
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
            ct.TensorType(name="past_hidden", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "qwen3_tts_lm_decode_v10.mlpackage"
    mlmodel.save(output_path)
    print(f"\n6. Saved to {output_path}")

    # Verify
    print("\n7. Verifying CoreML model...")
    loaded = ct.models.MLModel(output_path)
    test_result = loaded.predict({
        "cb0_id": np.array([[1995]], dtype=np.int32),
        "cb1_15_ids": np.array([[1159, 355, 22, 1174, 1093, 625, 1814, 1058, 905, 1846, 248, 1677, 889, 812, 901]], dtype=np.int32),
        "trailing_text_embed": tts_pad_embed.numpy().astype(np.float32),
        "kv_cache": kv_cache.numpy().astype(np.float32),
        "position": np.array([139], dtype=np.int32),
    })

    print(f"   CoreML logits shape: {test_result['logits'].shape}")
    print(f"   CoreML new_kv shape: {test_result['new_kv_cache'].shape}")
    print(f"   CoreML past_hidden shape: {test_result['past_hidden'].shape}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
