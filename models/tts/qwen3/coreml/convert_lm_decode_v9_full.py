#!/usr/bin/env python3
"""
Qwen3-TTS LM Decode V9 Full - With Code Predictor Integration

This version properly integrates code_predictor per decode step,
outputting all 16 codebook tokens at each step.

The key insight: code_predictor.generate() needs past_hidden context
to generate proper CB1-15 tokens. Running it on just CB0 IDs doesn't work.
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


class TracableDecodeV9Full(nn.Module):
    """
    Full decode with code_predictor integration.

    This properly handles past_hidden and generates all 16 codebooks per step.
    """

    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head
        self.code_predictor = talker.code_predictor

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
        Generate next token with code_predictor integration.

        Args:
            token_id: [B, 1] - current codec token ID (CB0)
            past_hidden: [B, 1, hidden_size] - hidden state from previous step
            trailing_text_embed: [B, 1, hidden_size] - tts_pad embedding
            kv_cache: [56, B, num_kv_heads, seq_len, head_dim]
            position: [B] - current position

        Returns:
            logits: [B, vocab_size]
            new_kv_cache: [56, B, num_kv_heads, seq_len+1, head_dim]
            new_past_hidden: [B, 1, hidden_size]
            all_codebooks: [B, 16] - all codebook tokens for this step
        """
        batch_size = token_id.shape[0]

        # Get CB0 embedding
        last_id_hidden = self.codec_embedding(token_id)  # [B, 1, hidden]

        # Run code_predictor to get CB1-15 using past_hidden context
        with torch.no_grad():
            predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)  # [B, 2, hidden]
            predictor_result = self.code_predictor.generate(
                inputs_embeds=predictor_input,
                max_new_tokens=self.num_code_groups - 1,  # 15
                do_sample=True,
                temperature=0.9,
                top_p=1.0,
                top_k=50,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )
            # predictor_result.sequences: [B, 15] - codebooks 1-15

        # Sum all codebook embeddings for the decoder input
        codec_hiddens = [last_id_hidden]
        for i in range(self.num_code_groups - 1):
            cb_embed = self.code_predictor.get_input_embeddings()[i](
                predictor_result.sequences[..., i : i + 1]
            )
            codec_hiddens.append(cb_embed)

        codec_hiddens = torch.cat(codec_hiddens, dim=1)  # [B, 16, hidden]
        inputs_embeds = codec_hiddens.sum(dim=1, keepdim=True)  # [B, 1, hidden]

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

        # This is new_past_hidden for next step
        new_past_hidden = hidden_states

        # Get logits for next CB0 token
        logits = self.codec_head(hidden_states).squeeze(1)

        # Build new KV cache
        new_kv_list = []
        for i in range(self.num_layers):
            old_key = kv_cache[i * 2]
            old_value = kv_cache[i * 2 + 1]
            new_kv_list.append(torch.cat([old_key, new_keys[i]], dim=2))
            new_kv_list.append(torch.cat([old_value, new_values[i]], dim=2))
        new_kv_cache = torch.stack(new_kv_list, dim=0)

        # Build all_codebooks: [B, 16]
        all_codebooks = torch.cat(
            [token_id, predictor_result.sequences], dim=1
        )  # [B, 16]

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
    print("Qwen3-TTS LM Decode V9 Full - With Code Predictor")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b", device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker

    print("\n2. Creating wrapper...")
    wrapper = TracableDecodeV9Full(talker)
    wrapper.eval()

    print("\n3. Testing wrapper...")
    # Test with V9-compatible shapes
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

    print("\n4. NOTE: Cannot convert to CoreML due to code_predictor.generate()")
    print("   The generate() call uses do_sample=True which is not traceable.")
    print("   This model must be run in PyTorch for correct operation.")
    print("   ")
    print("   For CoreML, we need a different approach:")
    print("   1. Pre-compute code_predictor weights for greedy decoding")
    print("   2. Or use a simplified model that sacrifices some quality")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
