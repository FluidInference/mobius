# Qwen3-TTS LM Decode V4
# Main transformer only - takes pre-computed inputs_embeds
# Code predictor embedding computation is done externally

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_KV_LEN = 500  # Max prefill + generated tokens


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


class TracableDecodeV4(nn.Module):
    """Main transformer decode only - takes pre-computed inputs_embeds."""

    def __init__(self, talker):
        super().__init__()
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

    def forward(self, inputs_embeds: torch.Tensor, kv_cache: torch.Tensor,
                position: torch.Tensor) -> tuple:
        """
        Run main transformer decode with pre-computed inputs_embeds.

        Args:
            inputs_embeds: [B, 1, hidden_size] - pre-computed sum of codebook embeddings + trailing_text
            kv_cache: [56, B, num_kv_heads, seq_len, head_dim]
            position: [B] - current position in sequence

        Returns:
            logits: [B, vocab_size]
            new_kv_cache: [56, B, num_kv_heads, seq_len+1, head_dim]
            past_hidden: [B, 1, hidden_size] - for next step
        """
        batch_size = inputs_embeds.shape[0]
        hidden_states = inputs_embeds

        # Position embeddings for single token
        pos_1d = position.unsqueeze(0).expand(3, -1)  # [3, B]
        position_ids = pos_1d.unsqueeze(-1)  # [3, B, 1]
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

        # Past hidden for next step (after norm)
        past_hidden = hidden_states

        return logits, new_kv_cache, past_hidden

    def _run_layer_with_cache(self, layer, hidden_states, key_cache, value_cache, cos, sin):
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

    def _run_attention_with_cache(self, attn, hidden_states, key_cache, value_cache, cos, sin):
        bsz, q_len, _ = hidden_states.shape

        query_states = attn.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = attn.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        query_states = attn.q_norm(query_states).transpose(1, 2)
        key_states = attn.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb_simple(query_states, key_states, cos, sin)

        new_key = key_states
        new_value = value_states

        # Concatenate with cache
        full_key = torch.cat([key_cache, key_states], dim=2)
        full_value = torch.cat([value_cache, value_states], dim=2)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            full_key = full_key.repeat_interleave(n_rep, dim=1)
            full_value = full_value.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(query_states, full_key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, full_value)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)

        return attn_output, new_key, new_value


def compute_decode_inputs(talker, token_id, past_hidden, trailing_text_embed):
    """Compute inputs_embeds externally (for testing with PyTorch code_predictor)."""

    # Get first codebook embedding
    last_id_hidden = talker.model.codec_embedding(token_id)

    # Run code_predictor to get other codebook tokens
    with torch.no_grad():
        predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)
        predictor_result = talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=talker.config.num_code_groups - 1,
            do_sample=False,
            return_dict_in_generate=True,
        )

    # Get embeddings for all codebooks and sum them
    codec_hiddens = [last_id_hidden]
    for i in range(talker.config.num_code_groups - 1):
        cb_embed = talker.code_predictor.get_input_embeddings()[i](
            predictor_result.sequences[..., i:i+1]
        )
        codec_hiddens.append(cb_embed)

    codec_hiddens = torch.cat(codec_hiddens, dim=1)
    inputs_embeds = codec_hiddens.sum(dim=1, keepdim=True)

    # Add trailing text embedding
    inputs_embeds = inputs_embeds + trailing_text_embed

    return inputs_embeds


def convert_to_coreml(wrapper, hidden_size=1024, num_kv_heads=8, head_dim=128):
    """Convert to CoreML."""
    print("\n=== Converting to CoreML ===")

    wrapper.eval()

    # Example inputs
    example_embeds = torch.randn(1, 1, hidden_size)
    example_kv = torch.randn(56, 1, num_kv_heads, 20, head_dim)
    example_pos = torch.tensor([20])

    print("Tracing...")
    traced = torch.jit.trace(wrapper, (example_embeds, example_kv, example_pos))

    print("Converting...")
    kv_range = ct.RangeDim(lower_bound=1, upper_bound=MAX_KV_LEN, default=20)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, hidden_size), dtype=np.float32),
            ct.TensorType(name="kv_cache", shape=(56, 1, num_kv_heads, kv_range, head_dim), dtype=np.float32),
            ct.TensorType(name="position", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="new_kv_cache", dtype=np.float32),
            ct.TensorType(name="past_hidden", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "qwen3_tts_lm_decode_v4.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    import subprocess
    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")

    return mlmodel


def main():
    print("=" * 60)
    print("Qwen3-TTS LM Decode V4 - Main Transformer Only")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    talker = model.model.talker
    config = talker.config

    print("\n2. Creating wrapper...")
    wrapper = TracableDecodeV4(talker)
    wrapper.eval()

    # Get tts_pad_embed
    TTS_PAD_TOKEN_ID = 151671
    with torch.no_grad():
        tts_pad_ids = torch.tensor([[TTS_PAD_TOKEN_ID]])
        tts_pad_embed = talker.text_projection(talker.model.text_embedding(tts_pad_ids))

    print("\n3. Testing with sample inputs...")
    # Create sample inputs
    token_id = torch.tensor([[1995]])
    past_hidden = torch.randn(1, 1, config.hidden_size)
    kv_cache = torch.randn(56, 1, 8, 10, 128)
    position = torch.tensor([10])

    # Compute inputs_embeds using external function
    with torch.no_grad():
        inputs_embeds = compute_decode_inputs(talker, token_id, past_hidden, tts_pad_embed)

    # Run main transformer
    with torch.no_grad():
        logits, new_kv, new_past_hidden = wrapper(inputs_embeds, kv_cache, position)

    print(f"   inputs_embeds shape: {inputs_embeds.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   New KV shape: {new_kv.shape}")
    print(f"   Past hidden shape: {new_past_hidden.shape}")

    print("\n4. Converting to CoreML...")
    convert_to_coreml(wrapper, hidden_size=config.hidden_size,
                     num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim)
    print("\nDone!")


if __name__ == "__main__":
    main()
