# Qwen3-TTS LM → CoreML Conversion v5
# Correct 10-position sequence construction matching official model

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

# The prefill sequence is exactly 10 positions:
# Position 0-2: role prefix (text_embed + text_projection of first 3 text tokens)
# Position 3: tts_pad + codec_think_id
# Position 4: tts_pad + codec_think_bos_id
# Position 5: tts_pad + language_id
# Position 6: tts_pad + codec_think_eos_id
# Position 7: tts_pad + speaker_embed
# Position 8: tts_bos + codec_pad_id
# Position 9: first_text_token + codec_bos_id

MAX_TEXT_LENGTH = 128
PREFILL_POSITIONS = 10  # Fixed prefill length


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


class TracablePrefillV5(nn.Module):
    """Prefill with correct 10-position sequence construction.

    Matches official model's sequence: role_prefix + codec_think_tokens + speaker + text_start
    """

    def __init__(self, talker, config):
        super().__init__()
        # Text embedding and projection (for role prefix, tts_bos, tts_pad)
        self.text_embedding = talker.model.text_embedding
        self.text_projection = talker.text_projection

        # Codec embedding (for codec tokens)
        self.codec_embedding = talker.model.codec_embedding

        # Transformer layers
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head

        self.config = talker.config
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size

        # Special token IDs (from config)
        self.codec_think_id = self.config.codec_think_id  # 2154
        self.codec_think_bos_id = self.config.codec_think_bos_id  # 2156
        self.codec_think_eos_id = self.config.codec_think_eos_id  # 2157
        self.codec_pad_id = self.config.codec_pad_id  # 2148
        self.codec_bos_id = self.config.codec_bos_id  # 2149

        # Language ID for English
        self.english_language_id = self.config.codec_language_id["english"]  # 2163

    def forward(self, role_ids: torch.Tensor, first_text_id: torch.Tensor,
                tts_bos_embed: torch.Tensor, tts_pad_embed: torch.Tensor,
                speaker_embed: torch.Tensor) -> tuple:
        """
        Construct the exact 10-position prefill sequence.

        Args:
            role_ids: [B, 3] - role prefix token IDs (e.g., <|im_start|>assistant\n)
            first_text_id: [B, 1] - first text token after role
            tts_bos_embed: [B, 1, hidden_size] - TTS BOS embedding (pre-projected)
            tts_pad_embed: [B, 1, hidden_size] - TTS PAD embedding (pre-projected)
            speaker_embed: [B, 1024] - speaker embedding from reference audio

        Returns:
            logits: [B, vocab_size]
            kv_cache: [56, B, num_kv_heads, 10, head_dim]
        """
        batch_size = role_ids.shape[0]
        device = role_ids.device

        # === Position 0-2: Role prefix ===
        # text_embed + text_projection for first 3 text tokens
        role_text_embed = self.text_embedding(role_ids)  # [B, 3, text_hidden]
        role_projected = self.text_projection(role_text_embed)  # [B, 3, hidden_size]

        # === Position 3-6: Codec think tokens (tts_pad + codec_embed) ===
        codec_think_ids = torch.tensor([
            [self.codec_think_id, self.codec_think_bos_id,
             self.english_language_id, self.codec_think_eos_id]
        ], dtype=torch.long, device=device).expand(batch_size, -1)
        codec_think_embeds = self.codec_embedding(codec_think_ids)  # [B, 4, hidden]

        # Add tts_pad to each
        think_positions = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds  # [B, 4, hidden]

        # === Position 7: Speaker (tts_pad + speaker_embed) ===
        speaker_position = tts_pad_embed + speaker_embed.unsqueeze(1)  # [B, 1, hidden]

        # === Position 8: tts_bos + codec_pad ===
        codec_pad_ids = torch.tensor([[self.codec_pad_id]], dtype=torch.long, device=device).expand(batch_size, -1)
        codec_pad_embed = self.codec_embedding(codec_pad_ids)  # [B, 1, hidden]
        bos_pad_position = tts_bos_embed + codec_pad_embed  # [B, 1, hidden]

        # === Position 9: first_text_token + codec_bos ===
        first_text_embed = self.text_embedding(first_text_id)  # [B, 1, text_hidden]
        first_text_projected = self.text_projection(first_text_embed)  # [B, 1, hidden]
        codec_bos_ids = torch.tensor([[self.codec_bos_id]], dtype=torch.long, device=device).expand(batch_size, -1)
        codec_bos_embed = self.codec_embedding(codec_bos_ids)  # [B, 1, hidden]
        text_bos_position = first_text_projected + codec_bos_embed  # [B, 1, hidden]

        # === Concatenate all 10 positions ===
        hidden_states = torch.cat([
            role_projected,      # [B, 3, hidden] - positions 0-2
            think_positions,     # [B, 4, hidden] - positions 3-6
            speaker_position,    # [B, 1, hidden] - position 7
            bos_pad_position,    # [B, 1, hidden] - position 8
            text_bos_position,   # [B, 1, hidden] - position 9
        ], dim=1)  # [B, 10, hidden]

        seq_len = hidden_states.shape[1]  # Should be 10

        # === Position embeddings ===
        pos_1d = torch.arange(seq_len, device=device)
        position_ids = pos_1d.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # === Causal mask ===
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=hidden_states.dtype, device=device),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)

        # === Run transformer layers ===
        all_keys = []
        all_values = []

        for layer in self.layers:
            hidden_states, key, value = self._run_layer(
                layer, hidden_states, causal_mask, cos, sin
            )
            all_keys.append(key)
            all_values.append(value)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Get logits from last position
        logits = self.codec_head(hidden_states[:, -1:, :]).squeeze(1)

        # Stack KV cache
        kv_list = []
        for key, value in zip(all_keys, all_values):
            kv_list.append(key)
            kv_list.append(value)
        kv_cache = torch.stack(kv_list, dim=0)

        return logits, kv_cache

    def _run_layer(self, layer, hidden_states, mask, cos, sin):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, key, value = self._run_attention(layer.self_attn, hidden_states, mask, cos, sin)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, key, value

    def _run_attention(self, attn, hidden_states, mask, cos, sin):
        bsz, q_len, _ = hidden_states.shape

        query_states = attn.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = attn.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        query_states = attn.q_norm(query_states).transpose(1, 2)
        key_states = attn.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb_simple(query_states, key_states, cos, sin)

        key_out = key_states
        value_out = value_states

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)

        return attn_output, key_out, value_out


def verify_prefill(wrapper, model, text="Hello world"):
    """Verify wrapper matches PyTorch with same input."""
    print("\n=== Verification ===")

    talker = model.model.talker
    config = talker.config
    processor = model.processor

    # Special token IDs from config.json
    TTS_BOS_TOKEN_ID = 151672
    TTS_PAD_TOKEN_ID = 151671

    # Get text tokens
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"Input text: '{text}'")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0, :10].tolist()}...")

    # Extract special embeddings
    tts_special_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID]])
    tts_special_embed = talker.text_projection(talker.model.text_embedding(tts_special_ids))
    tts_bos_embed = tts_special_embed[:, 0:1, :]  # [1, 1, hidden]
    tts_pad_embed = tts_special_embed[:, 1:2, :]  # [1, 1, hidden]

    # Load speaker embedding
    speaker_embed = torch.from_numpy(np.load("speaker_embedding.npy")).unsqueeze(0)  # [1, 1024]

    # Extract role prefix and first text token
    role_ids = input_ids[:, :3]  # First 3 tokens
    first_text_id = input_ids[:, 3:4]  # Token at position 3

    print(f"Role IDs: {role_ids[0].tolist()}")
    print(f"First text ID: {first_text_id[0].tolist()}")

    with torch.no_grad():
        # Wrapper output
        wrapper_logits, wrapper_kv = wrapper(
            role_ids, first_text_id,
            tts_bos_embed, tts_pad_embed,
            speaker_embed
        )
        print(f"Wrapper logits shape: {wrapper_logits.shape}")
        print(f"Wrapper KV cache shape: {wrapper_kv.shape}")

        # Now run official model and compare
        # We need to construct the same sequence the official way
        codec_prefill_list = [[
            config.codec_think_id,
            config.codec_think_bos_id,
            config.codec_language_id["english"],
            config.codec_think_eos_id,
        ]]

        codec_input_emebdding_0 = talker.model.codec_embedding(
            torch.tensor(codec_prefill_list, dtype=torch.long)
        )
        codec_input_emebdding_1 = talker.model.codec_embedding(
            torch.tensor([[config.codec_pad_id, config.codec_bos_id]], dtype=torch.long)
        )
        codec_input_emebdding = torch.cat([
            codec_input_emebdding_0,
            speaker_embed.view(1, 1, -1),
            codec_input_emebdding_1
        ], dim=1)

        # Role prefix
        role_text_embed = talker.text_projection(talker.model.text_embedding(input_ids[:, :3]))

        # tts_pad * 5 + tts_bos, then add codec embeddings
        _talker_input_embed = torch.cat([
            tts_pad_embed.expand(-1, 5, -1),
            tts_bos_embed
        ], dim=1) + codec_input_emebdding[:, :-1]

        # Add first text token + codec_bos
        first_text_projected = talker.text_projection(talker.model.text_embedding(input_ids[:, 3:4]))
        text_bos_position = first_text_projected + codec_input_emebdding[:, -1:]

        # Full sequence
        talker_input_embed = torch.cat([
            role_text_embed,
            _talker_input_embed,
            text_bos_position
        ], dim=1)

        print(f"Official input embed shape: {talker_input_embed.shape}")

        # Run through official model
        outputs = talker.model(inputs_embeds=talker_input_embed, use_cache=False, return_dict=True)
        official_logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)

        print(f"Official logits shape: {official_logits.shape}")

        # Compare
        wrapper_token = torch.argmax(wrapper_logits, dim=-1).item()
        official_token = torch.argmax(official_logits, dim=-1).item()
        print(f"Wrapper first token: {wrapper_token}")
        print(f"Official first token: {official_token}")
        print(f"Match: {wrapper_token == official_token}")

        corr = np.corrcoef(wrapper_logits.numpy().flatten(), official_logits.numpy().flatten())[0, 1]
        print(f"Correlation: {corr:.6f}")

    return wrapper_token == official_token


def convert_to_coreml(wrapper, hidden_size=1024):
    """Convert to CoreML."""
    print("\n=== Converting to CoreML ===")

    wrapper.eval()

    # Example inputs
    example_role = torch.randint(0, 10000, (1, 3))
    example_first_text = torch.randint(0, 10000, (1, 1))
    example_tts_bos = torch.randn(1, 1, hidden_size)
    example_tts_pad = torch.randn(1, 1, hidden_size)
    example_speaker = torch.randn(1, hidden_size)

    print("Tracing...")
    traced = torch.jit.trace(wrapper, (
        example_role, example_first_text,
        example_tts_bos, example_tts_pad, example_speaker
    ))

    print("Converting...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="role_ids", shape=(1, 3), dtype=np.int32),
            ct.TensorType(name="first_text_id", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="tts_bos_embed", shape=(1, 1, hidden_size), dtype=np.float32),
            ct.TensorType(name="tts_pad_embed", shape=(1, 1, hidden_size), dtype=np.float32),
            ct.TensorType(name="speaker_embed", shape=(1, hidden_size), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="kv_cache", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "qwen3_tts_lm_prefill_v5.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    import subprocess
    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")

    return mlmodel


def main():
    print("=" * 60)
    print("Qwen3-TTS LM Prefill V5 - Correct 10-Position Sequence")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    talker = model.model.talker
    talker_config = talker.config

    print("\n2. Creating wrapper...")
    wrapper = TracablePrefillV5(talker, talker_config)
    wrapper.eval()

    print("\n3. Verifying...")
    is_valid = verify_prefill(wrapper, model)

    if is_valid:
        print("\n4. Converting to CoreML...")
        convert_to_coreml(wrapper, hidden_size=talker_config.hidden_size)
        print("\nDone!")
    else:
        print("\nVerification failed, skipping conversion")


if __name__ == "__main__":
    main()
