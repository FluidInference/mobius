# Qwen3-TTS LM → CoreML Conversion v6
# Non-streaming mode: embeds ALL text at once in prefill

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_TEXT_LENGTH = 128


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


class TracablePrefillV6(nn.Module):
    """Non-streaming prefill: embeds ALL text at once.

    Sequence layout:
    - [0:3] role prefix (text_embed)
    - [3:3+4] codec think tokens (tts_pad + codec_embed)
    - [3+4] speaker (tts_pad + speaker_embed)
    - [3+5] (tts_bos + codec_pad)
    - [3+6:3+6+text_len] all text tokens (text_embed + codec_pad for each)
    - [3+6+text_len] tts_eos + codec_pad
    - [3+6+text_len+1] tts_pad + codec_bos

    Then during decode, trailing_text_hidden is constant (tts_pad_embed).
    """

    def __init__(self, talker):
        super().__init__()
        self.text_embedding = talker.model.text_embedding
        self.text_projection = talker.text_projection
        self.codec_embedding = talker.model.codec_embedding
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head

        self.config = talker.config
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size

        # Special token IDs
        self.codec_think_id = self.config.codec_think_id
        self.codec_think_bos_id = self.config.codec_think_bos_id
        self.codec_think_eos_id = self.config.codec_think_eos_id
        self.codec_pad_id = self.config.codec_pad_id
        self.codec_bos_id = self.config.codec_bos_id
        self.english_language_id = self.config.codec_language_id["english"]

    def forward(self, role_ids: torch.Tensor, text_ids: torch.Tensor, text_length: torch.Tensor,
                tts_bos_embed: torch.Tensor, tts_pad_embed: torch.Tensor, tts_eos_embed: torch.Tensor,
                speaker_embed: torch.Tensor) -> tuple:
        """
        Non-streaming prefill: embed all text at once.
        Uses full MAX_TEXT_LENGTH with masking for variable-length support.

        Args:
            role_ids: [B, 3] - role prefix token IDs
            text_ids: [B, MAX_TEXT_LENGTH] - all text tokens (padded)
            text_length: [B] - actual text length
            tts_bos_embed: [B, 1, hidden] - TTS BOS embedding
            tts_pad_embed: [B, 1, hidden] - TTS PAD embedding
            tts_eos_embed: [B, 1, hidden] - TTS EOS embedding
            speaker_embed: [B, 1024] - speaker embedding

        Returns:
            logits: [B, vocab_size]
            kv_cache: [56, B, num_kv_heads, seq_len, head_dim]
        """
        batch_size = role_ids.shape[0]
        device = role_ids.device
        max_text_len = text_ids.shape[1]  # MAX_TEXT_LENGTH

        # === Position 0-2: Role prefix ===
        role_embed = self.text_projection(self.text_embedding(role_ids))  # [B, 3, hidden]

        # === Position 3-6: Codec think tokens ===
        codec_think_ids = torch.tensor([
            [self.codec_think_id, self.codec_think_bos_id,
             self.english_language_id, self.codec_think_eos_id]
        ], dtype=torch.long, device=device).expand(batch_size, -1)
        codec_think_embeds = self.codec_embedding(codec_think_ids)
        think_positions = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

        # === Position 7: Speaker ===
        speaker_position = tts_pad_embed + speaker_embed.unsqueeze(1)

        # === Position 8: tts_bos + codec_pad ===
        codec_pad_embed = self.codec_embedding(torch.tensor([[self.codec_pad_id]], dtype=torch.long, device=device))
        bos_pad_position = tts_bos_embed + codec_pad_embed

        # === Position 9 to 9+text_len-1: All text tokens + codec_pad ===
        # Embed all text tokens
        all_text_embed = self.text_projection(self.text_embedding(text_ids[:, :text_len]))  # [B, text_len, hidden]

        # Create codec_pad embeddings for all text positions
        codec_pad_for_text = self.codec_embedding(
            torch.full((batch_size, text_len), self.codec_pad_id, dtype=torch.long, device=device)
        )
        text_positions = all_text_embed + codec_pad_for_text  # [B, text_len, hidden]

        # === Position 9+text_len: tts_eos + codec_pad ===
        eos_pad_position = tts_eos_embed + codec_pad_embed

        # === Position 9+text_len+1: tts_pad + codec_bos ===
        codec_bos_embed = self.codec_embedding(torch.tensor([[self.codec_bos_id]], dtype=torch.long, device=device))
        pad_bos_position = tts_pad_embed + codec_bos_embed

        # === Concatenate all positions ===
        hidden_states = torch.cat([
            role_embed,          # [B, 3, hidden]
            think_positions,     # [B, 4, hidden]
            speaker_position,    # [B, 1, hidden]
            bos_pad_position,    # [B, 1, hidden]
            text_positions,      # [B, text_len, hidden]
            eos_pad_position,    # [B, 1, hidden]
            pad_bos_position,    # [B, 1, hidden]
        ], dim=1)

        seq_len = hidden_states.shape[1]  # 3 + 4 + 1 + 1 + text_len + 1 + 1 = text_len + 11

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
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

        # === Run transformer layers ===
        all_keys = []
        all_values = []

        for layer in self.layers:
            hidden_states, key, value = self._run_layer(
                layer, hidden_states, causal_mask, cos, sin
            )
            all_keys.append(key)
            all_values.append(value)

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


def verify_prefill(wrapper, model, text="Hello world, this is a test."):
    """Verify wrapper output."""
    print("\n=== Verification ===")

    talker = model.model.talker
    config = talker.config
    processor = model.processor

    # Special token IDs
    TTS_BOS_TOKEN_ID = 151672
    TTS_PAD_TOKEN_ID = 151671
    TTS_EOS_TOKEN_ID = 151673
    ROLE_PREFIX = [151644, 77091, 198]

    tokenizer = processor.tokenizer
    text_ids_list = tokenizer.encode(text, add_special_tokens=False)
    text_len = len(text_ids_list)
    print(f"Text: '{text}'")
    print(f"Text tokens: {text_len}")

    with torch.no_grad():
        # TTS embeddings
        tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
        tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
        tts_bos_embed = tts_embed[:, 0:1, :]
        tts_pad_embed = tts_embed[:, 1:2, :]
        tts_eos_embed = tts_embed[:, 2:3, :]

        # Speaker
        speaker_embed = torch.from_numpy(np.load("speaker_embedding.npy")).unsqueeze(0)

        # Inputs
        role_ids = torch.tensor([ROLE_PREFIX])
        text_ids_padded = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
        text_ids_padded[0, :text_len] = torch.tensor(text_ids_list)
        text_length = torch.tensor([text_len])

        # Wrapper output
        wrapper_logits, wrapper_kv = wrapper(
            role_ids, text_ids_padded, text_length,
            tts_bos_embed, tts_pad_embed, tts_eos_embed,
            speaker_embed
        )
        print(f"Wrapper logits shape: {wrapper_logits.shape}")
        print(f"Wrapper KV shape: {wrapper_kv.shape}")
        print(f"Wrapper first token: {torch.argmax(wrapper_logits, dim=-1).item()}")

        # Build official non_streaming_mode sequence for comparison
        codec_prefill = [[
            config.codec_think_id, config.codec_think_bos_id,
            config.codec_language_id["english"], config.codec_think_eos_id
        ]]
        codec_embed_0 = talker.model.codec_embedding(torch.tensor(codec_prefill))
        codec_embed_1 = talker.model.codec_embedding(torch.tensor([[config.codec_pad_id, config.codec_bos_id]]))
        codec_input_embedding = torch.cat([
            codec_embed_0, speaker_embed.view(1, 1, -1), codec_embed_1
        ], dim=1)

        # Role prefix
        role_embed = talker.text_projection(talker.model.text_embedding(torch.tensor([ROLE_PREFIX])))

        # Think + speaker + bos_pad (6 positions)
        _talker_embed = torch.cat([
            tts_pad_embed.expand(-1, 5, -1),
            tts_bos_embed
        ], dim=1) + codec_input_embedding[:, :-1]

        # All text + eos + final bos
        all_text_embed = talker.text_projection(talker.model.text_embedding(text_ids_padded[:, :text_len]))
        codec_pad_for_text = talker.model.codec_embedding(
            torch.full((1, text_len), config.codec_pad_id, dtype=torch.long)
        )
        text_positions = all_text_embed + codec_pad_for_text

        codec_pad_single = talker.model.codec_embedding(torch.tensor([[config.codec_pad_id]]))
        eos_position = tts_eos_embed + codec_pad_single
        final_bos_position = tts_pad_embed + codec_input_embedding[:, -1:]

        official_embed = torch.cat([
            role_embed,
            _talker_embed,
            text_positions,
            eos_position,
            final_bos_position
        ], dim=1)

        print(f"Official embed shape: {official_embed.shape}")

        # Run through model
        outputs = talker.model(inputs_embeds=official_embed, use_cache=False, return_dict=True)
        official_logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)
        print(f"Official first token: {torch.argmax(official_logits, dim=-1).item()}")

        corr = np.corrcoef(wrapper_logits.numpy().flatten(), official_logits.numpy().flatten())[0, 1]
        print(f"Correlation: {corr:.6f}")

    return torch.argmax(wrapper_logits, dim=-1).item() == torch.argmax(official_logits, dim=-1).item()


def convert_to_coreml(wrapper, hidden_size=1024):
    """Convert to CoreML."""
    print("\n=== Converting to CoreML ===")

    wrapper.eval()

    # Use flexible text length
    text_len_range = ct.RangeDim(lower_bound=1, upper_bound=MAX_TEXT_LENGTH, default=10)

    example_role = torch.randint(0, 10000, (1, 3))
    example_text = torch.randint(0, 10000, (1, MAX_TEXT_LENGTH))
    example_text_len = torch.tensor([10])
    example_bos = torch.randn(1, 1, hidden_size)
    example_pad = torch.randn(1, 1, hidden_size)
    example_eos = torch.randn(1, 1, hidden_size)
    example_spk = torch.randn(1, hidden_size)

    print("Tracing...")
    traced = torch.jit.trace(wrapper, (
        example_role, example_text, example_text_len,
        example_bos, example_pad, example_eos, example_spk
    ))

    print("Converting...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="role_ids", shape=(1, 3), dtype=np.int32),
            ct.TensorType(name="text_ids", shape=(1, MAX_TEXT_LENGTH), dtype=np.int32),
            ct.TensorType(name="text_length", shape=(1,), dtype=np.int32),
            ct.TensorType(name="tts_bos_embed", shape=(1, 1, hidden_size), dtype=np.float32),
            ct.TensorType(name="tts_pad_embed", shape=(1, 1, hidden_size), dtype=np.float32),
            ct.TensorType(name="tts_eos_embed", shape=(1, 1, hidden_size), dtype=np.float32),
            ct.TensorType(name="speaker_embed", shape=(1, hidden_size), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="kv_cache", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "qwen3_tts_lm_prefill_v6.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    import subprocess
    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")

    return mlmodel


def main():
    print("=" * 60)
    print("Qwen3-TTS LM Prefill V6 - Non-Streaming Mode")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    talker = model.model.talker

    print("\n2. Creating wrapper...")
    wrapper = TracablePrefillV6(talker)
    wrapper.eval()

    print("\n3. Verifying...")
    is_valid = verify_prefill(wrapper, model)

    if is_valid:
        print("\n4. Converting to CoreML...")
        convert_to_coreml(wrapper, hidden_size=talker.config.hidden_size)
        print("\nDone!")
    else:
        print("\nVerification failed, skipping conversion")


if __name__ == "__main__":
    main()
