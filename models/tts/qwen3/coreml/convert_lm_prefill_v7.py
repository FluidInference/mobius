# Qwen3-TTS LM → CoreML Conversion v7
# Non-streaming mode with fixed-length sequence and masking

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_TEXT_LENGTH = 128
# Fixed sequence layout:
# 0-2: role prefix (3)
# 3-6: codec think tokens (4)
# 7: speaker (1)
# 8: tts_bos + codec_pad (1)
# 9 to 9+MAX_TEXT_LENGTH-1: text tokens (MAX_TEXT_LENGTH)
# 9+MAX_TEXT_LENGTH: tts_eos + codec_pad (1)
# 9+MAX_TEXT_LENGTH+1: tts_pad + codec_bos (1)
# Total: 3 + 4 + 1 + 1 + MAX_TEXT_LENGTH + 1 + 1 = MAX_TEXT_LENGTH + 11

FIXED_SEQ_LEN = MAX_TEXT_LENGTH + 11


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


class TracablePrefillV7(nn.Module):
    """Non-streaming prefill with fixed sequence length and masking."""

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

        # === Build fixed-length sequence ===
        hidden_states = torch.zeros(batch_size, FIXED_SEQ_LEN, self.hidden_size,
                                   device=device, dtype=tts_bos_embed.dtype)

        # Position 0-2: Role prefix
        role_embed = self.text_projection(self.text_embedding(role_ids))
        hidden_states[:, 0:3, :] = role_embed

        # Position 3-6: Codec think tokens
        codec_think_ids = torch.tensor([
            [self.codec_think_id, self.codec_think_bos_id,
             self.english_language_id, self.codec_think_eos_id]
        ], dtype=torch.long, device=device).expand(batch_size, -1)
        codec_think_embeds = self.codec_embedding(codec_think_ids)
        hidden_states[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

        # Position 7: Speaker
        hidden_states[:, 7:8, :] = tts_pad_embed + speaker_embed.unsqueeze(1)

        # Position 8: tts_bos + codec_pad
        codec_pad_embed = self.codec_embedding(
            torch.tensor([[self.codec_pad_id]], dtype=torch.long, device=device).expand(batch_size, -1)
        )
        hidden_states[:, 8:9, :] = tts_bos_embed + codec_pad_embed

        # Position 9 to 9+MAX_TEXT_LENGTH-1: Text tokens + codec_pad
        all_text_embed = self.text_projection(self.text_embedding(text_ids))
        codec_pad_for_text = self.codec_embedding(
            torch.full((batch_size, MAX_TEXT_LENGTH), self.codec_pad_id, dtype=torch.long, device=device)
        )
        hidden_states[:, 9:9+MAX_TEXT_LENGTH, :] = all_text_embed + codec_pad_for_text

        # Position 9+MAX_TEXT_LENGTH: tts_eos + codec_pad (will be moved by scatter)
        eos_position_idx = 9 + MAX_TEXT_LENGTH
        hidden_states[:, eos_position_idx:eos_position_idx+1, :] = tts_eos_embed + codec_pad_embed

        # Position 9+MAX_TEXT_LENGTH+1: tts_pad + codec_bos (will be moved by scatter)
        codec_bos_embed = self.codec_embedding(
            torch.tensor([[self.codec_bos_id]], dtype=torch.long, device=device).expand(batch_size, -1)
        )
        bos_position_idx = 9 + MAX_TEXT_LENGTH + 1
        hidden_states[:, bos_position_idx:bos_position_idx+1, :] = tts_pad_embed + codec_bos_embed

        # Move eos and bos to correct positions based on text_length
        # eos goes to position 9 + text_length
        # bos goes to position 9 + text_length + 1
        # We need to scatter these embeddings to the right positions

        # First, save eos and bos embeds
        eos_embed = tts_eos_embed + codec_pad_embed  # [B, 1, hidden]
        bos_embed = tts_pad_embed + codec_bos_embed  # [B, 1, hidden]

        # Scatter eos to position 9 + text_length
        eos_idx = (9 + text_length).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        hidden_states.scatter_(1, eos_idx, eos_embed)

        # Scatter bos to position 10 + text_length
        bos_idx = (10 + text_length).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        hidden_states.scatter_(1, bos_idx, bos_embed)

        # === Create attention mask ===
        # actual_len = 9 + text_length + 2 = text_length + 11
        actual_len = text_length + 11  # [B]

        # Position embeddings (use full sequence length)
        pos_1d = torch.arange(FIXED_SEQ_LEN, device=device)
        position_ids = pos_1d.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN, dtype=hidden_states.dtype, device=device),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Padding mask
        q_pos = torch.arange(FIXED_SEQ_LEN, device=device).view(1, 1, FIXED_SEQ_LEN, 1)
        k_pos = torch.arange(FIXED_SEQ_LEN, device=device).view(1, 1, 1, FIXED_SEQ_LEN)
        actual_len_expanded = actual_len.view(batch_size, 1, 1, 1)

        padding_mask = torch.where(
            k_pos >= actual_len_expanded,
            torch.tensor(float("-inf"), dtype=hidden_states.dtype, device=device),
            torch.tensor(0.0, dtype=hidden_states.dtype, device=device),
        )

        combined_mask = causal_mask + padding_mask
        combined_mask = combined_mask.expand(batch_size, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)

        # === Run transformer layers ===
        all_keys = []
        all_values = []

        for layer in self.layers:
            hidden_states, key, value = self._run_layer(
                layer, hidden_states, combined_mask, cos, sin
            )
            all_keys.append(key)
            all_values.append(value)

        hidden_states = self.norm(hidden_states)

        # Get logits from the bos position (actual_len - 1)
        bos_position = (actual_len - 1).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        last_hidden = torch.gather(hidden_states, 1, bos_position)
        logits = self.codec_head(last_hidden).squeeze(1)

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

    TTS_BOS = 151672
    TTS_PAD = 151671
    TTS_EOS = 151673
    ROLE_PREFIX = [151644, 77091, 198]

    tokenizer = processor.tokenizer
    text_ids_list = tokenizer.encode(text, add_special_tokens=False)
    text_len = len(text_ids_list)
    print(f"Text: '{text}'")
    print(f"Text tokens: {text_len}")

    with torch.no_grad():
        # TTS embeddings
        tts_ids = torch.tensor([[TTS_BOS, TTS_PAD, TTS_EOS]])
        tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
        tts_bos_embed = tts_embed[:, 0:1, :]
        tts_pad_embed = tts_embed[:, 1:2, :]
        tts_eos_embed = tts_embed[:, 2:3, :]

        speaker_embed = torch.from_numpy(np.load("speaker_embedding.npy")).unsqueeze(0)

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
        wrapper_token = torch.argmax(wrapper_logits, dim=-1).item()
        print(f"Wrapper first token: {wrapper_token}")

        # Build official sequence for comparison
        input_id = torch.tensor([ROLE_PREFIX + text_ids_list])

        codec_prefill = [[config.codec_think_id, config.codec_think_bos_id,
                         config.codec_language_id["english"], config.codec_think_eos_id]]
        codec_embed_0 = talker.model.codec_embedding(torch.tensor(codec_prefill))
        codec_embed_1 = talker.model.codec_embedding(torch.tensor([[config.codec_pad_id, config.codec_bos_id]]))
        codec_input_embedding = torch.cat([
            codec_embed_0, speaker_embed.view(1, 1, -1), codec_embed_1
        ], dim=1)

        _role_embed = talker.text_projection(talker.model.text_embedding(input_id[:, :3]))

        _talker_embed = torch.cat([
            tts_pad_embed.expand(-1, 5, -1),
            tts_bos_embed
        ], dim=1) + codec_input_embedding[:, :-1]

        all_text_official = talker.text_projection(talker.model.text_embedding(input_id[:, 3:]))
        codec_pad_for_text = talker.model.codec_embedding(
            torch.full((1, text_len), config.codec_pad_id)
        )
        text_part = all_text_official + codec_pad_for_text

        eos_part = tts_eos_embed + talker.model.codec_embedding(torch.tensor([[config.codec_pad_id]]))
        final_bos = tts_pad_embed + codec_input_embedding[:, -1:]

        official_embed = torch.cat([_role_embed, _talker_embed, text_part, eos_part, final_bos], dim=1)

        outputs = talker.model(inputs_embeds=official_embed, use_cache=False, return_dict=True)
        official_logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)
        official_token = torch.argmax(official_logits, dim=-1).item()
        print(f"Official first token: {official_token}")

        corr = np.corrcoef(wrapper_logits.numpy().flatten(), official_logits.numpy().flatten())[0, 1]
        print(f"Correlation: {corr:.6f}")

    return wrapper_token == official_token


def convert_to_coreml(wrapper, hidden_size=1024):
    """Convert to CoreML."""
    print("\n=== Converting to CoreML ===")

    wrapper.eval()

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

    output_path = "qwen3_tts_lm_prefill_v7.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    import subprocess
    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")

    return mlmodel


def main():
    print("=" * 60)
    print("Qwen3-TTS LM Prefill V7 - Fixed Length with Masking")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    talker = model.model.talker

    print("\n2. Creating wrapper...")
    wrapper = TracablePrefillV7(talker)
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
