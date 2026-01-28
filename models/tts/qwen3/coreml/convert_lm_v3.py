# Qwen3-TTS LM → CoreML Conversion v3
# Adds attention masking to handle variable-length input

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_TEXT_LENGTH = 128
PREFILL_LEN = MAX_TEXT_LENGTH + 2  # lang + text + bos


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


class TracablePrefillV3(nn.Module):
    """Prefill with attention masking for variable-length input."""

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

        self.language_id = self.config.codec_language_id["english"]
        self.bos_id = self.config.codec_bos_id

    def forward(self, text_ids: torch.Tensor, text_length: torch.Tensor) -> tuple:
        """
        Args:
            text_ids: [B, MAX_TEXT_LENGTH] - padded text tokens
            text_length: [B] - actual text length (not including lang/bos)

        Returns:
            logits: [B, vocab_size]
            kv_cache: [56, B, num_kv_heads, seq_len, head_dim]
        """
        batch_size = text_ids.shape[0]
        max_text_len = text_ids.shape[1]

        # Create embeddings with BOS in the correct position
        # Layout: [lang, text[0], text[1], ..., text[n-1], bos, padding...]
        # BOS should be at position text_length + 1

        text_embed = self.text_embedding(text_ids)
        text_projected = self.text_projection(text_embed)  # [B, MAX_TEXT_LEN, hidden]

        # Language and BOS embeddings
        lang_ids = torch.full((batch_size, 1), self.language_id, dtype=torch.long, device=text_ids.device)
        bos_ids = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=text_ids.device)
        lang_embed = self.codec_embedding(lang_ids)  # [B, 1, hidden]
        bos_embed = self.codec_embedding(bos_ids)    # [B, 1, hidden]

        # Insert BOS at the correct position by manipulating the tensor
        # Result should be: [lang, text[0:text_len], bos, text[text_len:] (padding)]
        # But this is tricky with variable text_length...

        # Simpler approach: Create full sequence with BOS at each possible position,
        # then select the right one. But that's expensive.

        # For now, use a scatter approach:
        # 1. Create base: [lang, text (all 128), padding_embed]
        # 2. Scatter BOS at position text_length + 1

        # Actually, let's just reorganize: put lang first, then interleave text with potential BOS

        # Cleaner approach: Build the sequence properly
        # hidden_states[0] = lang
        # hidden_states[1:text_length+1] = text[0:text_length]
        # hidden_states[text_length+1] = bos
        # hidden_states[text_length+2:] = padding (zeros or text embeddings - they'll be masked)

        seq_len = max_text_len + 2  # Fixed sequence length

        # Start with all text embeddings shifted by 1 (to leave room for lang at pos 0)
        hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=text_ids.device, dtype=text_projected.dtype)
        hidden_states[:, 0:1, :] = lang_embed
        hidden_states[:, 1:max_text_len+1, :] = text_projected

        # Insert BOS at position text_length + 1
        # Use scatter for this
        bos_position = (text_length + 1).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        hidden_states.scatter_(dim=1, index=bos_position, src=bos_embed)

        # Position embeddings
        pos_1d = torch.arange(seq_len, device=hidden_states.device)
        position_ids = pos_1d.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Create attention mask with padding
        # Valid positions: [0, 1, ..., text_length, text_length+1] = lang + real_text + bos
        # For the last position (bos), it should only attend to valid positions

        # actual_len = 1 (lang) + text_length + 1 (bos) = text_length + 2
        actual_len = text_length + 2  # [B]

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=hidden_states.dtype, device=hidden_states.device),
            diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        # Create padding mask
        # For each position i, mask out positions j where j > actual_len - 1 (except for causal)
        # Actually we want: position j is valid if j < actual_len
        # But the bos position should be at actual_len - 1

        # Simpler approach: create mask where:
        # - Position 0 (lang) is always valid
        # - Positions 1 to text_length are valid text
        # - Position text_length + 1 is bos (the position we predict from)
        # - Positions text_length + 2 to seq_len - 1 are padding (should be masked)

        # For query at position q, key at position k:
        # Valid if: k < actual_len AND k <= q (causal)

        # Create position indices
        q_pos = torch.arange(seq_len, device=hidden_states.device).view(1, 1, seq_len, 1)
        k_pos = torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
        actual_len_expanded = actual_len.view(batch_size, 1, 1, 1)

        # Padding mask: k >= actual_len → mask out
        # Use where to avoid NaN from False * -inf
        padding_mask = torch.where(
            k_pos >= actual_len_expanded,
            torch.tensor(float("-inf"), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.tensor(0.0, dtype=hidden_states.dtype, device=hidden_states.device),
        )

        # Combine causal and padding masks
        combined_mask = causal_mask + padding_mask
        combined_mask = combined_mask.expand(batch_size, 1, seq_len, seq_len)

        # Run through layers
        all_keys = []
        all_values = []

        for layer in self.layers:
            hidden_states, key, value = self._run_layer(
                layer, hidden_states, combined_mask, cos, sin
            )
            all_keys.append(key)
            all_values.append(value)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Get logits from the bos position (actual_len - 1)
        # For simplicity, we use the last position of the VALID sequence
        # This requires gathering from the right position

        # Create index for gathering: [B, 1]
        bos_position = (actual_len - 1).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        last_hidden = torch.gather(hidden_states, dim=1, index=bos_position)
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


def verify_prefill(wrapper, talker, text_ids, text_length):
    """Verify wrapper matches PyTorch with same input."""
    print("\n=== Verification ===")

    with torch.no_grad():
        # Wrapper output
        wrapper_logits, _ = wrapper(text_ids, text_length)
        print(f"Wrapper logits shape: {wrapper_logits.shape}")

        # PyTorch reference (no padding)
        actual_text = text_ids[0, :text_length[0].item()]
        text_embed = talker.model.text_embedding(actual_text.unsqueeze(0))
        text_projected = talker.text_projection(text_embed)

        lang_embed = talker.model.codec_embedding(torch.tensor([[talker.config.codec_language_id["english"]]]))
        bos_embed = talker.model.codec_embedding(torch.tensor([[talker.config.codec_bos_id]]))

        combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
        outputs = talker.model(inputs_embeds=combined, use_cache=False, return_dict=True)
        pytorch_logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)

        print(f"PyTorch logits shape: {pytorch_logits.shape}")

        # Compare
        wrapper_token = torch.argmax(wrapper_logits, dim=-1).item()
        pytorch_token = torch.argmax(pytorch_logits, dim=-1).item()
        print(f"Wrapper first token: {wrapper_token}")
        print(f"PyTorch first token: {pytorch_token}")
        print(f"Match: {wrapper_token == pytorch_token}")

        corr = np.corrcoef(wrapper_logits.numpy().flatten(), pytorch_logits.numpy().flatten())[0, 1]
        print(f"Correlation: {corr:.6f}")

    return wrapper_token == pytorch_token


def convert_to_coreml(wrapper):
    """Convert to CoreML."""
    print("\n=== Converting to CoreML ===")

    wrapper.eval()

    example_text = torch.randint(0, 10000, (1, MAX_TEXT_LENGTH))
    example_length = torch.tensor([10])

    print("Tracing...")
    traced = torch.jit.trace(wrapper, (example_text, example_length))

    print("Converting...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="text_ids", shape=(1, MAX_TEXT_LENGTH), dtype=np.int32),
            ct.TensorType(name="text_length", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="kv_cache", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "qwen3_tts_lm_prefill_v3.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    import subprocess
    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")

    return mlmodel


def main():
    print("=" * 60)
    print("Qwen3-TTS LM Prefill V3 - With Attention Masking")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    talker = model.model.talker
    processor = model.processor

    print("\n2. Creating wrapper...")
    wrapper = TracablePrefillV3(talker)
    wrapper.eval()

    print("\n3. Verifying...")
    text = "Hello world"
    inputs = processor(text=text, return_tensors="pt")
    text_ids = inputs.input_ids
    text_len = text_ids.shape[1]

    # Pad
    padded = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
    padded[0, :text_len] = text_ids[0]
    text_length = torch.tensor([text_len])

    is_valid = verify_prefill(wrapper, talker, padded, text_length)

    if is_valid:
        print("\n4. Converting to CoreML...")
        convert_to_coreml(wrapper)
        print("\nDone!")
    else:
        print("\nVerification failed, skipping conversion")


if __name__ == "__main__":
    main()
