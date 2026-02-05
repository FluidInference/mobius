# Qwen3-TTS LM → CoreML Conversion v9
# Non-streaming mode with CORRECT sequence based on official model tracing:
#
# Official sequence structure:
# - Position 0-2: role prefix (text embed only, NO codec!)
# - Position 3: tts_pad + think
# - Position 4: tts_pad + think_bos
# - Position 5: tts_pad + lang
# - Position 6: tts_pad + think_eos
# - Position 7: tts_pad + speaker
# - Position 8: tts_bos + codec_pad
# - Position 9 to 9+text_len-1: text_embed + codec_pad
# - Position 9+text_len: tts_eos + codec_pad
# - Position 9+text_len+1: tts_pad + codec_bos
#
# Total: 3 + 6 + text_len + 2 = text_len + 11

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_TEXT_LENGTH = 128
# CORRECT: text_len + 11
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


class TracablePrefillV9(nn.Module):
    """Non-streaming prefill with CORRECT sequence layout matching official model."""

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

    def _run_layer(self, layer, hidden_states, mask, cos, sin):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape

        q = layer.self_attn.q_proj(hidden_states)
        k = layer.self_attn.k_proj(hidden_states)
        v = layer.self_attn.v_proj(hidden_states)

        # Reshape to (batch, seq, heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Q/K normalization (critical - was missing!)
        q = layer.self_attn.q_norm(q)
        k = layer.self_attn.k_norm(k)

        # Transpose to (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb_simple(q, k, cos, sin)

        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_expanded = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
            k_expanded = k_expanded.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v_expanded = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
            v_expanded = v_expanded.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
        else:
            k_expanded = k
            v_expanded = v

        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_expanded)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)  # -1 handles GQA
        attn_output = layer.self_attn.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, k, v

    def forward(self, role_ids: torch.Tensor, text_ids: torch.Tensor, text_length: torch.Tensor,
                tts_bos_embed: torch.Tensor, tts_pad_embed: torch.Tensor, tts_eos_embed: torch.Tensor,
                speaker_embed: torch.Tensor) -> tuple:
        batch_size = role_ids.shape[0]
        device = role_ids.device

        # === Build fixed-length sequence ===
        hidden_states = torch.zeros(batch_size, FIXED_SEQ_LEN, self.hidden_size,
                                   device=device, dtype=tts_bos_embed.dtype)

        # Position 0-2: Role prefix (text embed ONLY, no codec!)
        role_embed = self.text_projection(self.text_embedding(role_ids))
        hidden_states[:, 0:3, :] = role_embed

        # Position 3-6: tts_pad + Codec think tokens (4 positions)
        # think_ids = [think, think_bos, lang, think_eos]
        codec_think_ids = torch.tensor([
            [self.codec_think_id, self.codec_think_bos_id,
             self.english_language_id, self.codec_think_eos_id]
        ], dtype=torch.long, device=device).expand(batch_size, -1)
        codec_think_embeds = self.codec_embedding(codec_think_ids)
        hidden_states[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

        # Position 7: tts_pad + Speaker (CORRECTED from tts_bos!)
        hidden_states[:, 7:8, :] = tts_pad_embed + speaker_embed.unsqueeze(1)

        # Position 8: tts_bos + codec_pad (NEW!)
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

        # Position 9+MAX_TEXT_LENGTH: tts_eos + codec_pad (placeholder)
        eos_position_idx = 9 + MAX_TEXT_LENGTH
        hidden_states[:, eos_position_idx:eos_position_idx+1, :] = tts_eos_embed + codec_pad_embed

        # Position 9+MAX_TEXT_LENGTH+1: tts_pad + codec_bos (placeholder)
        codec_bos_embed = self.codec_embedding(
            torch.tensor([[self.codec_bos_id]], dtype=torch.long, device=device).expand(batch_size, -1)
        )
        bos_position_idx = 9 + MAX_TEXT_LENGTH + 1
        hidden_states[:, bos_position_idx:bos_position_idx+1, :] = tts_pad_embed + codec_bos_embed

        # Move eos and bos to correct positions based on text_length
        eos_embed = tts_eos_embed + codec_pad_embed
        bos_embed = tts_pad_embed + codec_bos_embed

        # Scatter eos to position 9 + text_length
        eos_idx = (9 + text_length).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        hidden_states.scatter_(1, eos_idx, eos_embed)

        # Scatter bos to position 10 + text_length
        bos_idx = (10 + text_length).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        hidden_states.scatter_(1, bos_idx, bos_embed)

        # === Create attention mask ===
        # actual_len = 9 + text_length + 2 = text_length + 11
        actual_len = text_length + 11

        # Position embeddings
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

        # Get logits from bos position (actual_len - 1)
        bos_position = (actual_len - 1).view(batch_size, 1, 1).expand(-1, -1, self.hidden_size)
        last_hidden = torch.gather(hidden_states, 1, bos_position)
        logits = self.codec_head(last_hidden).squeeze(1)

        # Stack KV cache: [56, B, num_kv_heads, FIXED_SEQ_LEN, head_dim]
        # IMPORTANT: Only positions 0 to actual_len-1 are valid!
        # Caller must slice to kv_cache[:, :, :, :actual_len, :] before passing to decode
        kv_list = []
        for k, v in zip(all_keys, all_values):
            kv_list.append(k)
            kv_list.append(v)
        kv_cache = torch.stack(kv_list, dim=0)

        # Also return past_hidden for decode phase
        return logits, kv_cache, last_hidden


def main():
    print("=" * 60)
    print("Qwen3-TTS LM Prefill V9 - CORRECT Sequence")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel
    import numpy as np

    tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    processor = tts_model.processor
    talker = tts_model.model.talker

    wrapper = TracablePrefillV9(talker)
    wrapper.eval()

    # Test inputs
    TTS_BOS_TOKEN_ID = 151672
    TTS_PAD_TOKEN_ID = 151671
    TTS_EOS_TOKEN_ID = 151673
    ROLE_PREFIX = [151644, 77091, 198]

    text = "Hello world, this is a test of the text to speech system."
    tokenizer = processor.tokenizer
    text_ids_list = tokenizer.encode(text, add_special_tokens=False)
    text_len = len(text_ids_list)

    print(f"\nText: '{text}'")
    print(f"Text tokens: {text_len}")
    print(f"Expected sequence length: {text_len + 11}")

    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
    text_ids[0, :text_len] = torch.tensor(text_ids_list)
    text_length = torch.tensor([text_len], dtype=torch.long)

    with torch.no_grad():
        tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
        tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
        tts_bos_embed = tts_embed[:, 0:1, :]
        tts_pad_embed = tts_embed[:, 1:2, :]
        tts_eos_embed = tts_embed[:, 2:3, :]

    speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
    speaker_embed = torch.from_numpy(speaker_embed).to(torch.float32)

    # Test wrapper
    print("\nTesting wrapper...")
    with torch.no_grad():
        logits, kv_cache, past_hidden = wrapper(role_ids, text_ids, text_length,
                                    tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed)
    print(f"Wrapper output shapes: logits={logits.shape}, kv_cache={kv_cache.shape}, past_hidden={past_hidden.shape}")

    suppress_mask = np.zeros(talker.config.vocab_size, dtype=bool)
    suppress_mask[2048:] = True
    suppress_mask[talker.config.codec_eos_token_id] = False

    logits_np = logits.numpy().copy()
    logits_np[0, suppress_mask] = -float('inf')
    first_token = int(np.argmax(logits_np))
    print(f"Wrapper first token: {first_token}")

    # Compare with official
    print("\nComparing with official model...")

    # Run official generate
    input_text = tts_model._build_assistant_text(text)
    full_input_ids = tts_model._tokenize_texts([input_text])[0]

    voice_clone_prompt = {
        'ref_spk_embedding': [speaker_embed.squeeze(0)],
        'x_vector_only_mode': [True],
        'icl_mode': [False],
        'ref_code': None,
    }

    with torch.no_grad():
        result = tts_model.model.generate(
            input_ids=[full_input_ids],
            languages=['english'],
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True,
            max_new_tokens=20,
            do_sample=False,
        )
    codes_list, _ = result
    codes = codes_list[0]
    official_first = codes[0, 0].item()
    print(f"Official first token: {official_first}")
    print(f"Official first 10 tokens: {codes[:10, 0].tolist()}")

    if first_token == official_first:
        print("\nMATCH! First tokens are identical.")
    else:
        print(f"\nMISMATCH: wrapper={first_token}, official={official_first}")
        top10 = np.argsort(logits_np[0])[-10:][::-1]
        print(f"Wrapper top 10: {top10.tolist()}")

    # Trace for CoreML
    print("\nTracing for CoreML...")
    traced = torch.jit.trace(
        wrapper,
        (role_ids, text_ids, text_length, tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed)
    )

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="role_ids", shape=(1, 3), dtype=np.int32),
            ct.TensorType(name="text_ids", shape=(1, MAX_TEXT_LENGTH), dtype=np.int32),
            ct.TensorType(name="text_length", shape=(1,), dtype=np.int32),
            ct.TensorType(name="tts_bos_embed", shape=(1, 1, 1024), dtype=np.float32),
            ct.TensorType(name="tts_pad_embed", shape=(1, 1, 1024), dtype=np.float32),
            ct.TensorType(name="tts_eos_embed", shape=(1, 1, 1024), dtype=np.float32),
            ct.TensorType(name="speaker_embed", shape=(1, 1024), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="kv_cache", dtype=np.float32),
            ct.TensorType(name="past_hidden", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.save("qwen3_tts_lm_prefill_v9.mlpackage")
    print("Saved: qwen3_tts_lm_prefill_v9.mlpackage")

    import subprocess
    result = subprocess.run(['du', '-sh', 'qwen3_tts_lm_prefill_v9.mlpackage'], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")
    print("Done!")


if __name__ == "__main__":
    main()
