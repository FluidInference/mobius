# %% [markdown]
# # Qwen3-TTS LM (Talker) → CoreML Conversion v2
#
# Avoids M-RoPE interleaved indexing that causes CoreML issues.
# For text-only (where position IDs are identical across dimensions),
# we can simplify to standard RoPE.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
import numpy as np

# Configuration
MAX_TEXT_LENGTH = 128
MAX_CODEC_LENGTH = 256
MAX_SEQ_LENGTH = MAX_TEXT_LENGTH + MAX_CODEC_LENGTH + 2
PREFILL_LEN = MAX_TEXT_LENGTH + 2

print(f"Max text length: {MAX_TEXT_LENGTH}")
print(f"Max codec length: {MAX_CODEC_LENGTH}")
print(f"Max total sequence: {MAX_SEQ_LENGTH}")

# %%
# Load the model
from qwen_tts import Qwen3TTSModel

print("\nLoading model...")
model = Qwen3TTSModel.from_pretrained(
    "./model_0.6b",
    device_map="cpu",
    torch_dtype=torch.float32,
)
talker = model.model.talker
processor = model.processor
config = talker.config

print("Model loaded!")
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")


# %%
# Implement CoreML-friendly RoPE
def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_simple(q, k, cos, sin):
    """
    Apply rotary position embedding (CoreML-compatible).

    For text-only where all 3 M-RoPE dimensions have same positions,
    we can use the first dimension's cos/sin directly.

    Args:
        q: [B, num_heads, seq_len, head_dim]
        k: [B, num_kv_heads, seq_len, head_dim]
        cos: [3, B, seq_len, head_dim] or [B, seq_len, head_dim]
        sin: [3, B, seq_len, head_dim] or [B, seq_len, head_dim]
    """
    # Use first dimension if M-RoPE format
    if cos.dim() == 4:
        cos = cos[0]  # [B, seq_len, head_dim]
        sin = sin[0]

    # Expand for num_heads dimension: [B, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# %%
# Traceable Prefill model
class TracablePrefillV2(nn.Module):
    """Traceable prefill with simplified RoPE."""

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

    def forward(self, text_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_ids.shape[0]

        # Create embeddings
        text_embed = self.text_embedding(text_ids)
        text_projected = self.text_projection(text_embed)

        lang_ids = torch.full((batch_size, 1), self.language_id, dtype=torch.long, device=text_ids.device)
        bos_ids = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=text_ids.device)
        lang_embed = self.codec_embedding(lang_ids)
        bos_embed = self.codec_embedding(bos_ids)

        hidden_states = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
        seq_len = hidden_states.shape[1]

        # Position embeddings (M-RoPE format)
        pos_1d = torch.arange(seq_len, device=hidden_states.device)
        position_ids = pos_1d.unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)

        # Run through layers
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

        # Logits
        logits = self.codec_head(hidden_states[:, -1:, :]).squeeze(1)

        # Stack KV cache - flatten to rank 5 for CoreML
        # Original: [num_layers, 2, B, num_kv_heads, seq_len, head_dim]
        # Flattened: [num_layers * 2, B, num_kv_heads, seq_len, head_dim]
        # Layout: [K0, V0, K1, V1, ..., K27, V27]
        kv_list = []
        for key, value in zip(all_keys, all_values):
            kv_list.append(key)
            kv_list.append(value)
        kv_cache = torch.stack(kv_list, dim=0)  # [56, B, num_kv_heads, seq_len, head_dim]

        return logits, kv_cache

    def _create_causal_mask(self, seq_len, dtype, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def _run_layer(self, layer, hidden_states, causal_mask, cos, sin):
        residual = hidden_states

        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, key, value = self._run_attention(
            layer.self_attn, hidden_states, causal_mask, cos, sin
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, key, value

    def _run_attention(self, attn, hidden_states, causal_mask, cos, sin):
        bsz, q_len, _ = hidden_states.shape

        # Project and reshape
        query_states = attn.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = attn.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Apply QK-norm (important for accuracy!)
        query_states = attn.q_norm(query_states).transpose(1, 2)
        key_states = attn.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply simplified RoPE (CoreML-compatible)
        query_states, key_states = apply_rotary_pos_emb_simple(query_states, key_states, cos, sin)

        # Store KV before GQA expansion
        key_out = key_states
        value_out = value_states

        # GQA expansion
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)

        return attn_output, key_out, value_out


# %%
# Traceable Decode model
class TracableDecodeV2(nn.Module):
    """Traceable decode with simplified RoPE."""

    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head

        self.config = talker.config
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim

    def forward(
        self,
        token_id: torch.Tensor,
        kv_cache: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_id: [B, 1] codec token
            kv_cache: [num_layers * 2, B, num_kv_heads, cache_len, head_dim]
                      Layout: [K0, V0, K1, V1, ..., K27, V27]
            position: [B] current position
        """
        batch_size = token_id.shape[0]

        # Embed token
        hidden_states = self.codec_embedding(token_id)

        # Position embeddings
        position_ids = position.unsqueeze(1)  # [B, 1]
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, 1]
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Run through layers
        new_kv_list = []

        for i, layer in enumerate(self.layers):
            # KV cache layout: [K0, V0, K1, V1, ...]
            cached_key = kv_cache[i * 2]
            cached_value = kv_cache[i * 2 + 1]

            hidden_states, new_key, new_value = self._run_decode_layer(
                layer, hidden_states, cached_key, cached_value, cos, sin
            )
            new_kv_list.append(new_key)
            new_kv_list.append(new_value)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Logits
        logits = self.codec_head(hidden_states).squeeze(1)

        # Stack new KV cache (flattened)
        new_kv_cache = torch.stack(new_kv_list, dim=0)

        return logits, new_kv_cache

    def _run_decode_layer(self, layer, hidden_states, cached_key, cached_value, cos, sin):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        bsz = hidden_states.shape[0]

        # Project and reshape
        query_states = layer.self_attn.q_proj(hidden_states).view(bsz, 1, self.num_heads, self.head_dim)
        key_states = layer.self_attn.k_proj(hidden_states).view(bsz, 1, self.num_kv_heads, self.head_dim)
        value_states = layer.self_attn.v_proj(hidden_states).view(bsz, 1, self.num_kv_heads, self.head_dim)

        # Apply QK-norm (important for accuracy!)
        query_states = layer.self_attn.q_norm(query_states).transpose(1, 2)
        key_states = layer.self_attn.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply simplified RoPE
        query_states, key_states = apply_rotary_pos_emb_simple(query_states, key_states, cos, sin)

        # Concatenate with cache
        full_keys = torch.cat([cached_key, key_states], dim=2)
        full_values = torch.cat([cached_value, value_states], dim=2)

        # GQA expansion for attention
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            attn_keys = full_keys.repeat_interleave(n_rep, dim=1)
            attn_values = full_values.repeat_interleave(n_rep, dim=1)
        else:
            attn_keys = full_keys
            attn_values = full_values

        # Attention
        attn_weights = torch.matmul(query_states, attn_keys.transpose(2, 3)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, attn_values)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, 1, -1)
        attn_output = layer.self_attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, full_keys, full_values


# %%
# Test models
print("\n=== Testing V2 Models ===")

prefill_v2 = TracablePrefillV2(talker)
prefill_v2.eval()

decode_v2 = TracableDecodeV2(talker)
decode_v2.eval()

test_text_ids = torch.tensor([[9707, 1879]])  # "Hello world"

with torch.no_grad():
    logits, kv_cache = prefill_v2(test_text_ids)

print(f"Prefill logits shape: {logits.shape}")
print(f"KV cache shape: {kv_cache.shape}")
print(f"Top 5 tokens: {torch.topk(logits[0], 5).indices.tolist()}")

# Compare with original
with torch.no_grad():
    text_embed = talker.model.text_embedding(test_text_ids)
    text_projected = talker.text_projection(text_embed)
    lang_embed = talker.model.codec_embedding(torch.tensor([[config.codec_language_id["english"]]]))
    bos_embed = talker.model.codec_embedding(torch.tensor([[config.codec_bos_id]]))
    combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
    outputs = talker.model(inputs_embeds=combined, use_cache=True, return_dict=True)
    orig_logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)

print(f"Original top 5: {torch.topk(orig_logits[0], 5).indices.tolist()}")
diff = (logits - orig_logits).abs().max().item()
print(f"Diff from original: {diff}")


# %%
# Test decode
print("\n=== Testing V2 Decode ===")

first_token = torch.argmax(logits, dim=-1, keepdim=True)
initial_pos = kv_cache.shape[3]  # seq_len is at dim 3 now (flattened format)

with torch.no_grad():
    dec_logits, dec_kv = decode_v2(first_token, kv_cache, torch.tensor([initial_pos]))

print(f"Decode logits shape: {dec_logits.shape}")
print(f"Decode KV shape: {dec_kv.shape}")

# Multi-step generation
generated = [first_token.item()]
current_kv = kv_cache
current_pos = initial_pos

for step in range(10):
    with torch.no_grad():
        next_logits, current_kv = decode_v2(
            torch.tensor([[generated[-1]]]),
            current_kv,
            torch.tensor([current_pos])
        )
        next_token = torch.argmax(next_logits, dim=-1).item()
        generated.append(next_token)
        current_pos += 1

        if next_token == config.codec_eos_token_id:
            print(f"EOS at step {step}")
            break

print(f"Generated: {generated}")


# %%
# Trace models
print("\n=== Tracing V2 Models ===")

example_text = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)

try:
    with torch.no_grad():
        traced_prefill = torch.jit.trace(prefill_v2, example_text, strict=False)
    print("Prefill traced successfully!")
except Exception as e:
    print(f"Prefill tracing failed: {e}")
    import traceback
    traceback.print_exc()

example_token = torch.zeros((1, 1), dtype=torch.long)
# Flattened KV cache: [num_layers * 2, B, num_kv_heads, seq_len, head_dim]
example_kv = torch.zeros((config.num_hidden_layers * 2, 1, config.num_key_value_heads, PREFILL_LEN, config.head_dim))
example_pos = torch.tensor([PREFILL_LEN])

try:
    with torch.no_grad():
        traced_decode = torch.jit.trace(decode_v2, (example_token, example_kv, example_pos), strict=False)
    print("Decode traced successfully!")
except Exception as e:
    print(f"Decode tracing failed: {e}")
    import traceback
    traceback.print_exc()


# %%
# Convert Prefill to CoreML
print("\n=== Converting Prefill to CoreML ===")

prefill_inputs = [
    ct.TensorType(name="text_ids", shape=(1, MAX_TEXT_LENGTH), dtype=np.int32),
]

try:
    mlmodel_prefill = ct.convert(
        traced_prefill,
        inputs=prefill_inputs,
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="kv_cache")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    print("Prefill CoreML conversion successful!")

    output_path = Path("qwen3_tts_lm_prefill.mlpackage")
    mlmodel_prefill.save(str(output_path))
    print(f"Saved to: {output_path}")
    print(f"Size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e6:.1f} MB")

except Exception as e:
    print(f"Prefill CoreML conversion failed: {e}")
    import traceback
    traceback.print_exc()


# %%
# Convert Decode to CoreML
print("\n=== Converting Decode to CoreML ===")

decode_inputs = [
    ct.TensorType(name="token_id", shape=(1, 1), dtype=np.int32),
    ct.TensorType(
        name="kv_cache",
        shape=(
            config.num_hidden_layers * 2,  # Flattened: K0, V0, K1, V1, ...
            1,
            config.num_key_value_heads,
            ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LENGTH, default=PREFILL_LEN),
            config.head_dim,
        ),
        dtype=np.float32,
    ),
    ct.TensorType(name="position", shape=(1,), dtype=np.int32),
]

try:
    mlmodel_decode = ct.convert(
        traced_decode,
        inputs=decode_inputs,
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="new_kv_cache")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    print("Decode CoreML conversion successful!")

    output_path = Path("qwen3_tts_lm_decode.mlpackage")
    mlmodel_decode.save(str(output_path))
    print(f"Saved to: {output_path}")
    print(f"Size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e6:.1f} MB")

except Exception as e:
    print(f"Decode CoreML conversion failed: {e}")
    import traceback
    traceback.print_exc()


# %%
print("\n=== Conversion Complete ===")
