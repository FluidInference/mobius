# %% [markdown]
# # Qwen3-TTS LM (Talker) → CoreML Conversion
#
# Converts the autoregressive LM (text → codec codes) to CoreML.
# Creates two models:
# 1. Prefill: Process text prompt, return first logits + KV cache
# 2. Decode: Single-token generation with KV cache

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
import numpy as np

# Configuration
MAX_TEXT_LENGTH = 128  # Max text tokens
MAX_CODEC_LENGTH = 256  # Max generated codec tokens (~20s at 12.5Hz)
MAX_SEQ_LENGTH = MAX_TEXT_LENGTH + MAX_CODEC_LENGTH + 2  # +2 for lang + bos

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
print(f"Num attention heads: {config.num_attention_heads}")
print(f"Num KV heads: {config.num_key_value_heads}")
print(f"Head dim: {config.head_dim}")

# %%
# Analyze the model structure
print("\n=== Model Structure ===")

# Text embedding
print(f"Text embedding: {talker.model.text_embedding.weight.shape}")  # [text_vocab, 2048]
print(f"Text projection: {type(talker.text_projection).__name__}")  # MLP: 2048 -> 2048 -> 1024

# Codec embedding
print(f"Codec embedding: {talker.model.codec_embedding.weight.shape}")  # [vocab_size, 1024]

# Codec head
print(f"Codec head: {talker.codec_head.weight.shape}")  # [vocab_size, 1024]


# %%
# Create CoreML-compatible wrapper for the Talker
# Uses the original model's forward but extracts KV cache
class TalkerPrefillCoreML(nn.Module):
    """
    Prefill model: processes text prompt and returns first codec token logits + KV cache.
    """

    def __init__(self, talker):
        super().__init__()
        self.talker = talker
        self.config = talker.config

        # Special tokens
        self.language_id = self.config.codec_language_id["english"]
        self.bos_id = self.config.codec_bos_id

    def forward(self, text_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_ids: [B, text_len] text token IDs

        Returns:
            logits: [B, vocab_size] logits for first codec token
            kv_cache: [num_layers, 2, B, num_kv_heads, seq_len, head_dim]
        """
        batch_size = text_ids.shape[0]

        # 1. Create embeddings
        text_embed = self.talker.model.text_embedding(text_ids)  # [B, text_len, 2048]
        text_projected = self.talker.text_projection(text_embed)  # [B, text_len, 1024]

        # Language and BOS tokens
        lang_ids = torch.full((batch_size, 1), self.language_id, dtype=torch.long, device=text_ids.device)
        bos_ids = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=text_ids.device)
        lang_embed = self.talker.model.codec_embedding(lang_ids)
        bos_embed = self.talker.model.codec_embedding(bos_ids)

        # Combine: [lang, text..., bos]
        combined_embeds = torch.cat([lang_embed, text_projected, bos_embed], dim=1)

        # 2. Forward through model
        outputs = self.talker.model(
            inputs_embeds=combined_embeds,
            use_cache=True,
            return_dict=True,
        )

        # 3. Get logits for last position
        logits = self.talker.codec_head(outputs.last_hidden_state[:, -1:, :])
        logits = logits.squeeze(1)  # [B, vocab_size]

        # 4. Extract KV cache
        kv = outputs.past_key_values
        # Convert DynamicCache to tensors
        keys = []
        values = []
        for i in range(len(kv)):
            k, v = kv[i]
            keys.append(k)
            values.append(v)

        # Stack: [num_layers, B, num_kv_heads, seq_len, head_dim]
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        # Combine: [num_layers, 2, B, num_kv_heads, seq_len, head_dim]
        kv_cache = torch.stack([keys, values], dim=1)

        return logits, kv_cache


# %%
# Test the prefill wrapper
print("\n=== Testing Prefill Wrapper ===")

prefill_model = TalkerPrefillCoreML(talker)
prefill_model.eval()

# Test input
test_text_ids = torch.tensor([[9707, 1879]])  # "Hello world"
print(f"Input text IDs: {test_text_ids.shape}")

with torch.no_grad():
    logits, kv_cache = prefill_model(test_text_ids)

print(f"Logits shape: {logits.shape}")
print(f"KV cache shape: {kv_cache.shape}")
print(f"Top 5 tokens: {torch.topk(logits[0], 5).indices.tolist()}")


# %%
# Verify against original direct call
print("\n=== Verifying Against Original ===")

with torch.no_grad():
    # Original forward
    text_embed = talker.model.text_embedding(test_text_ids)
    text_projected = talker.text_projection(text_embed)
    lang_embed = talker.model.codec_embedding(torch.tensor([[config.codec_language_id["english"]]]))
    bos_embed = talker.model.codec_embedding(torch.tensor([[config.codec_bos_id]]))
    combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)

    outputs = talker.model(inputs_embeds=combined, use_cache=True, return_dict=True)
    orig_logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)

    print(f"Original logits shape: {orig_logits.shape}")
    print(f"Original top 5: {torch.topk(orig_logits[0], 5).indices.tolist()}")

    # Compare
    diff = (logits - orig_logits).abs().max().item()
    print(f"Max logits diff: {diff}")


# %%
# Create Decode model - uses past_key_values for generation
class TalkerDecodeCoreML(nn.Module):
    """
    Decode model: single-token generation with KV cache.
    """

    def __init__(self, talker):
        super().__init__()
        self.talker = talker
        self.config = talker.config

    def forward(
        self,
        token_id: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_id: [B, 1] codec token ID
            kv_cache: [num_layers, 2, B, num_kv_heads, cache_len, head_dim]

        Returns:
            logits: [B, vocab_size]
            new_kv_cache: [num_layers, 2, B, num_kv_heads, cache_len+1, head_dim]
        """
        # 1. Embed token
        token_embed = self.talker.model.codec_embedding(token_id)  # [B, 1, 1024]

        # 2. Convert kv_cache tensor to DynamicCache format
        from transformers.cache_utils import DynamicCache

        num_layers = kv_cache.shape[0]
        past_kv = DynamicCache()
        for i in range(num_layers):
            past_kv.update(kv_cache[i, 0], kv_cache[i, 1], i)

        # 3. Forward with cache
        outputs = self.talker.model(
            inputs_embeds=token_embed,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )

        # 4. Get logits
        logits = self.talker.codec_head(outputs.last_hidden_state)
        logits = logits.squeeze(1)  # [B, vocab_size]

        # 5. Extract updated KV cache
        new_kv = outputs.past_key_values
        keys = []
        values = []
        for i in range(len(new_kv)):
            k, v = new_kv[i]
            keys.append(k)
            values.append(v)

        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        new_kv_cache = torch.stack([keys, values], dim=1)

        return logits, new_kv_cache


# %%
# Test the decode wrapper
print("\n=== Testing Decode Wrapper ===")

decode_model = TalkerDecodeCoreML(talker)
decode_model.eval()

# Get first token from prefill logits
first_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
print(f"First generated token: {first_token.item()}")

with torch.no_grad():
    new_logits, new_kv_cache = decode_model(first_token, kv_cache)

print(f"New logits shape: {new_logits.shape}")
print(f"New KV cache shape: {new_kv_cache.shape}")
print(f"Next top 5: {torch.topk(new_logits[0], 5).indices.tolist()}")


# %%
# Generate a few tokens to verify
print("\n=== Multi-step Generation ===")

generated = [first_token.item()]
current_kv = kv_cache

for step in range(10):
    with torch.no_grad():
        next_logits, current_kv = decode_model(
            torch.tensor([[generated[-1]]]),
            current_kv,
        )
        next_token = torch.argmax(next_logits, dim=-1).item()
        generated.append(next_token)

        if next_token == config.codec_eos_token_id:
            print(f"EOS reached at step {step}")
            break

print(f"Generated tokens: {generated}")
print(f"EOS token: {config.codec_eos_token_id}")


# %%
# Try tracing prefill model
print("\n=== Tracing Prefill Model ===")

# The problem: DynamicCache is not traceable
# Solution: We need to create a version that outputs static tensors

# First, let's check if the model forward can be traced with inputs_embeds
class TracablePrefill(nn.Module):
    """Traceable prefill that avoids DynamicCache issues."""

    def __init__(self, talker):
        super().__init__()
        # Copy all necessary components
        self.text_embedding = talker.model.text_embedding
        self.text_projection = talker.text_projection
        self.codec_embedding = talker.model.codec_embedding
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head

        self.config = talker.config
        self.language_id = self.config.codec_language_id["english"]
        self.bos_id = self.config.codec_bos_id

        # Get M-RoPE config
        self.mrope_section = self.config.rope_scaling["mrope_section"]
        self.mrope_interleaved = self.config.rope_scaling.get("interleaved", False)

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
                layer, hidden_states, causal_mask, (cos, sin)
            )
            all_keys.append(key)
            all_values.append(value)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Logits
        logits = self.codec_head(hidden_states[:, -1:, :]).squeeze(1)

        # Stack KV cache
        keys = torch.stack(all_keys, dim=0)
        values = torch.stack(all_values, dim=0)
        kv_cache = torch.stack([keys, values], dim=1)

        return logits, kv_cache

    def _create_causal_mask(self, seq_len, dtype, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

    def _run_layer(self, layer, hidden_states, causal_mask, position_embeddings):
        residual = hidden_states

        hidden_states = layer.input_layernorm(hidden_states)
        attn_output, key, value = self._run_attention(
            layer.self_attn, hidden_states, causal_mask, position_embeddings
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, key, value

    def _run_attention(self, attn, hidden_states, causal_mask, position_embeddings):
        from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

        bsz, q_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.mrope_section, self.mrope_interleaved
        )

        # Store KV before GQA expansion
        key_out = key_states.clone()
        value_out = value_states.clone()

        # GQA expansion
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (head_dim ** 0.5)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)

        return attn_output, key_out, value_out


# %%
# Test traceable prefill
print("\n=== Testing Traceable Prefill ===")

traceable_prefill = TracablePrefill(talker)
traceable_prefill.eval()

with torch.no_grad():
    tr_logits, tr_kv = traceable_prefill(test_text_ids)

print(f"Traceable logits shape: {tr_logits.shape}")
print(f"Traceable KV cache shape: {tr_kv.shape}")

# Compare with original wrapper
diff = (logits - tr_logits).abs().max().item()
print(f"Diff from original prefill: {diff}")


# %%
# Trace the traceable prefill
print("\n=== Tracing Traceable Prefill ===")

example_text = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)

try:
    with torch.no_grad():
        traced_prefill = torch.jit.trace(
            traceable_prefill,
            example_text,
            strict=False,
        )
    print("Prefill tracing successful!")

    # Verify
    with torch.no_grad():
        test_input = torch.randint(0, 1000, (1, 10))
        orig_logits, orig_kv = traceable_prefill(test_input)
        traced_logits, traced_kv = traced_prefill(test_input)

        logits_diff = (orig_logits - traced_logits).abs().max().item()
        kv_diff = (orig_kv - traced_kv).abs().max().item()
        print(f"Logits diff: {logits_diff}")
        print(f"KV cache diff: {kv_diff}")

except Exception as e:
    print(f"Prefill tracing failed: {e}")
    import traceback
    traceback.print_exc()


# %%
# Create traceable decode
class TracableDecode(nn.Module):
    """Traceable decode with static KV cache."""

    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.rotary_emb = talker.model.rotary_emb
        self.codec_head = talker.codec_head

        self.config = talker.config
        self.mrope_section = self.config.rope_scaling["mrope_section"]
        self.mrope_interleaved = self.config.rope_scaling.get("interleaved", False)

    def forward(
        self,
        token_id: torch.Tensor,
        kv_cache: torch.Tensor,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_id: [B, 1] codec token
            kv_cache: [num_layers, 2, B, num_kv_heads, cache_len, head_dim]
            position: [B] current position

        Returns:
            logits: [B, vocab_size]
            new_kv_cache: [num_layers, 2, B, num_kv_heads, cache_len+1, head_dim]
        """
        batch_size = token_id.shape[0]
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        # Embed token
        hidden_states = self.codec_embedding(token_id)  # [B, 1, hidden]

        # Position embeddings
        position_ids = position.unsqueeze(1)  # [B, 1]
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, 1]
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Run through layers
        new_keys = []
        new_values = []

        for i, layer in enumerate(self.layers):
            cached_key = kv_cache[i, 0]  # [B, num_kv_heads, cache_len, head_dim]
            cached_value = kv_cache[i, 1]

            hidden_states, new_key, new_value = self._run_decode_layer(
                layer, hidden_states, cached_key, cached_value, (cos, sin)
            )
            new_keys.append(new_key)
            new_values.append(new_value)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Logits
        logits = self.codec_head(hidden_states).squeeze(1)

        # Stack new KV
        keys = torch.stack(new_keys, dim=0)
        values = torch.stack(new_values, dim=0)
        new_kv_cache = torch.stack([keys, values], dim=1)

        return logits, new_kv_cache

    def _run_decode_layer(self, layer, hidden_states, cached_key, cached_value, position_embeddings):
        from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        bsz = hidden_states.shape[0]
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        # Project
        query_states = layer.self_attn.q_proj(hidden_states)
        key_states = layer.self_attn.k_proj(hidden_states)
        value_states = layer.self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, 1, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.mrope_section, self.mrope_interleaved
        )

        # Concatenate with cache
        full_keys = torch.cat([cached_key, key_states], dim=2)
        full_values = torch.cat([cached_value, value_states], dim=2)

        # GQA expansion for attention
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            attn_keys = full_keys.repeat_interleave(n_rep, dim=1)
            attn_values = full_values.repeat_interleave(n_rep, dim=1)
        else:
            attn_keys = full_keys
            attn_values = full_values

        # Attention (no mask - can attend to all)
        attn_weights = torch.matmul(query_states, attn_keys.transpose(2, 3)) / (head_dim ** 0.5)
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
# Test traceable decode
print("\n=== Testing Traceable Decode ===")

traceable_decode = TracableDecode(talker)
traceable_decode.eval()

# Use KV cache from traceable prefill
initial_position = tr_kv.shape[4]  # seq_len
first_token = torch.argmax(tr_logits, dim=-1, keepdim=True)

with torch.no_grad():
    dec_logits, dec_kv = traceable_decode(first_token, tr_kv, torch.tensor([initial_position]))

print(f"Decode logits shape: {dec_logits.shape}")
print(f"Decode KV cache shape: {dec_kv.shape}")


# %%
# Multi-step generation with traceable models
print("\n=== Multi-step Generation (Traceable) ===")

generated = [first_token.item()]
current_kv = tr_kv
current_pos = initial_position

for step in range(10):
    with torch.no_grad():
        next_logits, current_kv = traceable_decode(
            torch.tensor([[generated[-1]]]),
            current_kv,
            torch.tensor([current_pos])
        )
        next_token = torch.argmax(next_logits, dim=-1).item()
        generated.append(next_token)
        current_pos += 1

        if next_token == config.codec_eos_token_id:
            print(f"EOS reached at step {step}")
            break

print(f"Generated tokens: {generated}")


# %%
# Trace decode model
print("\n=== Tracing Decode Model ===")

PREFILL_LEN = MAX_TEXT_LENGTH + 2

example_token = torch.zeros((1, 1), dtype=torch.long)
example_kv = torch.zeros((
    config.num_hidden_layers,
    2,
    1,
    config.num_key_value_heads,
    PREFILL_LEN,
    config.head_dim
))
example_pos = torch.tensor([PREFILL_LEN])

try:
    with torch.no_grad():
        traced_decode = torch.jit.trace(
            traceable_decode,
            (example_token, example_kv, example_pos),
            strict=False,
        )
    print("Decode tracing successful!")

except Exception as e:
    print(f"Decode tracing failed: {e}")
    import traceback
    traceback.print_exc()


# %%
# Convert Prefill to CoreML
print("\n=== Converting Prefill to CoreML ===")

prefill_inputs = [
    ct.TensorType(
        name="text_ids",
        shape=(1, MAX_TEXT_LENGTH),
        dtype=np.int32,
    ),
]

prefill_outputs = [
    ct.TensorType(name="logits"),
    ct.TensorType(name="kv_cache"),
]

try:
    mlmodel_prefill = ct.convert(
        traced_prefill,
        inputs=prefill_inputs,
        outputs=prefill_outputs,
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
    ct.TensorType(
        name="token_id",
        shape=(1, 1),
        dtype=np.int32,
    ),
    ct.TensorType(
        name="kv_cache",
        shape=(
            config.num_hidden_layers,
            2,
            1,
            config.num_key_value_heads,
            ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LENGTH, default=PREFILL_LEN),
            config.head_dim,
        ),
        dtype=np.float32,
    ),
    ct.TensorType(
        name="position",
        shape=(1,),
        dtype=np.int32,
    ),
]

decode_outputs = [
    ct.TensorType(name="logits"),
    ct.TensorType(name="new_kv_cache"),
]

try:
    mlmodel_decode = ct.convert(
        traced_decode,
        inputs=decode_inputs,
        outputs=decode_outputs,
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
print(f"Created:")
print(f"  - qwen3_tts_lm_prefill.mlpackage (text -> first logits + KV cache)")
print(f"  - qwen3_tts_lm_decode.mlpackage (token + KV -> next logits + updated KV)")
