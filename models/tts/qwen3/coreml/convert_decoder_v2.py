# %% [markdown]
# # Qwen3-TTS 12Hz Tokenizer Decoder → CoreML Conversion (v2)
#
# Uses pre-computed attention masks to avoid vmap tracing issues.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
import numpy as np

# Configuration
MAX_CODE_LENGTH = 125  # ~10 seconds at 12.5Hz
SAMPLE_RATE = 24000
UPSAMPLE_RATE = 1920

print(f"Max code length: {MAX_CODE_LENGTH}")
print(f"Max audio length: {MAX_CODE_LENGTH * UPSAMPLE_RATE / SAMPLE_RATE:.1f}s")

# %%
# Load the tokenizer model
from qwen_tts import Qwen3TTSTokenizer

print("Loading tokenizer...")
tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "./tokenizer_12hz",
    device_map="cpu",
)
decoder = tokenizer.model.decoder
decoder.eval()

print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")

# %%
# Create pre-computed attention masks
def create_causal_mask_simple(seq_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a simple causal mask [1, 1, seq_len, seq_len]"""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def create_sliding_window_mask(seq_len: int, window_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a sliding window causal mask [1, 1, seq_len, seq_len]"""
    # Start with all -inf (can't attend)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)

    for i in range(seq_len):
        # Can attend to positions from max(0, i - window_size + 1) to i (inclusive)
        start = max(0, i - window_size + 1)
        end = i + 1  # inclusive
        mask[i, start:end] = 0.0  # Allow attention (0 = no mask)

    return mask.unsqueeze(0).unsqueeze(0)


# %%
# Create CoreML-compatible decoder wrapper
class Qwen3DecoderCoreML(nn.Module):
    """
    CoreML-compatible decoder that pre-computes attention masks.
    """

    def __init__(self, decoder, max_length: int = MAX_CODE_LENGTH):
        super().__init__()
        self.max_length = max_length

        # Copy decoder components
        self.quantizer = decoder.quantizer
        self.pre_conv = decoder.pre_conv
        self.pre_transformer = decoder.pre_transformer
        self.upsample = decoder.upsample
        self.decoder_blocks = decoder.decoder

        # Get config
        self.config = decoder.pre_transformer.config
        self.sliding_window = getattr(self.config, "sliding_window", 72)

        # Pre-compute masks for max length
        # After pre_conv, the sequence length changes
        # We'll compute masks dynamically based on actual length

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, 16, T] codec codes

        Returns:
            audio: [B, 1, T'] audio waveform
        """
        batch_size = codes.shape[0]
        seq_len = codes.shape[2]

        # 1. Quantizer decode: codes -> embeddings
        hidden = self.quantizer.decode(codes)  # [B, codebook_dim, T]

        # 2. Pre-conv
        hidden = self.pre_conv(hidden)  # [B, latent_dim, T]

        # 3. Transformer (the tricky part)
        hidden = hidden.transpose(1, 2)  # [B, T, latent_dim]

        # Create attention masks
        transformer_seq_len = hidden.shape[1]
        causal_mask = create_causal_mask_simple(transformer_seq_len, hidden.dtype).to(hidden.device)
        sliding_mask = create_sliding_window_mask(transformer_seq_len, self.sliding_window, hidden.dtype).to(
            hidden.device
        )

        # Expand for batch
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
        sliding_mask = sliding_mask.expand(batch_size, -1, -1, -1)

        # Create mask dict
        mask_dict = {
            "full_attention": causal_mask,
            "sliding_attention": sliding_mask,
        }

        # Run transformer with pre-computed masks
        hidden = self._run_transformer(hidden, mask_dict)

        hidden = hidden.transpose(1, 2)  # [B, latent_dim, T]

        # 4. Upsample
        for upsample_block in self.upsample:
            for layer in upsample_block:
                hidden = layer(hidden)

        # 5. Decoder blocks
        for block in self.decoder_blocks:
            hidden = block(hidden)

        return hidden

    def _run_transformer(self, hidden_states: torch.Tensor, mask_dict: dict) -> torch.Tensor:
        """Run transformer layers with pre-computed masks."""
        transformer = self.pre_transformer

        # Input projection
        hidden_states = transformer.input_proj(hidden_states)

        # Position embeddings
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = transformer.rotary_emb(hidden_states, position_ids)

        # Run layers
        for layer in transformer.layers:
            # Get the right mask for this layer
            attn_type = getattr(layer, "attention_type", "full_attention")
            attention_mask = mask_dict.get(attn_type, mask_dict["full_attention"])

            # Run layer (simplified - skip cache)
            hidden_states = self._run_layer(layer, hidden_states, attention_mask, position_embeddings)

        # Output projection
        hidden_states = transformer.norm(hidden_states)
        hidden_states = transformer.output_proj(hidden_states)

        return hidden_states

    def _run_layer(
        self,
        layer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple,
    ) -> torch.Tensor:
        """Run a single transformer layer."""
        residual = hidden_states

        # Self attention
        hidden_states = layer.input_layernorm(hidden_states)

        # Run attention manually to avoid the mask creation
        attn_output = self._run_attention(layer.self_attn, hidden_states, attention_mask, position_embeddings)

        # Layer scale and residual
        attn_output = layer.self_attn_layer_scale(attn_output)
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.mlp_layer_scale(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _run_attention(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple,
    ) -> torch.Tensor:
        """Run attention with explicit mask."""
        bsz, q_len, _ = hidden_states.shape

        # Get config values
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim

        # Projections
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)

        # Reshape to [B, num_heads, T, head_dim]
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states = self._apply_rotary_pos_emb(query_states, cos, sin)
        key_states = self._apply_rotary_pos_emb(key_states, cos, sin)

        # Expand KV for GQA if needed
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        # Attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (attn.head_dim**0.5)

        # Add attention mask
        attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # Output projection
        attn_output = attn.o_proj(attn_output)

        return attn_output

    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings."""
        # x: [B, num_heads, T, head_dim]
        # cos, sin: [1, T, head_dim]
        cos = cos.unsqueeze(1)  # [1, 1, T, head_dim]
        sin = sin.unsqueeze(1)  # [1, 1, T, head_dim]

        # rotate_half: splits and rotates
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (rotated * sin)


# %%
# Test the wrapper
print("\n=== Testing CoreML Wrapper ===")

coreml_decoder = Qwen3DecoderCoreML(decoder)
coreml_decoder.eval()

test_codes = torch.randint(0, 2048, (1, 16, 10))
print(f"Input shape: {test_codes.shape}")

with torch.no_grad():
    # Test original
    orig_output = decoder(test_codes)
    print(f"Original output shape: {orig_output.shape}")

    # Test wrapper
    wrapper_output = coreml_decoder(test_codes)
    print(f"Wrapper output shape: {wrapper_output.shape}")

    # Compare
    diff = (orig_output - wrapper_output).abs().max().item()
    print(f"Max diff: {diff}")

# %%
# Trace with torch.jit
print("\n=== Tracing with torch.jit ===")

example_codes = torch.randint(0, 2048, (1, 16, MAX_CODE_LENGTH))

try:
    with torch.no_grad():
        traced_model = torch.jit.trace(
            coreml_decoder,
            example_codes,
            strict=False,
        )
    print("Tracing successful!")

    # Verify
    with torch.no_grad():
        traced_output = traced_model(example_codes)
        original_output = coreml_decoder(example_codes)
        diff = (traced_output - original_output).abs().max().item()
        print(f"Max diff traced vs wrapper: {diff}")

except Exception as e:
    print(f"Tracing failed: {e}")
    import traceback

    traceback.print_exc()

# %%
# Convert to CoreML
print("\n=== Converting to CoreML ===")

inputs = [
    ct.TensorType(
        name="codes",
        shape=(1, 16, MAX_CODE_LENGTH),
        dtype=np.int32,
    ),
]

outputs = [
    ct.TensorType(name="audio"),
]

try:
    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    print("CoreML conversion successful!")

    output_path = Path("qwen3_tts_decoder_10s.mlpackage")
    mlmodel.save(str(output_path))
    print(f"Saved to: {output_path}")
    print(f"Model size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e6:.1f} MB")

except Exception as e:
    print(f"CoreML conversion failed: {e}")
    import traceback

    traceback.print_exc()
