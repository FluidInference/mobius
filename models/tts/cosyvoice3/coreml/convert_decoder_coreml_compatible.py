"""
Custom CoreML-compatible decoder - like we did for ISTFT.

Instead of using dynamic loops/operations that coremltools can't convert,
we explicitly unroll all 24 layers using only CoreML-friendly operations.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
from huggingface_hub import hf_hub_download

REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
sys.path.insert(0, str(REPO_PATH))

print("="*80)
print("Custom CoreML-Compatible Decoder (ISTFT Approach)")
print("="*80)
print("Explicitly unrolling all 24 layers to avoid dynamic operations")

# Load model
REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CACHE_DIR = Path.home() / ".cache" / "cosyvoice3_analysis"

print("\nLoading LLM checkpoint...")
llm_path = hf_hub_download(repo_id=REPO_ID, filename="llm.pt", cache_dir=CACHE_DIR)
llm_state = torch.load(llm_path, map_location="cpu", weights_only=True)

from transformers import Qwen2ForCausalLM, Qwen2Config

llm_config = Qwen2Config(
    hidden_size=896,
    num_hidden_layers=24,
    vocab_size=151936,
    intermediate_size=4864,
    num_attention_heads=14,
    num_key_value_heads=2,
)

print(f"Config: {llm_config.num_hidden_layers} layers")

llm_model = Qwen2ForCausalLM(llm_config)
llm_model.load_state_dict(llm_state, strict=False)
llm_model.eval()

# AnemllRMSNorm
class AnemllRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps
        self.dim = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, [doubled.shape[-1]], eps=self.eps)
        normed = normed[..., : self.dim]
        return normed * self.weight

def patch_rms_norms(module: nn.Module):
    for name, child in list(module.named_children()):
        if "RMSNorm" in type(child).__name__ and hasattr(child, "weight"):
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-6))
            setattr(module, name, AnemllRMSNorm(child.weight.data, eps=eps))
        else:
            patch_rms_norms(child)

print("Patching RMSNorm...")
patch_rms_norms(llm_model)

# CoreML-compatible attention (no dynamic ops)
class CoreMLAttention(nn.Module):
    """Single attention layer with all ops CoreML-friendly."""

    def __init__(self, layer, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.layer = layer
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(self, hidden_states, cos, sin, attention_mask):
        attn = self.layer.self_attn
        bsz, q_len, hidden_size = hidden_states.shape

        # QKV projections
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        # Reshape to [batch, seq, num_heads, head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary - cos/sin are [batch, 1, seq, head_dim], broadcast to all heads
        # cos/sin broadcast from [1,1,seq,64] -> [1,14,seq,64] for Q and [1,2,seq,64] for K/V
        q = q * cos + self.rotate_half_simple(q) * sin
        k = k * cos + self.rotate_half_simple(k) * sin

        # Repeat KV for GQA - use static ops only
        # 14 Q heads, 2 KV heads -> repeat each KV 7 times
        k = k.repeat(1, 7, 1, 1)  # [1, 2, seq, dim] -> [1, 14, seq, dim]
        v = v.repeat(1, 7, 1, 1)

        # Attention
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, hidden_size)
        attn_output = attn.o_proj(attn_output)

        return attn_output

    @staticmethod
    def rotate_half_simple(x):
        """Rotate half - purely tensor ops, no indexing."""
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        return torch.cat([-x2, x1], dim=-1)

# CoreML-compatible decoder layer
class CoreMLDecoderLayer(nn.Module):
    """Single decoder layer with CoreML-friendly ops."""

    def __init__(self, layer, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.attention = CoreMLAttention(layer, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp

    def forward(self, hidden_states, cos, sin, attention_mask):
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention
        attn_output = self.attention(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + attn_output

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

# Explicitly unrolled decoder (no loops!)
class CoreMLExplicitDecoder(nn.Module):
    """All 24 layers explicitly written out - no loops, no dynamic ops."""

    def __init__(self, layers, config):
        super().__init__()
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads

        # Create 24 individual layer attributes (not a list - avoid loops)
        for i in range(24):
            setattr(self, f'layer_{i}', CoreMLDecoderLayer(layers[i], num_heads, num_kv_heads, head_dim))

    def forward(self, hidden_states, cos, sin, attention_mask):
        # Explicitly call each layer (no loops!)
        hidden_states = self.layer_0(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_1(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_2(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_3(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_4(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_5(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_6(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_7(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_8(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_9(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_10(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_11(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_12(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_13(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_14(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_15(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_16(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_17(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_18(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_19(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_20(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_21(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_22(hidden_states, cos, sin, attention_mask)
        hidden_states = self.layer_23(hidden_states, cos, sin, attention_mask)

        return hidden_states

print("\nCreating CoreML-compatible decoder...")
coreml_decoder = CoreMLExplicitDecoder(llm_model.model.layers, llm_config)
coreml_decoder.eval()

# Trace
print("\nTracing custom decoder (all 24 layers unrolled)...")
batch_size = 1
seq_len = 10
hidden_size = llm_config.hidden_size
head_dim = hidden_size // llm_config.num_attention_heads

hidden_states = torch.randn(batch_size, seq_len, hidden_size)
cos = torch.randn(batch_size, 1, seq_len, head_dim)  # [1, 1, seq, head_dim] broadcasts to all heads
sin = torch.randn(batch_size, 1, seq_len, head_dim)
attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)

print(f"Input: {hidden_states.shape}")

with torch.inference_mode():
    print("Tracing...")
    traced = torch.jit.trace(coreml_decoder, (hidden_states, cos, sin, attention_mask))

print("✓ Tracing complete!")

# Convert to CoreML
print("\nConverting to CoreML...")
print("This uses only CoreML-compatible operations (like custom ISTFT)")

try:
    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name='hidden_states', shape=(1, ct.RangeDim(1, 512), hidden_size), dtype=np.float16),
            ct.TensorType(name='cos', shape=(1, 1, ct.RangeDim(1, 512), head_dim), dtype=np.float16),
            ct.TensorType(name='sin', shape=(1, 1, ct.RangeDim(1, 512), head_dim), dtype=np.float16),
            ct.TensorType(name='attention_mask', shape=(1, 1, ct.RangeDim(1, 512), ct.RangeDim(1, 512)), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name='output_hidden', dtype=np.float16)],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )

    output_path = "cosyvoice_llm_decoder_coreml.mlpackage"
    coreml_model.save(output_path)

    import os
    size_mb = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(output_path)
        for filename in filenames
    ) / 1024 / 1024

    print(f"\n✓ SUCCESS - Saved: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")

    print("\n" + "="*80)
    print("CoreML-Compatible Decoder Complete!")
    print("="*80)
    print("\n28 files → 5 files:")
    print("  1. cosyvoice_llm_embedding.mlpackage")
    print("  2. cosyvoice_llm_decoder_coreml.mlpackage  ← NEW")
    print("  3. cosyvoice_llm_lm_head.mlpackage")
    print("  4. flow_decoder.mlpackage")
    print("  5. converted/hift_vocoder.mlpackage")

except Exception as e:
    print(f"\n✗ Conversion failed: {e}")
    print("\nIf this still fails, the issue is deeper than loops/dynamic ops.")
    import traceback
    traceback.print_exc()
