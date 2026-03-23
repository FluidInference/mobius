#!/usr/bin/env python3
"""
Argmax-style CodeDecoder CoreML Conversion (Stateful KV Cache, W8A16)

Reverse-engineered from Argmax TTSKit's CodeDecoder MIL program (6531 lines):

CodeDecoder (28-layer transformer, single-token decode with stateful KV cache):
  Inputs:
    - cache_length [1] (int32): current position in sequence
    - input_embeds [1, 1024, 1, 1] (fp16): embedding from CodeEmbedder/MultiCodeEmbedder
    - key_padding_mask [1, 256] (fp16): mask for valid KV positions
    - kv_cache_update_mask [1, 256] (fp16): one-hot mask for write position
  States (CoreML stateful):
    - self_attn_key_cache [1, 28672, 1, 256] (fp16): all 28 layers × 1024 KV-heads packed
    - self_attn_value_cache [1, 28672, 1, 256] (fp16): same

  Outputs:
    - logits [1, 1, 3072] (fp16): next CB0 token logits
    - hidden_states [1, 1024, 1, 1] (fp16): for code predictor

Architecture details from MIL:
  - 28 layers, 16 Q heads, 8 KV heads, head_dim=128, hidden=1024
  - RMSNorm on Q and K (per-head normalization)
  - RoPE via precomputed cos/sin weight tables [256, 128]
  - All linear layers as 1x1 convolutions (W8A16 palettized)
  - KV cache: [1, 28*1024, 1, 256] = [1, 28672, 1, 256] packed along dim=1
  - Cache update via element-wise: new_cache = cache * (1 - update_mask) + new_kv * update_mask
  - SwiGLU MLP: gate_proj → SiLU, up_proj, multiply, down_proj
  - GQA: 16 Q heads / 8 KV heads = 2:1 repeat

Usage:
    python convert_argmax_code_decoder.py [--model-path ./model_0.6b]
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import argparse

MAX_SEQ_LEN = 256  # Maximum sequence length (Argmax uses 256)


def patch_rmsnorm_for_trace():
    """Monkey-patch Qwen3-TTS RMSNorm to avoid dynamic dtype casts.

    The original RMSNorm does hidden.to(float32) → compute → hidden.to(input_dtype).
    The .to(input_dtype) generates aten::Int ops that coremltools can't handle.
    We replace it with a version that stays in float32 (ct handles precision).
    """
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSRMSNorm

    def rmsnorm_forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.float() * hidden_states

    Qwen3TTSRMSNorm.forward = rmsnorm_forward


def patch_coremltools_int_cast():
    """Fix coremltools _cast to handle multi-dimensional constant arrays.

    The aten::Int op can receive multi-element arrays in traced graphs
    (e.g., from .size() calls). coremltools 9.0 only handles scalars.
    This patch extracts the scalar value from the array.
    """
    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    original_cast = ct_ops._cast

    def patched_cast(context, node, dtype, dtype_name):
        inputs = ct_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    val = val.item()
                # If multi-element, try to convert the first element (shouldn't happen)
            if not isinstance(val, dtype):
                val = dtype(val)
            res = mb.const(val=val, name=node.name)
            context.add(res, node.name)
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res, node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res, node.name)

    ct_ops._cast = patched_cast


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE. cos/sin: [1, head_dim] expanded to match q/k."""
    # q: [num_heads, head_dim, 1, 1] (ANE layout)
    # cos/sin: [1, head_dim, 1] → broadcast
    q_embed = (q * cos) + (rotate_half_4d(q) * sin)
    k_embed = (k * cos) + (rotate_half_4d(k) * sin)
    return q_embed, k_embed


def rotate_half_4d(x):
    """rotate_half for 4D ANE tensor [heads, head_dim, 1, 1]."""
    half = x.shape[1] // 2
    x1 = x[:, :half, :, :]
    x2 = x[:, half:, :, :]
    return torch.cat((-x2, x1), dim=1)


class ArgmaxCodeDecoder(nn.Module):
    """
    Single-token transformer decoder matching Argmax's CodeDecoder architecture.

    Uses explicit KV cache inputs/outputs (not CoreML stateful API in PyTorch).
    The CoreML stateful conversion is handled during ct.convert() via state specs.

    Data layout: ANE-friendly [1, C, 1, S] format throughout.
    """

    def __init__(self, talker):
        super().__init__()
        self.layers = talker.model.layers
        self.norm = talker.model.norm
        self.codec_head = talker.codec_head

        # Extract inv_freq for manual RoPE computation
        # The talker's RoPE expects 3D position_ids which doesn't work for our use case
        rotary_emb = talker.model.rotary_emb
        self.register_buffer('inv_freq', rotary_emb.inv_freq.clone())
        self.attention_scaling = rotary_emb.attention_scaling

        config = talker.config
        self.num_heads = config.num_attention_heads      # 16
        self.num_kv_heads = config.num_key_value_heads   # 8
        self.head_dim = config.head_dim                  # 128
        self.hidden_size = config.hidden_size            # 1024
        self.num_layers = config.num_hidden_layers       # 28
        self.kv_dim = self.num_kv_heads * self.head_dim  # 1024

    def forward(
        self,
        input_embeds: torch.Tensor,       # [1, 1024, 1, 1]
        cache_length: torch.Tensor,       # [1]
        key_padding_mask: torch.Tensor,   # [1, MAX_SEQ_LEN]
        kv_cache_update_mask: torch.Tensor,  # [1, MAX_SEQ_LEN]
        key_cache: torch.Tensor,          # [1, 28672, 1, MAX_SEQ_LEN]
        value_cache: torch.Tensor,        # [1, 28672, 1, MAX_SEQ_LEN]
    ):
        """
        Returns:
            logits: [1, 1, 3072]
            hidden_states: [1, 1024, 1, 1]
            new_key_cache: [1, 28672, 1, MAX_SEQ_LEN]
            new_value_cache: [1, 28672, 1, MAX_SEQ_LEN]
        """
        # Manual RoPE computation for current position
        # (avoids talker's 3D-position RoPE which expects [3, batch, seq] format)
        position_ids = cache_length.unsqueeze(0).float()  # [1, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float()  # [1, dim/2, 1]
        freqs = (inv_freq_expanded @ position_ids.unsqueeze(-1)).transpose(1, 2)  # [1, 1, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [1, 1, head_dim]
        cos_full = (emb.cos() * self.attention_scaling).to(input_embeds.dtype)
        sin_full = (emb.sin() * self.attention_scaling).to(input_embeds.dtype)
        # cos_full/sin_full: [1, 1, 128]

        hidden = input_embeds  # [1, 1024, 1, 1]

        new_key_slices = []
        new_value_slices = []

        for i, layer in enumerate(self.layers):
            start = i * self.kv_dim
            end = (i + 1) * self.kv_dim
            layer_key_cache = key_cache[:, start:end, :, :]    # [1, 1024, 1, 256]
            layer_value_cache = value_cache[:, start:end, :, :] # [1, 1024, 1, 256]

            hidden, new_key, new_value = self._transformer_layer(
                layer, hidden, layer_key_cache, layer_value_cache,
                cos_full, sin_full, key_padding_mask, kv_cache_update_mask
            )
            new_key_slices.append(new_key)
            new_value_slices.append(new_value)

        # Final RMSNorm
        hidden_3d = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)  # [1, 1, 1024]
        normed = self.norm(hidden_3d)  # [1, 1, 1024]

        # Logits
        logits = self.codec_head(normed)  # [1, 1, 3072]

        # Reconstruct full caches with scatter-style update
        new_key_full = self._update_cache(key_cache, new_key_slices, kv_cache_update_mask)
        new_value_full = self._update_cache(value_cache, new_value_slices, kv_cache_update_mask)

        # hidden_states output is post-norm (what talker passes to code_predictor)
        hidden_out = normed.squeeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 1024, 1, 1]
        return logits, hidden_out, new_key_full, new_value_full

    def _transformer_layer(
        self, layer, hidden, key_cache, value_cache, cos, sin, key_padding_mask,
        kv_cache_update_mask
    ):
        """Single transformer layer with cached KV attention.

        Args:
            hidden: [1, 1024, 1, 1] — ANE layout
            key_cache: [1, 1024, 1, 256]
            value_cache: [1, 1024, 1, 256]
            cos: [1, 1, 128]
            sin: [1, 1, 128]
            key_padding_mask: [1, 256]
            kv_cache_update_mask: [1, 256] — one-hot mask for current position
        """
        residual = hidden

        # RMSNorm (convert from ANE layout)
        h_3d = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)  # [1, 1, 1024]
        h_normed = layer.input_layernorm(h_3d)  # [1, 1, 1024]

        attn = layer.self_attn

        # Q/K/V projections
        q = attn.q_proj(h_normed)  # [1, 1, 2048] (16 heads × 128)
        k = attn.k_proj(h_normed)  # [1, 1, 1024] (8 heads × 128)
        v = attn.v_proj(h_normed)  # [1, 1, 1024]

        # Reshape for multi-head
        q = q.view(1, 1, self.num_heads, self.head_dim)     # [1, 1, 16, 128]
        k = k.view(1, 1, self.num_kv_heads, self.head_dim)  # [1, 1, 8, 128]
        v = v.view(1, 1, self.num_kv_heads, self.head_dim)  # [1, 1, 8, 128]

        # QK norm (RMSNorm per head)
        q = attn.q_norm(q)
        k = attn.k_norm(k)

        # Transpose to [1, heads, 1, head_dim]
        q = q.transpose(1, 2)  # [1, 16, 1, 128]
        k = k.transpose(1, 2)  # [1, 8, 1, 128]
        v = v.transpose(1, 2)  # [1, 8, 1, 128]

        # Apply RoPE
        q, k = self._apply_rope(q, k, cos, sin)

        # New key/value for this position
        new_key = k   # [1, 8, 1, 128]
        new_value = v  # [1, 8, 1, 128]

        # Write current K/V to cache BEFORE computing attention
        # This ensures the current position's K/V participates in attention
        nk_flat = new_key.reshape(1, self.kv_dim, 1, 1)   # [1, 1024, 1, 1]
        nv_flat = new_value.reshape(1, self.kv_dim, 1, 1)
        update_mask = kv_cache_update_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, 256]
        key_cache = key_cache * (1.0 - update_mask) + nk_flat * update_mask
        value_cache = value_cache * (1.0 - update_mask) + nv_flat * update_mask

        # Reshape updated cache for attention
        # key_cache: [1, 1024, 1, 256] → [1, 8, 128, 256]
        kc = key_cache.squeeze(2)  # [1, 1024, 256]
        kc = kc.view(1, self.num_kv_heads, self.head_dim, MAX_SEQ_LEN)  # [1, 8, 128, 256]

        vc = value_cache.squeeze(2)  # [1, 1024, 256]
        vc = vc.view(1, self.num_kv_heads, self.head_dim, MAX_SEQ_LEN)  # [1, 8, 128, 256]

        # GQA expansion: 8 KV heads → 16 Q heads (repeat 2x)
        n_rep = self.num_heads // self.num_kv_heads
        kc = kc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, MAX_SEQ_LEN)
        vc = vc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, MAX_SEQ_LEN)

        # Attention: Q @ K^T / sqrt(d)
        # q: [1, 16, 1, 128], kc: [1, 16, 128, 256]
        attn_weights = torch.matmul(q, kc) / (self.head_dim ** 0.5)  # [1, 16, 1, 256]

        # Apply key padding mask (Argmax uses additive mask with -inf for padding)
        # key_padding_mask: [1, 256] → [1, 1, 1, 256]
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights + mask  # masked positions have -inf

        attn_weights = torch.softmax(attn_weights.float(), dim=-1)

        # Attention output: weights @ V
        # vc layout for matmul: need [1, 16, 256, 128]
        vc_t = vc.transpose(-1, -2)  # [1, 16, 256, 128]
        attn_output = torch.matmul(attn_weights, vc_t)  # [1, 16, 1, 128]

        # Reshape: [1, 16, 1, 128] → [1, 1, 2048]
        attn_output = attn_output.transpose(1, 2).reshape(1, 1, -1)

        # Output projection
        attn_output = attn.o_proj(attn_output)  # [1, 1, 1024]

        # Add residual
        attn_4d = attn_output.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 1024, 1, 1]
        hidden = residual + attn_4d

        # MLP with residual
        residual = hidden
        h_3d = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)  # [1, 1, 1024]
        h_normed = layer.post_attention_layernorm(h_3d)
        mlp_out = layer.mlp(h_normed)  # [1, 1, 1024]
        mlp_4d = mlp_out.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        hidden = residual + mlp_4d

        # Return new KV in ANE-friendly layout [1, kv_dim, 1, 1]
        nk = nk_flat
        nv = nv_flat

        return hidden, nk, nv

    def _apply_rope(self, q, k, cos, sin):
        """Apply rotary position embeddings.

        q: [1, 16, 1, 128], k: [1, 8, 1, 128]
        cos/sin: [1, 1, 128]
        """
        cos = cos.unsqueeze(1)  # [1, 1, 1, 128]
        sin = sin.unsqueeze(1)  # [1, 1, 1, 128]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def _update_cache(self, old_cache, new_slices, update_mask):
        """Update KV cache using element-wise masking (Argmax style).

        old_cache: [1, 28672, 1, 256]
        new_slices: list of 28 tensors [1, 1024, 1, 1]
        update_mask: [1, 256] — one-hot for write position

        Argmax approach:
            cache = cache * (1 - mask) + new_kv * mask
        This overwrites exactly one position.
        """
        new_kv = torch.cat(new_slices, dim=1)  # [1, 28672, 1, 1]
        mask = update_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, 256]
        updated = old_cache * (1.0 - mask) + new_kv * mask
        return updated


def main():
    parser = argparse.ArgumentParser(description="Convert Argmax-style CodeDecoder")
    parser.add_argument("--model-path", default="./model_0.6b", help="Path to Qwen3-TTS model")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--quantize-w8", action="store_true", help="Apply W8A16 palettization")
    parser.add_argument("--float32", action="store_true", help="Use FLOAT32 compute precision (avoids fp16 underflow)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip CoreML verification")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    # Patch RMSNorm and coremltools before loading model
    patch_rmsnorm_for_trace()
    patch_coremltools_int_cast()

    print("=" * 60)
    print("Argmax-style CodeDecoder Conversion (28-layer, stateful)")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker

    num_layers = talker.config.num_hidden_layers
    num_kv_heads = talker.config.num_key_value_heads
    head_dim = talker.config.head_dim
    kv_dim = num_kv_heads * head_dim  # 1024

    print(f"   Layers: {num_layers}")
    print(f"   Heads: {talker.config.num_attention_heads} Q, {num_kv_heads} KV")
    print(f"   Head dim: {head_dim}")
    print(f"   KV dim per layer: {kv_dim}")
    print(f"   Total KV dim: {num_layers * kv_dim}")  # 28672

    # 2. Create wrapper
    print("\n2. Creating CodeDecoder wrapper...")
    wrapper = ArgmaxCodeDecoder(talker)
    wrapper.eval()

    # 3. Test
    print("\n3. Testing wrapper...")
    test_embeds = torch.randn(1, 1024, 1, 1)
    test_cache_len = torch.tensor([5])
    test_key_mask = torch.zeros(1, MAX_SEQ_LEN)
    test_key_mask[0, 6:] = float('-inf')  # positions 0-5 valid, rest masked
    test_update_mask = torch.zeros(1, MAX_SEQ_LEN)
    test_update_mask[0, 5] = 1.0  # write at position 5
    test_key_cache = torch.randn(1, num_layers * kv_dim, 1, MAX_SEQ_LEN)
    test_value_cache = torch.randn(1, num_layers * kv_dim, 1, MAX_SEQ_LEN)

    with torch.no_grad():
        logits, hidden, new_kc, new_vc = wrapper(
            test_embeds, test_cache_len, test_key_mask, test_update_mask,
            test_key_cache, test_value_cache
        )
    print(f"   logits: {logits.shape}")         # [1, 1, 3072]
    print(f"   hidden: {hidden.shape}")         # [1, 1024, 1, 1]
    print(f"   new_key_cache: {new_kc.shape}")  # [1, 28672, 1, 256]
    print(f"   new_value_cache: {new_vc.shape}")

    # 4. Trace
    print("\n4. Tracing...")
    example_inputs = (
        test_embeds, test_cache_len, test_key_mask, test_update_mask,
        test_key_cache, test_value_cache
    )
    traced = torch.jit.trace(wrapper, example_inputs, strict=False)

    # Verify trace
    with torch.no_grad():
        t_logits, t_hidden, t_kc, t_vc = traced(*example_inputs)
    diff = (t_logits - logits).abs().max().item()
    print(f"   Trace logits diff: {diff}")

    # 5. Convert to CoreML
    print("\n5. Converting to CoreML...")
    inputs = [
        ct.TensorType(name="input_embeds", shape=(1, 1024, 1, 1), dtype=np.float16),
        ct.TensorType(name="cache_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="key_padding_mask", shape=(1, MAX_SEQ_LEN), dtype=np.float16),
        ct.TensorType(name="kv_cache_update_mask", shape=(1, MAX_SEQ_LEN), dtype=np.float16),
        ct.TensorType(name="key_cache", shape=(1, num_layers * kv_dim, 1, MAX_SEQ_LEN), dtype=np.float16),
        ct.TensorType(name="value_cache", shape=(1, num_layers * kv_dim, 1, MAX_SEQ_LEN), dtype=np.float16),
    ]

    outputs = [
        ct.TensorType(name="logits", dtype=np.float16),
        ct.TensorType(name="hidden_states", dtype=np.float16),
        ct.TensorType(name="new_key_cache", dtype=np.float16),
        ct.TensorType(name="new_value_cache", dtype=np.float16),
    ]

    precision = ct.precision.FLOAT32 if args.float32 else ct.precision.FLOAT16
    print(f"   Compute precision: {precision}")
    ml_model = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision,
    )

    # 6. Optional W8A16 quantization
    if args.quantize_w8:
        print("\n6. Applying W8A16 palettized quantization...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        op_config = OpPalettizerConfig(
            mode="kmeans",
            nbits=8,
            weight_threshold=512,
        )
        opt_config = OptimizationConfig(global_config=op_config)
        ml_model = palettize_weights(ml_model, config=opt_config)
        print("   W8A16 palettization applied")

    cd_path = f"{args.output_dir}/CodeDecoder.mlpackage"
    ml_model.save(cd_path)
    print(f"   Saved: {cd_path}")

    # 7. Verify CoreML
    if not args.skip_verify:
        print("\n7. Verifying CoreML model...")
        loaded = ct.models.MLModel(cd_path)
        result = loaded.predict({
            "input_embeds": test_embeds.numpy().astype(np.float16),
            "cache_length": test_cache_len.numpy().astype(np.int32),
            "key_padding_mask": test_key_mask.numpy().astype(np.float16),
            "kv_cache_update_mask": test_update_mask.numpy().astype(np.float16),
            "key_cache": test_key_cache.numpy().astype(np.float16),
            "value_cache": test_value_cache.numpy().astype(np.float16),
        })
        print(f"   logits shape: {result['logits'].shape}")
        print(f"   hidden_states shape: {result['hidden_states'].shape}")
        cml_logits = result['logits'].astype(np.float32)
        pt_logits = logits.detach().numpy().astype(np.float32)
        diff = np.abs(cml_logits - pt_logits).max()
        print(f"   Max logits diff: {diff}")

    print("\n" + "=" * 60)
    print(f"Done! Model saved: {cd_path}")
    print("=" * 60)
    print("\nNote: To match Argmax exactly, convert this to stateful on the Swift side")
    print("by using MLState API with the key_cache and value_cache as state tensors.")


if __name__ == "__main__":
    main()
