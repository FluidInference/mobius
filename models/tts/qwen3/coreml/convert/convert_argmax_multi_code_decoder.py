#!/usr/bin/env python3
"""
Argmax-style MultiCodeDecoder CoreML Conversion

Reverse-engineered from Argmax TTSKit's MultiCodeDecoder MIL program (1368 lines):

MultiCodeDecoder (5-layer code predictor transformer):
  Inputs:
    - cache_length [1] (int32): current position
    - input_embeds [1, 1024, 1, 1] (fp16): embedding from MultiCodeEmbedder
    - key_cache [1, 5120, 1, 16] (fp16): 5 layers × 1024 KV-heads packed
    - key_padding_mask [1, 16] (fp16): mask for valid KV positions
    - kv_cache_update_mask [1, 16] (fp16): one-hot for write position
    - value_cache [1, 5120, 1, 16] (fp16): same layout as key_cache

  Outputs:
    - all_logits [1, 15, 2048] (fp16): logits for all 15 codebooks (CB1-CB15)
    - hidden_states [1, 1024, 1, 1] (fp16): final hidden state
    - new_key_cache [1, 5120, 1, 16] (fp16): updated
    - new_value_cache [1, 5120, 1, 16] (fp16): updated

Architecture from MIL:
  - 5 layers, 16 Q heads, 8 KV heads, head_dim=128, hidden=1024
  - Same structure as CodeDecoder but with max_seq=16 (only 17 positions needed)
  - 15 separate lm_heads (lm_heads_0 through lm_heads_14)
  - RoPE via cos/sin tables [16, 128] (not palettized, just fp16 const)
  - W8A16 palettized weights

Usage:
    python convert_argmax_multi_code_decoder.py [--model-path ./model_0.6b]
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import argparse

MAX_CP_SEQ_LEN = 16  # code predictor max: 2 initial + 15 generated - 1


def patch_rmsnorm_for_trace():
    """Monkey-patch Qwen3-TTS RMSNorm to avoid dynamic dtype casts."""
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSRMSNorm

    def rmsnorm_forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.float() * hidden_states

    Qwen3TTSRMSNorm.forward = rmsnorm_forward


def patch_coremltools_int_cast():
    """Fix coremltools _cast to handle multi-dimensional constant arrays."""
    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    def patched_cast(context, node, dtype, dtype_name):
        inputs = ct_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
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


class ArgmaxMultiCodeDecoder(nn.Module):
    """
    5-layer code predictor transformer matching Argmax's MultiCodeDecoder.

    Takes a single embedding, processes through 5 transformer layers with KV cache,
    and produces logits from all 15 lm_heads simultaneously.
    """

    def __init__(self, code_predictor, codec_embedding):
        super().__init__()
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.lm_heads = code_predictor.lm_head

        # Extract inv_freq for manual RoPE computation
        rotary_emb = code_predictor.model.rotary_emb
        self.register_buffer('inv_freq', rotary_emb.inv_freq.clone())
        self.attention_scaling = rotary_emb.attention_scaling

        attn0 = self.layers[0].self_attn
        cfg = attn0.config
        self.num_heads = cfg.num_attention_heads       # 16
        self.num_kv_heads = cfg.num_key_value_heads    # 8
        self.head_dim = attn0.head_dim                 # 128
        self.num_layers = len(self.layers)             # 5
        self.kv_dim = self.num_kv_heads * self.head_dim  # 1024
        self.hidden_size = cfg.hidden_size  # 1024

    def forward(
        self,
        input_embeds: torch.Tensor,       # [1, 1024, 1, 1]
        cache_length: torch.Tensor,       # [1]
        key_cache: torch.Tensor,          # [1, 5120, 1, 16]
        key_padding_mask: torch.Tensor,   # [1, 16]
        kv_cache_update_mask: torch.Tensor,  # [1, 16]
        value_cache: torch.Tensor,        # [1, 5120, 1, 16]
    ):
        """
        Returns:
            all_logits: [1, 15, 2048]
            hidden_states: [1, 1024, 1, 1]
            new_key_cache: [1, 5120, 1, 16]
            new_value_cache: [1, 5120, 1, 16]
        """
        # Manual RoPE for current position (avoids 3D position_ids issue)
        position_ids = cache_length.unsqueeze(0).float()  # [1, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float()  # [1, dim/2, 1]
        freqs = (inv_freq_expanded @ position_ids.unsqueeze(-1)).transpose(1, 2)  # [1, 1, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [1, 1, head_dim]
        cos = (emb.cos() * self.attention_scaling).to(input_embeds.dtype)
        sin = (emb.sin() * self.attention_scaling).to(input_embeds.dtype)

        hidden = input_embeds  # [1, 1024, 1, 1]

        new_key_slices = []
        new_value_slices = []

        for i, layer in enumerate(self.layers):
            start = i * self.kv_dim
            end = (i + 1) * self.kv_dim
            layer_kc = key_cache[:, start:end, :, :]     # [1, 1024, 1, 16]
            layer_vc = value_cache[:, start:end, :, :]

            hidden, nk, nv = self._transformer_layer(
                layer, hidden, layer_kc, layer_vc, cos, sin, key_padding_mask,
                kv_cache_update_mask
            )
            new_key_slices.append(nk)
            new_value_slices.append(nv)

        # Final norm
        h_3d = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)  # [1, 1, 1024]
        normed = self.norm(h_3d)  # [1, 1, 1024]

        # All 15 lm_heads
        all_logits = []
        for head in self.lm_heads:
            logits = head(normed)  # [1, 1, 2048]
            all_logits.append(logits.squeeze(1))  # [1, 2048]
        all_logits = torch.stack(all_logits, dim=1)  # [1, 15, 2048]

        # Update caches
        new_key_full = self._update_cache(key_cache, new_key_slices, kv_cache_update_mask)
        new_value_full = self._update_cache(value_cache, new_value_slices, kv_cache_update_mask)

        return all_logits, hidden, new_key_full, new_value_full

    def _transformer_layer(
        self, layer, hidden, key_cache, value_cache, cos, sin, key_padding_mask,
        kv_cache_update_mask
    ):
        """Single transformer layer with cached KV attention.

        Args:
            hidden: [1, 1024, 1, 1] — ANE layout
            key_cache: [1, 1024, 1, 16]
            value_cache: [1, 1024, 1, 16]
            cos: [1, 1, 128]
            sin: [1, 1, 128]
            key_padding_mask: [1, 16]
            kv_cache_update_mask: [1, 16] — one-hot mask for current position
        """
        residual = hidden

        # RMSNorm
        h_3d = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)  # [1, 1, 1024]
        h_normed = layer.input_layernorm(h_3d)

        attn = layer.self_attn

        # Q/K/V
        q = attn.q_proj(h_normed).view(1, 1, self.num_heads, self.head_dim)
        k = attn.k_proj(h_normed).view(1, 1, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(h_normed).view(1, 1, self.num_kv_heads, self.head_dim)

        # QK norm
        if hasattr(attn, 'q_norm'):
            q = attn.q_norm(q)
            k = attn.k_norm(k)

        q = q.transpose(1, 2)  # [1, 16, 1, 128]
        k = k.transpose(1, 2)  # [1, 8, 1, 128]
        v = v.transpose(1, 2)  # [1, 8, 1, 128]

        # RoPE
        cos_r = cos.unsqueeze(1)  # [1, 1, 1, 128]
        sin_r = sin.unsqueeze(1)
        q = (q * cos_r) + (rotate_half(q) * sin_r)
        k = (k * cos_r) + (rotate_half(k) * sin_r)

        new_key = k
        new_value = v

        # Write current K/V to cache BEFORE computing attention
        # This ensures the current position's K/V participates in attention
        nk_flat = new_key.reshape(1, self.kv_dim, 1, 1)   # [1, 1024, 1, 1]
        nv_flat = new_value.reshape(1, self.kv_dim, 1, 1)
        update_mask = kv_cache_update_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, 16]
        key_cache = key_cache * (1.0 - update_mask) + nk_flat * update_mask
        value_cache = value_cache * (1.0 - update_mask) + nv_flat * update_mask

        # Reshape updated cache for attention
        # key_cache: [1, 1024, 1, 16] → [1, 8, 128, 16]
        kc = key_cache.squeeze(2).view(1, self.num_kv_heads, self.head_dim, MAX_CP_SEQ_LEN)
        vc = value_cache.squeeze(2).view(1, self.num_kv_heads, self.head_dim, MAX_CP_SEQ_LEN)

        # GQA
        n_rep = self.num_heads // self.num_kv_heads
        kc = kc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, MAX_CP_SEQ_LEN)
        vc = vc.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(1, self.num_heads, self.head_dim, MAX_CP_SEQ_LEN)

        # Attention
        attn_weights = torch.matmul(q, kc) / (self.head_dim ** 0.5)  # [1, 16, 1, 16]
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, 16]
        attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights.float(), dim=-1)

        vc_t = vc.transpose(-1, -2)  # [1, 16, 16, 128]
        attn_output = torch.matmul(attn_weights, vc_t)  # [1, 16, 1, 128]
        attn_output = attn_output.transpose(1, 2).reshape(1, 1, -1)
        attn_output = attn.o_proj(attn_output)  # [1, 1, 1024]

        attn_4d = attn_output.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        hidden = residual + attn_4d

        # MLP
        residual = hidden
        h_3d = hidden.squeeze(-1).squeeze(-1).unsqueeze(0)
        h_normed = layer.post_attention_layernorm(h_3d)
        mlp_out = layer.mlp(h_normed)
        mlp_4d = mlp_out.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        hidden = residual + mlp_4d

        # Return new KV in ANE layout
        nk = nk_flat
        nv = nv_flat
        return hidden, nk, nv

    def _update_cache(self, old_cache, new_slices, update_mask):
        """Element-wise cache update."""
        new_kv = torch.cat(new_slices, dim=1)  # [1, 5120, 1, 1]
        mask = update_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, 16]
        return old_cache * (1.0 - mask) + new_kv * mask


def main():
    parser = argparse.ArgumentParser(description="Convert Argmax-style MultiCodeDecoder")
    parser.add_argument("--model-path", default="./model_0.6b", help="Path to Qwen3-TTS model")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--quantize-w8", action="store_true", help="Apply W8A16 palettization")
    parser.add_argument("--skip-verify", action="store_true", help="Skip CoreML verification")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    patch_rmsnorm_for_trace()
    patch_coremltools_int_cast()

    print("=" * 60)
    print("Argmax-style MultiCodeDecoder Conversion (5-layer CP)")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model_path, device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker
    cp = talker.code_predictor
    codec_embedding = talker.model.codec_embedding

    print(f"   CP layers: {len(cp.model.layers)}")
    print(f"   CP lm_heads: {len(cp.lm_head)}")

    # 2. Create wrapper
    print("\n2. Creating MultiCodeDecoder wrapper...")
    wrapper = ArgmaxMultiCodeDecoder(cp, codec_embedding)
    wrapper.eval()

    kv_dim = wrapper.kv_dim
    num_layers = wrapper.num_layers
    total_kv = num_layers * kv_dim  # 5120

    # 3. Test
    print("\n3. Testing wrapper...")
    test_embeds = torch.randn(1, 1024, 1, 1)
    test_cache_len = torch.tensor([2])
    test_kc = torch.randn(1, total_kv, 1, MAX_CP_SEQ_LEN)
    test_mask = torch.zeros(1, MAX_CP_SEQ_LEN)
    test_mask[0, 3:] = float('-inf')
    test_update = torch.zeros(1, MAX_CP_SEQ_LEN)
    test_update[0, 2] = 1.0
    test_vc = torch.randn(1, total_kv, 1, MAX_CP_SEQ_LEN)

    with torch.no_grad():
        all_logits, hidden, new_kc, new_vc = wrapper(
            test_embeds, test_cache_len, test_kc, test_mask, test_update, test_vc
        )
    print(f"   all_logits: {all_logits.shape}")     # [1, 15, 2048]
    print(f"   hidden: {hidden.shape}")              # [1, 1024, 1, 1]
    print(f"   new_key_cache: {new_kc.shape}")       # [1, 5120, 1, 16]
    print(f"   new_value_cache: {new_vc.shape}")

    # 4. Trace
    print("\n4. Tracing...")
    example = (test_embeds, test_cache_len, test_kc, test_mask, test_update, test_vc)
    traced = torch.jit.trace(wrapper, example, strict=False)

    with torch.no_grad():
        t_logits, _, _, _ = traced(*example)
    diff = (t_logits - all_logits).abs().max().item()
    print(f"   Trace diff: {diff}")

    # 5. Convert
    print("\n5. Converting to CoreML...")
    inputs = [
        ct.TensorType(name="input_embeds", shape=(1, 1024, 1, 1), dtype=np.float16),
        ct.TensorType(name="cache_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="key_cache", shape=(1, total_kv, 1, MAX_CP_SEQ_LEN), dtype=np.float16),
        ct.TensorType(name="key_padding_mask", shape=(1, MAX_CP_SEQ_LEN), dtype=np.float16),
        ct.TensorType(name="kv_cache_update_mask", shape=(1, MAX_CP_SEQ_LEN), dtype=np.float16),
        ct.TensorType(name="value_cache", shape=(1, total_kv, 1, MAX_CP_SEQ_LEN), dtype=np.float16),
    ]

    outputs = [
        ct.TensorType(name="all_logits", dtype=np.float16),
        ct.TensorType(name="hidden_states", dtype=np.float16),
        ct.TensorType(name="new_key_cache", dtype=np.float16),
        ct.TensorType(name="new_value_cache", dtype=np.float16),
    ]

    # FLOAT32 precision is required here because the CodeDecoder outputs
    # post-RMSNorm hidden_states with large outlier values (up to ~100 due to
    # RMSNorm weights up to ~20). In FLOAT16 precision, the Q/K/V matmuls inside
    # the MCD layers produce NaN from intermediate overflow with these inputs.
    # The MCD is only 5 layers so the perf impact of fp32 is minimal.
    ml_model = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
    )

    # 6. W8A16 quantization
    if args.quantize_w8:
        print("\n6. Applying W8A16 palettized quantization...")
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        op_config = OpPalettizerConfig(mode="kmeans", nbits=8, weight_threshold=512)
        opt_config = OptimizationConfig(global_config=op_config)
        ml_model = palettize_weights(ml_model, config=opt_config)
        print("   W8A16 palettization applied")

    mcd_path = f"{args.output_dir}/MultiCodeDecoder.mlpackage"
    ml_model.save(mcd_path)
    print(f"   Saved: {mcd_path}")

    # 7. Verify
    if not args.skip_verify:
        print("\n7. Verifying CoreML model...")
        loaded = ct.models.MLModel(mcd_path)
        result = loaded.predict({
            "input_embeds": test_embeds.numpy().astype(np.float16),
            "cache_length": test_cache_len.numpy().astype(np.int32),
            "key_cache": test_kc.numpy().astype(np.float16),
            "key_padding_mask": test_mask.numpy().astype(np.float16),
            "kv_cache_update_mask": test_update.numpy().astype(np.float16),
            "value_cache": test_vc.numpy().astype(np.float16),
        })
        print(f"   all_logits shape: {result['all_logits'].shape}")
        cml_logits = result['all_logits'].astype(np.float32)
        pt_logits = all_logits.detach().numpy().astype(np.float32)
        diff = np.abs(cml_logits - pt_logits).max()
        print(f"   Max diff: {diff}")

    print("\n" + "=" * 60)
    print(f"Done! Model saved: {mcd_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
