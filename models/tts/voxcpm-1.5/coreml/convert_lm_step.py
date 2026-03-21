"""Convert the VoxCPM 1.5 LM step (base_lm + residual_lm) to CoreML.

This model processes one token through both the base LM (24 layers) and
residual LM (6 layers), updating KV caches for each.

For prefill, this is called once per conditioning token.
For generation, this is called once per autoregressive step.

Input:
  embed: [1, 1024] - input embedding (text or audio feature)
  position: [1] - int32 position in sequence
  base_k0..base_k23, base_v0..base_v23: [1, 2, max_len, 64] - base LM KV cache
  res_k0..res_k5, res_v0..res_v5: [1, 2, max_len, 64] - residual LM KV cache

Output:
  lm_hidden: [1, 1024] - base LM output after FSQ
  res_hidden: [1, 1024] - residual LM output
  stop_logit: [1, 2] - stop prediction logits
  Updated KV caches (same shapes as inputs)
"""

import time
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ_LEN = 512  # Max sequence length for KV cache


def patch_gqa_attention_step(model: nn.Module):
    """Patch the attention forward_step to avoid GQA SDPA and in-place cache updates."""
    from voxcpm.modules.minicpm4.model import MiniCPMAttention, apply_rotary_pos_emb

    def patched_forward_step(
        self,
        hidden_states: torch.Tensor,       # [B, hidden]
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        position_id: torch.Tensor,          # scalar or [1]
        kv_cache: Tuple[torch.Tensor, torch.Tensor],  # (key_cache, value_cache)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_cache, value_cache = kv_cache

        # Functional cache update (no in-place ops for CoreML)
        pos = position_id.long()
        # Create index for scatter
        idx = pos.view(1, 1, 1, 1).expand(bsz, self.num_key_value_heads, 1, self.head_dim)
        new_key_cache = key_cache.scatter(2, idx, key_states)
        new_value_cache = value_cache.scatter(2, idx, value_states)

        # Expand KV heads for MHA
        num_groups = self.num_heads // self.num_key_value_heads
        expanded_k = new_key_cache.repeat_interleave(num_groups, dim=1)
        expanded_v = new_value_cache.repeat_interleave(num_groups, dim=1)

        # Causal mask
        attn_mask = torch.arange(new_key_cache.size(2), device=query_states.device).unsqueeze(0) <= pos.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # [1, 1, 1, max_len]

        attn_output = F.scaled_dot_product_attention(
            query_states, expanded_k, expanded_v,
            attn_mask=attn_mask.expand(-1, self.num_heads, -1, -1),
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, (new_key_cache, new_value_cache)

    for module in model.modules():
        if isinstance(module, MiniCPMAttention):
            import types
            module.forward_step = types.MethodType(patched_forward_step, module)


class TraceableLMStep(nn.Module):
    """Single-step LM processing with explicit KV cache I/O.

    Combines base_lm step + FSQ + residual_lm step + stop head.
    """

    def __init__(self, tts_model):
        super().__init__()
        self.base_lm = tts_model.base_lm
        self.residual_lm = tts_model.residual_lm
        self.fsq_layer = tts_model.fsq_layer
        self.stop_proj = tts_model.stop_proj
        self.stop_actn = tts_model.stop_actn
        self.stop_head = tts_model.stop_head

        self.num_base_layers = len(self.base_lm.layers)
        self.num_res_layers = len(self.residual_lm.layers)

    def forward(
        self,
        embed: torch.Tensor,           # [1, 1024]
        position: torch.Tensor,        # [1] int
        # Base LM KV caches: 24 pairs
        base_k0: torch.Tensor, base_v0: torch.Tensor,
        base_k1: torch.Tensor, base_v1: torch.Tensor,
        base_k2: torch.Tensor, base_v2: torch.Tensor,
        base_k3: torch.Tensor, base_v3: torch.Tensor,
        base_k4: torch.Tensor, base_v4: torch.Tensor,
        base_k5: torch.Tensor, base_v5: torch.Tensor,
        base_k6: torch.Tensor, base_v6: torch.Tensor,
        base_k7: torch.Tensor, base_v7: torch.Tensor,
        base_k8: torch.Tensor, base_v8: torch.Tensor,
        base_k9: torch.Tensor, base_v9: torch.Tensor,
        base_k10: torch.Tensor, base_v10: torch.Tensor,
        base_k11: torch.Tensor, base_v11: torch.Tensor,
        base_k12: torch.Tensor, base_v12: torch.Tensor,
        base_k13: torch.Tensor, base_v13: torch.Tensor,
        base_k14: torch.Tensor, base_v14: torch.Tensor,
        base_k15: torch.Tensor, base_v15: torch.Tensor,
        base_k16: torch.Tensor, base_v16: torch.Tensor,
        base_k17: torch.Tensor, base_v17: torch.Tensor,
        base_k18: torch.Tensor, base_v18: torch.Tensor,
        base_k19: torch.Tensor, base_v19: torch.Tensor,
        base_k20: torch.Tensor, base_v20: torch.Tensor,
        base_k21: torch.Tensor, base_v21: torch.Tensor,
        base_k22: torch.Tensor, base_v22: torch.Tensor,
        base_k23: torch.Tensor, base_v23: torch.Tensor,
        # Residual LM KV caches: 8 pairs
        res_k0: torch.Tensor, res_v0: torch.Tensor,
        res_k1: torch.Tensor, res_v1: torch.Tensor,
        res_k2: torch.Tensor, res_v2: torch.Tensor,
        res_k3: torch.Tensor, res_v3: torch.Tensor,
        res_k4: torch.Tensor, res_v4: torch.Tensor,
        res_k5: torch.Tensor, res_v5: torch.Tensor,
        res_k6: torch.Tensor, res_v6: torch.Tensor,
        res_k7: torch.Tensor, res_v7: torch.Tensor,
    ):
        base_caches = [
            (base_k0, base_v0), (base_k1, base_v1), (base_k2, base_v2), (base_k3, base_v3),
            (base_k4, base_v4), (base_k5, base_v5), (base_k6, base_v6), (base_k7, base_v7),
            (base_k8, base_v8), (base_k9, base_v9), (base_k10, base_v10), (base_k11, base_v11),
            (base_k12, base_v12), (base_k13, base_v13), (base_k14, base_v14), (base_k15, base_v15),
            (base_k16, base_v16), (base_k17, base_v17), (base_k18, base_v18), (base_k19, base_v19),
            (base_k20, base_v20), (base_k21, base_v21), (base_k22, base_v22), (base_k23, base_v23),
        ]
        res_caches = [
            (res_k0, res_v0), (res_k1, res_v1), (res_k2, res_v2), (res_k3, res_v3),
            (res_k4, res_v4), (res_k5, res_v5), (res_k6, res_v6), (res_k7, res_v7),
        ]

        # === Base LM forward_step ===
        position_emb = self.base_lm.rope_emb(position)
        hidden = embed

        new_base_caches = []
        for i, layer in enumerate(self.base_lm.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attn_out, new_kv = layer.self_attn.forward_step(
                hidden, position_emb, position, base_caches[i]
            )
            new_base_caches.append(new_kv)

            scale = layer.scale_depth / (layer.num_hidden_layers ** 0.5) if layer.use_mup else 1.0
            hidden = residual + attn_out * scale

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden * scale

        lm_hidden = self.base_lm.norm(hidden)

        # FSQ
        lm_hidden_fsq = self.fsq_layer(lm_hidden)

        # Stop head
        stop_logit = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))

        # === Residual LM forward_step ===
        res_input = lm_hidden_fsq + embed  # FSQ output + original embedding
        position_emb_res = self.residual_lm.rope_emb(position)
        res_hidden = res_input

        new_res_caches = []
        for i, layer in enumerate(self.residual_lm.layers):
            residual = res_hidden
            res_hidden = layer.input_layernorm(res_hidden)
            attn_out, new_kv = layer.self_attn.forward_step(
                res_hidden, position_emb_res, position, res_caches[i]
            )
            new_res_caches.append(new_kv)

            scale = layer.scale_depth / (layer.num_hidden_layers ** 0.5) if layer.use_mup else 1.0
            res_hidden = residual + attn_out * scale

            residual = res_hidden
            res_hidden = layer.post_attention_layernorm(res_hidden)
            res_hidden = layer.mlp(res_hidden)
            res_hidden = residual + res_hidden * scale

        res_hidden = self.residual_lm.norm(res_hidden)

        # Flatten outputs: lm_hidden_fsq, res_hidden, stop_logit, then all cache tensors
        outputs = [lm_hidden_fsq, res_hidden, stop_logit]
        for k, v in new_base_caches:
            outputs.extend([k, v])
        for k, v in new_res_caches:
            outputs.extend([k, v])

        return tuple(outputs)


def main():
    print("=== Converting LM Step to CoreML ===\n")

    print("[1/5] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    tts = model.tts_model
    tts = tts.float().cpu().eval()

    # Patch attention for CoreML compatibility
    patch_gqa_attention_step(tts.base_lm)
    patch_gqa_attention_step(tts.residual_lm)

    lm_step = TraceableLMStep(tts)
    lm_step.eval()

    # Test inputs
    num_kv_heads = 2
    head_dim = 64
    embed = torch.randn(1, 1024)
    position = torch.tensor([0])

    # Initialize empty KV caches
    cache_shape = (1, num_kv_heads, MAX_SEQ_LEN, head_dim)
    base_caches = [torch.zeros(cache_shape) for _ in range(48)]  # 24 layers * 2 (k,v)
    res_caches = [torch.zeros(cache_shape) for _ in range(16)]   # 8 layers * 2 (k,v)

    all_inputs = [embed, position] + base_caches + res_caches

    print("[2/5] Testing PyTorch forward...")
    with torch.no_grad():
        outputs = lm_step(*all_inputs)
    lm_hidden = outputs[0]
    res_hidden = outputs[1]
    stop_logit = outputs[2]
    print(f"  lm_hidden: {lm_hidden.shape}, res_hidden: {res_hidden.shape}")
    print(f"  stop_logit: {stop_logit.shape}")
    print(f"  Total output tensors: {len(outputs)} (3 + 48 base caches + 16 res caches)")

    print("\n[3/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(lm_step, all_inputs)

    # Verify trace
    with torch.no_grad():
        traced_outputs = traced(*all_inputs)
    max_diff = max((o1 - o2).abs().max().item() for o1, o2 in zip(outputs[:3], traced_outputs[:3]))
    print(f"  Trace parity (first 3 outputs): max diff = {max_diff:.2e}")

    # Convert to CoreML
    print("\n[4/5] Converting to CoreML...")
    t0 = time.time()

    ct_inputs = [
        ct.TensorType(name="embed", shape=(1, 1024)),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
    ]
    for i in range(24):
        ct_inputs.append(ct.TensorType(name=f"base_k{i}", shape=cache_shape))
        ct_inputs.append(ct.TensorType(name=f"base_v{i}", shape=cache_shape))
    for i in range(8):
        ct_inputs.append(ct.TensorType(name=f"res_k{i}", shape=cache_shape))
        ct_inputs.append(ct.TensorType(name=f"res_v{i}", shape=cache_shape))

    ct_outputs = [
        ct.TensorType(name="lm_hidden"),
        ct.TensorType(name="res_hidden"),
        ct.TensorType(name="stop_logit"),
    ]
    for i in range(24):
        ct_outputs.append(ct.TensorType(name=f"out_base_k{i}"))
        ct_outputs.append(ct.TensorType(name=f"out_base_v{i}"))
    for i in range(8):
        ct_outputs.append(ct.TensorType(name=f"out_res_k{i}"))
        ct_outputs.append(ct.TensorType(name=f"out_res_v{i}"))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    out_path = "lm_step.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Validate
    print("\n[5/5] Validating...")
    input_dict = {"embed": embed.numpy(), "position": position.numpy().astype(np.int32)}
    for i in range(24):
        input_dict[f"base_k{i}"] = base_caches[i * 2].numpy()
        input_dict[f"base_v{i}"] = base_caches[i * 2 + 1].numpy()
    for i in range(8):
        input_dict[f"res_k{i}"] = res_caches[i * 2].numpy()
        input_dict[f"res_v{i}"] = res_caches[i * 2 + 1].numpy()

    coreml_pred = mlmodel.predict(input_dict)
    coreml_lm = coreml_pred["lm_hidden"]
    coreml_res = coreml_pred["res_hidden"]

    diff_lm = np.abs(lm_hidden.numpy() - coreml_lm).max()
    diff_res = np.abs(res_hidden.numpy() - coreml_res).max()
    corr_lm = np.corrcoef(lm_hidden.numpy().flatten(), coreml_lm.flatten())[0, 1]
    corr_res = np.corrcoef(res_hidden.numpy().flatten(), coreml_res.flatten())[0, 1]

    print(f"  lm_hidden: max_diff={diff_lm:.4e}, corr={corr_lm:.6f}")
    print(f"  res_hidden: max_diff={diff_res:.4e}, corr={corr_res:.6f}")

    if corr_lm > 0.999 and corr_res > 0.999:
        print("  PASS: Excellent correlation")
    else:
        print("  WARN: Check correlations")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
