"""Convert the VoxCPM 1.5 residual LM step to CoreML.

Processes one token through the 8-layer residual LM.

Input:
  embed: [1, 1024] - FSQ output + original embedding
  position: [1] - int32 position in sequence
  k0..k7, v0..v7: [1, 2, max_len, 64] - KV cache per layer

Output:
  res_hidden: [1, 1024] - residual LM output (normalized)
  Updated KV caches (same shapes as inputs)
"""

import math
import time
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ_LEN = 512


def patch_gqa_attention_step(model: nn.Module):
    """Patch the attention forward_step to avoid GQA SDPA and in-place cache updates."""
    from voxcpm.modules.minicpm4.model import MiniCPMAttention, apply_rotary_pos_emb

    def patched_forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        position_id: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
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

        pos = position_id.long()
        idx = pos.view(1, 1, 1, 1).expand(bsz, self.num_key_value_heads, 1, self.head_dim)
        new_key_cache = key_cache.scatter(2, idx, key_states)
        new_value_cache = value_cache.scatter(2, idx, value_states)

        num_groups = self.num_heads // self.num_key_value_heads
        expanded_k = new_key_cache.repeat_interleave(num_groups, dim=1)
        expanded_v = new_value_cache.repeat_interleave(num_groups, dim=1)

        seq_range = torch.arange(new_key_cache.size(2), device=query_states.device)
        attn_mask = seq_range.unsqueeze(0) <= pos.unsqueeze(1)
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

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


class TraceableResidualLMStep(nn.Module):
    """Single-step residual LM processing with explicit KV cache I/O."""

    def __init__(self, tts_model):
        super().__init__()
        self.residual_lm = tts_model.residual_lm
        self.num_layers = len(self.residual_lm.layers)

    def forward(
        self,
        embed: torch.Tensor,
        position: torch.Tensor,
        k0: torch.Tensor, v0: torch.Tensor,
        k1: torch.Tensor, v1: torch.Tensor,
        k2: torch.Tensor, v2: torch.Tensor,
        k3: torch.Tensor, v3: torch.Tensor,
        k4: torch.Tensor, v4: torch.Tensor,
        k5: torch.Tensor, v5: torch.Tensor,
        k6: torch.Tensor, v6: torch.Tensor,
        k7: torch.Tensor, v7: torch.Tensor,
    ):
        caches = [
            (k0, v0), (k1, v1), (k2, v2), (k3, v3),
            (k4, v4), (k5, v5), (k6, v6), (k7, v7),
        ]

        position_emb = self.residual_lm.rope_emb(position)
        hidden = embed

        new_caches = []
        for i, layer in enumerate(self.residual_lm.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attn_out, new_kv = layer.self_attn.forward_step(
                hidden, position_emb, position, caches[i]
            )
            new_caches.append(new_kv)

            scale = layer.scale_depth / math.sqrt(layer.num_hidden_layers) if layer.use_mup else 1.0
            hidden = residual + attn_out * scale

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden * scale

        res_hidden = self.residual_lm.norm(hidden)

        outputs = [res_hidden]
        for k, v in new_caches:
            outputs.extend([k, v])
        return tuple(outputs)


def main():
    print("=== Converting Residual LM Step to CoreML ===\n")

    print("[1/5] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    tts = model.tts_model.float().cpu().eval()

    patch_gqa_attention_step(tts.residual_lm)

    res_step = TraceableResidualLMStep(tts)
    res_step.eval()

    # Test inputs
    cache_shape = (1, 2, MAX_SEQ_LEN, 64)
    embed = torch.randn(1, 1024)
    position = torch.tensor([0])
    caches = [torch.zeros(cache_shape) for _ in range(16)]

    all_inputs = [embed, position] + caches

    print("[2/5] Testing PyTorch forward...")
    with torch.no_grad():
        outputs = res_step(*all_inputs)
    res_hidden = outputs[0]
    print(f"  res_hidden: {res_hidden.shape}, range [{res_hidden.min():.4f}, {res_hidden.max():.4f}]")
    print(f"  Total output tensors: {len(outputs)} (1 + 16 caches)")

    print("\n[3/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(res_step, all_inputs)

    with torch.no_grad():
        traced_outputs = traced(*all_inputs)
    max_diff = (outputs[0] - traced_outputs[0]).abs().max().item()
    print(f"  Trace parity: max diff = {max_diff:.2e}")

    print("\n[4/5] Converting to CoreML...")
    t0 = time.time()

    ct_inputs = [
        ct.TensorType(name="embed", shape=(1, 1024)),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
    ]
    for i in range(8):
        ct_inputs.append(ct.TensorType(name=f"k{i}", shape=cache_shape))
        ct_inputs.append(ct.TensorType(name=f"v{i}", shape=cache_shape))

    ct_outputs = [ct.TensorType(name="res_hidden")]
    for i in range(8):
        ct_outputs.append(ct.TensorType(name=f"out_k{i}"))
        ct_outputs.append(ct.TensorType(name=f"out_v{i}"))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    out_path = "residual_lm_step.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Validate
    print("\n[5/5] Validating...")
    input_dict = {"embed": embed.numpy(), "position": position.numpy().astype(np.int32)}
    for i in range(8):
        input_dict[f"k{i}"] = caches[i * 2].numpy()
        input_dict[f"v{i}"] = caches[i * 2 + 1].numpy()

    coreml_pred = mlmodel.predict(input_dict)
    coreml_res = coreml_pred["res_hidden"]

    diff = np.abs(res_hidden.numpy() - coreml_res)
    corr = np.corrcoef(res_hidden.numpy().flatten(), coreml_res.flatten())[0, 1]

    print(f"  res_hidden: max_diff={diff.max():.4e}, mean_diff={diff.mean():.4e}, corr={corr:.6f}")

    if corr > 0.999:
        print("  PASS: Excellent correlation")
    elif corr > 0.99:
        print("  WARN: Acceptable correlation")
    else:
        print("  FAIL: Poor correlation")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
