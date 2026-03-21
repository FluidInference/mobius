"""Convert the VoxCPM 1.5 base LM step to CoreML.

Processes one token through the 24-layer base LM, producing the normalized
hidden state, FSQ-quantized hidden state, and stop logits.

Input:
  embed: [1, 1024] - input embedding (text or audio feature)
  position: [1] - int32 position in sequence
  k0..k23, v0..v23: [1, 2, max_len, 64] - KV cache per layer

Output:
  lm_hidden: [1, 1024] - normalized base LM output (for stop head)
  lm_hidden_fsq: [1, 1024] - FSQ-quantized output (for residual LM input)
  stop_logit: [1, 2] - stop prediction logits
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


class TraceableBaseLMStep(nn.Module):
    """Single-step base LM processing with explicit KV cache I/O."""

    def __init__(self, tts_model):
        super().__init__()
        self.base_lm = tts_model.base_lm
        self.fsq_layer = tts_model.fsq_layer
        self.stop_proj = tts_model.stop_proj
        self.stop_actn = tts_model.stop_actn
        self.stop_head = tts_model.stop_head
        self.num_layers = len(self.base_lm.layers)

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
        k8: torch.Tensor, v8: torch.Tensor,
        k9: torch.Tensor, v9: torch.Tensor,
        k10: torch.Tensor, v10: torch.Tensor,
        k11: torch.Tensor, v11: torch.Tensor,
        k12: torch.Tensor, v12: torch.Tensor,
        k13: torch.Tensor, v13: torch.Tensor,
        k14: torch.Tensor, v14: torch.Tensor,
        k15: torch.Tensor, v15: torch.Tensor,
        k16: torch.Tensor, v16: torch.Tensor,
        k17: torch.Tensor, v17: torch.Tensor,
        k18: torch.Tensor, v18: torch.Tensor,
        k19: torch.Tensor, v19: torch.Tensor,
        k20: torch.Tensor, v20: torch.Tensor,
        k21: torch.Tensor, v21: torch.Tensor,
        k22: torch.Tensor, v22: torch.Tensor,
        k23: torch.Tensor, v23: torch.Tensor,
    ):
        caches = [
            (k0, v0), (k1, v1), (k2, v2), (k3, v3),
            (k4, v4), (k5, v5), (k6, v6), (k7, v7),
            (k8, v8), (k9, v9), (k10, v10), (k11, v11),
            (k12, v12), (k13, v13), (k14, v14), (k15, v15),
            (k16, v16), (k17, v17), (k18, v18), (k19, v19),
            (k20, v20), (k21, v21), (k22, v22), (k23, v23),
        ]

        position_emb = self.base_lm.rope_emb(position)
        hidden = embed

        new_caches = []
        for i, layer in enumerate(self.base_lm.layers):
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

        lm_hidden = self.base_lm.norm(hidden)
        lm_hidden_fsq = self.fsq_layer(lm_hidden)
        stop_logit = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))

        outputs = [lm_hidden, lm_hidden_fsq, stop_logit]
        for k, v in new_caches:
            outputs.extend([k, v])
        return tuple(outputs)


def main():
    print("=== Converting Base LM Step to CoreML ===\n")

    print("[1/5] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    tts = model.tts_model.float().cpu().eval()

    patch_gqa_attention_step(tts.base_lm)

    lm_step = TraceableBaseLMStep(tts)
    lm_step.eval()

    # Test inputs
    cache_shape = (1, 2, MAX_SEQ_LEN, 64)
    embed = torch.randn(1, 1024)
    position = torch.tensor([0])
    caches = [torch.zeros(cache_shape) for _ in range(48)]

    all_inputs = [embed, position] + caches

    print("[2/5] Testing PyTorch forward...")
    with torch.no_grad():
        outputs = lm_step(*all_inputs)
    lm_hidden = outputs[0]
    lm_hidden_fsq = outputs[1]
    stop_logit = outputs[2]
    print(f"  lm_hidden: {lm_hidden.shape}, range [{lm_hidden.min():.4f}, {lm_hidden.max():.4f}]")
    print(f"  lm_hidden_fsq: {lm_hidden_fsq.shape}, range [{lm_hidden_fsq.min():.4f}, {lm_hidden_fsq.max():.4f}]")
    print(f"  stop_logit: {stop_logit.shape}")
    print(f"  Total output tensors: {len(outputs)} (3 + 48 caches)")

    print("\n[3/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(lm_step, all_inputs)

    with torch.no_grad():
        traced_outputs = traced(*all_inputs)
    max_diff = max((o1 - o2).abs().max().item() for o1, o2 in zip(outputs[:3], traced_outputs[:3]))
    print(f"  Trace parity: max diff = {max_diff:.2e}")

    print("\n[4/5] Converting to CoreML...")
    t0 = time.time()

    ct_inputs = [
        ct.TensorType(name="embed", shape=(1, 1024)),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
    ]
    for i in range(24):
        ct_inputs.append(ct.TensorType(name=f"k{i}", shape=cache_shape))
        ct_inputs.append(ct.TensorType(name=f"v{i}", shape=cache_shape))

    ct_outputs = [
        ct.TensorType(name="lm_hidden"),
        ct.TensorType(name="lm_hidden_fsq"),
        ct.TensorType(name="stop_logit"),
    ]
    for i in range(24):
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

    out_path = "base_lm_step.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Validate
    print("\n[5/5] Validating...")
    input_dict = {"embed": embed.numpy(), "position": position.numpy().astype(np.int32)}
    for i in range(24):
        input_dict[f"k{i}"] = caches[i * 2].numpy()
        input_dict[f"v{i}"] = caches[i * 2 + 1].numpy()

    coreml_pred = mlmodel.predict(input_dict)

    for name, idx in [("lm_hidden", 0), ("lm_hidden_fsq", 1), ("stop_logit", 2)]:
        pt_val = outputs[idx].numpy()
        cm_val = coreml_pred[name]
        diff = np.abs(pt_val - cm_val)
        pt_flat = pt_val.flatten()
        cm_flat = cm_val.flatten()
        if np.std(pt_flat) > 1e-6 and np.std(cm_flat) > 1e-6:
            corr = np.corrcoef(pt_flat, cm_flat)[0, 1]
        else:
            corr = float('nan')
        print(f"  {name}: max_diff={diff.max():.4e}, mean_diff={diff.mean():.4e}, corr={corr:.6f}")

    corr_lm = np.corrcoef(outputs[0].numpy().flatten(), coreml_pred["lm_hidden"].flatten())[0, 1]
    corr_fsq = np.corrcoef(outputs[1].numpy().flatten(), coreml_pred["lm_hidden_fsq"].flatten())[0, 1]

    if corr_lm > 0.999 and corr_fsq > 0.99:
        print("  PASS: Excellent correlation")
    elif corr_lm > 0.99:
        print("  WARN: Acceptable correlation")
    else:
        print("  FAIL: Poor correlation")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
