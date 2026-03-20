"""Convert the VoxCPM 1.5 LocDiT estimator to CoreML.

The LocDiT estimates velocity fields for the flow matching diffusion process.
Called with batch=2 for classifier-free guidance (conditioned + unconditioned).

Input:
  x: [2, 64, patch_size=4] - noisy latent (conditioned + zero-cond copies)
  mu: [2, 1024] - LM hidden state (conditioned + zeros)
  t: [2] - current timestep
  cond: [2, 64, T'] - prefix conditioning features
  dt: [2] - delta time (zeros when mean_mode=False)

Output:
  velocity: [2, 64, patch_size=4] - estimated velocity field

The Euler solver loop runs in Python/Swift, calling this model N times
(default 10 timesteps).
"""

import time
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_gqa_attention(model: nn.Module):
    """Replace scaled_dot_product_attention with GQA-expanded version."""
    from voxcpm.modules.minicpm4.model import MiniCPMAttention, apply_rotary_pos_emb

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        num_groups = self.num_heads // self.num_key_value_heads
        key_states = key_states.repeat_interleave(num_groups, dim=1)
        value_states = value_states.repeat_interleave(num_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        past_key_value = (key_states, value_states)
        return attn_output, past_key_value

    for module in model.modules():
        if isinstance(module, MiniCPMAttention):
            import types
            module.forward = types.MethodType(patched_forward, module)


class TraceableLocDiT(nn.Module):
    """Wraps VoxCPMLocDiT for tracing with fixed shapes."""

    def __init__(self, estimator):
        super().__init__()
        self.in_proj = estimator.in_proj
        self.cond_proj = estimator.cond_proj
        self.out_proj = estimator.out_proj
        self.time_embeddings = estimator.time_embeddings
        self.time_mlp = estimator.time_mlp
        self.delta_time_mlp = estimator.delta_time_mlp
        self.decoder = estimator.decoder

    def forward(
        self,
        x: torch.Tensor,      # [B, 64, 4]
        mu: torch.Tensor,      # [B, 1024]
        t: torch.Tensor,       # [B]
        cond: torch.Tensor,    # [B, 64, T']
        dt: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        # Project x: [B, 64, 4] -> transpose -> [B, 4, 64] -> in_proj -> [B, 4, 1024]
        x_proj = self.in_proj(x.transpose(1, 2).contiguous())

        # Project cond: [B, 64, T'] -> transpose -> [B, T', 64] -> cond_proj -> [B, T', 1024]
        cond_proj = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond_proj.size(1)

        # Time embeddings
        t_emb = self.time_embeddings(t).to(x_proj.dtype)
        t_emb = self.time_mlp(t_emb)
        dt_emb = self.time_embeddings(dt).to(x_proj.dtype)
        dt_emb = self.delta_time_mlp(dt_emb)
        t_emb = t_emb + dt_emb

        # Concat: [mu+t, cond, x] along seq dim
        # mu+t: [B, 1024] -> [B, 1, 1024]
        combined = torch.cat([(mu + t_emb).unsqueeze(1), cond_proj, x_proj], dim=1)

        # Decoder (non-causal)
        hidden, _ = self.decoder(combined, is_causal=False)

        # Extract only the x portion (skip mu token and cond tokens)
        hidden = hidden[:, prefix + 1:, :]
        hidden = self.out_proj(hidden)

        return hidden.transpose(1, 2).contiguous()


def main():
    print("=== Converting LocDiT estimator to CoreML ===\n")

    # Load model
    print("[1/5] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    tts = model.tts_model
    est = tts.feat_decoder.estimator.float().cpu().eval()

    # Patch GQA
    patch_gqa_attention(est)

    traceable = TraceableLocDiT(est)
    traceable.eval()

    # Test inputs: batch=2 for CFG, patch_size=4, cond=last patch (patch_size=4)
    patch_size = 4
    cond_len = patch_size  # VoxCPM uses last patch as conditioning
    batch = 2  # conditioned + unconditioned for CFG

    test_x = torch.randn(batch, 64, patch_size)
    test_mu = torch.randn(batch, 1024)
    test_t = torch.tensor([0.5, 0.5])
    test_cond = torch.randn(batch, 64, cond_len)
    test_dt = torch.zeros(batch)

    with torch.no_grad():
        pt_output = traceable(test_x, test_mu, test_t, test_cond, test_dt)
    print(f"  PyTorch: output shape {pt_output.shape}")

    # Trace
    print("\n[2/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(traceable, (test_x, test_mu, test_t, test_cond, test_dt))

    with torch.no_grad():
        traced_output = traced(test_x, test_mu, test_t, test_cond, test_dt)
    max_diff = (pt_output - traced_output).abs().max().item()
    print(f"  Trace parity: max diff = {max_diff:.2e}")

    # Convert to CoreML
    print("\n[3/5] Converting to CoreML...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=(batch, 64, patch_size)),
            ct.TensorType(name="mu", shape=(batch, 1024)),
            ct.TensorType(name="t", shape=(batch,)),
            ct.TensorType(name="cond", shape=(batch, 64, cond_len)),
            ct.TensorType(name="dt", shape=(batch,)),
        ],
        outputs=[
            ct.TensorType(name="velocity"),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    out_path = "locdit_step.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Validate
    print("\n[4/5] Validating...")
    coreml_pred = mlmodel.predict({
        "x": test_x.numpy(),
        "mu": test_mu.numpy(),
        "t": test_t.numpy(),
        "cond": test_cond.numpy(),
        "dt": test_dt.numpy(),
    })
    coreml_output = coreml_pred["velocity"]
    diff = np.abs(pt_output.numpy() - coreml_output)
    corr = np.corrcoef(pt_output.numpy().flatten(), coreml_output.flatten())[0, 1]
    print(f"  Absolute: max={diff.max():.4e}, mean={diff.mean():.4e}")
    print(f"  Correlation: {corr:.6f}")

    if corr > 0.999:
        print("  PASS: Excellent correlation")
    elif corr > 0.99:
        print("  WARN: Good correlation")
    else:
        print("  FAIL: Poor correlation")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
