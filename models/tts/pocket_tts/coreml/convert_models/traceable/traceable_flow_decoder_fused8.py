"""Fused 8-step LSD flow decoder.

Bakes all 8 Euler integration steps into one traced graph so CoreML can run
the full velocity integration in a single inference call, eliminating the
7 round-trips that the current per-step graph requires.

LSD update rule (matches the per-step variant in `traceable_flow_decoder.py`):
    z_0 = latent_init
    for i in [0, 1, ..., 7]:
        s_i  = i / 8
        t_i  = (i + 1) / 8
        v_i  = flow_net(transformer_out, s_i, t_i, z_i)
        z_{i+1} = z_i + v_i * (1 / 8)
    final_latent = z_8
"""
import torch
import torch.nn as nn


class FusedFlowDecoder8(nn.Module):
    NUM_STEPS = 8

    def __init__(self, flow_net, ldim: int = 32):
        super().__init__()
        self.flow_net = flow_net
        self.ldim = ldim
        dt = 1.0 / self.NUM_STEPS
        # Constant time stamps registered as buffers so they are baked in.
        s_values = torch.tensor([[i * dt] for i in range(self.NUM_STEPS)], dtype=torch.float32)
        t_values = torch.tensor([[(i + 1) * dt] for i in range(self.NUM_STEPS)], dtype=torch.float32)
        self.register_buffer("s_values", s_values, persistent=False)
        self.register_buffer("t_values", t_values, persistent=False)
        self.dt = dt

    @classmethod
    def from_flowlm(cls, flow_lm) -> "FusedFlowDecoder8":
        return cls(flow_lm.flow_net, flow_lm.ldim)

    def forward(
        self,
        transformer_out: torch.Tensor,  # [B, 1024]
        latent_init: torch.Tensor,  # [B, 32]
    ) -> torch.Tensor:
        z = latent_init
        # Explicit unroll so torch.jit.trace produces 8 sequential ops
        # sharing the same flow_net weights.
        for i in range(self.NUM_STEPS):
            s = self.s_values[i : i + 1]  # [1, 1]
            t = self.t_values[i : i + 1]  # [1, 1]
            v = self.flow_net(transformer_out, s, t, z)
            z = z + v * self.dt
        return z
