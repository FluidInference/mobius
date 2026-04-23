"""ANE-compatible CausalConvPositionEmbedding for the Flow DiT port.

The host module in `verify/CosyVoice/cosyvoice/flow/DiT/modules.py:115`
is:

    Conv1d(dim=1024, dim, k=31, groups=16, padding=0) + Mish   # ×2
    with asymmetric causal F.pad(x, (k-1, 0, 0, 0)) on each input.

On ANE, `1 < groups < C_in` grouped Conv1d is rejected by ANEF, so the
whole module gets evicted to CPU. That pulls 20 convs, Mish chain
(`softplus → tanh → mul`), and the left-asymmetric `pad` into a CPU
island of ~77 ops in the trace-unrolled Euler loop.

Two prior attempts failed at `ANECCompile() FAILED (11)`:

  - Option A: 16 parallel `groups=1` Conv1d + Mish (math-equivalent).
    fp32 parity PASSED (cumulative MAE 2.907e-05) but ANEF couldn't
    tile the slice → 16×conv → concat pattern across 10 Euler steps.

  - Option C: same 16-way split + Mish→SiLU (not math-equivalent).
    Still failed, ruling out Mish as the cause.

Both attempts kept in git history; see /src/conv_pos_ane.py's prior
revisions.

This file implements Option L: **expand the grouped Conv1d to a single
`groups=1` dense Conv1d with a block-diagonal weight**. ANE-native
op, math-equivalent (matmul entries for off-diagonal blocks are zero
so they contribute nothing), but 16× more weights and 16× more FLOPs.

Memory cost per conv: (1024 × 1024 × 31) fp16 ≈ 64.5 MB × 2 convs
≈ 129 MB extra. FLOP cost per conv: ~16G per S=250 call × 2 × 10 steps
= 320G FLOPs. At ~15 TOPS ANE (M1 Pro) theoretical peak this is
~20 ms, real-world 2-5×, so ~100 ms added. If the 77-CPU-op island
was costing >100 ms of CPU + CPU↔ANE transfer, this is a net win.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ANECausalConvPositionEmbedding(nn.Module):
    """ANE-friendly replacement for host `CausalConvPositionEmbedding`.

    Same forward signature as host: `(B, S, C)` in, `(B, S, C)` out,
    optional boolean padding mask. Internally replaces the `groups=16`
    Conv1d with a `groups=1` Conv1d whose weight is
    `(C, C, K)` but only the 16 diagonal `(C/groups, C/groups, K)`
    blocks are nonzero.
    """

    def __init__(self, dim: int = 1024, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        assert kernel_size % 2 != 0, "kernel_size must be odd to match host"
        assert dim % groups == 0
        self.dim = dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, groups=1, padding=0, bias=True)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, groups=1, padding=0, bias=True)
        self.act1 = nn.Mish()
        self.act2 = nn.Mish()

    @classmethod
    @torch.no_grad()
    def from_host(cls, host: nn.Module) -> "ANECausalConvPositionEmbedding":
        """Build from a host `CausalConvPositionEmbedding`, porting weights
        via block-diagonal expansion."""
        host_conv1 = host.conv1[0]  # nn.Sequential[Conv1d, Mish]
        host_conv2 = host.conv2[0]
        dim = host_conv1.in_channels
        k = int(host.kernel_size)
        groups = host_conv1.groups
        port = cls(dim=dim, kernel_size=k, groups=groups)

        for dst, src in [(port.conv1, host_conv1), (port.conv2, host_conv2)]:
            cls._expand_grouped_weight(src, dst)
        return port

    @staticmethod
    @torch.no_grad()
    def _expand_grouped_weight(src: nn.Conv1d, dst: nn.Conv1d) -> None:
        """Copy host grouped Conv1d weights into dense Conv1d as block diag.

        Host weight shape: (C, C/groups, K).
        Dst  weight shape: (C, C, K) with block-diagonal structure.
        """
        C, Cg, K = src.weight.shape
        groups = src.groups
        assert C == groups * Cg
        # Start from zeros — off-diagonal blocks stay zero.
        W_full = torch.zeros(C, C, K, dtype=src.weight.dtype, device=src.weight.device)
        for g in range(groups):
            lo, hi = g * Cg, (g + 1) * Cg
            # Rows `lo:hi` receive weights from input channels `lo:hi` only.
            W_full[lo:hi, lo:hi, :] = src.weight[lo:hi, :, :]
        dst.weight.copy_(W_full)
        if src.bias is not None:
            dst.bias.copy_(src.bias)
        elif dst.bias is not None:
            dst.bias.zero_()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Match host forward exactly.
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        # (B, S, C) → (B, C, S)
        x = x.permute(0, 2, 1)

        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv1(x)
        x = self.act1(x)

        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv2(x)
        x = self.act2(x)

        out = x.permute(0, 2, 1)  # (B, S, C)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


@torch.no_grad()
def parity_check(host: nn.Module, port: nn.Module, tol: float = 1e-6) -> float:
    """Sanity: bit-exact equivalence of host grouped conv vs block-diag expansion.

    Matmul with zeros contributes nothing, so this should match to fp32
    rounding (~1e-7) — well below `tol`.
    """
    host.eval()
    port.eval()
    dim = host.conv1[0].in_channels
    x = torch.randn(1, 64, dim)  # (B, S, C)
    y_host = host(x)
    y_port = port(x)
    mae = (y_host - y_port).abs().mean().item()
    if mae > tol:
        raise RuntimeError(
            f"ANECausalConvPositionEmbedding parity failed: MAE {mae:.3e} > tol {tol:.0e}"
        )
    return mae
