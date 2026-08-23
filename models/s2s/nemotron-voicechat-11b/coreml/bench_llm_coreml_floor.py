#!/usr/bin/env python3
"""CoreML decode-step floor benchmark for the NemotronH-9B backbone.

Question: can a CoreML int4 single-token step of the 9B hybrid fit the 80 ms
full-duplex frame budget? (The MLX verdict rests on a riva-translate-4b
extrapolation; this measures the 9B geometry directly.)

Builds single-step graphs with the EXACT VoiceChat-11B dims (from the
checkpoint header) but random weights — kernel latency depends on shapes and
dtypes, not values:
  mamba_stack: 9x Mamba2 mixer step (of 27)     in_proj 4480->22656, out 10240->4480
  mlp_stack:   9x MLP step (of 25)              4480->15680 relu^2 ->4480
  attn_stack:  4x GQA attention step (of 4)     40q/8kv x 128, 1024-frame KV window
  heads:       norm_f + lm_head + function_head 2x 4480->131072

Full step ~= 3*mamba_stack + (25/9)*mlp_stack + attn_stack + heads.
Each stack is converted to mlprogram and int4 per-block quantized
(linear_symmetric, block 32 — NOT palettized; pal4 was pathological in the
riva-translate trial).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

# Exact VoiceChat-11B / NemotronH-9B dims (verified from safetensors header)
D_MODEL = 4480
D_INNER = 10240
N_HEADS = 128
HEADDIM = 80
D_STATE = 128
N_GROUPS = 8
D_CONV = 4
CONV_DIM = D_INNER + 2 * N_GROUPS * D_STATE  # 12288
IN_PROJ_OUT = 2 * D_INNER + 2 * N_GROUPS * D_STATE + N_HEADS  # 22656
MLP_INTER = 15680
ATTN_Q = 5120  # 40 heads x 128
ATTN_KV = 1024  # 8 heads x 128
ATTN_HEADDIM = 128
KV_WINDOW = 1024
VOCAB = 131072

app = typer.Typer(add_completion=False)


def rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w


class Mamba2Step(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_w = nn.Parameter(torch.randn(D_MODEL) * 0.02 + 1)
        self.in_proj = nn.Linear(D_MODEL, IN_PROJ_OUT, bias=False)
        self.conv_w = nn.Parameter(torch.randn(CONV_DIM, D_CONV) * 0.1)
        self.conv_b = nn.Parameter(torch.zeros(CONV_DIM))
        self.A_log = nn.Parameter(torch.zeros(N_HEADS))
        self.D = nn.Parameter(torch.ones(N_HEADS))
        self.dt_bias = nn.Parameter(torch.zeros(N_HEADS))
        self.gate_norm_w = nn.Parameter(torch.ones(D_INNER))
        self.out_proj = nn.Linear(D_INNER, D_MODEL, bias=False)

    def forward(self, hidden, conv_state, ssm_state):
        # hidden [1, D_MODEL]; conv_state [1, CONV_DIM, D_CONV-1]; ssm_state [1, N_HEADS, HEADDIM, D_STATE]
        x = rms_norm(hidden, self.norm_w)
        zxbcdt = self.in_proj(x)  # [1, 22656]
        z = zxbcdt[:, :D_INNER]
        xBC = zxbcdt[:, D_INNER : D_INNER + CONV_DIM]
        dt = zxbcdt[:, D_INNER + CONV_DIM :]  # [1, 128]

        window = torch.cat([conv_state, xBC.unsqueeze(-1)], dim=-1)  # [1, CONV_DIM, 4]
        new_conv_state = window[:, :, 1:]
        xBC = F.silu((window * self.conv_w).sum(-1) + self.conv_b)  # [1, CONV_DIM]

        xs = xBC[:, :D_INNER].reshape(1, N_HEADS, HEADDIM)
        B = xBC[:, D_INNER : D_INNER + N_GROUPS * D_STATE].reshape(1, N_GROUPS, 1, D_STATE)
        C = xBC[:, D_INNER + N_GROUPS * D_STATE :].reshape(1, N_GROUPS, 1, D_STATE)
        B = B.repeat(1, 1, N_HEADS // N_GROUPS, 1).reshape(1, N_HEADS, D_STATE)
        C = C.repeat(1, 1, N_HEADS // N_GROUPS, 1).reshape(1, N_HEADS, D_STATE)

        dt = F.softplus(dt + self.dt_bias)  # [1, 128]
        dA = torch.exp(dt * -torch.exp(self.A_log))  # [1, 128]
        new_ssm = ssm_state * dA[:, :, None, None] + xs[:, :, :, None] * (dt[:, :, None] * B)[:, :, None, :]
        y = (new_ssm * C[:, :, None, :]).sum(-1) + self.D[None, :, None] * xs  # [1, 128, 80]
        y = y.reshape(1, D_INNER)
        y = rms_norm(y * F.silu(z), self.gate_norm_w)
        return hidden + self.out_proj(y), new_conv_state, new_ssm


class MLPStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(D_MODEL))
        self.up = nn.Linear(D_MODEL, MLP_INTER, bias=False)
        self.down = nn.Linear(MLP_INTER, D_MODEL, bias=False)

    def forward(self, hidden):
        x = rms_norm(hidden, self.norm_w)
        return hidden + self.down(F.relu(self.up(x)).pow(2))


class AttnStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(D_MODEL))
        self.q = nn.Linear(D_MODEL, ATTN_Q, bias=False)
        self.k = nn.Linear(D_MODEL, ATTN_KV, bias=False)
        self.v = nn.Linear(D_MODEL, ATTN_KV, bias=False)
        self.o = nn.Linear(ATTN_Q, D_MODEL, bias=False)

    def forward(self, hidden, k_cache, v_cache):
        # k_cache/v_cache [1, 8, KV_WINDOW-1, 128] -> returns updated window
        x = rms_norm(hidden, self.norm_w)
        q = self.q(x).reshape(1, 40, 1, ATTN_HEADDIM)
        k = self.k(x).reshape(1, 8, 1, ATTN_HEADDIM)
        v = self.v(x).reshape(1, 8, 1, ATTN_HEADDIM)
        ks = torch.cat([k_cache, k], dim=2)  # [1, 8, W, 128]
        vs = torch.cat([v_cache, v], dim=2)
        kr = ks.repeat_interleave(5, dim=1)  # GQA 40/8
        vr = vs.repeat_interleave(5, dim=1)
        att = torch.softmax((q @ kr.transpose(-1, -2)) / (ATTN_HEADDIM**0.5), dim=-1)
        out = (att @ vr).reshape(1, ATTN_Q)
        return hidden + self.o(out), ks[:, :, 1:], vs[:, :, 1:]


class Heads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(D_MODEL))
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.function_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, hidden):
        x = rms_norm(hidden, self.norm_w)
        return self.lm_head(x), self.function_head(x)


class Stack(nn.Module):
    def __init__(self, kind: str, n: int) -> None:
        super().__init__()
        cls = {"mamba": Mamba2Step, "mlp": MLPStep, "attn": AttnStep}[kind]
        self.kind = kind
        self.blocks = nn.ModuleList([cls() for _ in range(n)])

    def forward(self, hidden, *states):
        outs = []
        if self.kind == "mamba":
            for i, b in enumerate(self.blocks):
                hidden, cs, ss = b(hidden, states[2 * i], states[2 * i + 1])
                outs += [cs, ss]
        elif self.kind == "attn":
            for i, b in enumerate(self.blocks):
                hidden, kc, vc = b(hidden, states[2 * i], states[2 * i + 1])
                outs += [kc, vc]
        else:
            for b in self.blocks:
                hidden = b(hidden)
        return (hidden, *outs)


def convert_int4(model: nn.Module, example: tuple, name: str, outdir: Path) -> Path:
    model.eval()
    traced = torch.jit.trace(model, example, strict=False)
    inputs = [ct.TensorType(name=f"in_{i}", shape=tuple(t.shape), dtype=np.float32) for i, t in enumerate(example)]
    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32
        )
    )
    ml = cto.linear_quantize_weights(ml, config=cfg)
    pkg = outdir / f"{name}.mlpackage"
    ml.save(str(pkg))
    reloaded = ct.models.MLModel(str(pkg))
    dest = outdir / f"{name}.mlmodelc"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(reloaded.get_compiled_model_path(), dest)
    del reloaded
    typer.echo(f"built {dest}")
    return dest


@app.command()
def build(
    outdir: Path = typer.Option(Path("build/llm_floor")),
    which: str = typer.Option("all", help="mamba|mlp|attn|heads|all"),
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    hidden = torch.randn(1, D_MODEL)

    if which in ("mamba", "all"):
        n = 9
        states = []
        for _ in range(n):
            states += [torch.randn(1, CONV_DIM, D_CONV - 1), torch.randn(1, N_HEADS, HEADDIM, D_STATE) * 0.1]
        convert_int4(Stack("mamba", n), (hidden, *states), "mamba_stack9", outdir)
    if which in ("mlp", "all"):
        convert_int4(Stack("mlp", 9), (hidden,), "mlp_stack9", outdir)
    if which in ("attn", "all"):
        n = 4
        states = []
        for _ in range(n):
            states += [
                torch.randn(1, 8, KV_WINDOW - 1, ATTN_HEADDIM) * 0.1,
                torch.randn(1, 8, KV_WINDOW - 1, ATTN_HEADDIM) * 0.1,
            ]
        convert_int4(Stack("attn", n), (hidden, *states), "attn_stack4", outdir)
    if which in ("heads", "all"):
        convert_int4(Heads(), (hidden,), "heads", outdir)


if __name__ == "__main__":
    app()
