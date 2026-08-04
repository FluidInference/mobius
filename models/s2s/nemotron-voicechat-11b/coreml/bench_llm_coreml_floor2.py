#!/usr/bin/env python3
"""Optimized CoreML 9B decode-step floor: stateful shards.

v1 (bench_llm_coreml_floor.py) measured 50.6 ms chained across 8 stateless
model calls — with ~280 MB/step of mamba/KV state crossing the host boundary.
This version:
  1. Keeps ALL recurrent state (mamba conv+ssm, attention KV window) on-device
     via iOS18 stateful prediction (ct.StateType, in-place buffer mutation).
  2. Interleaves layers into realistic shards: 7x Mamba2 + 6x MLP + 1x attention
     (real model = 27/25/4 over 56 layers; 4 shards approximate it exactly).
  3. Chains 4 shard instances + heads: only hidden [1,4480] (18 KB) crosses
     the host per call.

Same exact VoiceChat-11B geometry, random weights, int4 per-block.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer

D_MODEL = 4480
D_INNER = 10240
N_HEADS = 128
HEADDIM = 80
D_STATE = 128
N_GROUPS = 8
D_CONV = 4
CONV_DIM = D_INNER + 2 * N_GROUPS * D_STATE
IN_PROJ_OUT = 2 * D_INNER + 2 * N_GROUPS * D_STATE + N_HEADS
MLP_INTER = 15680
ATTN_Q = 5120
ATTN_KV_HEADS = 8
ATTN_HEADDIM = 128
KV_WINDOW = 1024
VOCAB = 131072

N_MAMBA, N_MLP, N_ATTN = 7, 6, 1  # per shard; x4 shards = 28/24/4 (~real 27/25/4)

app = typer.Typer(add_completion=False)


def rms_norm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w


class Shard(nn.Module):
    """7x Mamba2 + 6x MLP + 1x attention, all recurrent state in buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.m_norm = nn.ParameterList()
        self.m_in = nn.ModuleList()
        self.m_conv_w = nn.ParameterList()
        self.m_conv_b = nn.ParameterList()
        self.m_Alog = nn.ParameterList()
        self.m_D = nn.ParameterList()
        self.m_dtb = nn.ParameterList()
        self.m_gnorm = nn.ParameterList()
        self.m_out = nn.ModuleList()
        for i in range(N_MAMBA):
            self.m_norm.append(nn.Parameter(torch.ones(D_MODEL)))
            self.m_in.append(nn.Linear(D_MODEL, IN_PROJ_OUT, bias=False))
            self.m_conv_w.append(nn.Parameter(torch.randn(CONV_DIM, D_CONV) * 0.1))
            self.m_conv_b.append(nn.Parameter(torch.zeros(CONV_DIM)))
            self.m_Alog.append(nn.Parameter(torch.zeros(N_HEADS)))
            self.m_D.append(nn.Parameter(torch.ones(N_HEADS)))
            self.m_dtb.append(nn.Parameter(torch.zeros(N_HEADS)))
            self.m_gnorm.append(nn.Parameter(torch.ones(D_INNER)))
            self.m_out.append(nn.Linear(D_INNER, D_MODEL, bias=False))
            self.register_buffer(f"conv_state_{i}", torch.zeros(1, CONV_DIM, D_CONV - 1))
            self.register_buffer(f"ssm_state_{i}", torch.zeros(1, N_HEADS, HEADDIM, D_STATE))

        self.f_norm = nn.ParameterList()
        self.f_up = nn.ModuleList()
        self.f_down = nn.ModuleList()
        for _ in range(N_MLP):
            self.f_norm.append(nn.Parameter(torch.ones(D_MODEL)))
            self.f_up.append(nn.Linear(D_MODEL, MLP_INTER, bias=False))
            self.f_down.append(nn.Linear(MLP_INTER, D_MODEL, bias=False))

        self.a_norm = nn.Parameter(torch.ones(D_MODEL))
        self.a_q = nn.Linear(D_MODEL, ATTN_Q, bias=False)
        self.a_k = nn.Linear(D_MODEL, ATTN_KV_HEADS * ATTN_HEADDIM, bias=False)
        self.a_v = nn.Linear(D_MODEL, ATTN_KV_HEADS * ATTN_HEADDIM, bias=False)
        self.a_o = nn.Linear(ATTN_Q, D_MODEL, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, ATTN_KV_HEADS, KV_WINDOW, ATTN_HEADDIM))
        self.register_buffer("v_cache", torch.zeros(1, ATTN_KV_HEADS, KV_WINDOW, ATTN_HEADDIM))

    def mamba_step(self, i, hidden):
        conv_state = getattr(self, f"conv_state_{i}")
        ssm_state = getattr(self, f"ssm_state_{i}")
        x = rms_norm(hidden, self.m_norm[i])
        zxbcdt = self.m_in[i](x)
        z = zxbcdt[:, :D_INNER]
        xBC = zxbcdt[:, D_INNER : D_INNER + CONV_DIM]
        dt = zxbcdt[:, D_INNER + CONV_DIM :]

        window = torch.cat([conv_state, xBC.unsqueeze(-1)], dim=-1)
        conv_state[:, :, :] = window[:, :, 1:]
        xBC = F.silu((window * self.m_conv_w[i]).sum(-1) + self.m_conv_b[i])

        xs = xBC[:, :D_INNER].reshape(1, N_HEADS, HEADDIM)
        B = xBC[:, D_INNER : D_INNER + N_GROUPS * D_STATE].reshape(1, N_GROUPS, 1, D_STATE)
        C = xBC[:, D_INNER + N_GROUPS * D_STATE :].reshape(1, N_GROUPS, 1, D_STATE)
        B = B.repeat(1, 1, N_HEADS // N_GROUPS, 1).reshape(1, N_HEADS, D_STATE)
        C = C.repeat(1, 1, N_HEADS // N_GROUPS, 1).reshape(1, N_HEADS, D_STATE)

        dt = F.softplus(dt + self.m_dtb[i])
        dA = torch.exp(dt * -torch.exp(self.m_Alog[i]))
        new_ssm = ssm_state * dA[:, :, None, None] + xs[:, :, :, None] * (dt[:, :, None] * B)[:, :, None, :]
        ssm_state[:, :, :, :] = new_ssm
        y = (new_ssm * C[:, :, None, :]).sum(-1) + self.m_D[i][None, :, None] * xs
        y = rms_norm(y.reshape(1, D_INNER) * F.silu(z), self.m_gnorm[i])
        return hidden + self.m_out[i](y)

    def mlp_step(self, i, hidden):
        x = rms_norm(hidden, self.f_norm[i])
        return hidden + self.f_down[i](F.relu(self.f_up[i](x)).pow(2))

    def attn_step(self, hidden):
        x = rms_norm(hidden, self.a_norm)
        q = self.a_q(x).reshape(1, 40, 1, ATTN_HEADDIM)
        k = self.a_k(x).reshape(1, ATTN_KV_HEADS, 1, ATTN_HEADDIM)
        v = self.a_v(x).reshape(1, ATTN_KV_HEADS, 1, ATTN_HEADDIM)
        ks = torch.cat([self.k_cache[:, :, 1:], k], dim=2)
        vs = torch.cat([self.v_cache[:, :, 1:], v], dim=2)
        self.k_cache[:, :, :, :] = ks
        self.v_cache[:, :, :, :] = vs
        kr = ks.repeat_interleave(5, dim=1)
        vr = vs.repeat_interleave(5, dim=1)
        att = torch.softmax((q @ kr.transpose(-1, -2)) / (ATTN_HEADDIM**0.5), dim=-1)
        return hidden + self.a_o((att @ vr).reshape(1, ATTN_Q))

    def forward(self, hidden):
        # interleave roughly like the real model: M M MLP M MLP ... A
        order = ["m0", "f0", "m1", "f1", "m2", "f2", "m3", "a", "f3", "m4", "f4", "m5", "f5", "m6"]
        for tag in order:
            if tag[0] == "m":
                hidden = self.mamba_step(int(tag[1]), hidden)
            elif tag[0] == "f":
                hidden = self.mlp_step(int(tag[1]), hidden)
            else:
                hidden = self.attn_step(hidden)
        return hidden


class Heads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(D_MODEL))
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.function_head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, hidden):
        x = rms_norm(hidden, self.norm_w)
        return self.lm_head(x), self.function_head(x)


def quantize_and_save(ml, name: str, outdir: Path) -> None:
    cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32
        )
    )
    ml = cto.linear_quantize_weights(ml, config=cfg)
    ml.save(str(outdir / f"{name}.mlpackage"))
    typer.echo(f"saved {outdir}/{name}.mlpackage")


@app.command()
def build(outdir: Path = typer.Option(Path("build/llm_floor2"))) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    shard = Shard().eval()
    hidden = torch.randn(1, D_MODEL)
    traced = torch.jit.trace(shard, (hidden,), strict=False)

    states = []
    for i in range(N_MAMBA):
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, CONV_DIM, D_CONV - 1)), name=f"conv_state_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, N_HEADS, HEADDIM, D_STATE)), name=f"ssm_state_{i}"))
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, ATTN_KV_HEADS, KV_WINDOW, ATTN_HEADDIM)), name="k_cache"))
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, ATTN_KV_HEADS, KV_WINDOW, ATTN_HEADDIM)), name="v_cache"))

    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="hidden", shape=(1, D_MODEL), dtype=np.float32)],
        outputs=[ct.TensorType(name="hidden_out", dtype=np.float32)],
        states=states,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    quantize_and_save(ml, "shard14_stateful", outdir)

    heads = Heads().eval()
    traced_h = torch.jit.trace(heads, (hidden,), strict=False)
    mlh = ct.convert(
        traced_h,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="hidden", shape=(1, D_MODEL), dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    quantize_and_save(mlh, "heads", outdir)


@app.command()
def bench(outdir: Path = typer.Option(Path("build/llm_floor2")), steps: int = typer.Option(50)) -> None:
    cu = ct.ComputeUnit.CPU_AND_GPU
    shards = [ct.models.MLModel(str(outdir / "shard14_stateful.mlpackage"), compute_units=cu) for _ in range(4)]
    heads = ct.models.MLModel(str(outdir / "heads.mlpackage"), compute_units=cu)
    states = [s.make_state() for s in shards]
    hidden = np.random.randn(1, D_MODEL).astype(np.float32)

    def step():
        h = hidden
        for s, st in zip(shards, states):
            h = s.predict({"hidden": h}, state=st)["hidden_out"].astype(np.float32)
        heads.predict({"hidden": h})

    for _ in range(5):
        step()
    t0 = time.time()
    for _ in range(steps):
        step()
    dt = (time.time() - t0) / steps * 1000
    typer.echo(f"stateful sharded 9B step (4 shards + heads = 5 calls, states on-device): {dt:.1f} ms")


if __name__ == "__main__":
    app()
