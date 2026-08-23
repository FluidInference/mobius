#!/usr/bin/env python3
"""Measure int4 quantization quality on the REAL fine-tuned VoiceChat-11B LLM.

Dual-track forward through all 56 NemotronH layers, streamed one layer at a
time from components/llm.safetensors (bf16 -> fp32; peak RSS ~ 1 layer):
  track A: fp32 weights (baseline)
  track B: big matmuls quantized int4 linear_symmetric per-block-32 along the
           input axis (same scheme as coremltools OpLinearQuantizerConfig),
           then dequantized. Embeddings stay fp (matches deployment plan).

Both tracks share the same implementation, so any divergence from NVIDIA's
exact forward (e.g. positional handling in the 4 attention layers) cancels
out — the fp32-vs-int4 delta isolates quantization error alone.

Metrics: per-layer hidden cosine, final-logits top-1/top-5 agreement across
positions, KL(softmax_fp32 || softmax_int4), same for the function head.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import typer
from safetensors import safe_open

COMPONENTS = Path.home() / "Documents/models/voicechat-11b/components/llm.safetensors"
TOKENIZER = Path.home() / "Documents/models/nemotron-nano-9b-mlx-4bit/tokenizer.json"

D_MODEL = 4480
N_LAYERS = 56
ATTN_LAYERS = {14, 21, 30, 39}
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 40, 8, 128
N_HEADS, HEADDIM, D_STATE, N_GROUPS, D_INNER, D_CONV = 128, 80, 128, 8, 10240, 4

PROMPTS = [
    "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
    "Your job is to be helpful and harmless and have engaging conversations in English.",
    "Hello, do you know what color the sky is? I was wondering about that earlier today.",
    "Please call the weather tool for San Francisco and tell me if I need an umbrella.",
]

app = typer.Typer(add_completion=False)


QUANT_NBITS = 4
QUANT_BLOCK = 32
HEAD_NBITS = None  # None -> same as body


def quant_int4_pb32(w: torch.Tensor, nbits: int | None = None, block: int | None = None) -> torch.Tensor:
    """linear_symmetric intN, per-block along the input (last) axis."""
    nbits = nbits or QUANT_NBITS
    block = block or QUANT_BLOCK
    if nbits >= 16:  # fp16 cast — noise-floor reference, no int quant
        return w.half().float()
    qmax = 2 ** (nbits - 1) - 1
    out, inp = w.shape
    pad = (-inp) % block
    if pad:
        w = F.pad(w, (0, pad))
    blocks = w.reshape(out, -1, block)
    scale = blocks.abs().amax(-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(blocks / scale), -qmax - 1, qmax)
    dq = (q * scale).reshape(out, -1)[:, :inp]
    return dq


def rms_norm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w


def gated_group_norm(y, z, w):
    """Mamba gated norm: per-group RMSNorm (8 groups of 1280), eps 1e-5 —
    matches convert_llm_real.py / mlx-lm nemotron_h, NOT a full-width norm."""
    g = (y * F.silu(z)).reshape(*y.shape[:-1], N_GROUPS, D_INNER // N_GROUPS)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + 1e-5)
    return g.reshape(*y.shape) * w


class LayerRunner:
    """Runs one layer on both tracks; weights loaded lazily then freed."""

    def __init__(self, f):
        self.f = f

    def get(self, key, quant=False):
        t = self.f.get_tensor(key).to(torch.float32)
        if quant and t.dim() == 2:
            t = quant_int4_pb32(t)
        return t

    def mamba(self, i, h, quant):
        p = f"stt_model.llm.layers.{i}.mixer."
        norm_w = self.get(f"stt_model.llm.layers.{i}.norm.weight")
        in_proj = self.get(p + "in_proj.weight", quant)
        conv_w = self.get(p + "conv1d.weight").squeeze(1)  # [12288, 4]
        conv_b = self.get(p + "conv1d.bias")
        A_log, D = self.get(p + "A_log"), self.get(p + "D")
        dt_bias = self.get(p + "dt_bias")
        gnorm = self.get(p + "norm.weight")
        out_proj = self.get(p + "out_proj.weight", quant)

        T = h.shape[0]
        x = rms_norm(h, norm_w)
        zxbcdt = x @ in_proj.T  # [T, 22656]
        CONV_DIM = D_INNER + 2 * N_GROUPS * D_STATE
        z = zxbcdt[:, :D_INNER]
        xBC = zxbcdt[:, D_INNER : D_INNER + CONV_DIM]
        dt = zxbcdt[:, D_INNER + CONV_DIM :]

        # causal depthwise conv over time (kernel 4)
        xBC_pad = F.pad(xBC.T.unsqueeze(0), (D_CONV - 1, 0))  # [1, CONV_DIM, T+3]
        xBC = F.conv1d(xBC_pad, conv_w.unsqueeze(1), bias=conv_b, groups=CONV_DIM)[0].T
        xBC = F.silu(xBC)

        xs = xBC[:, :D_INNER].reshape(T, N_HEADS, HEADDIM)
        B = xBC[:, D_INNER : D_INNER + N_GROUPS * D_STATE].reshape(T, N_GROUPS, D_STATE)
        C = xBC[:, D_INNER + N_GROUPS * D_STATE :].reshape(T, N_GROUPS, D_STATE)
        B = B.repeat_interleave(N_HEADS // N_GROUPS, dim=1)  # [T, 128, 128]
        C = C.repeat_interleave(N_HEADS // N_GROUPS, dim=1)

        dt = F.softplus(dt + dt_bias)  # [T, 128]
        dA = torch.exp(dt * -torch.exp(A_log))  # [T, 128]

        state = torch.zeros(N_HEADS, HEADDIM, D_STATE)
        ys = []
        for t in range(T):
            state = state * dA[t][:, None, None] + xs[t][:, :, None] * (dt[t][:, None] * B[t])[:, None, :]
            ys.append((state * C[t][:, None, :]).sum(-1) + D[:, None] * xs[t])
        y = torch.stack(ys).reshape(T, D_INNER)
        y = gated_group_norm(y, z, gnorm)
        return h + y @ out_proj.T

    def mlp(self, i, h, quant):
        p = f"stt_model.llm.layers.{i}.mixer."
        norm_w = self.get(f"stt_model.llm.layers.{i}.norm.weight")
        up = self.get(p + "up_proj.weight", quant)
        down = self.get(p + "down_proj.weight", quant)
        x = rms_norm(h, norm_w)
        return h + F.relu(x @ up.T).pow(2) @ down.T

    def attn(self, i, h, quant):
        p = f"stt_model.llm.layers.{i}.mixer."
        norm_w = self.get(f"stt_model.llm.layers.{i}.norm.weight")
        qw = self.get(p + "q_proj.weight", quant)
        kw = self.get(p + "k_proj.weight", quant)
        vw = self.get(p + "v_proj.weight", quant)
        ow = self.get(p + "o_proj.weight", quant)
        T = h.shape[0]
        x = rms_norm(h, norm_w)
        q = (x @ qw.T).reshape(T, N_Q_HEADS, HEAD_DIM).transpose(0, 1)
        k = (x @ kw.T).reshape(T, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        v = (x @ vw.T).reshape(T, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        k = k.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=0)
        v = v.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=0)
        att = (q @ k.transpose(-1, -2)) / HEAD_DIM**0.5
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        att = torch.softmax(att + mask, dim=-1)
        out = (att @ v).transpose(0, 1).reshape(T, N_Q_HEADS * HEAD_DIM)
        return h + out @ ow.T


@app.command()
def main(nbits: int = typer.Option(4), block: int = typer.Option(32), head_nbits: int = typer.Option(0)) -> None:
    global QUANT_NBITS, QUANT_BLOCK, HEAD_NBITS
    QUANT_NBITS, QUANT_BLOCK = nbits, block
    HEAD_NBITS = head_nbits or nbits
    typer.echo(f"scheme: body int{nbits} pb{block}, heads int{HEAD_NBITS}")
    torch.set_grad_enabled(False)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))

    with safe_open(COMPONENTS, framework="pt", device="cpu") as f:
        runner = LayerRunner(f)
        embed = f.get_tensor("stt_model.embed_tokens.weight").to(torch.float32)

        all_top1, all_top5, all_kl, all_fn_top1 = [], [], [], []
        for prompt in PROMPTS:
            ids = tok.encode(prompt).ids[:64]
            h_fp = embed[ids]  # embeddings stay fp in both tracks
            h_q = h_fp.clone()

            cos_by_layer = []
            for i in range(N_LAYERS):
                kind = (
                    "attn" if i in ATTN_LAYERS
                    else "mamba" if f"stt_model.llm.layers.{i}.mixer.A_log" in f.keys()
                    else "mlp"
                )
                fn = getattr(runner, kind)
                h_fp = fn(i, h_fp, quant=False)
                h_q = fn(i, h_q, quant=True)
                cos = F.cosine_similarity(h_fp, h_q, dim=-1).mean().item()
                cos_by_layer.append(cos)

            norm_f = runner.get("stt_model.llm.norm_f.weight")
            lm_fp32 = runner.get("stt_model.lm_head.weight")
            lm_int4 = quant_int4_pb32(lm_fp32, nbits=HEAD_NBITS)
            fn_fp32 = runner.get("stt_model.function_head.weight")
            fn_int4 = quant_int4_pb32(fn_fp32, nbits=HEAD_NBITS)

            xf, xq = rms_norm(h_fp, norm_f), rms_norm(h_q, norm_f)
            logits_fp = xf @ lm_fp32.T
            logits_q = xq @ lm_int4.T
            fn_fp = xf @ fn_fp32.T
            fn_q = xq @ fn_int4.T

            top1 = (logits_fp.argmax(-1) == logits_q.argmax(-1)).float().mean().item()
            top5_fp = logits_fp.topk(5, dim=-1).indices
            top5 = (logits_q.argmax(-1, keepdim=True) == top5_fp).any(-1).float().mean().item()
            kl = F.kl_div(
                F.log_softmax(logits_q, -1), F.log_softmax(logits_fp, -1),
                log_target=True, reduction="batchmean",
            ).item()
            fn_top1 = (fn_fp.argmax(-1) == fn_q.argmax(-1)).float().mean().item()

            all_top1.append(top1)
            all_top5.append(top5)
            all_kl.append(kl)
            all_fn_top1.append(fn_top1)
            typer.echo(
                f"prompt[{len(ids)} tok]: top1 {top1:.3f}  top5 {top5:.3f}  KL {kl:.4f}  "
                f"fn_top1 {fn_top1:.3f}  final cos {cos_by_layer[-1]:.5f}  min cos {min(cos_by_layer):.5f}"
            )

        typer.echo(
            f"\nOVERALL int4-pb32 vs fp32 (real fine-tuned weights): "
            f"top1 {np.mean(all_top1):.3f}  top5 {np.mean(all_top5):.3f}  "
            f"KL {np.mean(all_kl):.4f}  function-head top1 {np.mean(all_fn_top1):.3f}"
        )


if __name__ == "__main__":
    app()
