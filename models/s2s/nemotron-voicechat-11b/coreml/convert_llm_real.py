#!/usr/bin/env python3
"""Convert the REAL fine-tuned VoiceChat-11B LLM to CoreML (sharded, stateful, int8).

4 shards of 14 layers (0-13 / 14-27 / 28-41 / 42-55, exact NemotronH layer
types from the checkpoint) + heads (norm_f + lm_head + function_head).
Single-token step; all recurrent state on-device (ct.StateType):
  per mamba layer:  conv_state [1,12288,3], ssm_state [1,128,80,128]
  per attn layer:   k/v window [1,8,1024,128] + shared pos counter [1]
Attention masks zero-filled window slots via the pos state (exact vs the
unbounded-attention reference for contexts < 1024; beyond that it becomes an
82s sliding window — deployment choice, Mamba carries long-range state).

Semantics verified against mlx-lm nemotron_h + base config: RMSNorm eps 1e-5,
gated Mamba norm is PER-GROUP (group_size 1280), attention has NO RoPE,
MLP is squared-ReLU, dt = softplus(dt + dt_bias), A = -exp(A_log).

Embeddings are exported to embed_tokens_fp16.npy for host-side lookup.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from safetensors import safe_open

LLM_SLICE = Path.home() / "Documents/models/voicechat-11b/components/llm.safetensors"
TOKENIZER = Path.home() / "Documents/models/nemotron-nano-9b-mlx-4bit/tokenizer.json"

D_MODEL = 4480
N_LAYERS = 56
ATTN_LAYERS = {14, 21, 30, 39}
EPS = 1e-5
N_HEADS, HEADDIM, D_STATE, N_GROUPS, D_INNER, D_CONV = 128, 80, 128, 8, 10240, 4
CONV_DIM = D_INNER + 2 * N_GROUPS * D_STATE
GROUP_SIZE = D_INNER // N_GROUPS  # gated-norm group
N_Q, N_KV, AHD = 40, 8, 128
KV_WINDOW = 1024
VOCAB = 131072
SHARDS = [(0, 14), (14, 28), (28, 42), (42, 56)]

app = typer.Typer(add_completion=False)


def rms_norm(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def layer_kind(f, i):
    if i in ATTN_LAYERS:
        return "attn"
    return "mamba" if f"stt_model.llm.layers.{i}.mixer.A_log" in f.keys() else "mlp"


class RealShard(nn.Module):
    def __init__(self, f, start: int, end: int) -> None:
        super().__init__()
        self.layer_ids = list(range(start, end))
        self.kinds = {}
        g = lambda k: nn.Parameter(f.get_tensor(k).to(torch.float32), requires_grad=False)
        for i in self.layer_ids:
            kind = layer_kind(f, i)
            self.kinds[i] = kind
            p = f"stt_model.llm.layers.{i}."
            setattr(self, f"norm_{i}", g(p + "norm.weight"))
            if kind == "mamba":
                setattr(self, f"in_proj_{i}", g(p + "mixer.in_proj.weight"))
                setattr(self, f"conv_w_{i}", nn.Parameter(f.get_tensor(p + "mixer.conv1d.weight").to(torch.float32).squeeze(1), requires_grad=False))
                setattr(self, f"conv_b_{i}", g(p + "mixer.conv1d.bias"))
                setattr(self, f"A_log_{i}", g(p + "mixer.A_log"))
                setattr(self, f"Dp_{i}", g(p + "mixer.D"))
                setattr(self, f"dt_bias_{i}", g(p + "mixer.dt_bias"))
                setattr(self, f"gnorm_{i}", g(p + "mixer.norm.weight"))
                setattr(self, f"out_proj_{i}", g(p + "mixer.out_proj.weight"))
                self.register_buffer(f"conv_state_{i}", torch.zeros(1, CONV_DIM, D_CONV - 1))
                self.register_buffer(f"ssm_state_{i}", torch.zeros(1, N_HEADS, HEADDIM, D_STATE))
            elif kind == "mlp":
                setattr(self, f"up_{i}", g(p + "mixer.up_proj.weight"))
                setattr(self, f"down_{i}", g(p + "mixer.down_proj.weight"))
            else:
                setattr(self, f"qw_{i}", g(p + "mixer.q_proj.weight"))
                setattr(self, f"kw_{i}", g(p + "mixer.k_proj.weight"))
                setattr(self, f"vw_{i}", g(p + "mixer.v_proj.weight"))
                setattr(self, f"ow_{i}", g(p + "mixer.o_proj.weight"))
                self.register_buffer(f"k_cache_{i}", torch.zeros(1, N_KV, KV_WINDOW, AHD))
                self.register_buffer(f"v_cache_{i}", torch.zeros(1, N_KV, KV_WINDOW, AHD))
        if any(k == "attn" for k in self.kinds.values()):
            self.register_buffer("pos", torch.zeros(1))

    def mamba_step(self, i, h):
        conv_state = getattr(self, f"conv_state_{i}")
        ssm_state = getattr(self, f"ssm_state_{i}")
        x = rms_norm(h, getattr(self, f"norm_{i}"))
        zxbcdt = x @ getattr(self, f"in_proj_{i}").T
        z = zxbcdt[:, :D_INNER]
        xBC = zxbcdt[:, D_INNER : D_INNER + CONV_DIM]
        dt = zxbcdt[:, D_INNER + CONV_DIM :]

        window = torch.cat([conv_state, xBC.unsqueeze(-1)], dim=-1)
        conv_state[:, :, :] = window[:, :, 1:]
        xBC = F.silu((window * getattr(self, f"conv_w_{i}")).sum(-1) + getattr(self, f"conv_b_{i}"))

        xs = xBC[:, :D_INNER].reshape(1, N_HEADS, HEADDIM)
        B = xBC[:, D_INNER : D_INNER + N_GROUPS * D_STATE].reshape(1, N_GROUPS, 1, D_STATE)
        C = xBC[:, D_INNER + N_GROUPS * D_STATE :].reshape(1, N_GROUPS, 1, D_STATE)
        B = B.repeat(1, 1, N_HEADS // N_GROUPS, 1).reshape(1, N_HEADS, D_STATE)
        C = C.repeat(1, 1, N_HEADS // N_GROUPS, 1).reshape(1, N_HEADS, D_STATE)

        dt = F.softplus(dt + getattr(self, f"dt_bias_{i}"))
        dA = torch.exp(dt * -torch.exp(getattr(self, f"A_log_{i}")))
        new_ssm = ssm_state * dA[:, :, None, None] + xs[:, :, :, None] * (dt[:, :, None] * B)[:, :, None, :]
        ssm_state[:, :, :, :] = new_ssm
        y = (new_ssm * C[:, :, None, :]).sum(-1) + getattr(self, f"Dp_{i}")[None, :, None] * xs
        # gated per-group RMSNorm (group_size 1280)
        y = (y.reshape(1, D_INNER) * F.silu(z)).reshape(1, N_GROUPS, GROUP_SIZE)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
        y = y.reshape(1, D_INNER) * getattr(self, f"gnorm_{i}")
        return h + y @ getattr(self, f"out_proj_{i}").T

    def mlp_step(self, i, h):
        x = rms_norm(h, getattr(self, f"norm_{i}"))
        return h + F.relu(x @ getattr(self, f"up_{i}").T).pow(2) @ getattr(self, f"down_{i}").T

    def attn_step(self, i, h):
        k_cache = getattr(self, f"k_cache_{i}")
        v_cache = getattr(self, f"v_cache_{i}")
        x = rms_norm(h, getattr(self, f"norm_{i}"))
        q = (x @ getattr(self, f"qw_{i}").T).reshape(1, N_Q, 1, AHD)
        k = (x @ getattr(self, f"kw_{i}").T).reshape(1, N_KV, 1, AHD)
        v = (x @ getattr(self, f"vw_{i}").T).reshape(1, N_KV, 1, AHD)
        ks = torch.cat([k_cache[:, :, 1:], k], dim=2)
        vs = torch.cat([v_cache[:, :, 1:], v], dim=2)
        k_cache[:, :, :, :] = ks
        v_cache[:, :, :, :] = vs
        valid = torch.clamp(self.pos + 1.0, max=float(KV_WINDOW))  # tokens in window incl. current
        idx = torch.arange(KV_WINDOW, dtype=torch.float32)
        neg = torch.where(idx < (KV_WINDOW - valid), torch.full_like(idx, -3e4), torch.zeros_like(idx))
        kr = ks.repeat_interleave(N_Q // N_KV, dim=1)
        vr = vs.repeat_interleave(N_Q // N_KV, dim=1)
        att = torch.softmax((q @ kr.transpose(-1, -2)) / AHD**0.5 + neg, dim=-1)
        return h + (att @ vr).reshape(1, N_Q * AHD) @ getattr(self, f"ow_{i}").T

    def forward(self, hidden):
        for i in self.layer_ids:
            kind = self.kinds[i]
            if kind == "mamba":
                hidden = self.mamba_step(i, hidden)
            elif kind == "mlp":
                hidden = self.mlp_step(i, hidden)
            else:
                hidden = self.attn_step(i, hidden)
        if hasattr(self, "pos"):
            self.pos[:] = self.pos + 1.0
        return hidden


class RealHeads(nn.Module):
    def __init__(self, f) -> None:
        super().__init__()
        self.norm_f = nn.Parameter(f.get_tensor("stt_model.llm.norm_f.weight").to(torch.float32), requires_grad=False)
        self.lm = nn.Parameter(f.get_tensor("stt_model.lm_head.weight").to(torch.float32), requires_grad=False)
        self.fn = nn.Parameter(f.get_tensor("stt_model.function_head.weight").to(torch.float32), requires_grad=False)

    def forward(self, hidden):
        x = rms_norm(hidden, self.norm_f)
        return x @ self.lm.T, x @ self.fn.T


def convert_module(mod, name: str, outdir: Path, has_states: bool) -> None:
    mod.eval()
    hidden = torch.randn(1, D_MODEL)
    traced = torch.jit.trace(mod, (hidden,), strict=False)
    states = []
    if has_states:
        for bname, buf in mod.named_buffers():
            states.append(ct.StateType(wrapped_type=ct.TensorType(shape=tuple(buf.shape)), name=bname))
    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="hidden", shape=(1, D_MODEL), dtype=np.float32)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in (["hidden_out"] if has_states else ["text_logits", "function_logits"])],
        states=states or None,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    cfg = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_block", block_size=32
        )
    )
    ml = cto.linear_quantize_weights(ml, config=cfg)
    ml.save(str(outdir / f"{name}.mlpackage"))
    typer.echo(f"saved {outdir / name}.mlpackage")


@app.command()
def convert(
    shard: int = typer.Option(-1, help="0-3 for one shard, 4 for heads+embed, -1 for all"),
    outdir: Path = typer.Option(Path("build/llm_real")),
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with safe_open(LLM_SLICE, framework="pt", device="cpu") as f:
        jobs = range(5) if shard < 0 else [shard]
        for j in jobs:
            if j < 4:
                start, end = SHARDS[j]
                typer.echo(f"building shard {j} (layers {start}-{end - 1})...")
                mod = RealShard(f, start, end)
                convert_module(mod, f"shard{j}", outdir, has_states=True)
                del mod
            else:
                typer.echo("building heads + exporting embeddings...")
                convert_module(RealHeads(f), "heads", outdir, has_states=False)
                emb = f.get_tensor("stt_model.embed_tokens.weight").to(torch.float16).numpy()
                np.save(outdir / "embed_tokens_fp16.npy", emb)
                typer.echo(f"saved {outdir}/embed_tokens_fp16.npy {emb.shape}")
            gc.collect()


@app.command()
def test(
    outdir: Path = typer.Option(Path("build/llm_real")),
    gen_tokens: int = typer.Option(24),
) -> None:
    """Prefill a real prompt token-by-token through the CoreML pipeline, compare
    per-position argmax vs the fp32 torch reference, then greedy-generate."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    prompt = "You are an AI voice assistant developed by NVIDIA. Your name is"
    ids = tok.encode(prompt).ids
    typer.echo(f"prompt: {len(ids)} tokens")

    emb = np.load(outdir / "embed_tokens_fp16.npy")
    cu = ct.ComputeUnit.CPU_AND_GPU
    shards = [ct.models.MLModel(str(outdir / f"shard{j}.mlpackage"), compute_units=cu) for j in range(4)]
    heads = ct.models.MLModel(str(outdir / "heads.mlpackage"), compute_units=cu)
    states = [s.make_state() for s in shards]

    def coreml_step(token_id: int):
        h = emb[token_id][None, :].astype(np.float32)
        for s, st in zip(shards, states):
            h = s.predict({"hidden": h}, state=st)["hidden_out"].astype(np.float32)
        out = heads.predict({"hidden": h})
        return out["text_logits"]

    # torch reference (fp32): one shard in RAM at a time, all timesteps through it
    with safe_open(LLM_SLICE, framework="pt", device="cpu") as f:
        hs = torch.from_numpy(emb[ids].astype(np.float32))
        with torch.no_grad():
            for s, e in SHARDS:
                mod = RealShard(f, s, e)
                hs = torch.cat([mod(hs[t : t + 1]) for t in range(hs.shape[0])])
                del mod
                gc.collect()
            headm = RealHeads(f)
            ref_argmax = [int(headm(hs[t : t + 1])[0].argmax()) for t in range(hs.shape[0])]
            del headm
            gc.collect()

    cm_argmax = []
    for t in ids:
        cm_argmax.append(int(np.argmax(coreml_step(t))))
    agree = np.mean([a == b for a, b in zip(ref_argmax, cm_argmax)])
    typer.echo(f"prefill argmax agreement (coreml int8 vs torch fp32): {agree:.3f}")
    if agree < 1.0:  # int8 is exactly lossless — any flip is a regression
        typer.echo("PARITY FAIL: prefill argmax mismatch")
        raise typer.Exit(1)

    cur = cm_argmax[-1]
    gen = [cur]
    for _ in range(gen_tokens - 1):
        cur = int(np.argmax(coreml_step(cur)))
        gen.append(cur)
    typer.echo(f"greedy continuation: {tok.decode(gen)!r}")


if __name__ == "__main__":
    app()
