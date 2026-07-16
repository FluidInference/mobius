"""ANE decode-unblock experiment: current decoder_step vs a §6.3 host-owned-cache rewrite.

Tests whether the documented decoder_step ANE failure (ANECCompile on real incrementing
position, SWIFT_PORT_FINDINGS.md) is escaped by the Surgical Inference §6.3 boundary:
the graph does NO in-graph cache mutation and NO position-driven mask compares — it reads
the past cache, concatenates the current K/V, attends against a host-supplied additive
mask, and outputs ONLY the current-step K/V slice [B,1,H,D]. The host owns the append and
the mask.

ANE admission is structural (op/shape/state pattern), not weight-dependent, so this uses
random weights at Magpie's real dims — no NeMo model / gated download needed. Both variants
share dims so the old-fails / new-passes comparison is clean.

Builds:
  build_ane/old_decoder_step.mlpackage  — current design (in-graph blend + pos compares)
  build_ane/new_decoder_step.mlpackage  — §6.3 host-owned cache (current-slice outputs)
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

# Import the PRODUCTION current graph (no NeMo at module import time).
HERE = Path(__file__).resolve()
COREML_DIR = HERE.parents[2]  # models/tts/magpie/coreml
sys.path.insert(0, str(COREML_DIR))
from traceable.traceable_decoder_step import TraceableDecoderStep, MASK_NEG  # noqa: E402

# Magpie mrt-357m decoder dims (confirmed: 12 layers, d_model 768, H12 x D64, cache 1x512x12x64).
N_LAYERS = 12
D_MODEL = 768
SA_HEADS = 12
D_HEAD = D_MODEL // SA_HEADS  # 64
D_FFN = 3072
XA_HEADS = 12
XA_D_MEM = 768
MAX_SEQ = 512
T_ENC = 256
NUM_CODEBOOKS = 8
NUM_TOKENS = 2024
OUT_DIM = NUM_CODEBOOKS * NUM_TOKENS  # 16192


# ---------------------------------------------------------------------------
# NEW: §6.3 host-owned-cache single-step decoder.
# ---------------------------------------------------------------------------
class HostCacheSelfAttention(nn.Module):
    """Causal self-attn that reads past cache, concatenates current K/V, and returns
    ONLY the current-step K/V slice. No in-graph cache write, no position compares."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.d = d_model // n_heads
        self.scale = self.d ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * self.d, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d, d_model, bias=False)

    def forward(self, x, past_k, past_v, attn_mask):
        """x: [B,1,d_model]; past_k/past_v: [B,max_seq,H,D] (host-owned, current NOT written);
        attn_mask: [B,1,1,max_seq+1] additive (0 keep, MASK_NEG drop), last col = current."""
        B = x.shape[0]
        qkv = self.qkv_proj(x).view(B, 1, 3, self.h, self.d)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each [B,1,H,D]

        k_all = torch.cat([past_k, k], dim=1)  # [B, max_seq+1, H, D] — current at last index
        v_all = torch.cat([past_v, v], dim=1)

        q4 = q.transpose(1, 2)             # [B,H,1,D]
        k4 = k_all.permute(0, 2, 1, 3)     # [B,H,max_seq+1,D]
        v4 = v_all.permute(0, 2, 1, 3)

        attn = torch.matmul(q4, k4.transpose(-2, -1)) * self.scale + attn_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v4).transpose(1, 2).reshape(B, 1, -1)
        out = self.o_proj(out)
        return out, k, v  # current-step slice only


class HostCacheCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_mem):
        super().__init__()
        self.h = n_heads
        self.d = d_model // n_heads
        self.scale = self.d ** -0.5
        self.q_proj = nn.Linear(d_model, n_heads * self.d, bias=False)
        self.kv_proj = nn.Linear(d_mem, 2 * n_heads * self.d, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d, d_model, bias=False)

    def forward(self, x, memory, mem_mask_add):
        B, Tq, _ = x.shape
        Tm = memory.shape[1]
        q = self.q_proj(x).view(B, Tq, self.h, self.d).transpose(1, 2)
        kv = self.kv_proj(memory).view(B, Tm, 2, self.h, self.d)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale + mem_mask_add
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Tq, -1)
        return self.o_proj(out)


class HostCacheFFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ffn, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ffn, d_model, 1, bias=False)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv2(self.act(self.conv1(x)))
        return x.transpose(1, 2)


class HostCacheLayer(nn.Module):
    def __init__(self, d_model, d_ffn, sa_heads, xa_heads, xa_d_mem):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model, bias=False)
        self.self_attn = HostCacheSelfAttention(d_model, sa_heads)
        self.norm_xa_q = nn.LayerNorm(d_model, bias=False)
        self.norm_xa_m = nn.LayerNorm(xa_d_mem, bias=False)
        self.cross_attn = HostCacheCrossAttention(d_model, xa_heads, xa_d_mem)
        self.norm_ff = nn.LayerNorm(d_model, bias=False)
        self.ffn = HostCacheFFN(d_model, d_ffn)

    def forward(self, x, past_k, past_v, attn_mask, memory, mem_mask_add):
        sa, nk, nv = self.self_attn(self.norm_sa(x), past_k, past_v, attn_mask)
        x = x + sa
        x = x + self.cross_attn(self.norm_xa_q(x), self.norm_xa_m(memory), mem_mask_add)
        x = x + self.ffn(self.norm_ff(x))
        return x, nk, nv


class HostCacheDecoderStep(nn.Module):
    """§6.3 boundary: caches as read-only inputs, host-supplied masks, current-slice outputs."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            HostCacheLayer(D_MODEL, D_FFN, SA_HEADS, XA_HEADS, XA_D_MEM) for _ in range(N_LAYERS)
        ])
        self.norm_out = nn.LayerNorm(D_MODEL, bias=False)
        self.final_proj = nn.Linear(D_MODEL, OUT_DIM)

    def forward(self, audio_embed, encoder_output, mem_mask_add, attn_mask,
                *caches):
        # caches = (pk0, pv0, pk1, pv1, ..., pk11, pv11)
        x = audio_embed
        new_slices = []
        for i, layer in enumerate(self.layers):
            pk, pv = caches[2 * i], caches[2 * i + 1]
            x, nk, nv = layer(x, pk, pv, attn_mask, encoder_output, mem_mask_add)
            new_slices += [nk, nv]
        logits = self.final_proj(self.norm_out(x))
        return tuple([logits] + new_slices)


# ---------------------------------------------------------------------------
# Build + convert
# ---------------------------------------------------------------------------
def build_old():
    dec = TraceableDecoderStep(
        n_layers=N_LAYERS, d_model=D_MODEL, d_ffn=D_FFN, sa_n_heads=SA_HEADS,
        xa_n_heads=XA_HEADS, xa_d_memory=XA_D_MEM, max_seq_len=MAX_SEQ,
        num_codebooks=NUM_CODEBOOKS, num_tokens_per_codebook=NUM_TOKENS,
    ).eval()

    audio = torch.randn(1, 1, D_MODEL)
    enc = torch.randn(1, T_ENC, D_MODEL)
    enc_mask = torch.ones(1, T_ENC, dtype=torch.bool)
    args = (audio, enc, enc_mask)
    for _ in range(N_LAYERS):
        args += (torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD),
                 torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD),
                 torch.tensor([0.0]))
    with torch.no_grad():
        traced = torch.jit.trace(dec, args)

    inputs = [
        ct.TensorType(name="audio_embed", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="encoder_output", shape=(1, T_ENC, D_MODEL)),
        ct.TensorType(name="encoder_mask", shape=(1, T_ENC), dtype=np.bool_),
    ]
    for i in range(N_LAYERS):
        inputs += [
            ct.TensorType(name=f"cache_k{i}", shape=(1, MAX_SEQ, SA_HEADS, D_HEAD)),
            ct.TensorType(name=f"cache_v{i}", shape=(1, MAX_SEQ, SA_HEADS, D_HEAD)),
            ct.TensorType(name=f"position{i}", shape=(1,)),
        ]
    return ct.convert(traced, inputs=inputs, convert_to="mlprogram",
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS17)


def build_new():
    dec = HostCacheDecoderStep().eval()
    audio = torch.randn(1, 1, D_MODEL)
    enc = torch.randn(1, T_ENC, D_MODEL)
    mem_mask_add = torch.zeros(1, 1, 1, T_ENC)
    attn_mask = torch.zeros(1, 1, 1, MAX_SEQ + 1)
    caches = ()
    for _ in range(N_LAYERS):
        caches += (torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD),
                   torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD))
    args = (audio, enc, mem_mask_add, attn_mask) + caches
    with torch.no_grad():
        traced = torch.jit.trace(dec, args)

    inputs = [
        ct.TensorType(name="audio_embed", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="encoder_output", shape=(1, T_ENC, D_MODEL)),
        ct.TensorType(name="mem_mask_add", shape=(1, 1, 1, T_ENC)),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, MAX_SEQ + 1)),
    ]
    for i in range(N_LAYERS):
        inputs += [
            ct.TensorType(name=f"cache_k{i}", shape=(1, MAX_SEQ, SA_HEADS, D_HEAD)),
            ct.TensorType(name=f"cache_v{i}", shape=(1, MAX_SEQ, SA_HEADS, D_HEAD)),
        ]
    return ct.convert(traced, inputs=inputs, convert_to="mlprogram",
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS17)


def main():
    out = Path(__file__).parent / "build_ane"
    out.mkdir(parents=True, exist_ok=True)
    print("Building OLD (current production graph) ...")
    build_old().save(str(out / "old_decoder_step.mlpackage"))
    print("Building NEW (§6.3 host-owned cache) ...")
    build_new().save(str(out / "new_decoder_step.mlpackage"))
    print(f"Saved both to {out}")


if __name__ == "__main__":
    main()
