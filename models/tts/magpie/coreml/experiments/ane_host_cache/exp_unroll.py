"""Does §6.3 unblock the N=2 AR unroll (Trial 4a) on ANE — the real speed lever?

The §6.3 host-owned cache makes the *decoder* per-step 40% faster, but that doesn't cut the
number of autoregressive iterations. Cutting iterations needs an in-graph unroll: run N
(decoder → sample → embed) frames per CoreML call. Trial 4a (blend-based decoder) tried N=2
and ANECompile failed. This reconverts the unroll with the §6.3 decoder to isolate the
blocker: is it the cache mutation (§6.3 fixes) or the LT sampling tail (topk/cumsum/int32,
which §6.3 does not touch)?

Builds N=1 and N=2 §6.3 unrolls (decoder + faithful LT sampler + audio_embed feedback,
random weights) and probes ANE admission + per-frame latency + per-op device placement.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from exp_convert import (  # noqa: E402
    HostCacheLayer, N_LAYERS, D_MODEL, D_FFN, SA_HEADS, D_HEAD,
    XA_HEADS, XA_D_MEM, MAX_SEQ, T_ENC, MASK_NEG,
)

LOCAL_DIM = 256
NUM_CB = 8
NUM_CODES = 2024
TOP_K = 80


class DecoderBody(nn.Module):
    """12 §6.3 layers over one audio_embed; returns hidden + current-step slices."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            HostCacheLayer(D_MODEL, D_FFN, SA_HEADS, XA_HEADS, XA_D_MEM) for _ in range(N_LAYERS)
        ])

    def forward(self, x, enc, mem_add, attn_mask, past_ks, past_vs):
        nks, nvs = [], []
        for i, layer in enumerate(self.layers):
            x, nk, nv = layer(x, past_ks[i], past_vs[i], attn_mask, enc, mem_add)
            nks.append(nk); nvs.append(nv)
        return x, nks, nvs


class MiniLT(nn.Module):
    """Faithful-enough local-transformer sampling tail: the ANE-hostile ops
    (topk, cumsum, cumsum==1 argmax, one-hot @ table) that Trial 4a fuses per frame."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(D_MODEL, LOCAL_DIM)
        self.qkv = nn.Linear(LOCAL_DIM, 3 * LOCAL_DIM, bias=False)
        self.o = nn.Linear(LOCAL_DIM, LOCAL_DIM, bias=False)
        self.out_w = nn.Parameter(torch.randn(NUM_CB, NUM_CODES, LOCAL_DIM) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(NUM_CB, NUM_CODES))
        self.register_buffer("proj_emb", torch.randn(NUM_CB, NUM_CODES, LOCAL_DIM) * 0.02)
        self.register_buffer("arange", torch.arange(NUM_CODES, dtype=torch.int32))
        self.scale = LOCAL_DIM ** -0.5

    def forward(self, hidden, uniforms):
        seq = self.in_proj(hidden.reshape(1, D_MODEL))  # [1, LOCAL_DIM]
        codes = []
        for cb in range(NUM_CB):
            qkv = self.qkv(seq)
            q, k, v = qkv.split(LOCAL_DIM, dim=-1)
            attn = (q @ k.t()) * self.scale
            T = seq.shape[0]
            causal = torch.tril(torch.ones(T, T, dtype=attn.dtype))
            attn = (attn + (1.0 - causal) * MASK_NEG).softmax(dim=-1)
            out = self.o(attn @ v)
            last = out[cb] if out.shape[0] > cb else out[-1]
            logits = last @ self.out_w[cb].t() + self.out_b[cb]
            # ANE-hostile sampling tail (the suspected blocker):
            top_v, top_i = torch.topk(logits, TOP_K, dim=-1)
            probs = top_v.softmax(dim=-1)
            cdf = probs.cumsum(dim=-1)
            u = uniforms[cb].reshape(())
            ge = (cdf >= u).to(torch.int32)
            slot = (ge.cumsum(dim=-1) == 1).to(top_i.dtype)
            code = (top_i * slot).sum().to(torch.int32)
            codes.append(code)
            onehot = (self.arange == code).to(self.proj_emb.dtype)
            seq = torch.cat([seq, (onehot @ self.proj_emb[cb]).unsqueeze(0)], dim=0)
        return torch.stack(codes)


class UnrollN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.dec = DecoderBody()
        self.lt = MiniLT()
        self.register_buffer("audio_emb", torch.randn(NUM_CB, NUM_CODES, D_MODEL) * 0.02)
        self.register_buffer("arange", torch.arange(NUM_CODES, dtype=torch.int32))

    def embed(self, codes):
        acc = torch.zeros(D_MODEL)
        for cb in range(NUM_CB):
            onehot = (self.arange == codes[cb]).to(self.audio_emb.dtype)
            acc = acc + onehot @ self.audio_emb[cb]
        return (acc / NUM_CB).view(1, 1, D_MODEL)

    def forward(self, audio_embed, enc, mem_add, attn_mask, unis, *caches):
        pks = list(caches[0::2]); pvs = list(caches[1::2])
        x = audio_embed
        all_codes = []
        for it in range(self.n):
            hidden, nks, nvs = self.dec(x, enc, mem_add, attn_mask, pks, pvs)
            codes = self.lt(hidden, unis[it])
            all_codes.append(codes)
            # NOTE: fixed-size cache across iters (no in-graph grow) — keeps shapes
            # constant so the ANE-admission/latency structure is what we measure;
            # the AR chain is preserved via the embed feedback below.
            if it < self.n - 1:
                x = self.embed(codes)
        return tuple(all_codes)


def build(n):
    m = UnrollN(n).eval()
    audio = torch.randn(1, 1, D_MODEL)
    enc = torch.randn(1, T_ENC, D_MODEL)
    mem = torch.zeros(1, 1, 1, T_ENC)
    mask = torch.zeros(1, 1, 1, MAX_SEQ + 1)
    unis = torch.rand(n, NUM_CB)
    caches = ()
    for _ in range(N_LAYERS):
        caches += (torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD), torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD))
    args = (audio, enc, mem, mask, unis) + caches
    with torch.no_grad():
        traced = torch.jit.trace(m, args)
    inputs = [
        ct.TensorType(name="audio_embed", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="encoder_output", shape=(1, T_ENC, D_MODEL)),
        ct.TensorType(name="mem_mask_add", shape=(1, 1, 1, T_ENC)),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, MAX_SEQ + 1)),
        ct.TensorType(name="uniforms", shape=(n, NUM_CB)),
    ]
    for i in range(N_LAYERS):
        inputs += [ct.TensorType(name=f"cache_k{i}", shape=(1, MAX_SEQ, SA_HEADS, D_HEAD)),
                   ct.TensorType(name=f"cache_v{i}", shape=(1, MAX_SEQ, SA_HEADS, D_HEAD))]
    return ct.convert(traced, inputs=inputs, convert_to="mlprogram",
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS17)


def device_breakdown(pkg):
    try:
        from coremltools.models.compute_plan import MLComputePlan
        _keep = ct.models.MLModel(pkg)
        plan = MLComputePlan.load_from_path(_keep.get_compiled_model_path(),
                                            compute_units=ct.ComputeUnit.CPU_AND_NE)
        c = {}
        for fn in plan.model_structure.program.functions.values():
            for op in fn.block.operations:
                du = plan.get_compute_device_usage_for_mlprogram_operation(op)
                d = type(du.preferred_compute_device).__name__ if du else "None"
                c[d] = c.get(d, 0) + 1
        ne, cpu = c.get("MLNeuralEngineComputeDevice", 0), c.get("MLCPUComputeDevice", 0)
        tot = ne + cpu + c.get("MLGPUComputeDevice", 0)
        return f"ANE {ne} / CPU {cpu} of {tot} assigned -> {100*ne/tot:.0f}% ANE" if tot else "n/a"
    except Exception as e:
        return f"unavailable: {str(e)[:70]}"


def main():
    out = Path(__file__).parent / "build_ane"
    out.mkdir(parents=True, exist_ok=True)
    for n in [1, 2]:
        print(f"\n=== §6.3 unroll N={n} ({n} frame(s)/call) ===", flush=True)
        try:
            mdl = build(n)
        except Exception as e:
            print(f"  CONVERT FAIL: {str(e)[:120]}", flush=True); continue
        p = str(out / f"unroll_n{n}.mlpackage")
        mdl.save(p)
        # latency on ANE
        for cu in ["CPU_AND_NE", "CPU_ONLY"]:
            try:
                mm = ct.models.MLModel(p, compute_units=getattr(ct.ComputeUnit, cu))
                feed = {"audio_embed": np.random.randn(1,1,D_MODEL).astype(np.float32)*0.1,
                        "encoder_output": np.random.randn(1,T_ENC,D_MODEL).astype(np.float32)*0.1,
                        "mem_mask_add": np.zeros((1,1,1,T_ENC),np.float32),
                        "attn_mask": np.zeros((1,1,1,MAX_SEQ+1),np.float32),
                        "uniforms": np.random.rand(n,NUM_CB).astype(np.float32)}
                for i in range(N_LAYERS):
                    feed[f"cache_k{i}"] = np.zeros((1,MAX_SEQ,SA_HEADS,D_HEAD),np.float32)
                    feed[f"cache_v{i}"] = np.zeros((1,MAX_SEQ,SA_HEADS,D_HEAD),np.float32)
                ts = []
                for _ in range(20):
                    t0 = time.time(); mm.predict(feed); ts.append((time.time()-t0)*1000)
                ts = np.array(ts[2:])
                print(f"  {cu:12s}: p50 {np.percentile(ts,50):.1f}ms  -> {np.percentile(ts,50)/n:.1f}ms/frame", flush=True)
            except Exception as e:
                mk = "ANECompile FAIL" if ("-14" in str(e) or "ANECompile" in str(e) or "ANECCompile" in str(e)) else str(e)[:80]
                print(f"  {cu:12s}: FAIL {mk}", flush=True)
        print(f"  placement   : {device_breakdown(p)}", flush=True)


if __name__ == "__main__":
    main()
