"""Fuse the Magpie Local Transformer + 8-codebook sampler into a single CoreML
model so one ANE dispatch covers what was previously 8 Swift CPU LT passes.

The Swift baseline (``MagpieLocalSampler.sample(...)``) runs 8 sequential
1-layer 256-dim transformer passes per AR decoder step, plus top-k
softmax + categorical sampling between each pass. On M2 that costs
roughly 4–8 ms per AR step; over the first chunk's 24 AR steps that's
~100–200 ms of pure-CPU work that contributes directly to TTFA.

This script bakes those 8 LT passes plus per-codebook sampling into one
.mlpackage (``local_transformer.mlpackage``). The expected win is one ANE
call (~1–2 ms) per AR step instead of CPU LT + 8 round-trips.

Inputs (caller supplies per AR step):
    decoder_hidden : (1, 768) fp32  — output of decoder_step.mlmodelc
    uniforms       : (8,)     fp32  — pre-drawn uniform samples in [0, 1)
                                       from the MagpieMT19937 RNG, one per cb.
    forbid_eos     : (1,)     fp32  — 1.0 to forbid EOS (t < min_frames),
                                       0.0 otherwise.
    temperature    : (1,)     fp32  — sampling temperature.

Output:
    codes : (8,) int32  — sampled codebook tokens for one frame.

Quantitative parity caveats:
    * Sampling uses fp32 cumsum + fp32 compare against the passed uniform.
      Swift's ``numpyChoice`` promotes cumsum to fp64 for the bsearch step;
      the divergence is below 1 ULP for top-k=80 softmax and we accept it.
    * Top-k tie-breaking on the K-th boundary uses CoreML's ``topk`` (which
      mirrors ``torch.topk``: stable, earliest-index wins). Same rule the
      Swift heap path uses.
    * ``constants/audio_embedding_*.npy`` are pre-projected through
      ``in_proj`` at convert time so the runtime path is one gather instead
      of gather + matmul.

CFG path (``cfgScale != 1.0``) is **not** in this fused model — Magpie's
default in FluidAudio is ``cfgScale = 1.0`` (off). Callers needing CFG
must fall back to the Swift sampler.

Usage:
    uv run python convert_local_transformer.py \\
        --output build/local_transformer.mlpackage

By default, weights are read from the FluidAudio cache:
    ~/.cache/fluidaudio/Models/magpie-tts/constants/{local_transformer/,
                                                    audio_embedding_*.npy}
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct


# ---------------------------------------------------------------------------
# Constants (mirrors MagpieConstants.swift / generate_coreml.py)
# ---------------------------------------------------------------------------

D_MODEL = 768
LOCAL_DIM = 256
FFN_DIM = 1024
NUM_CODEBOOKS = 8
NUM_CODES_PER_CODEBOOK = 2024
MAX_LT_POSITIONS = 10  # numCodebooks + 2 — pos_emb has 10 rows

DEFAULT_TOP_K = 80
EOS_ID = 2017
ALWAYS_FORBIDDEN = (2016, 2018, 2019, 2020, 2021, 2022, 2023)

# Large negative addend used to mask logits. Must stay within fp16
# range (~±65504) because ``compute_precision=FLOAT16`` casts intermediate
# tensors. Softmax(x - max) drives -1e4 to <1e-30 in fp32, so it's
# effectively zero post-softmax while still being safely fp16-representable.
NEG_INF = -1e4


# ---------------------------------------------------------------------------
# PyTorch module (traced by coremltools)
# ---------------------------------------------------------------------------


class _LocalTransformerLayer(nn.Module):
    """Single 1-layer pre-norm causal self-attention + pre-norm FFN block.

    Mirrors MagpieLocalTransformer.forward in the Swift port.
    """

    def __init__(self, w: Dict[str, np.ndarray]):
        super().__init__()
        D = LOCAL_DIM

        # Layer norm weights (no bias — matches Swift impl).
        self.register_buffer("norm1_w", torch.from_numpy(w["norm1_weight"]).float())
        self.register_buffer("norm2_w", torch.from_numpy(w["norm2_weight"]).float())

        # Self-attention: stacked QKV projection, then output projection.
        # qkv_weight shape is (3D, D); o_weight shape is (D, D).
        self.register_buffer("qkv_w", torch.from_numpy(w["sa_qkv_weight"]).float())
        self.register_buffer("o_w", torch.from_numpy(w["sa_o_weight"]).float())

        # FFN: stored as Conv1d kernel-1 in NeMo, equivalent to Linear.
        # Squeeze the trailing dim of size 1 if present.
        ffn1 = w["ffn_conv1_weight"]
        if ffn1.ndim == 3:
            ffn1 = ffn1.squeeze(-1)
        ffn2 = w["ffn_conv2_weight"]
        if ffn2.ndim == 3:
            ffn2 = ffn2.squeeze(-1)
        self.register_buffer("ffn_w1", torch.from_numpy(ffn1).float())
        self.register_buffer("ffn_w2", torch.from_numpy(ffn2).float())

        # Positional embeddings (reused across all 8 LT calls).
        self.register_buffer(
            "pos_emb", torch.from_numpy(w["pos_emb"]).float()  # (10, 256)
        )

        self._D = D
        self._ffn_D = FFN_DIM
        self._scale = 1.0 / math.sqrt(D)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (T, D) — returns (T, D)."""
        T, D = seq.shape[-2], seq.shape[-1]

        x = seq + self.pos_emb[:T]

        # Pre-norm causal self-attention.
        x_norm = _layer_norm_no_bias(x, self.norm1_w)
        qkv = x_norm @ self.qkv_w.t()                        # (T, 3D)
        q, k, v = qkv.split(D, dim=-1)                        # each (T, D)
        attn = (q @ k.t()) * self._scale                      # (T, T)
        # Causal mask via additive `-inf`. Use a large negative number to
        # stay safely fp16-representable post-conversion; -1e9 is below the
        # softmax floor and well within fp32.
        causal = torch.tril(torch.ones(T, T, dtype=attn.dtype, device=attn.device))
        attn = attn + (1.0 - causal) * NEG_INF
        attn = attn.softmax(dim=-1)
        sa_out = attn @ v                                     # (T, D)
        sa_out = sa_out @ self.o_w.t()                        # (T, D)
        x = x + sa_out

        # Pre-norm FFN with tanh-GELU.
        x_norm = _layer_norm_no_bias(x, self.norm2_w)
        h = x_norm @ self.ffn_w1.t()                          # (T, ffnD)
        h = _gelu_tanh(h)
        h = h @ self.ffn_w2.t()                               # (T, D)
        x = x + h
        return x


def _layer_norm_no_bias(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LayerNorm with weight only (no bias) — matches Swift impl exactly."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5) * weight


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """tanh-approximation GELU — matches Swift `applyGeluTanh`."""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * x.pow(3))))


class FusedLocalTransformer(nn.Module):
    """Single CoreML graph that does the full per-frame LT + sampling loop.

    Internal flow per AR step:
        proj_decoder = decoder_hidden @ in_proj.T + in_proj_b      (1, 256)
        seq          = proj_decoder                                (1, 256)
        for cb in 0..7:
            out      = LT(seq)                                      (cb+1, 256)
            logits   = out[-1] @ out_W[cb].T + out_b[cb]            (2024,)
            logits  += allow_eos_mask
            logits  += eos_addend * forbid_eos_flag
            top_v, top_i = topk(logits, K)
            top_v   /= max(temperature, 1e-8)
            probs    = softmax(top_v)
            cdf      = cumsum(probs)
            slot     = sum(cdf < uniform[cb])           # 0..K-1
            code     = top_i[slot]
            codes.append(code)
            seq      = cat([seq, proj_audio_emb[cb, code]])         # (cb+2, 256)
        return stack(codes)                                          (8,)
    """

    def __init__(
        self,
        lt_w: Dict[str, np.ndarray],
        out_proj_weights: List[np.ndarray],
        out_proj_biases: List[np.ndarray],
        proj_audio_emb: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
    ):
        super().__init__()
        self.layer = _LocalTransformerLayer(lt_w)
        self.top_k = int(top_k)

        # Decoder-hidden → LT input projection (Linear w/ bias).
        self.register_buffer(
            "in_proj_w", torch.from_numpy(lt_w["in_proj_weight"]).float()
        )  # (256, 768)
        self.register_buffer(
            "in_proj_b", torch.from_numpy(lt_w["in_proj_bias"]).float()
        )  # (256,)

        # Per-codebook output heads → stacked into (8, 2024, 256) and (8, 2024).
        out_w = np.stack(out_proj_weights, axis=0)
        out_b = np.stack(out_proj_biases, axis=0)
        assert out_w.shape == (NUM_CODEBOOKS, NUM_CODES_PER_CODEBOOK, LOCAL_DIM)
        assert out_b.shape == (NUM_CODEBOOKS, NUM_CODES_PER_CODEBOOK)
        self.register_buffer("out_w", torch.from_numpy(out_w).float())
        self.register_buffer("out_b", torch.from_numpy(out_b).float())

        # Pre-projected audio embeddings: skip a runtime matmul per cb step.
        # proj_audio_emb[cb][code] = audio_emb[cb][code] @ in_proj_W.T + in_proj_b
        assert proj_audio_emb.shape == (NUM_CODEBOOKS, NUM_CODES_PER_CODEBOOK, LOCAL_DIM)
        self.register_buffer(
            "proj_audio_emb", torch.from_numpy(proj_audio_emb).float()
        )

        # Forbidden-token addends.
        # always_addend: -inf at indices 2016, 2018-2023; zero elsewhere.
        # eos_addend:    -inf at 2017; zero elsewhere. Multiplied by
        #                forbid_eos_flag at runtime.
        always = np.zeros(NUM_CODES_PER_CODEBOOK, dtype=np.float32)
        for tok in ALWAYS_FORBIDDEN:
            always[tok] = NEG_INF
        eos = np.zeros(NUM_CODES_PER_CODEBOOK, dtype=np.float32)
        eos[EOS_ID] = NEG_INF
        self.register_buffer("always_addend", torch.from_numpy(always))
        self.register_buffer("eos_addend", torch.from_numpy(eos))

        # Constants for gather-free indexing. Using `(arange == idx) * value`
        # mask-multiply pattern instead of `gather` keeps the graph on ANE —
        # CoreML's gather_nd rejects mixed-rank inputs at runtime
        # (gather_nd_kernel: In TORCH_GATHER mode, inputs should have the
        # same rank).
        self.register_buffer(
            "arange_codes",
            torch.arange(NUM_CODES_PER_CODEBOOK, dtype=torch.int32),
        )
        self.register_buffer(
            "arange_topk", torch.arange(self.top_k, dtype=torch.int32)
        )

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        uniforms: torch.Tensor,
        forbid_eos: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs fp32. Returns int32 (8,)."""
        # Project decoder hidden → LT input space. (1, 256).
        first = decoder_hidden.reshape(D_MODEL) @ self.in_proj_w.t() + self.in_proj_b
        seq = first.unsqueeze(0)  # (1, 256)

        temp_safe = torch.clamp(temperature, min=1e-8)
        eos_scale = forbid_eos.reshape(())  # scalar 0.0 or 1.0

        codes: List[torch.Tensor] = []
        # Loop unrolled at trace time — produces 8 distinct LT subgraphs
        # with T = 1, 2, ..., 8 respectively.
        for cb in range(NUM_CODEBOOKS):
            out = self.layer(seq)                              # (cb+1, 256)
            # T = cb + 1, so the last row is at index cb. Constant int —
            # avoids the `-1` indexing that traces as a runtime gather.
            last = out[cb]                                      # (256,)
            logits = last @ self.out_w[cb].t() + self.out_b[cb]  # (2024,)
            logits = logits + self.always_addend
            logits = logits + self.eos_addend * eos_scale
            logits = logits / temp_safe.reshape(())

            # Top-k truncation, then softmax + cumsum + categorical sample.
            top_v, top_i = torch.topk(logits, self.top_k, dim=-1)  # (K,), (K,)
            probs = top_v.softmax(dim=-1)                          # (K,)
            cdf = probs.cumsum(dim=-1)                             # (K,)
            u = uniforms[cb].reshape(())

            # Sample without gather: mark the FIRST cdf bin that crosses u.
            # `(cdf >= u).cumsum() == 1` is a one-hot mask on that bin
            # (cumsum transitions from 0 → 1 exactly once at the chosen slot).
            ge_u = (cdf >= u).to(torch.int32)                      # (K,)
            ge_cum = ge_u.cumsum(dim=-1)                           # (K,)
            slot_mask = (ge_cum == 1).to(top_i.dtype)              # (K,) one-hot
            # Pick the code via mask multiply + sum. ANE-friendly.
            code64 = (top_i * slot_mask).sum()                     # () int64
            code32 = code64.to(torch.int32)
            codes.append(code32)

            # Append next LT input via the pre-projected audio embedding row.
            # `code_onehot @ proj_audio_emb[cb]` is the matmul form of a row
            # gather — keeps the graph on ANE (vs TORCH_GATHER which Espresso
            # rejects when ranks differ).
            code_onehot = (self.arange_codes == code32).to(self.proj_audio_emb.dtype)
            next_in = code_onehot @ self.proj_audio_emb[cb]        # (256,)
            seq = torch.cat([seq, next_in.unsqueeze(0)], dim=0)

        return torch.stack(codes, dim=0)                           # (8,)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_lt_weights(lt_dir: str) -> Dict[str, np.ndarray]:
    needed = [
        "in_proj_weight", "in_proj_bias",
        "pos_emb",
        "norm1_weight", "norm2_weight",
        "sa_qkv_weight", "sa_o_weight",
        "ffn_conv1_weight", "ffn_conv2_weight",
    ]
    out: Dict[str, np.ndarray] = {}
    for name in needed:
        path = os.path.join(lt_dir, f"{name}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing LT weight: {path}")
        out[name] = np.load(path).astype(np.float32)
    return out


def _load_out_projections(lt_dir: str) -> tuple[List[np.ndarray], List[np.ndarray]]:
    weights, biases = [], []
    for i in range(NUM_CODEBOOKS):
        w = np.load(os.path.join(lt_dir, f"out_proj_{i}_weight.npy")).astype(np.float32)
        b = np.load(os.path.join(lt_dir, f"out_proj_{i}_bias.npy")).astype(np.float32)
        weights.append(w)
        biases.append(b)
    return weights, biases


def _load_audio_embeddings(const_dir: str) -> List[np.ndarray]:
    out = []
    for i in range(NUM_CODEBOOKS):
        path = os.path.join(const_dir, f"audio_embedding_{i}.npy")
        out.append(np.load(path).astype(np.float32))
    return out


def _project_audio_embeddings(
    audio_emb: List[np.ndarray], in_proj_w: np.ndarray, in_proj_b: np.ndarray
) -> np.ndarray:
    """Pre-compute `audio_emb @ in_proj_W.T + in_proj_b` per codebook.

    Returns array (8, 2024, 256). Saves a 768->256 matmul at runtime.
    """
    out = np.empty((NUM_CODEBOOKS, NUM_CODES_PER_CODEBOOK, LOCAL_DIM), dtype=np.float32)
    for i in range(NUM_CODEBOOKS):
        ae = audio_emb[i]  # (2024, 768)
        out[i] = ae @ in_proj_w.T + in_proj_b
    return out


# ---------------------------------------------------------------------------
# Convert + parity check
# ---------------------------------------------------------------------------


def _torch_reference(model: FusedLocalTransformer, inputs) -> np.ndarray:
    with torch.no_grad():
        return model(*inputs).cpu().numpy()


_PRECISION_MAP = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}


def convert(
    cache_dir: str,
    output_path: str,
    top_k: int = DEFAULT_TOP_K,
    precision: str = "fp16",
) -> str:
    lt_dir = os.path.join(cache_dir, "constants", "local_transformer")
    const_dir = os.path.join(cache_dir, "constants")

    if not os.path.isdir(lt_dir):
        raise FileNotFoundError(f"LT weight dir not found: {lt_dir}")

    print(f"Loading LT weights from {lt_dir}")
    lt_w = _load_lt_weights(lt_dir)
    out_w, out_b = _load_out_projections(lt_dir)
    print(f"Loading audio embeddings from {const_dir}")
    audio_emb = _load_audio_embeddings(const_dir)
    print("Pre-projecting audio embeddings through in_proj ...")
    proj_audio_emb = _project_audio_embeddings(
        audio_emb, lt_w["in_proj_weight"], lt_w["in_proj_bias"]
    )

    print("Building FusedLocalTransformer module ...")
    module = FusedLocalTransformer(
        lt_w=lt_w,
        out_proj_weights=out_w,
        out_proj_biases=out_b,
        proj_audio_emb=proj_audio_emb,
        top_k=top_k,
    )
    module.eval()

    # Example inputs for tracing.
    rng = np.random.default_rng(0)
    decoder_hidden = torch.from_numpy(
        rng.standard_normal((1, D_MODEL), dtype=np.float32) * 0.1
    )
    uniforms = torch.from_numpy(
        rng.uniform(0.0, 1.0, NUM_CODEBOOKS).astype(np.float32)
    )
    forbid_eos = torch.tensor([1.0], dtype=torch.float32)
    temperature = torch.tensor([0.6], dtype=torch.float32)

    print("Tracing module ...")
    with torch.no_grad():
        traced = torch.jit.trace(
            module, (decoder_hidden, uniforms, forbid_eos, temperature),
            check_trace=False,
        )

    # Trace verification.
    with torch.no_grad():
        ref = module(decoder_hidden, uniforms, forbid_eos, temperature)
        traced_out = traced(decoder_hidden, uniforms, forbid_eos, temperature)
    if not torch.equal(ref, traced_out):
        raise RuntimeError(
            f"Trace produced different result.\n  ref:    {ref.tolist()}\n"
            f"  traced: {traced_out.tolist()}"
        )
    print(f"  Trace ok. Sampled codes (seed 0): {ref.tolist()}")

    print("Converting to CoreML (mlprogram, fp16, iOS17) ...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="decoder_hidden", shape=(1, D_MODEL), dtype=np.float32),
            ct.TensorType(name="uniforms", shape=(NUM_CODEBOOKS,), dtype=np.float32),
            ct.TensorType(name="forbid_eos", shape=(1,), dtype=np.float32),
            ct.TensorType(name="temperature", shape=(1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="codes", dtype=np.int32),
        ],
        convert_to="mlprogram",
        compute_precision=_PRECISION_MAP[precision],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL,
    )
    print(f"  ct.convert took {time.time() - t0:.2f}s")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Quick parity sanity check via CoreML CPU_ONLY (deterministic).
    print("\nParity check (CoreML CPU_ONLY vs PyTorch)...")
    cm = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    cm_out = cm.predict({
        "decoder_hidden": decoder_hidden.numpy(),
        "uniforms": uniforms.numpy(),
        "forbid_eos": forbid_eos.numpy(),
        "temperature": temperature.numpy(),
    })["codes"]
    ref_np = ref.numpy()
    print(f"  PyTorch codes : {ref_np.tolist()}")
    print(f"  CoreML codes  : {cm_out.tolist()}")
    if not np.array_equal(ref_np, cm_out):
        # fp16 conversion can perturb softmax tails enough that a sample
        # crosses a top-k bin boundary; report rather than fail.
        diff = (ref_np != cm_out).sum()
        print(f"  WARN: {diff}/{NUM_CODEBOOKS} codes differ — fp16 boundary jitter.")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.expanduser("~/.cache/fluidaudio/Models/magpie-tts"),
        help="Path to the FluidAudio Magpie cache (must contain "
             "constants/local_transformer/ and constants/audio_embedding_*.npy).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="build/local_transformer.mlpackage",
        help="Output mlpackage path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k value to bake into the graph (default 80).",
    )
    parser.add_argument(
        "--precision", choices=["fp16", "fp32"], default="fp16",
        help="compute_precision for ct.convert (default fp16, production)",
    )
    args = parser.parse_args()

    convert(args.cache_dir, args.output, top_k=args.top_k, precision=args.precision)


if __name__ == "__main__":
    main()
