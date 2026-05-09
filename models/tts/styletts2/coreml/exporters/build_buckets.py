"""Build per-bucket bert + fused_diffusion_sampler mlpackages.

Why
---
Bert and fused_diffusion_sampler are the two stages whose token axis
(T_TOK) cannot be `ct.RangeDim` — HF Albert and the cross-attention in
the diffusion sampler both contain shape ops the MLProgram CPU backend
rejects under data-dependent shapes (see notes in `coreml/exporters/convert.py`
on bert and diffusion_unet).

iteration_3 ships both at T=57 (the captured shape from the default
"StyleTTS 2 is a text to speech model." text). Sentences whose phoneme
count exceeds 57 are silently truncated by the runtime, which clips
the prosody / breaks paragraph-length inputs.

This driver builds the same two stages at multiple fixed T_TOK sizes
in one pass, producing per-bucket mlpackages that the runtime selects
between based on the actual token count (smallest bucket >= count).
No EnumeratedShapes — separate mlpackages, simpler conversion (each
graph is fully static), only one bucket loaded at a time.

Output layout
-------------
    coreml/packages/bert_fp16_t64.mlpackage
    coreml/packages/bert_fp16_t128.mlpackage
    coreml/packages/bert_fp16_t256.mlpackage
    coreml/packages/fused_diffusion_sampler_fp16_t64.mlpackage
    coreml/packages/fused_diffusion_sampler_fp16_t128.mlpackage
    coreml/packages/fused_diffusion_sampler_fp16_t256.mlpackage

Disk delta vs single T=57 build (fp16):
    bert_fp16            ~12 MB ->  3 buckets ~36 MB total (+24 MB)
    fused_diffusion_sampler_fp16  ~47 MB ->  3 buckets ~141 MB total (+94 MB)
    Combined extra: ~118 MB

Run
---
    cd models/tts/styletts2
    uv run python coreml/exporters/build_buckets.py
    uv run python coreml/exporters/build_buckets.py --buckets 64,128
    uv run python coreml/exporters/build_buckets.py --precision fp32 --stages bert
"""

from __future__ import annotations

import argparse
import sys
import time
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coreml.exporters import convert as _convert  # noqa: F401  (installs MIL patches)
from coreml._runtime import HERE, build_runtime, stage_example_inputs
from coreml.exporters.fuse_diffusion_sampler import FusedDiffusionSampler
from coreml.wrappers import BertWrapper

PACKAGES_DIR = HERE / "coreml" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_to(t: torch.Tensor, target_T: int, *, dim: int = -1, value: int = 0) -> torch.Tensor:
    """Right-pad a tensor along `dim` to length `target_T` with `value`.

    Errors if `t.shape[dim] > target_T` — caller picks the bucket.
    """
    cur = t.shape[dim]
    if cur > target_T:
        raise ValueError(
            f"input length {cur} exceeds bucket size {target_T} along dim {dim}"
        )
    if cur == target_T:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = target_T - cur
    pad = torch.full(pad_shape, value, dtype=t.dtype)
    return torch.cat([t, pad], dim=dim)


def _metric(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    return {
        "shape": tuple(a.shape),
        "mse": float(np.mean(diff * diff)),
        "max_abs_delta": float(np.max(np.abs(diff))),
        "cos": cos,
    }


# ---------------------------------------------------------------------------
# Per-bucket converters
# ---------------------------------------------------------------------------


def convert_bert_bucket(rt, *, t_tok: int, precision: str) -> Path:
    """Convert bert at a given fixed T_TOK.

    Pads the captured (tokens, attention_mask) to T_TOK with 0 (pad
    token) / 0 (mask out) and traces. Output names match convert.py's
    bert stage.
    """
    import coremltools as ct

    print(f"\n=== bert bucket T={t_tok} ({precision}) ===")
    tokens, attn = stage_example_inputs("bert", rt)
    real_T = int(tokens.shape[1])
    if real_T > t_tok:
        raise SystemExit(
            f"bucket T={t_tok} smaller than captured tokens T={real_T}; "
            "use a larger reference text or larger bucket."
        )
    print(f"  captured tokens: {tuple(tokens.shape)}, padding to T={t_tok}")
    tokens_pad = _pad_to(tokens, t_tok, dim=1, value=0)
    attn_pad = _pad_to(attn, t_tok, dim=1, value=0)

    wrapper = BertWrapper(rt.model.bert, rt.model.bert_encoder)
    with torch.no_grad():
        eager_out = wrapper(tokens_pad, attn_pad)
    if not isinstance(eager_out, tuple):
        eager_out = (eager_out,)
    for i, t in enumerate(eager_out):
        print(f"  eager out [{i}]: {tuple(t.shape)} {t.dtype}")

    print("  tracing ...")
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, (tokens_pad, attn_pad), check_trace=False, strict=False
        )

    ct_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )
    inputs = [
        ct.TensorType(name="tokens", shape=tuple(tokens_pad.shape), dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=tuple(attn_pad.shape), dtype=np.int32),
    ]
    print(f"  ct.convert ({precision}, fixed T={t_tok}) ...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct_precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  ct.convert: {time.time() - t0:.1f}s")

    suffix_prec = "_fp16" if precision == "fp16" else ""
    out_path = PACKAGES_DIR / f"bert{suffix_prec}_t{t_tok}.mlpackage"
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    size_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6
    print(f"  saved {out_path.relative_to(HERE)} ({size_mb:.1f} MB)")
    return out_path


def convert_sampler_bucket(rt, *, t_tok: int, precision: str, seed: int = 0) -> Path:
    """Convert fused_diffusion_sampler at a given fixed T_TOK.

    Synthesizes a deterministic `embedding [1, T_TOK, 768]` for trace +
    parity. Real conditioning at inference time is bert_dur padded to
    the same T_TOK. The graph is `T_TOK`-static so embeddings of the
    matching shape are required.
    """
    import coremltools as ct

    print(f"\n=== fused_diffusion_sampler bucket T={t_tok} ({precision}) ===")
    fused = FusedDiffusionSampler(rt.model.diffusion.diffusion)
    g = torch.Generator()
    g.manual_seed(seed)
    noise_init = torch.randn(1, 1, 256, generator=g, dtype=torch.float32)
    noises_aux = torch.randn(4, 1, 1, 256, generator=g, dtype=torch.float32)
    embedding = torch.randn(1, t_tok, 768, generator=g, dtype=torch.float32)
    features = torch.randn(1, 256, generator=g, dtype=torch.float32)
    inputs = (noise_init, noises_aux, embedding, features)
    print(f"  embedding axis: T={t_tok}")

    with torch.no_grad():
        eager = fused(*inputs)
    print(f"  eager out: {tuple(eager.shape)} {eager.dtype}")

    print("  tracing ...")
    fused.eval()
    with torch.no_grad():
        traced = torch.jit.trace(fused, inputs, check_trace=False, strict=False)

    ct_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )
    descs = [
        ct.TensorType(name="noise_init", shape=tuple(noise_init.shape), dtype=np.float32),
        ct.TensorType(name="noises_aux", shape=tuple(noises_aux.shape), dtype=np.float32),
        ct.TensorType(name="embedding", shape=tuple(embedding.shape), dtype=np.float32),
        ct.TensorType(name="features", shape=tuple(features.shape), dtype=np.float32),
    ]
    print(f"  ct.convert ({precision}, fixed T={t_tok}) ...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=descs,
        convert_to="mlprogram",
        compute_precision=ct_precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  ct.convert: {time.time() - t0:.1f}s")

    suffix_prec = "_fp16" if precision == "fp16" else ""
    out_path = (
        PACKAGES_DIR
        / f"fused_diffusion_sampler{suffix_prec}_t{t_tok}.mlpackage"
    )
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    size_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / 1e6
    print(f"  saved {out_path.relative_to(HERE)} ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parse_buckets(s: str) -> list[int]:
    out = sorted({int(x) for x in s.split(",") if x.strip()})
    if not out:
        raise ValueError("no buckets parsed")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buckets",
        default="64,128,256",
        help="comma-separated T_TOK sizes (default 64,128,256)",
    )
    parser.add_argument(
        "--stages",
        default="bert,sampler",
        help="comma-separated subset of {bert, sampler} (default both)",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "fp32"],
        help="output precision (default fp16, matches iteration_3)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    buckets = _parse_buckets(args.buckets)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    bad = stages - {"bert", "sampler"}
    if bad:
        raise SystemExit(f"unknown stages: {sorted(bad)}")

    print(f"buckets: {buckets}")
    print(f"stages: {sorted(stages)}")
    print(f"precision: {args.precision}")

    rt = build_runtime()
    real_T = int(rt.captures.tokens.shape[1])
    print(f"\nRuntime ready: captured tokens T={real_T}")
    print("(buckets must all be >= captured T to pad cleanly)")

    failures: list[tuple[str, int, str]] = []
    summary: list[tuple[str, int, Path, float]] = []
    for t_tok in buckets:
        if "bert" in stages:
            try:
                p = convert_bert_bucket(rt, t_tok=t_tok, precision=args.precision)
                size_mb = sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1e6
                summary.append(("bert", t_tok, p, size_mb))
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                failures.append(("bert", t_tok, f"{type(e).__name__}: {e}"))
        if "sampler" in stages:
            try:
                p = convert_sampler_bucket(
                    rt, t_tok=t_tok, precision=args.precision, seed=args.seed
                )
                size_mb = sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) / 1e6
                summary.append(("sampler", t_tok, p, size_mb))
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                failures.append(("sampler", t_tok, f"{type(e).__name__}: {e}"))

    print("\n=== summary ===")
    print(f"  {'stage':<10} {'T_TOK':>6} {'size (MB)':>12}  path")
    total = 0.0
    for stage, t_tok, p, mb in summary:
        print(f"  {stage:<10} {t_tok:>6} {mb:>10.1f}    {p.relative_to(HERE)}")
        total += mb
    print(f"  {'TOTAL':<10} {'':>6} {total:>10.1f} MB")

    if failures:
        print("\n=== failures ===")
        for stage, t_tok, msg in failures:
            print(f"  {stage:<10} T={t_tok}  {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
