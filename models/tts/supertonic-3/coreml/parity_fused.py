"""Parity: fused 8-step VectorEstimator vs 8 sequential single-step calls.

Runs identical seeded inputs (masks=1) through:
  (a) the single-step reference model, looped 8x with denoised->noisy
      feedback (exactly what Supertonic3Synthesizer does), and
  (b) the fused model, one call,
both on CPU_ONLY (deterministic), and reports max_abs / mean_abs / SNR of
the FINAL latent. fp16 LSD is precision-sensitive across 8 accumulated
steps; <5e-2 max_abs on the final latent is the acceptance band.

Both .mlpackage and compiled .mlmodelc paths are accepted (the shipped
HF-cache artifacts are mlmodelc).

Usage:
    python3.11 -m coreml.parity_fused SINGLE_STEP_MODEL FUSED_MODEL [--steps 8] [--seeds 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np

L_DIM = 144


def _load(path: Path, units=ct.ComputeUnit.CPU_ONLY):
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(path), compute_units=units)
    return ct.models.MLModel(str(path), compute_units=units)


def _inputs(seed: int, L: int, T: int):
    rng = np.random.default_rng(seed)
    return {
        "noisy_latent": rng.standard_normal((1, L_DIM, L)).astype(np.float32),
        "text_emb": rng.standard_normal((1, 256, T)).astype(np.float32),
        "style_ttl": rng.standard_normal((1, 50, 256)).astype(np.float32),
        "latent_mask": np.ones((1, 1, L), dtype=np.float32),
        "text_mask": np.ones((1, 1, T), dtype=np.float32),
    }


def _run_loop(model, feeds: dict, steps: int) -> np.ndarray:
    total = np.full((1,), float(steps), dtype=np.float32)
    noisy = feeds["noisy_latent"]
    for k in range(steps):
        out = model.predict({
            **feeds,
            "noisy_latent": noisy.astype(np.float32),
            "current_step": np.full((1,), float(k), dtype=np.float32),
            "total_step": total,
        })
        noisy = out["denoised_latent"]
    return noisy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("single", type=Path)
    p.add_argument("fused", type=Path)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--L", type=int, default=128)
    p.add_argument("--T", type=int, default=128)
    args = p.parse_args()

    single = _load(args.single)
    fused = _load(args.fused)

    print(f"single: {args.single.name}   fused: {args.fused.name}   steps={args.steps}")
    print(f"{'seed':>4} {'max_abs':>10} {'mean_abs':>10} {'SNR_dB':>8} {'ref_std':>8}")
    print("-" * 48)
    worst = 0.0
    for seed in range(args.seeds):
        feeds = _inputs(seed, args.L, args.T)
        ref = _run_loop(single, feeds, args.steps)
        out = fused.predict(feeds)["denoised_latent"]
        diff = (out - ref).astype(np.float64)
        max_abs = float(np.abs(diff).max())
        mean_abs = float(np.abs(diff).mean())
        snr = 10 * np.log10((ref.astype(np.float64) ** 2).mean() / max((diff ** 2).mean(), 1e-30))
        worst = max(worst, max_abs)
        print(f"{seed:>4} {max_abs:>10.3e} {mean_abs:>10.3e} {snr:>8.1f} {ref.std():>8.4f}")
    band = 5e-2
    print(f"\nworst max_abs = {worst:.3e}  ({'WITHIN' if worst < band else 'OUTSIDE'} the {band:g} band)")


if __name__ == "__main__":
    main()
