"""Numerical parity check: Snake polynomial replacements vs original sin² form.

For each polynomial Snake variant, sweep `α·x` over the codec operating range
and report:
  - max abs error vs original `x + (1/α)·sin²(α·x)`
  - mean abs error
  - max relative error (where |original| > 1e-3)

Codec activations after LayerNorm typically have α·x ∈ roughly [-2, 2]. A few
extreme values can exceed that, so we also sweep [-π, π] for full picture.

Run:
    uv run python nanocodec_experiments/snake_parity.py
"""
from __future__ import annotations

import math

import torch

from nanocodec_experiments import modules as mods


def snake_reference(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return x + (1.0 / (alpha + 1e-9)) * torch.sin(alpha * x).pow(2)


def evaluate(name: str, fn, x: torch.Tensor, alpha: torch.Tensor) -> dict:
    ref = snake_reference(x, alpha)
    out = fn(x, alpha)
    err = (out - ref).abs()
    denom = ref.abs().clamp(min=1e-3)
    rel = err / denom
    return {
        "name": name,
        "max_abs": err.max().item(),
        "mean_abs": err.mean().item(),
        "max_rel": rel.max().item(),
    }


def main() -> int:
    # x sweep: 4096 points across the range, alpha = 1 (typical trained value)
    alpha = torch.ones(1)

    ranges = {
        "codec [-2, 2]": (-2.0, 2.0),
        "moderate [-π/2, π/2]": (-math.pi / 2, math.pi / 2),
        "wide [-π, π]": (-math.pi, math.pi),
    }

    half_pi = math.pi / 2

    def taylor5_clipped(x, a):
        ax = torch.clamp(a * x, -half_pi, half_pi)
        y2 = ax * ax
        return x + (y2 - y2 ** 2 / 3.0 + 2.0 * y2 ** 3 / 45.0) / (a + 1e-9)

    variants = {
        "SnakePolyTaylor (3rd)": lambda x, a: x + a * x * x * (1.0 - (a * x) ** 2 / 3.0),
        "SnakeNoSinPow (3rd)": lambda x, a: x + a * x * x - (a * a * a / 3.0) * (x ** 4),
        "SnakeTaylor5 (5th)": lambda x, a: (
            x
            + a * x ** 2
            - (a ** 3 / 3.0) * x ** 4
            + (2.0 * a ** 5 / 45.0) * x ** 6
        ),
        "SnakeTaylor5Clipped": taylor5_clipped,
        "SnakeTaylor7 (7th)": lambda x, a: (
            x
            + a * x ** 2
            - (a ** 3 / 3.0) * x ** 4
            + (2.0 * a ** 5 / 45.0) * x ** 6
            - (8.0 * a ** 7 / 315.0) * x ** 8
        ),
    }

    print(f"{'variant':25s}  {'range':25s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'max_rel':>10s}")
    for range_name, (lo, hi) in ranges.items():
        x = torch.linspace(lo, hi, 4096)
        for vname, fn in variants.items():
            r = evaluate(vname, fn, x, alpha)
            print(f"{vname:25s}  {range_name:25s}  "
                  f"{r['max_abs']:10.4e}  {r['mean_abs']:10.4e}  {r['max_rel']:10.4e}")

    # Also report at varying alpha — check if higher alpha hurts higher-order more
    print("\nAt α=2 (post-training drift), codec [-1, 1]:")
    alpha2 = torch.full((1,), 2.0)
    x = torch.linspace(-1.0, 1.0, 4096)
    for vname, fn in variants.items():
        r = evaluate(vname, fn, x, alpha2)
        print(f"  {vname:25s}  max_abs={r['max_abs']:10.4e}  max_rel={r['max_rel']:10.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
