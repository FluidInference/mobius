"""Trial 4 — convert the fused 5-step ADPM2 diffusion sampler.

Builds `coreml/packages/fused_diffusion_sampler{,_fp16}.mlpackage`.

Why
---
The Python ADPM2 loop dispatches `diffusion_unet` 8x per utterance
(num_steps=5 -> 4 iters x 2 calls each). The Karras schedule, the per-
iter `(sigma_up, sigma_down, sigma_mid)`, and the loop control flow are
all constants of `num_steps`. Bake them into a single traced graph and
dispatch the whole sampler in one call.

The 4 stochastic noise injections that the per-step path draws via
`torch.randn` are passed in as one `[4, 1, 1, 256]` input tensor so the
graph stays deterministic and the runner can reproduce the exact RNG
sequence by drawing those tensors up-front under a fixed seed.

Pseudocode (matches `coreml/fusions.md` Trial 4):

    fused_sampler(noise_init, noises_aux, embedding, features) -> s_pred
      x = sigmas[0] * noise_init                       # bake sigmas[0]
      for i in range(num_steps - 1):                   # baked, num_steps=5
          x_dn   = unet(x, sigmas[i], embedding, features)
          d      = (x - x_dn) / sigmas[i]
          x_mid  = x + d * (sigma_mids[i] - sigmas[i])
          x_mid_dn = unet(x_mid, sigma_mids[i], embedding, features)
          d_mid  = (x_mid - x_mid_dn) / sigma_mids[i]
          x      = x + d_mid * (sigma_downs[i] - sigmas[i])
          x      = x + noises_aux[i] * sigma_ups[i]    # stochastic aux noise
      return x

CoreML inputs / output
----------------------
| name        | shape           | dtype | meaning |
|-------------|-----------------|-------|---------|
| noise_init  | `[1, 1, 256]`   | f32   | `torch.randn(1, 256).unsqueeze(1)` at seed 0 |
| noises_aux  | `[4, 1, 1, 256]`| f32   | per-iter stochastic noise |
| embedding   | `[1, T_TOK, 768]` | f32 | `bert_dur` (padded to T_TOK) |
| features    | `[1, 256]`      | f32   | `ref_s` |

Output: `s_pred [1, 1, 256]`.

Token axis (T_TOK) is fixed at the captured shape (default 57). HF
Albert / sampler cross-attention reject `ct.RangeDim` on the token
axis — see `convert.py` notes on bert and diffusion_unet. To support
multiple bucket sizes, run this script once per bucket with --t-tok.

Run
---
    cd models/tts/styletts2
    uv run python coreml/exporters/fuse_diffusion_sampler.py
    uv run python coreml/exporters/fuse_diffusion_sampler.py --precision fp16
    uv run python coreml/exporters/fuse_diffusion_sampler.py --t-tok 128 --suffix _t128
"""

from __future__ import annotations

import argparse
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coreml.exporters import convert as _convert  # noqa: F401  (installs MIL patches)
from coreml._runtime import HERE, build_runtime
from coreml.wrappers import DiffusionDenoiseStepWrapper

PACKAGES_DIR = HERE / "coreml" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Karras + ADPM2 schedule helpers (mirror coreml/inference.py)
# ---------------------------------------------------------------------------


def _karras_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: float) -> torch.Tensor:
    rho_inv = 1.0 / rho
    steps = torch.arange(num_steps, dtype=torch.float32)
    sigmas = (
        sigma_max ** rho_inv
        + (steps / (num_steps - 1)) * (sigma_min ** rho_inv - sigma_max ** rho_inv)
    ) ** rho
    return torch.cat([sigmas, torch.zeros(1)])  # F.pad(..., value=0.0)


def _adpm2_get_sigmas(sigma: float, sigma_next: float, rho: float = 1.0):
    sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
    sigma_down = sqrt(max(sigma_next ** 2 - sigma_up ** 2, 0.0))
    sigma_mid = ((sigma ** (1 / rho) + sigma_down ** (1 / rho)) / 2) ** rho
    return sigma_up, sigma_down, sigma_mid


# ---------------------------------------------------------------------------
# Fused module
# ---------------------------------------------------------------------------


class FusedDiffusionSampler(nn.Module):
    """5-step ADPM2 sampler baked into one nn.Module.

    Reuses `DiffusionDenoiseStepWrapper` unchanged as the inner step
    (single source of truth for the per-call denoise math).
    """

    def __init__(
        self,
        kdiffusion: nn.Module,
        *,
        num_steps: int = 5,
        sigma_min: float = 0.0001,
        sigma_max: float = 3.0,
        rho_schedule: float = 9.0,
        rho_sampler: float = 1.0,
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")
        self.num_steps = int(num_steps)
        self.unet_step = DiffusionDenoiseStepWrapper(kdiffusion)

        sigmas = _karras_sigmas(num_steps, sigma_min, sigma_max, rho_schedule)
        sigma_ups = torch.zeros(num_steps - 1)
        sigma_downs = torch.zeros(num_steps - 1)
        sigma_mids = torch.zeros(num_steps - 1)
        for i in range(num_steps - 1):
            su, sd, sm = _adpm2_get_sigmas(
                float(sigmas[i]), float(sigmas[i + 1]), rho_sampler
            )
            sigma_ups[i] = su
            sigma_downs[i] = sd
            sigma_mids[i] = sm

        # Non-persistent buffers — baked into the trace as constants.
        self.register_buffer("sigmas", sigmas, persistent=False)
        self.register_buffer("sigma_ups", sigma_ups, persistent=False)
        self.register_buffer("sigma_downs", sigma_downs, persistent=False)
        self.register_buffer("sigma_mids", sigma_mids, persistent=False)
        self.eval()

    def forward(
        self,
        noise_init: torch.Tensor,   # [1, 1, 256]
        noises_aux: torch.Tensor,   # [num_steps - 1, 1, 1, 256]
        embedding: torch.Tensor,    # [1, T_TOK, 768]
        features: torch.Tensor,     # [1, 256]
    ) -> torch.Tensor:
        x = self.sigmas[0] * noise_init
        for i in range(self.num_steps - 1):
            sigma_i = self.sigmas[i].view(1)
            sigma_mid_i = self.sigma_mids[i].view(1)
            x_dn = self.unet_step(x, sigma_i, embedding, features)
            d = (x - x_dn) / sigma_i
            x_mid = x + d * (self.sigma_mids[i] - self.sigmas[i])
            x_mid_dn = self.unet_step(x_mid, sigma_mid_i, embedding, features)
            d_mid = (x_mid - x_mid_dn) / sigma_mid_i
            x = x + d_mid * (self.sigma_downs[i] - self.sigmas[i])
            x = x + noises_aux[i] * self.sigma_ups[i]
        return x


# ---------------------------------------------------------------------------
# Reference path (per-step ADPM2 loop in eager Python — for parity check)
# ---------------------------------------------------------------------------


def _reference_per_step(
    kdiffusion: nn.Module,
    noise_init: torch.Tensor,
    noises_aux: torch.Tensor,
    embedding: torch.Tensor,
    features: torch.Tensor,
    *,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho_schedule: float,
    rho_sampler: float,
) -> torch.Tensor:
    """Eager replica of `_adpm2_sample` in coreml/inference.py, but with
    the stochastic noises supplied as an input tensor (so this can be
    bit-compared to the fused module without RNG entanglement)."""
    step = DiffusionDenoiseStepWrapper(kdiffusion)
    sigmas = _karras_sigmas(num_steps, sigma_min, sigma_max, rho_schedule)
    x = sigmas[0] * noise_init
    for i in range(num_steps - 1):
        s_i = float(sigmas[i])
        s_n = float(sigmas[i + 1])
        s_up, s_dn, s_mid = _adpm2_get_sigmas(s_i, s_n, rho_sampler)
        sigma_i = torch.tensor([s_i], dtype=torch.float32)
        sigma_mid_t = torch.tensor([s_mid], dtype=torch.float32)
        x_dn = step(x, sigma_i, embedding, features)
        d = (x - x_dn) / s_i
        x_mid = x + d * (s_mid - s_i)
        x_mid_dn = step(x_mid, sigma_mid_t, embedding, features)
        d_mid = (x_mid - x_mid_dn) / s_mid
        x = x + d_mid * (s_dn - s_i)
        x = x + noises_aux[i] * s_up
    return x


# ---------------------------------------------------------------------------
# Convert + bench
# ---------------------------------------------------------------------------


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
        "rms_a": float(np.sqrt(np.mean(a * a))),
        "rms_b": float(np.sqrt(np.mean(b * b))),
        "cos": cos,
    }


def _build_inputs(t_tok: int, seed: int) -> tuple:
    """Synthesize the 4 trace inputs at a given token-axis size.

    `noise_init` and `noises_aux` are drawn under a fixed seed so eager
    and CoreML can be byte-compared. `embedding` and `features` are
    stand-ins (the parity check only needs *some* deterministic input;
    real conditioning comes from bert_dur + ref_s at inference time).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    noise_init = torch.randn(1, 1, 256, generator=g, dtype=torch.float32)
    noises_aux = torch.randn(4, 1, 1, 256, generator=g, dtype=torch.float32)
    embedding = torch.randn(1, t_tok, 768, generator=g, dtype=torch.float32)
    features = torch.randn(1, 256, generator=g, dtype=torch.float32)
    return noise_init, noises_aux, embedding, features


def convert_fused(
    *,
    precision: str,
    t_tok: int,
    suffix: str,
    seed: int,
    sigma_min: float,
    sigma_max: float,
    rho_schedule: float,
    rho_sampler: float,
    num_steps: int,
) -> Path:
    import coremltools as ct

    print(f"=== Trial 4 convert: fused_diffusion_sampler ({precision}, T={t_tok}) ===")
    rt = build_runtime()
    fused = FusedDiffusionSampler(
        rt.model.diffusion.diffusion,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho_schedule=rho_schedule,
        rho_sampler=rho_sampler,
    )

    inputs = _build_inputs(t_tok, seed)
    noise_init, noises_aux, embedding, features = inputs
    print(f"  noise_init = {tuple(noise_init.shape)} {noise_init.dtype}")
    print(f"  noises_aux = {tuple(noises_aux.shape)} {noises_aux.dtype}")
    print(f"  embedding  = {tuple(embedding.shape)} {embedding.dtype}")
    print(f"  features   = {tuple(features.shape)} {features.dtype}")

    # Eager parity: fused vs reference per-step path. The two should be
    # byte-equal because the only stochastic component (noises_aux) is
    # a shared input tensor.
    with torch.no_grad():
        eager_fused = fused(*inputs)
        eager_per_step = _reference_per_step(
            rt.model.diffusion.diffusion,
            *inputs,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho_schedule=rho_schedule,
            rho_sampler=rho_sampler,
        )
    m = _metric(eager_per_step.numpy(), eager_fused.numpy())
    print(
        f"  eager parity (fused vs per-step): cos={m['cos']:.6f} "
        f"max|d|={m['max_abs_delta']:.3e}"
    )
    if m["max_abs_delta"] > 1e-4:
        raise SystemExit(
            "ABORT: fused module does not match per-step reference. "
            "Investigate before converting."
        )

    print("  tracing ...")
    fused.eval()
    with torch.no_grad():
        traced = torch.jit.trace(fused, inputs, check_trace=False, strict=False)

    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
    ct_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )
    print(f"  ct.convert ({precision}, fixed shapes) ...")
    descs = [
        ct.TensorType(name="noise_init", shape=tuple(noise_init.shape), dtype=np.float32),
        ct.TensorType(name="noises_aux", shape=tuple(noises_aux.shape), dtype=np.float32),
        ct.TensorType(name="embedding", shape=tuple(embedding.shape), dtype=np.float32),
        ct.TensorType(name="features", shape=tuple(features.shape), dtype=np.float32),
    ]
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
    out_path = PACKAGES_DIR / f"fused_diffusion_sampler{suffix_prec}{suffix}.mlpackage"
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"  saved {out_path.relative_to(HERE)}")
    return out_path, inputs, eager_fused


def bench(out_path: Path, inputs: tuple, eager_out: torch.Tensor) -> None:
    import coremltools as ct

    feed = {
        "noise_init": inputs[0].detach().numpy().astype(np.float32),
        "noises_aux": inputs[1].detach().numpy().astype(np.float32),
        "embedding": inputs[2].detach().numpy().astype(np.float32),
        "features": inputs[3].detach().numpy().astype(np.float32),
    }
    eager_np = eager_out.detach().numpy().astype(np.float32)

    placements = [
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("ALL", ct.ComputeUnit.ALL),
    ]

    print("\n=== Trial 4 bench (fused_diffusion_sampler) ===")
    for name, units in placements:
        print(f"\n  --- {name} ---")
        t0 = time.time()
        m = ct.models.MLModel(str(out_path), compute_units=units)
        load_ms = (time.time() - t0) * 1000.0

        for _ in range(3):
            m.predict(feed)

        timings = []
        for _ in range(8):
            t1 = time.time()
            out = m.predict(feed)
            timings.append((time.time() - t1) * 1000.0)
        timings.sort()
        out_arr = np.asarray(list(out.values())[0])
        met = _metric(eager_np, out_arr)
        med = timings[len(timings) // 2]
        avg = sum(timings) / len(timings)
        spread = timings[-1] - timings[0]
        print(
            f"  load={load_ms:6.0f}ms  warm: min={timings[0]:6.1f} med={med:6.1f} "
            f"avg={avg:6.1f} max={timings[-1]:6.1f}  spread={spread:5.1f} ms"
        )
        print(
            f"  parity vs eager: cos={met['cos']:.6f}  max|d|={met['max_abs_delta']:.3e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="output mlpackage precision (default fp32)",
    )
    parser.add_argument(
        "--t-tok",
        type=int,
        default=57,
        help="token axis size baked into the graph (must match bert padding)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="filename suffix appended after the precision suffix (e.g. _t128)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--sigma-min", type=float, default=0.0001)
    parser.add_argument("--sigma-max", type=float, default=3.0)
    parser.add_argument("--rho-schedule", type=float, default=9.0)
    parser.add_argument("--rho-sampler", type=float, default=1.0)
    parser.add_argument("--no-bench", action="store_true", help="skip the bench loop")
    args = parser.parse_args()

    out_path, inputs, eager_out = convert_fused(
        precision=args.precision,
        t_tok=args.t_tok,
        suffix=args.suffix,
        seed=args.seed,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho_schedule=args.rho_schedule,
        rho_sampler=args.rho_sampler,
        num_steps=args.num_steps,
    )
    if not args.no_bench:
        bench(out_path, inputs, eager_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
