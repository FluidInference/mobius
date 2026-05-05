"""Progressive nanocodec subgraph probe.

Goal: find the size/structure threshold at which the patched (Taylor5Clipped)
nanocodec graph stops compiling for ANE (i.e. transitions from a clean
ANE plan to ``ANECCompile() FAILED`` / catch-all "ANE not available").

We do this by building progressively larger HiFi-GAN-style subgraphs that
mirror the magpie nanocodec decoder body (5 upsample stages, channels
864 -> 27, total 96 Snake instances) but vary in depth, then converting
each with the existing trace -> ct.convert -> xcrun compile -> coreml-cli
pipeline.

Specs (in increasing complexity):

  res_block_27          : 1 ResidualBlock (Conv1d -> Snake -> Conv1d, +residual) @ C=27
  hifigan_resblock_27   : HiFiGANResBlock = 3 ResidualBlocks chained (dilations 1,3,5)
  hifigan_reslayer_27   : HiFiGANResLayer = 3 HiFiGANResBlocks averaged (kernels 3,7,11)
  stage_27              : Activation -> ConvTranspose1d -> ResLayer (single stage, smallest channels)
  body_2stage           : last 2 stages chained (54->27 -> 27)
  body_3stage           : last 3 stages chained (108->54->27 -> 27)
  body_4stage           : last 4 stages chained (216->...->27)
  body_5stage           : full body (864->...->27)
  full_decoder          : pre_conv + body_5stage + post_act + post_conv + tanh

Each spec is converted from a small synthetic build (no NeMo dependency at
trace time) using Taylor5Clipped Snake activation. We keep convolution
weights random — we only care about static graph topology / ANE plan.

Run:
    uv run python nanocodec_experiments/nano_subgraph_probe.py
    uv run python nanocodec_experiments/nano_subgraph_probe.py --only stage_27 body_2stage
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "build" / "subgraph_packages"
MLC_DIR = HERE / "build" / "subgraph_compiled"
RESULTS_DIR = HERE / "results"
RAW_DIR = RESULTS_DIR / "subgraph_raw"
LEDGER_PATH = RESULTS_DIR / "subgraph_ledger.json"

COREML_CLI_DIR = HERE.parent.parent.parent.parent.parent / "tools" / "coreml-cli"


# ---------------------------------------------------------------------------
# Building blocks (mirror NeMo HiFi-GAN decoder structure with Taylor5Clipped)
# ---------------------------------------------------------------------------

_HALF_PI = 1.5707963267948966


class SnakeTaylor5Clipped(nn.Module):
    """Taylor-5 expansion of x + (1/α)·sin²(α·x) clamped at α·x = ±π/2."""
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        a = self.alpha
        ax = torch.clamp(a * x, -_HALF_PI, _HALF_PI)
        y2 = ax * ax
        y4 = y2 * y2
        y6 = y4 * y2
        sin2 = y2 - y4 / 3.0 + 2.0 * y6 / 45.0
        return x + sin2 / (a + 1e-9)


def _conv1d(in_c: int, out_c: int, k: int, dilation: int = 1, bias: bool = True) -> nn.Conv1d:
    pad = ((k - 1) * dilation) // 2
    return nn.Conv1d(in_c, out_c, kernel_size=k, dilation=dilation, padding=pad, bias=bias)


class ResidualBlock(nn.Module):
    """Conv1d -> Snake -> Conv1d with residual add (matches NeMo HiFiGAN ResidualBlock)."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.input_act  = SnakeTaylor5Clipped(channels)
        self.input_conv = _conv1d(channels, channels, kernel_size, dilation=dilation)
        self.skip_act   = SnakeTaylor5Clipped(channels)
        self.skip_conv  = _conv1d(channels, channels, kernel_size, dilation=1)

    def forward(self, x):
        y = self.input_act(x)
        y = self.input_conv(y)
        y = self.skip_act(y)
        y = self.skip_conv(y)
        return x + y


class HiFiGANResBlock(nn.Module):
    """3 ResidualBlocks chained, dilations [1, 3, 5]."""
    def __init__(self, channels: int, kernel_size: int = 3, dilations=(1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class HiFiGANResLayer(nn.Module):
    """3 HiFiGANResBlocks averaged across kernel sizes [3, 7, 11]."""
    def __init__(self, channels: int, kernel_sizes=(3, 7, 11), dilations=(1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList([
            HiFiGANResBlock(channels, kernel_size=k, dilations=dilations)
            for k in kernel_sizes
        ])

    def forward(self, x):
        outs = [b(x) for b in self.blocks]
        return sum(outs) / float(len(outs))


class UpsampleStage(nn.Module):
    """Activation -> ConvTranspose1d -> HiFiGANResLayer (one decoder upsample stage)."""
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int):
        super().__init__()
        self.act = SnakeTaylor5Clipped(in_c)
        # ConvTranspose1d with same-padding. Kernel = 2*stride per HiFiGAN.
        pad = (kernel - stride) // 2
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad)
        self.res_layer = HiFiGANResLayer(out_c)

    def forward(self, x):
        x = self.act(x)
        x = self.up(x)
        x = self.res_layer(x)
        return x


# magpie nanocodec decoder shapes, observed from loaded model:
# stage:        in    out   kernel  stride
# stage[0]:    864 -> 432   16      8
# stage[1]:    432 -> 216   16      8
# stage[2]:    216 -> 108    8      4
# stage[3]:    108 ->  54    4      2
# stage[4]:     54 ->  27    4      2
STAGES = [
    (864, 432, 16, 8),
    (432, 216, 16, 8),
    (216, 108,  8, 4),
    (108,  54,  4, 2),
    ( 54,  27,  4, 2),
]
INPUT_DIM = 432   # Magpie codec encoded_dim (768 → 864 via pre_conv? actual fits in graph)


class Body(nn.Module):
    """Concatenation of the last `n_stages` upsample stages."""
    def __init__(self, n_stages: int):
        super().__init__()
        if n_stages < 1 or n_stages > len(STAGES):
            raise ValueError(n_stages)
        used = STAGES[-n_stages:]
        self.stages = nn.ModuleList([
            UpsampleStage(in_c, out_c, k, s) for (in_c, out_c, k, s) in used
        ])
        self.in_channels = used[0][0]

    def forward(self, x):
        for s in self.stages:
            x = s(x)
        return x


class FullDecoder(nn.Module):
    """pre_conv -> 5-stage body -> post_act -> post_conv -> tanh."""
    def __init__(self, input_dim: int = INPUT_DIM, pre_kernel: int = 7):
        super().__init__()
        first_in = STAGES[0][0]
        self.pre_conv = _conv1d(input_dim, first_in, pre_kernel)
        self.body = Body(n_stages=len(STAGES))
        last_out = STAGES[-1][1]
        self.post_act = SnakeTaylor5Clipped(last_out)
        self.post_conv = _conv1d(last_out, 1, 3)
        self.out_act = nn.Tanh()

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.body(x)
        x = self.post_act(x)
        x = self.post_conv(x)
        return self.out_act(x)


# ---------------------------------------------------------------------------
# Spec table
# ---------------------------------------------------------------------------

# Frame counts: codec produces 1024 samples per token at 21.5 fps.
# Use T_in such that the full decoder output matches a few hundred ms of audio.
# Stages strides multiply: 8*8*4*2*2 = 256×.
# For a body that uses last N stages, total stride = product of those strides.
_STRIDES = [s for *_, s in STAGES]


def _t_in_for_body(n_stages: int, t_out: int = 4096) -> int:
    """Pick T such that t_out (post-stages) is roughly fixed.

    body uses last n_stages of STAGES; stride = product of their strides.
    """
    used = _STRIDES[-n_stages:]
    total_stride = 1
    for s in used:
        total_stride *= s
    return max(1, t_out // total_stride)


def build_specs() -> list[tuple]:
    """Return list of (name, builder, input_shape, notes)."""
    specs: list[tuple] = []

    # Smallest channel block (27ch) — should match Phase A snake_taylor5_clipped_block
    specs.append((
        "res_block_27",
        lambda: ResidualBlock(27).eval(),
        (1, 27, 1024),
        "1 ResidualBlock @ C=27 (Conv→Snake→Conv + skip)",
    ))
    specs.append((
        "hifigan_resblock_27",
        lambda: HiFiGANResBlock(27).eval(),
        (1, 27, 1024),
        "HiFiGANResBlock = 3 ResBlocks chained (dilations 1,3,5)",
    ))
    specs.append((
        "hifigan_reslayer_27",
        lambda: HiFiGANResLayer(27).eval(),
        (1, 27, 1024),
        "HiFiGANResLayer = 3 ResBlocks averaged (kernels 3,7,11)",
    ))

    # Single upsample stage at smallest channels
    in_c, out_c, k, s = STAGES[-1]
    t_in = _t_in_for_body(1)
    specs.append((
        "stage_27",
        lambda in_c=in_c, out_c=out_c, k=k, s=s: UpsampleStage(in_c, out_c, k, s).eval(),
        (1, in_c, t_in),
        f"single upsample stage {in_c}->{out_c} k={k} s={s}",
    ))

    # Progressive body: last N stages
    for n in (2, 3, 4, 5):
        t_in = _t_in_for_body(n)
        in_c = STAGES[-n][0]
        specs.append((
            f"body_{n}stage",
            lambda n=n: Body(n).eval(),
            (1, in_c, t_in),
            f"last {n} stages (in_c={in_c}, T={t_in})",
        ))

    # Full decoder (pre_conv + body_5stage + post_act + post_conv + tanh)
    specs.append((
        "full_decoder",
        lambda: FullDecoder().eval(),
        (1, INPUT_DIM, 256),
        f"pre_conv + 5 stages + post (input_dim={INPUT_DIM}, T=256)",
    ))

    # Bisection: body_5stage at large T (matching full_decoder body output)
    # body_5stage @ T=8 was 99% ANE (output T=4096).
    # full_decoder @ T=256 was 0% ANE  (output T=262144).
    # Test body alone with the same large-T input that full_decoder uses post-pre_conv.
    specs.append((
        "body_5stage_T256",
        lambda: Body(5).eval(),
        (1, STAGES[0][0], 256),
        "body_5stage with input T=256 (output T=262144). Memory test vs T=8 baseline.",
    ))

    # Bisection: body_5stage at intermediate T to find the size threshold
    specs.append((
        "body_5stage_T32",
        lambda: Body(5).eval(),
        (1, STAGES[0][0], 32),
        "body_5stage with input T=32  (output T=16384).",
    ))
    specs.append((
        "body_5stage_T64",
        lambda: Body(5).eval(),
        (1, STAGES[0][0], 64),
        "body_5stage with input T=64  (output T=32768).",
    ))
    specs.append((
        "body_5stage_T128",
        lambda: Body(5).eval(),
        (1, STAGES[0][0], 128),
        "body_5stage with input T=128 (output T=65536).",
    ))

    # Bisection: full_decoder at small T to test if memory alone causes failure
    specs.append((
        "full_decoder_T8",
        lambda: FullDecoder().eval(),
        (1, INPUT_DIM, 8),
        "full_decoder with input T=8 (output T=2048). If 99% ANE, failure is purely memory.",
    ))

    # Refine threshold: T=8 (out 4096) passes, T=32 (out 16384) fails.
    # Probe T=16 (out 8192), T=24 (out 12288), T=20 (out 10240).
    specs.append((
        "body_5stage_T16",
        lambda: Body(5).eval(),
        (1, STAGES[0][0], 16),
        "body_5stage with input T=16 (output T=8192).",
    ))
    specs.append((
        "body_5stage_T20",
        lambda: Body(5).eval(),
        (1, STAGES[0][0], 20),
        "body_5stage with input T=20 (output T=10240).",
    ))
    specs.append((
        "body_5stage_T24",
        lambda: Body(5).eval(),
        (1, STAGES[0][0], 24),
        "body_5stage with input T=24 (output T=12288).",
    ))

    return specs


# ---------------------------------------------------------------------------
# Conversion / probe pipeline (matches analyze.py)
# ---------------------------------------------------------------------------

def trace_and_convert(model: nn.Module, input_shape: tuple, pkg_path: Path) -> None:
    x = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, (x,), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=input_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="y")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    if pkg_path.exists():
        shutil.rmtree(pkg_path)
    pkg_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(pkg_path))


def compile_to_mlmodelc(pkg_path: Path, mlc_dir: Path) -> Path:
    if mlc_dir.exists():
        shutil.rmtree(mlc_dir)
    mlc_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(mlc_dir)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = mlc_dir / f"{pkg_path.stem}.mlmodelc"
    if not out.exists():
        raise RuntimeError(f"compile produced no mlmodelc at {out}")
    return out


def run_fallback_json(mlmodelc: Path, plan_timeout: float = 600.0) -> dict:
    cmd = [
        "uv", "run", "coreml-cli",
        str(mlmodelc),
        "--fallback",
        "--json",
        "--plan-timeout", str(plan_timeout),
    ]
    proc = subprocess.run(
        cmd, cwd=str(COREML_CLI_DIR), check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"coreml-cli failed for {mlmodelc.name}\n"
            f"stdout: {proc.stdout.decode(errors='ignore')}\n"
            f"stderr: {proc.stderr.decode(errors='ignore')}"
        )
    text = proc.stdout.decode(errors="ignore").strip()
    # Strip any trailing stderr leak after the JSON closes
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        idx = text.rfind("\n}\n")
        if idx >= 0:
            return json.loads(text[:idx + 2])
        idx = text.rfind("}")
        if idx >= 0:
            return json.loads(text[:idx + 1])
        raise


def summarize(fb: dict) -> dict:
    models = fb.get("models") or []
    fallback = (models[0] if models else {}).get("fallback") or {}
    total = fallback.get("total_ops")
    ane_ops = fallback.get("ane_ops")
    cpu_ops = fallback.get("cpu_ops")
    ane_pct = fallback.get("ane_percent")
    if ane_pct is None and total:
        ane_pct = round(100.0 * (ane_ops or 0) / total, 2)
    reasons = fallback.get("reasons") or []
    return {
        "ane_percent": ane_pct,
        "total_ops": total,
        "ane_ops": ane_ops,
        "cpu_ops": cpu_ops,
        "top_rejections": [
            (r.get("reason"), r.get("count"), r.get("op_types"))
            for r in reasons[:5]
        ],
    }


def run_one(name: str, builder: Callable, input_shape: tuple, notes: str) -> dict:
    pkg = PKG_DIR / f"{name}.mlpackage"
    mlc = MLC_DIR / name
    raw = RAW_DIR / f"{name}.json"

    print(f"\n[{name}] {notes}  input={input_shape}")
    t0 = time.time()
    model = builder()
    trace_and_convert(model, input_shape, pkg)
    t1 = time.time()
    mlmodelc = compile_to_mlmodelc(pkg, mlc)
    t2 = time.time()
    fb = run_fallback_json(mlmodelc)
    t3 = time.time()

    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(fb, indent=2))

    s = summarize(fb)
    s.update({
        "name": name,
        "notes": notes,
        "input_shape": list(input_shape),
        "timings_s": {
            "convert": round(t1 - t0, 2),
            "compile": round(t2 - t1, 2),
            "fallback": round(t3 - t2, 2),
        },
    })
    print(f"  -> ANE {s['ane_percent']}%  cpu_ops={s['cpu_ops']}/{s['total_ops']}  "
          f"convert={s['timings_s']['convert']}s "
          f"compile={s['timings_s']['compile']}s "
          f"fallback={s['timings_s']['fallback']}s")
    for reason, n, op_types in s["top_rejections"][:3]:
        ops_short = ", ".join(f"{k}:{v}" for k, v in (op_types or {}).items())
        print(f"     - {n} x {reason}  [{ops_short}]")
    return s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip", nargs="*", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_specs()
    if args.only:
        specs = [s for s in specs if s[0] in args.only]
    if args.skip:
        specs = [s for s in specs if s[0] not in args.skip]
    if not specs:
        print("No specs selected.", file=sys.stderr)
        return 1

    ledger = {"runs": []}
    if LEDGER_PATH.exists():
        try:
            ledger = json.loads(LEDGER_PATH.read_text())
        except Exception:
            ledger = {"runs": []}
    runs_by_name = {r.get("name"): r for r in ledger.get("runs", [])}

    for name, builder, shape, notes in specs:
        try:
            runs_by_name[name] = run_one(name, builder, shape, notes)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            runs_by_name[name] = {"name": name, "notes": notes, "error": str(e)}

    ledger["runs"] = list(runs_by_name.values())
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2))
    print(f"\nWrote {LEDGER_PATH}")

    print("\n=== Subgraph ANE summary ===")
    fmt = "{:24s}  {:>7s}  {:>14s}  {}"
    print(fmt.format("name", "ANE %", "cpu/total", "notes"))
    for r in ledger["runs"]:
        if "error" in r:
            print(fmt.format(r["name"], "ERR", "-", r.get("error", "")[:60]))
            continue
        pct = r.get("ane_percent")
        cpu = r.get("cpu_ops")
        total = r.get("total_ops")
        print(fmt.format(
            r["name"],
            f"{pct}" if pct is not None else "?",
            f"{cpu}/{total}" if cpu is not None else "?",
            r.get("notes", ""),
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
