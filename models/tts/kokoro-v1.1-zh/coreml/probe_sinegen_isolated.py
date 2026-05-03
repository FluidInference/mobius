"""Isolate the divergence source inside CoreMLSineGenV2.

Build standalone CoreML models for each candidate cause and find which one
diverges from PyTorch:
    M0. Just `sin` of a fixed input tensor (CoreML sin op accuracy)
    M1. sin(cumsum(rad_values_down))                 — sin precision @ 39k rad
    M2. interpolate(linear) of a precomputed tensor   — interp accuracy
    M3. Full CoreMLSineGenV2 (current convert.py code)

For each, compare CoreML output to PyTorch on the same input.

Usage:
    uv run python probe_sinegen_isolated.py
"""
import math
import pathlib
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def diff(a, b, label):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    rms_d = float(np.sqrt(np.mean((a - b) ** 2)))
    rel = rms_d / max(rms_a, 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1]) if rms_a > 0 and rms_b > 0 else 0.0
    print(f"  [{label:55s}] rel={rel:.3e} corr={corr:.7f}  rms_pt={rms_a:.3e} rms_cm={rms_b:.3e}")


def convert(model, name, input_tensors, input_specs, output_names):
    """Trace, convert, save, return CoreML model."""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, input_tensors, strict=False)
    ml = ct.convert(traced,
                    inputs=input_specs,
                    outputs=[ct.TensorType(name=n) for n in output_names],
                    convert_to="mlprogram",
                    minimum_deployment_target=ct.target.iOS17,
                    compute_precision=ct.precision.FLOAT32,
                    compute_units=ct.ComputeUnit.CPU_ONLY)
    out_path = f"build/probe/{name}.mlpackage"
    pathlib.Path("build/probe").mkdir(parents=True, exist_ok=True)
    ml.save(out_path)
    return ct.models.MLModel(out_path, compute_units=ct.ComputeUnit.CPU_ONLY)


def run_pytorch(module, *inputs):
    module.eval()
    with torch.no_grad():
        return module(*inputs)


# Build a realistic input similar to the noise-stage input.
upsample_scale = 300
sampling_rate = 24000
harmonic_num = 8
T_a = 139  # T from the actual zm_009 test
T2 = T_a * 2
F0 = (100 + 50 * torch.sin(torch.arange(T2).float() / 30)).abs().clamp(min=10).unsqueeze(0)
upsample = nn.Upsample(scale_factor=upsample_scale)
f0_full = upsample(F0[:, None]).transpose(1, 2)  # (1, T2*300, 1)
T_full = f0_full.shape[1]
print(f"f0_full: {tuple(f0_full.shape)}, T_full={T_full}")
harmonics = torch.arange(1, harmonic_num + 2, dtype=torch.float32)
fn = f0_full * harmonics.view(1, 1, -1)


# ───────── M0: sin only (input = some large precomputed phase tensor) ─────────
print("\n=== M0: pure sin of large argument ===")
class JustSin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# Create an input with magnitudes ranging up to ~22000 rad
phase_test = torch.linspace(0, 22000, 10000).unsqueeze(0).unsqueeze(0)  # (1, 1, 10000)
m0 = convert(JustSin(), "M0_sin",
             (phase_test,),
             [ct.TensorType(name="x", shape=(1, 1, 10000), dtype=np.float32)],
             ["y"])
pt0 = run_pytorch(JustSin(), phase_test).numpy()
cm0 = np.array(m0.predict({"x": phase_test.numpy().astype(np.float32)})["y"])
diff(pt0, cm0, "JustSin: sin(0..22000 rad)")

# Same with bounded input
phase_test_bounded = torch.linspace(0, 2 * math.pi, 10000).unsqueeze(0).unsqueeze(0)
pt0b = run_pytorch(JustSin(), phase_test_bounded).numpy()
cm0b = np.array(m0.predict({"x": phase_test_bounded.numpy().astype(np.float32)})["y"])
diff(pt0b, cm0b, "JustSin: sin(0..2π rad)")


# ───────── M1: cumsum + scale + sin (no interpolate) ─────────
print("\n=== M1: cumsum-then-sin pattern (no interpolate) ===")
class CumsumSin(nn.Module):
    def __init__(self, sr, us):
        super().__init__()
        self.sr = sr
        self.us = us
    def forward(self, fn):
        rad_values = fn / self.sr
        rv = rad_values.transpose(1, 2)
        rv_down = F.avg_pool1d(rv, kernel_size=self.us, stride=self.us)
        rad_down = rv_down.transpose(1, 2)
        phase = torch.cumsum(rad_down, dim=1) * 2 * math.pi
        ph = phase.transpose(1, 2) * self.us
        return torch.sin(ph)

T2_dim = ct.RangeDim(lower_bound=2, upper_bound=4000, default=T2)
m1 = convert(CumsumSin(sampling_rate, upsample_scale), "M1_cumsum_sin",
             (fn,),
             [ct.TensorType(name="fn", shape=(1, T_full, harmonic_num + 1), dtype=np.float32)],
             ["y"])
pt1 = run_pytorch(CumsumSin(sampling_rate, upsample_scale), fn).numpy()
cm1 = np.array(m1.predict({"fn": fn.numpy().astype(np.float32)})["y"])
diff(pt1, cm1, "M1: sin(cumsum*2π*US) — no interp")


# ───────── M2: cumsum + scale + WRAP + sin (no interpolate, with wrap) ─────────
print("\n=== M2: cumsum-then-WRAP-then-sin (no interpolate) ===")
class CumsumWrapSin(nn.Module):
    def __init__(self, sr, us):
        super().__init__()
        self.sr = sr
        self.us = us
    def forward(self, fn):
        rad_values = fn / self.sr
        rv = rad_values.transpose(1, 2)
        rv_down = F.avg_pool1d(rv, kernel_size=self.us, stride=self.us)
        rad_down = rv_down.transpose(1, 2)
        cycles_down = torch.cumsum(rad_down, dim=1)
        cyc = cycles_down.transpose(1, 2) * self.us
        cyc_frac = cyc - torch.floor(cyc)
        return torch.sin(cyc_frac * 2 * math.pi)

m2 = convert(CumsumWrapSin(sampling_rate, upsample_scale), "M2_cumsum_wrap_sin",
             (fn,),
             [ct.TensorType(name="fn", shape=(1, T_full, harmonic_num + 1), dtype=np.float32)],
             ["y"])
pt2 = run_pytorch(CumsumWrapSin(sampling_rate, upsample_scale), fn).numpy()
cm2 = np.array(m2.predict({"fn": fn.numpy().astype(np.float32)})["y"])
diff(pt2, cm2, "M2: sin((cumsum*US) - floor) — wrap, no interp")


# ───────── M3: cumsum + scale + interpolate + sin (current original) ─────────
print("\n=== M3: cumsum + interpolate + sin (CURRENT original code) ===")
class CumsumInterpSin(nn.Module):
    def __init__(self, sr, us):
        super().__init__()
        self.sr = sr
        self.us = us
    def forward(self, fn):
        rad_values = fn / self.sr
        rv = rad_values.transpose(1, 2)
        rv_down = F.avg_pool1d(rv, kernel_size=self.us, stride=self.us)
        rad_down = rv_down.transpose(1, 2)
        phase = torch.cumsum(rad_down, dim=1) * 2 * math.pi
        ph = phase.transpose(1, 2) * self.us
        ph_up = F.interpolate(ph, scale_factor=float(self.us), mode="linear", align_corners=False)
        return torch.sin(ph_up.transpose(1, 2))

m3 = convert(CumsumInterpSin(sampling_rate, upsample_scale), "M3_cumsum_interp_sin",
             (fn,),
             [ct.TensorType(name="fn", shape=(1, T_full, harmonic_num + 1), dtype=np.float32)],
             ["y"])
pt3 = run_pytorch(CumsumInterpSin(sampling_rate, upsample_scale), fn).numpy()
cm3 = np.array(m3.predict({"fn": fn.numpy().astype(np.float32)})["y"])
diff(pt3, cm3, "M3: sin(interp(cumsum*2π*US)) — full pipeline")


# ───────── M4: cumsum + interpolate + WRAP + sin (proposed fix) ─────────
print("\n=== M4: cumsum + interpolate + WRAP + sin (FIX) ===")
class CumsumInterpWrapSin(nn.Module):
    def __init__(self, sr, us):
        super().__init__()
        self.sr = sr
        self.us = us
    def forward(self, fn):
        rad_values = fn / self.sr
        rv = rad_values.transpose(1, 2)
        rv_down = F.avg_pool1d(rv, kernel_size=self.us, stride=self.us)
        rad_down = rv_down.transpose(1, 2)
        cycles_down = torch.cumsum(rad_down, dim=1)
        cyc = cycles_down.transpose(1, 2) * self.us
        cyc_up = F.interpolate(cyc, scale_factor=float(self.us), mode="linear", align_corners=False)
        cyc_full = cyc_up.transpose(1, 2)
        cyc_frac = cyc_full - torch.floor(cyc_full)
        return torch.sin(cyc_frac * 2 * math.pi)

m4 = convert(CumsumInterpWrapSin(sampling_rate, upsample_scale), "M4_cumsum_interp_wrap_sin",
             (fn,),
             [ct.TensorType(name="fn", shape=(1, T_full, harmonic_num + 1), dtype=np.float32)],
             ["y"])
pt4 = run_pytorch(CumsumInterpWrapSin(sampling_rate, upsample_scale), fn).numpy()
cm4 = np.array(m4.predict({"fn": fn.numpy().astype(np.float32)})["y"])
diff(pt4, cm4, "M4: sin(((interp(cumsum*US)) - floor) * 2π) — FIX candidate")


# ───────── M5: just interpolate (linear) — does CoreML's interp diverge? ─────────
print("\n=== M5: interpolate-only (CoreML linear interp accuracy) ===")
class JustInterpolate(nn.Module):
    def __init__(self, us):
        super().__init__()
        self.us = us
    def forward(self, x):
        return F.interpolate(x, scale_factor=float(self.us), mode="linear", align_corners=False)

# Input: large linearly-increasing tensor like phase
T_down = T2  # downsampled length
phase_lin = torch.cumsum(torch.full((1, 1, T_down), 0.075 * 2 * math.pi * upsample_scale), dim=2)
print(f"  phase_lin shape={tuple(phase_lin.shape)}, max={phase_lin.max().item():.0f}")
m5 = convert(JustInterpolate(upsample_scale), "M5_interp",
             (phase_lin,),
             [ct.TensorType(name="x", shape=(1, 1, T_down), dtype=np.float32)],
             ["y"])
pt5 = run_pytorch(JustInterpolate(upsample_scale), phase_lin).numpy()
cm5 = np.array(m5.predict({"x": phase_lin.numpy().astype(np.float32)})["y"])
diff(pt5, cm5, "M5: just interpolate(linear, scale=300) on 0..40000 ramp")


# ───────── M6: cumsum + scale + interpolate (no sin) ─────────
print("\n=== M6: cumsum + interpolate (no sin) — same input as M3 but expose phase ===")
class CumsumInterp(nn.Module):
    def __init__(self, sr, us):
        super().__init__()
        self.sr = sr
        self.us = us
    def forward(self, fn):
        rad_values = fn / self.sr
        rv = rad_values.transpose(1, 2)
        rv_down = F.avg_pool1d(rv, kernel_size=self.us, stride=self.us)
        rad_down = rv_down.transpose(1, 2)
        phase = torch.cumsum(rad_down, dim=1) * 2 * math.pi
        ph = phase.transpose(1, 2) * self.us
        ph_up = F.interpolate(ph, scale_factor=float(self.us), mode="linear", align_corners=False)
        return ph_up.transpose(1, 2)

m6 = convert(CumsumInterp(sampling_rate, upsample_scale), "M6_cumsum_interp_phase",
             (fn,),
             [ct.TensorType(name="fn", shape=(1, T_full, harmonic_num + 1), dtype=np.float32)],
             ["y"])
pt6 = run_pytorch(CumsumInterp(sampling_rate, upsample_scale), fn).numpy()
cm6 = np.array(m6.predict({"fn": fn.numpy().astype(np.float32)})["y"])
diff(pt6, cm6, "M6: phase only (cumsum+interp) — pre-sin")
print(f"  pt6 max={np.abs(pt6).max():.1f}  cm6 max={np.abs(cm6).max():.1f}")
