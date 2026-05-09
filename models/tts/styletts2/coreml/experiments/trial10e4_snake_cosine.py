"""Trial 10e4 — Snake → cosine-identity rewrite, ship candidate.

Trial 10e3 ablation 2 (Snake → identity) confirmed Snake activation is
the planner partitioning trigger. That ablation drops audio quality
entirely — it's diagnostic-only.

The weight-preserving fix is Kokoro-ANE's cosine-identity rewrite:

    x + (1/α) sin²(αx)  ≡  x + (1 - cos(2αx)) / (2α)

mathematically identical at fp32 (validated bit-equivalent in Phase 3b
of an earlier StyleTTS2 push). This trial converts decoder_upsample at
T_mel=50 with the cosine-identity rewrite applied to AdaINResBlock1 and
Generator inline Snakes, then probes ANE acceptance.

If ANE still accepts (calling_count drops vs baseline 180, no E5RT
FAILED in stderr, fast warm predict), the rewrite is ship-ready and
issue #59 outcome (1) lands.

Run:
    cd models/tts/styletts2
    uv run python coreml/experiments/trial10e4_snake_cosine.py
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coreml.exporters import convert as _convert  # noqa: F401  (installs MIL patches)
from coreml._runtime import HERE, build_runtime, stage_example_inputs
from coreml.wrappers import build_wrapper

import coremltools as ct

PACKAGES_DIR = HERE / "coreml" / "packages"
T_MEL = 50
HOP = 300


def install_snake_cosine_resblock() -> None:
    """Patch AdaINResBlock1.forward to use cos-identity for both Snake calls."""
    from Modules.hifigan import AdaINResBlock1  # type: ignore
    if hasattr(AdaINResBlock1, "_e3_forward_original"):
        return
    AdaINResBlock1._e3_forward_original = AdaINResBlock1.forward  # type: ignore[attr-defined]

    def _forward_cos(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2,
            self.alpha1, self.alpha2,
        ):
            xt = n1(x, s)
            # Snake → cosine identity:
            #   xt + (1/a1) sin²(a1 xt)  ≡  xt + (1 - cos(2 a1 xt)) / (2 a1)
            xt = xt + (1.0 - torch.cos(2.0 * a1 * xt)) / (2.0 * a1)
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1.0 - torch.cos(2.0 * a2 * xt)) / (2.0 * a2)
            xt = c2(xt)
            x = xt + x
        return x

    AdaINResBlock1.forward = _forward_cos


def restore_resblock() -> None:
    from Modules.hifigan import AdaINResBlock1  # type: ignore
    if hasattr(AdaINResBlock1, "_e3_forward_original"):
        AdaINResBlock1.forward = AdaINResBlock1._e3_forward_original  # type: ignore[attr-defined]
        del AdaINResBlock1._e3_forward_original  # type: ignore[attr-defined]


def install_snake_cosine_generator(wrapper: nn.Module) -> None:
    """Replace the Generator.forward installed by `_patch_generator_use_har`
    with a variant whose two inline Snakes are emitted as cosine identity."""
    gen = wrapper.generator
    if not hasattr(gen, "_e3_forward_original"):
        gen._e3_forward_original = gen.forward  # type: ignore[attr-defined]

    def _forward_cos(self, x, s, har_source, _f0_unused):
        for i in range(self.num_upsamples):
            a = self.alphas[i]
            x = x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a)
            x_source = self.noise_convs[i](har_source)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        a = self.alphas[self.num_upsamples]
        x = x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    gen.forward = types.MethodType(_forward_cos, gen)  # type: ignore[assignment]


def restore_generator(wrapper: nn.Module) -> None:
    gen = wrapper.generator
    if hasattr(gen, "_e3_forward_original"):
        gen.forward = gen._e3_forward_original  # type: ignore[attr-defined]
        del gen._e3_forward_original  # type: ignore[attr-defined]


def main() -> None:
    print("[10e4] loading runtime + base inputs ...", flush=True)
    rt = build_runtime()
    full_inputs = stage_example_inputs("decoder_upsample", rt)
    x_pre, ref, har = full_inputs
    cropped = (
        x_pre[:, :, :T_MEL].contiguous(),
        ref.contiguous(),
        har[:, :, :T_MEL * HOP].contiguous(),
    )

    install_snake_cosine_resblock()
    wrapper = build_wrapper("decoder_upsample", rt.model)
    install_snake_cosine_generator(wrapper)

    out_path = PACKAGES_DIR / "trial10e4_snake_cosine_identity.mlpackage"
    print(f"[10e4] tracing + converting ...", flush=True)
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, cropped, check_trace=False, strict=False
        )
    inputs = [
        ct.TensorType(name="x_pre", shape=tuple(cropped[0].shape), dtype=np.float32),
        ct.TensorType(name="ref", shape=tuple(cropped[1].shape), dtype=np.float32),
        ct.TensorType(name="har_source", shape=tuple(cropped[2].shape), dtype=np.float32),
    ]
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"[10e4] ct.convert: {time.time() - t0:.1f}s", flush=True)
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))

    restore_resblock()
    restore_generator(wrapper)

    # Probe with .cpuAndNeuralEngine — interactive (foreground), so stderr
    # captures the ANE compile messages reliably.
    deref = Path(tempfile.gettempdir()) / "probe_a4_cos.mlpackage"
    if deref.exists():
        shutil.rmtree(deref)
    shutil.copytree(out_path, deref, symlinks=False)

    print(f"\n[10e4] probing {out_path.name} on .cpuAndNeuralEngine ...", flush=True)
    probe_code = f"""
import sys, numpy as np, time
import coremltools as ct
print('[probe] loading ...', file=sys.stderr, flush=True)
t0 = time.time()
m = ct.models.MLModel({str(deref)!r}, compute_units=ct.ComputeUnit.CPU_AND_NE)
print(f'[probe] load_ms={{(time.time()-t0)*1000:.0f}}', file=sys.stderr, flush=True)
print('[probe] predicting ...', file=sys.stderr, flush=True)
t0 = time.time()
out = m.predict({{'x_pre': np.zeros((1,512,{T_MEL}),np.float32),
                 'ref':   np.zeros((1,128),np.float32),
                 'har_source': np.zeros((1,1,{T_MEL*HOP}),np.float32)}})
print(f'[probe] predict_ms={{(time.time()-t0)*1000:.0f}}', file=sys.stderr, flush=True)
for i in range(5):
    t0 = time.time()
    m.predict({{'x_pre': np.zeros((1,512,{T_MEL}),np.float32),
                'ref':   np.zeros((1,128),np.float32),
                'har_source': np.zeros((1,1,{T_MEL*HOP}),np.float32)}})
    print(f'[probe] warm[{{i}}]_ms={{(time.time()-t0)*1000:.0f}}', file=sys.stderr, flush=True)
"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proc = subprocess.run(
        ["uv", "run", "python", "-c", probe_code],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "MLLOG": "1", "OS_ACTIVITY_MODE": "info"},
        cwd=str(_HERE),
    )
    print("\n[10e4] probe stderr (filtered):")
    for line in proc.stderr.splitlines():
        if "scikit-learn" in line or "Torch version" in line:
            continue
        print(f"  {line}")

    # Capture fragmentation count via log show.
    log_proc = subprocess.run(
        ["log", "show",
         "--predicate", 'subsystem == "com.apple.ane" AND category == "compiler"',
         "--start", ts,
         "--info", "--debug",
         "--style", "compact"],
        capture_output=True, text=True, timeout=120,
    )
    calling_count = log_proc.stdout.count("Calling ANE compiler")
    success_count = log_proc.stdout.count("SUCCESS: model=")
    print(f"\n[10e4] log_show: calling={calling_count}  success={success_count}  "
          f"(baseline 180 / 89; ablation2 identity = 2 / 1)")

    e5rt_failed = "ANECCompile" in proc.stderr and "FAILED" in proc.stderr
    print(f"[10e4] E5RT in stderr: {'FAILED' if e5rt_failed else 'no failure'}")

    flipped = (calling_count < 90) and not e5rt_failed
    print(f"\n[10e4] verdict: {'FLIP — Snake → cosine identity preserves ANE acceptance' if flipped else 'NO FLIP — cosine identity rewrite did not land ANE'}")


if __name__ == "__main__":
    main()
