"""Trial 10e3 — op bisection by ablation (decoder_upsample / ANE planner).

Trial 10e2 narrowed the blocker from "rejected op" to "planner
fragmentation": ANECompilerService accepts every subgraph it sees, but
the MIL→ANEF planner produces too many small subgraphs to link. This
script walks the prioritized ablation list to find which graph property
forces over-fragmentation.

Per ablation:
    1. Install monkey-patch (idempotent; original method stashed on the class).
    2. Build wrapper at T_mel = 50 (smallest, fastest compile).
    3. Trace + convert at fp16 fixed shapes.
    4. Save throwaway mlpackage to coreml/packages/trial10e3_a<n>_<name>.mlpackage.
    5. Restore patch.
    6. Probe: load with .cpuAndNeuralEngine, run one predict, capture stderr.
    7. Count `Calling ANE compiler` lines via `log show --start <ts>`.
    8. Compare to baseline (Trial 10e1 T_mel=50 mlpackage, same probe).

Decision per ablation:
    * count drops + ANECCompile succeeds (no E5RT FAILED in stderr): FLIP.
      Stop. Identifies the offender. Document, propose weight-preserving
      rewrite.
    * count drops but E5RT still fails: that op contributes to fragmentation
      but isn't the only cause; continue.
    * count unchanged: that op isn't the primary fragmenter; continue.

Walk full list:
    1. AdaIN drop affine (88 instances; scalar-per-feature broadcast)
    2. Snake → identity (101 sin+pow boundaries)
    3. AdaINResBlock1 residual fold (in-place mods)
    4. Halve upsample stack (depth ablation)

Diagnostic-only — these graphs are throwaway. No parity checks, no
listening. Quality irrelevant.

Run:
    cd models/tts/styletts2
    uv run python coreml/exporters/trial10e3_bisection.py
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coreml.exporters import convert as _convert  # noqa: F401  installs MIL patches
from coreml._runtime import HERE, build_runtime, stage_example_inputs
from coreml.wrappers import build_wrapper

import coremltools as ct

PACKAGES_DIR = HERE / "coreml" / "packages"
T_MEL = 50  # smallest probe size — fastest compile, no width-error noise
HOP = 300
# Baseline `Calling ANE compiler` count established from the prior sweep
# at T_mel=50 (no ablations); used as the FLIP threshold reference.
BASELINE_CALLING_COUNT = 180

CALLING_RE = re.compile(r"Calling ANE compiler")
SUCCESS_RE = re.compile(r"SUCCESS: model=")
ANE_FAIL_RE = re.compile(r"ANECCompile.*FAILED")


# ---------------------------------------------------------------------------
# Idempotent monkey-patch installers + restorers
# ---------------------------------------------------------------------------


def install_adain_drop_affine() -> None:
    """Ablation 1: AdaIN1d.forward returns InstanceNorm only (no gamma/beta)."""
    from Modules.hifigan import AdaIN1d  # type: ignore
    if hasattr(AdaIN1d, "_e3_forward_original"):
        return
    AdaIN1d._e3_forward_original = AdaIN1d.forward  # type: ignore[attr-defined]

    def _forward_no_affine(self, x, s):
        return self.norm(x)

    AdaIN1d.forward = _forward_no_affine


def restore_adain() -> None:
    from Modules.hifigan import AdaIN1d  # type: ignore
    if hasattr(AdaIN1d, "_e3_forward_original"):
        AdaIN1d.forward = AdaIN1d._e3_forward_original  # type: ignore[attr-defined]
        del AdaIN1d._e3_forward_original  # type: ignore[attr-defined]


def install_snake_identity_resblock() -> None:
    """Ablation 2 (part 1): AdaINResBlock1.forward without Snake activations."""
    from Modules.hifigan import AdaINResBlock1  # type: ignore
    if hasattr(AdaINResBlock1, "_e3_forward_original"):
        return
    AdaINResBlock1._e3_forward_original = AdaINResBlock1.forward  # type: ignore[attr-defined]

    def _forward_no_snake(self, x, s):
        for c1, c2, n1, n2, _a1, _a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2,
            self.alpha1, self.alpha2,
        ):
            xt = n1(x, s)
            # Snake → identity (was: xt = xt + (1/a1) * sin(a1*xt)**2)
            xt = c1(xt)
            xt = n2(xt, s)
            # Snake → identity
            xt = c2(xt)
            x = xt + x
        return x

    AdaINResBlock1.forward = _forward_no_snake


def restore_resblock() -> None:
    from Modules.hifigan import AdaINResBlock1  # type: ignore
    if hasattr(AdaINResBlock1, "_e3_forward_original"):
        AdaINResBlock1.forward = AdaINResBlock1._e3_forward_original  # type: ignore[attr-defined]
        del AdaINResBlock1._e3_forward_original  # type: ignore[attr-defined]


def install_snake_identity_generator(wrapper: nn.Module) -> None:
    """Ablation 2 (part 2): replace the Generator.forward installed by
    `_patch_generator_use_har` with a Snake-less variant. Acts on the live
    wrapper instance (not the class), since `_patch_generator_use_har` binds
    a closure to the gen instance."""
    gen = wrapper.generator
    if not hasattr(gen, "_e3_forward_original"):
        gen._e3_forward_original = gen.forward  # type: ignore[attr-defined]

    def _forward_no_snake(self, x, s, har_source, _f0_unused):
        for i in range(self.num_upsamples):
            # Snake → identity (was: x = x + (1/alphas[i]) * sin(alphas[i]*x)**2)
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
        # Snake → identity (was: x = x + (1/alphas[i+1]) * sin(alphas[i+1]*x)**2)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    gen.forward = types.MethodType(_forward_no_snake, gen)  # type: ignore[assignment]


def install_residual_fold_resblock() -> None:
    """Ablation 3: AdaINResBlock1 without the per-iteration residual `x = xt + x`.
    Replace with x = xt (drop the skip). Pure diagnostic."""
    from Modules.hifigan import AdaINResBlock1  # type: ignore
    if hasattr(AdaINResBlock1, "_e3_forward_original"):
        return
    AdaINResBlock1._e3_forward_original = AdaINResBlock1.forward  # type: ignore[attr-defined]

    def _forward_no_residual(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2,
            self.alpha1, self.alpha2,
        ):
            xt = n1(x, s)
            xt = xt + (1.0 / a1) * (torch.sin(a1 * xt) ** 2)
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1.0 / a2) * (torch.sin(a2 * xt) ** 2)
            xt = c2(xt)
            x = xt  # ← residual dropped (was: x = xt + x)
        return x

    AdaINResBlock1.forward = _forward_no_residual


def install_halve_upsample(wrapper: nn.Module) -> None:
    """Ablation 4: drop the last 2 ups stages + their resblocks.
    Diagnostic-only: output channel count is wrong (not projected by conv_post),
    so we replace conv_post with a 1x1 projection from the post-ups[1] channel
    count to 1. tanh as usual.
    """
    gen = wrapper.generator
    if not hasattr(gen, "_e3_forward_original"):
        gen._e3_forward_original = gen.forward  # type: ignore[attr-defined]

    # post-ups[1] has upsample_initial_channel // 4 channels (512/4 = 128)
    post_ups1_ch = gen.ups[0].out_channels // 2  # ups[0] outputs 256 → ups[1] outputs 128
    head = nn.Conv1d(post_ups1_ch, 1, kernel_size=1, bias=True)
    head.weight.data.zero_()
    head.bias.data.zero_()
    gen._e3_head = head

    def _forward_halved(self, x, s, har_source, _f0_unused):
        for i in range(2):  # only ups[0], ups[1]
            x = x + (1.0 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
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
        x = self._e3_head(x)
        x = torch.tanh(x)
        return x

    gen.forward = types.MethodType(_forward_halved, gen)  # type: ignore[assignment]


def restore_generator(wrapper: nn.Module) -> None:
    gen = wrapper.generator
    if hasattr(gen, "_e3_forward_original"):
        gen.forward = gen._e3_forward_original  # type: ignore[attr-defined]
        del gen._e3_forward_original  # type: ignore[attr-defined]
    if hasattr(gen, "_e3_head"):
        del gen._e3_head  # type: ignore[attr-defined]


def restore_all(wrapper: nn.Module) -> None:
    restore_adain()
    restore_resblock()
    restore_generator(wrapper)


# ---------------------------------------------------------------------------
# Convert + probe driver
# ---------------------------------------------------------------------------


def _convert_at_t_mel(
    rt,
    cropped_inputs: tuple,
    out_path: Path,
) -> Path:
    """Build wrapper post-monkey-patch, trace + convert at fixed shapes, save."""
    wrapper = build_wrapper("decoder_upsample", rt.model)
    return _convert_with_wrapper(wrapper, cropped_inputs, out_path), wrapper


def _convert_with_wrapper(wrapper, cropped_inputs: tuple, out_path: Path) -> Path:
    x_pre, ref, har = cropped_inputs
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, cropped_inputs, check_trace=False, strict=False
        )
    inputs = [
        ct.TensorType(name="x_pre", shape=tuple(x_pre.shape), dtype=np.float32),
        ct.TensorType(name="ref", shape=tuple(ref.shape), dtype=np.float32),
        ct.TensorType(name="har_source", shape=tuple(har.shape), dtype=np.float32),
    ]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    return out_path


def _probe_with_log_show(pkg_path: Path, t_mel: int) -> dict:
    """Load mlpackage with .cpuAndNeuralEngine, predict once, capture stderr +
    `log show` output between the start/end timestamps. Return a dict with:
    `predict_ok`, `e5rt_failed`, `calling_count`, `success_count`, `width_errors`,
    `stderr_tail`, `compile_seconds`."""
    deref = Path(tempfile.gettempdir()) / pkg_path.name
    if deref.exists():
        shutil.rmtree(deref)
    shutil.copytree(pkg_path, deref, symlinks=False)

    # Capture timestamp before triggering compile.
    t0 = datetime.datetime.now()
    ts_start = t0.strftime("%Y-%m-%d %H:%M:%S")

    # Use a subprocess so stderr is cleanly captured regardless of Python's
    # internal redirection.
    probe_code = f"""
import os, sys, numpy as np, time
import coremltools as ct
print('[probe] loading ...', file=sys.stderr, flush=True)
m = ct.models.MLModel({str(deref)!r}, compute_units=ct.ComputeUnit.CPU_AND_NE)
print('[probe] predicting ...', file=sys.stderr, flush=True)
ti = time.time()
out = m.predict({{'x_pre': np.zeros((1,512,{t_mel}),np.float32),
                 'ref':   np.zeros((1,128),np.float32),
                 'har_source': np.zeros((1,1,{t_mel*HOP}),np.float32)}})
print(f'[probe] predict_ms={{(time.time()-ti)*1000:.0f}}', file=sys.stderr, flush=True)
"""
    proc = subprocess.run(
        ["uv", "run", "python", "-c", probe_code],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "MLLOG": "1", "OS_ACTIVITY_MODE": "info"},
        cwd=str(_HERE),
    )
    t1 = datetime.datetime.now()
    ts_end = t1.strftime("%Y-%m-%d %H:%M:%S")
    compile_secs = (t1 - t0).total_seconds()

    stderr = proc.stderr
    e5rt_failed = bool(ANE_FAIL_RE.search(stderr))
    width_errors = re.findall(r"Tensor width goes beyond limit supported \((\d+)", stderr)
    predict_ok = "[probe] predict_ms=" in stderr and proc.returncode == 0

    # log show retroactively reads the persistent log store between timestamps.
    log_proc = subprocess.run(
        ["log", "show",
         "--predicate", 'subsystem == "com.apple.ane" AND category == "compiler"',
         "--start", ts_start, "--end", ts_end,
         "--info", "--debug",
         "--style", "compact"],
        capture_output=True, text=True, timeout=120,
    )
    log_text = log_proc.stdout
    calling_count = len(CALLING_RE.findall(log_text))
    success_count = len(SUCCESS_RE.findall(log_text))

    shutil.rmtree(deref, ignore_errors=True)
    return {
        "predict_ok": predict_ok,
        "e5rt_failed": e5rt_failed,
        "calling_count": calling_count,
        "success_count": success_count,
        "width_errors": [int(w) for w in width_errors],
        "compile_seconds": compile_secs,
        "stderr_tail": "\n".join(stderr.strip().splitlines()[-8:]),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    name: str
    pkg_path: Path | None
    convert_seconds: float
    probe: dict
    flipped: bool
    notes: str = ""


def run_one(label: str, install_fn, restore_fn, rt, cropped_inputs, *, requires_wrapper=False) -> AblationResult:
    name = label.replace(" ", "_").replace("/", "_")
    out_path = PACKAGES_DIR / f"trial10e3_{name}.mlpackage"
    print(f"\n{'='*60}\n[trial10e3] {label}\n{'='*60}", flush=True)

    t0 = time.time()
    if requires_wrapper:
        # Patch needs the live wrapper instance; build wrapper first, then patch.
        wrapper = build_wrapper("decoder_upsample", rt.model)
        install_fn(wrapper)
    else:
        install_fn()
        wrapper = build_wrapper("decoder_upsample", rt.model)

    try:
        _convert_with_wrapper(wrapper, cropped_inputs, out_path)
        convert_secs = time.time() - t0
        print(f"[trial10e3] {label}: convert {convert_secs:.1f}s; running probe ...", flush=True)
        probe = _probe_with_log_show(out_path, T_MEL)
    finally:
        restore_fn(wrapper) if requires_wrapper else restore_fn()
        # Always restore generator state too (some installers may have touched it).
        restore_generator(wrapper)

    # FLIP signal: fragmentation count must drop > 50 % vs baseline AND no
    # width errors. (subprocess stderr is unreliable for E5RT detection — the
    # `coreml`/`espresso` os_log entries don't always propagate through
    # subprocess.PIPE, so e5rt_failed is a noisy signal. Fragmentation count
    # from `log show` is the load-bearing metric.)
    notes = ""
    flipped = (
        probe["calling_count"] < BASELINE_CALLING_COUNT * 0.5
        and not probe["width_errors"]
    )
    if probe["calling_count"] >= BASELINE_CALLING_COUNT * 0.5:
        notes = f"fragmentation unchanged ({probe['calling_count']} ≈ baseline {BASELINE_CALLING_COUNT})"
    elif probe["width_errors"]:
        notes = f"width errors at lower fragmentation: {set(probe['width_errors'])}"
    else:
        notes = f"fragmentation dropped {BASELINE_CALLING_COUNT} → {probe['calling_count']}"

    print(
        f"[trial10e3] {label}: predict_ok={probe['predict_ok']} "
        f"e5rt_failed={probe['e5rt_failed']} "
        f"calling={probe['calling_count']} success={probe['success_count']} "
        f"widths={set(probe['width_errors'])} "
        f"compile_s={probe['compile_seconds']:.1f} -> {'FLIP' if flipped else 'no flip'}",
        flush=True,
    )

    return AblationResult(
        name=label, pkg_path=out_path, convert_seconds=convert_secs,
        probe=probe, flipped=flipped, notes=notes,
    )


def main() -> None:
    print("[trial10e3] loading runtime + base inputs ...", flush=True)
    rt = build_runtime()
    full_inputs = stage_example_inputs("decoder_upsample", rt)
    x_pre, ref, har = full_inputs
    cropped = (
        x_pre[:, :, :T_MEL].contiguous(),
        ref.contiguous(),
        har[:, :, :T_MEL * HOP].contiguous(),
    )
    print(
        f"[trial10e3] cropped inputs: x_pre={tuple(cropped[0].shape)} "
        f"har={tuple(cropped[2].shape)}"
    )

    # Baseline: probe the existing trial10e1 T_mel=50 mlpackage (1D Conv,
    # all ablations OFF). Reuses what's already on disk.
    print("\n=== BASELINE: trial10e1 T_mel=50 (no ablations) ===", flush=True)
    baseline_pkg = PACKAGES_DIR / "decoder_upsample_trial10e1_fp16_tmel50.mlpackage"
    if not baseline_pkg.exists():
        sys.exit(f"baseline mlpackage missing: {baseline_pkg}\n"
                 "run trial10e1_t_mel_cap.py first")
    baseline_probe = _probe_with_log_show(baseline_pkg, T_MEL)
    print(
        f"[baseline] predict_ok={baseline_probe['predict_ok']} "
        f"e5rt_failed={baseline_probe['e5rt_failed']} "
        f"calling={baseline_probe['calling_count']} "
        f"success={baseline_probe['success_count']} "
        f"compile_s={baseline_probe['compile_seconds']:.1f}",
        flush=True,
    )

    results: list[AblationResult] = []

    # Ablation 1: AdaIN drop affine
    r = run_one(
        "ablation1_adain_drop_affine",
        install_adain_drop_affine,
        restore_adain,
        rt, cropped,
    )
    results.append(r)
    if r.flipped:
        return _summary(baseline_probe, results)

    # Ablation 2: Snake → identity (resblock + generator)
    def install_snake_full(wrapper):
        install_snake_identity_resblock()
        install_snake_identity_generator(wrapper)

    def restore_snake_full(wrapper):
        restore_resblock()
        restore_generator(wrapper)

    r = run_one(
        "ablation2_snake_identity",
        install_snake_full, restore_snake_full,
        rt, cropped, requires_wrapper=True,
    )
    results.append(r)
    if r.flipped:
        return _summary(baseline_probe, results)

    # Ablation 3: AdaINResBlock1 residual drop
    r = run_one(
        "ablation3_residual_fold",
        install_residual_fold_resblock, restore_resblock,
        rt, cropped,
    )
    results.append(r)
    if r.flipped:
        return _summary(baseline_probe, results)

    # Ablation 4: halve upsample stack
    r = run_one(
        "ablation4_halve_upsample",
        install_halve_upsample, restore_generator,
        rt, cropped, requires_wrapper=True,
    )
    results.append(r)

    return _summary(baseline_probe, results)


def _summary(baseline: dict, results: list[AblationResult]) -> None:
    print(f"\n\n{'='*60}\n[trial10e3] SUMMARY\n{'='*60}", flush=True)
    print(
        f"  {'ablation':<32} | {'calling':>7} | {'success':>7} | "
        f"{'e5rt':>5} | {'width':>5} | {'compile_s':>9} | {'flip?':>5}"
    )
    print(
        f"  {'baseline (no ablation)':<32} | "
        f"{baseline['calling_count']:>7} | "
        f"{baseline['success_count']:>7} | "
        f"{'FAIL' if baseline['e5rt_failed'] else 'pass':>5} | "
        f"{len(set(baseline['width_errors'])):>5} | "
        f"{baseline['compile_seconds']:>8.1f}s |"
    )
    for r in results:
        p = r.probe
        print(
            f"  {r.name:<32} | "
            f"{p['calling_count']:>7} | "
            f"{p['success_count']:>7} | "
            f"{'FAIL' if p['e5rt_failed'] else 'pass':>5} | "
            f"{len(set(p['width_errors'])):>5} | "
            f"{p['compile_seconds']:>8.1f}s | "
            f"{'YES' if r.flipped else '':>5}"
        )

    flippers = [r for r in results if r.flipped]
    if flippers:
        print(f"\n[trial10e3] FLIP found: {flippers[0].name}")
    else:
        print(f"\n[trial10e3] NO FLIP across all {len(results)} ablations.")
        print("           Issue #59 outcome (2): structural to HiFi-GAN itself.")
        print("           Vocoder swap (Vocos / iSTFT) is the path forward.")


if __name__ == "__main__":
    main()
