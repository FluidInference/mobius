#!/usr/bin/env python
"""Convert a published LocalVQE PyTorch checkpoint to a streaming Core ML model.

    uv run python convert-coreml.py --ckpt localvqe-v1.3-4.8M.pt --frames 1 4 16 \
        --output-dir build

Each ``--frames N`` value produces one ``.mlpackage`` (and compiled
``.mlmodelc``) that consumes ``256*N`` new mic + far-end samples per call and
returns ``256*N`` enhanced samples plus every recurrent state as explicit
``in_*`` / ``out_*`` tensors (see ``localvqe_coreml/streaming.py``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from localvqe_coreml.common import load_model
from localvqe_coreml.streaming import HOP, StreamingLocalVQE

MODEL_NAMES = {
    "localvqe-v1.3-4.8M.pt": "localvqe-v1.3-4.8M",
    "localvqe-v1.2-1.3M.pt": "localvqe-v1.2-1.3M",
}


def model_basename(ckpt: Path, frames: int, precision: str) -> str:
    base = MODEL_NAMES.get(ckpt.name, ckpt.stem)
    ms = frames * HOP * 1000 // 16000
    # fp32 is the shipped precision (fp16 loses ~70 dB of parity, see README).
    tag = f"{ms}ms" if precision == "fp32" else f"{ms}ms-{precision}"
    return f"{base}-{tag}"


def convert(sm: StreamingLocalVQE, precision: str) -> ct.models.MLModel:
    T = sm.frames
    mic = torch.zeros(1, HOP * T)
    ref = torch.zeros(1, HOP * T)
    states = sm.zero_states()
    sm.eval()
    with torch.no_grad():
        traced = torch.jit.trace(sm, (mic, ref, *states), check_trace=False)

    inputs = [
        ct.TensorType(name="mic", shape=(1, HOP * T), dtype=np.float32),
        ct.TensorType(name="ref", shape=(1, HOP * T), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="enhanced", dtype=np.float32)]
    for name, shape in sm.state_spec:
        inputs.append(ct.TensorType(name=f"in_{name}", shape=shape, dtype=np.float32))
        outputs.append(ct.TensorType(name=f"out_{name}", dtype=np.float32))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    return mlmodel


def annotate(mlmodel: ct.models.MLModel, sm: StreamingLocalVQE, ckpt: Path, precision: str) -> None:
    T = sm.frames
    mlmodel.author = "LocalAI (model) / FluidInference (Core ML conversion)"
    mlmodel.license = "Apache-2.0"
    mlmodel.version = ckpt.stem
    mlmodel.short_description = (
        f"LocalVQE {ckpt.stem}: neural acoustic echo cancellation + noise suppression + "
        f"dereverberation, 16 kHz, streaming {T}x256-sample hops per call with explicit state."
    )
    md = mlmodel.user_defined_metadata
    md["source"] = "https://github.com/localai-org/LocalVQE"
    md["weights"] = f"https://huggingface.co/LocalAI-io/LocalVQE/blob/main/{ckpt.name}"
    md["sample_rate"] = "16000"
    md["hop"] = str(HOP)
    md["frames_per_call"] = str(T)
    md["samples_per_call"] = str(HOP * T)
    md["output_delay_samples"] = str(HOP)
    md["ola_scale"] = str(sm.ola_scale)
    md["precision"] = precision
    md["state_names"] = json.dumps([n for n, _ in sm.state_spec])


def compile_modelc(mlpackage: Path, out_dir: Path) -> Path:
    """Compile with coremltools (works without a full Xcode install)."""
    modelc = out_dir / (mlpackage.stem + ".mlmodelc")
    if modelc.exists():
        shutil.rmtree(modelc)
    compiled = ct.models.utils.compile_model(str(mlpackage))
    shutil.copytree(compiled, modelc)
    shutil.rmtree(compiled, ignore_errors=True)
    return modelc


def coreml_step_fn(mlmodel: ct.models.MLModel, sm: StreamingLocalVQE):
    names = [n for n, _ in sm.state_spec]

    def step(mic, ref, states):
        feed = {"mic": mic.numpy(), "ref": ref.numpy()}
        for n, s in zip(names, states):
            feed[f"in_{n}"] = np.ascontiguousarray(s.numpy() if torch.is_tensor(s) else s, dtype=np.float32)
        out = mlmodel.predict(feed)
        new_states = [torch.from_numpy(np.asarray(out[f"out_{n}"], dtype=np.float32)) for n in names]
        return torch.from_numpy(np.asarray(out["enhanced"], dtype=np.float32)), new_states

    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--arch-version", type=int, default=3)
    ap.add_argument("--frames", type=int, nargs="+", default=[1])
    ap.add_argument("--precision", choices=["fp16", "fp32"], nargs="+", default=["fp32"])
    ap.add_argument("--output-dir", type=Path, default=Path("build"))
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, arch_version=args.arch_version)
    for frames in args.frames:
        for precision in args.precision:
            sm = StreamingLocalVQE(model, frames=frames, ola_scale=1.0)
            name = model_basename(args.ckpt, frames, precision)
            t0 = time.time()
            mlmodel = convert(sm, precision)
            annotate(mlmodel, sm, args.ckpt, precision)
            pkg = args.output_dir / f"{name}.mlpackage"
            if pkg.exists():
                shutil.rmtree(pkg)
            mlmodel.save(str(pkg))
            print(f"[ok] {pkg}  ({time.time() - t0:.1f}s, {len(sm.state_spec)} state tensors)")
            if not args.no_compile:
                modelc = compile_modelc(pkg, args.output_dir)
                print(f"[ok] {modelc}")


if __name__ == "__main__":
    main()
