#!/usr/bin/env python
"""Parity + timing of a converted Core ML LocalVQE against the PyTorch reference.

    uv run python verify-coreml.py --ckpt localvqe-v1.3-4.8M.pt \
        --model build/localvqe-v1.3-4.8M-16ms.mlpackage --mic dt_mic.wav --ref dt_ref.wav
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import coremltools as ct, numpy as np, torch
from localvqe_coreml.common import load_model, read_wav, write_wav
from localvqe_coreml.streaming import HOP, StreamingLocalVQE, run_streaming

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, type=Path)
ap.add_argument("--model", required=True, type=Path, nargs="+")
ap.add_argument("--mic", required=True, type=Path)
ap.add_argument("--ref", required=True, type=Path)
ap.add_argument("--compute-units", nargs="+", default=["CPU_ONLY", "CPU_AND_NE", "ALL"])
ap.add_argument("--save-dir", type=Path)
args = ap.parse_args()

model = load_model(args.ckpt)
mic = torch.from_numpy(read_wav(args.mic))[None]
ref = torch.from_numpy(read_wav(args.ref))[None]
n = mic.shape[1] - mic.shape[1] % HOP
mic, ref = mic[:, :n], ref[:, :n]
with torch.no_grad():
    ref_out = (model.decoder(model(mic, ref), length=n)[0].numpy()) * 2.0  # GGML level (ola_scale=1)
peak = np.abs(ref_out).max()
print(f"clip {n/16000:.2f}s  mic peak {mic.abs().max():.3f}  enhanced peak {peak:.3f}")


def snr_db(a, b):
    return 10 * np.log10(np.sum(b**2) / (np.sum((a - b) ** 2) + 1e-20))


for mpath in args.model:
    frames = int(mpath.stem.split("-")[-1].replace("ms", "").replace("-fp32", "")) if False else None
    for cu in args.compute_units:
        mlmodel = ct.models.MLModel(str(mpath), compute_units=getattr(ct.ComputeUnit, cu))
        T = int(mlmodel.user_defined_metadata["frames_per_call"])
        sm = StreamingLocalVQE(model, frames=T)
        names = [k for k, _ in sm.state_spec]
        calls = []

        def step(a, b, states):
            feed = {"mic": a.numpy(), "ref": b.numpy()}
            for k, s in zip(names, states):
                feed[f"in_{k}"] = np.ascontiguousarray(s.numpy() if torch.is_tensor(s) else s, dtype=np.float32)
            t0 = time.perf_counter()
            out = mlmodel.predict(feed)
            calls.append(time.perf_counter() - t0)
            return (torch.from_numpy(np.asarray(out["enhanced"], dtype=np.float32)),
                    [np.asarray(out[f"out_{k}"], dtype=np.float32) for k in names])

        y = run_streaming(sm, mic, ref, step_fn=step)[0].numpy()
        calls = np.array(calls[3:]) * 1000
        d = np.abs(y - ref_out)
        print(f"{mpath.name:42s} {cu:10s} frames={T:2d}  max|diff|={d.max():.2e} ({d.max()/peak*100:.3f}% of peak)  "
              f"SNR={snr_db(y, ref_out):6.1f} dB  per-call p50={np.median(calls):.2f}ms p99={np.percentile(calls,99):.2f}ms  "
              f"RTFx={T*HOP/16000*1000/np.median(calls):.0f}x")
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            write_wav(args.save_dir / f"{mpath.stem}-{cu}.wav", y)
if args.save_dir:
    write_wav(args.save_dir / "torch-reference.wav", ref_out)
