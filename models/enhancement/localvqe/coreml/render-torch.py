#!/usr/bin/env python
"""Render a blind-set layout through the PyTorch reference (whole-clip forward),
writing aligned float32 WAVs at the GGML output level (x2). --no-fold runs
the AlignBlock softmax at temperature 1.0 instead of the trained value (the upstream README warns this loses several
dB of far-end-single-talk ERLE) — a diagnostic for reproducing older
published numbers.

    uv run python render-torch.py --ckpt localvqe-v1.2-1.3M.pt --blind-dir blind --out renders/torch-v1.2 [--no-fold]
"""
from __future__ import annotations
import argparse, multiprocessing as mp
from pathlib import Path
import numpy as np, torch
from localvqe_coreml.common import load_model, read_wav, write_wav

_model = None
def _init(ckpt, fold, dmax, arch):
    global _model
    torch.set_num_threads(1)
    _model = _load(ckpt, fold, dmax, arch)

def _load(ckpt, fold, dmax, arch=3):
    from localvqe_coreml.upstream.model import LocalVQE
    d = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = dict(d["model_config"]); cfg.pop("transform", None)
    m = LocalVQE(**cfg, arch_version=arch, dmax=dmax); m.load_state_dict(d["model_state_dict"]); m.eval()
    if fold:
        m.align.fold_temperature()
    else:
        with torch.no_grad():
            m.align.temperature.fill_(1.0)  # ignore the trained softmax temperature (upstream "default 1.0")
    return m

def one(job):
    mic_p, lpb_p, out_p = job
    if out_p.exists():
        return
    import soundfile as sf
    mic, _ = sf.read(str(mic_p), dtype="float32"); lpb, _ = sf.read(str(lpb_p), dtype="float32")
    n = min(len(mic), len(lpb)); n -= n % 256
    with torch.no_grad():
        # temperature: unfolded model divides by the trained temperature inside forward
        # (upstream default path); folded model has temperature 1.0 and folded weights.
        y = _model.decoder(_model(torch.from_numpy(mic[:n])[None], torch.from_numpy(lpb[:n])[None]), length=n)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_p, (y[0].numpy() * 2.0).astype(np.float32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--blind-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--dmax", type=int, default=64, help="AlignBlock delay window in frames (v1.2/v1.3: 64; v1.1: 32)")
    ap.add_argument("--arch-version", type=int, default=3, help="3 = SiLU (v1.2+), 2 = ReLU6 (v1.1 reference; diagnostic)")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    jobs = []
    for mic in sorted(args.blind_dir.glob("*/*_mic.flac")):
        stem = mic.name[:-9]
        jobs.append((mic, mic.with_name(stem + "_lpb.flac"), args.out / mic.parent.name / f"{stem}_enh.wav"))
    with mp.Pool(args.jobs, initializer=_init, initargs=(args.ckpt, not args.no_fold, args.dmax, args.arch_version)) as p:
        p.map(one, jobs, chunksize=2)
    print(f"rendered {sum(1 for j in jobs if j[2].exists())} / {len(jobs)} -> {args.out}")

if __name__ == "__main__":
    main()
