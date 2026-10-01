"""Warm per-call latency of every MOSS-TTS-Nano CoreML model across compute units (M-series host)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np

HERE = Path(__file__).resolve().parent
CU = {"all": ct.ComputeUnit.ALL, "ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu": ct.ComputeUnit.CPU_ONLY}


def feed_for(ml, seq_default: int | None = None) -> dict:
    spec = ml.get_spec()
    feed = {}
    for i in spec.description.input:
        t = i.type.multiArrayType
        shape = []
        for k, d in enumerate(t.shape):
            if t.shapeRange.sizeRanges and k < len(t.shapeRange.sizeRanges):
                r = t.shapeRange.sizeRanges[k]
                shape.append(seq_default if (seq_default and r.upperBound != r.lowerBound) else d)
            else:
                shape.append(d)
        if t.dataType == t.INT32:
            arr = np.zeros(shape, np.int32)
            if i.name == "input_ids":
                arr[..., 0] = 3
                arr[..., 1:] = 1024
            if i.name == "input_len":
                arr[:] = 200
            if i.name == "cur_len":
                arr[:] = 300
        else:
            arr = np.zeros(shape, np.float32)
            if i.name in ("text_temperature", "audio_temperature", "repetition_penalty"):
                arr[:] = 1.0
            if i.name == "audio_top_p":
                arr[:] = 0.8
            if i.name in ("text_u", "audio_u"):
                arr[:] = 0.5
            if i.name == "audio":
                arr = np.random.default_rng(0).standard_normal(shape).astype(np.float32) * 0.1
        feed[i.name] = arr
    return feed


def bench(path: Path, units: list[str], iters: int, seq_default: int | None = None) -> None:
    row = [path.stem]
    for cu in units:
        try:
            ml = ct.models.MLModel(str(path), compute_units=CU[cu])
            feed = feed_for(ml, seq_default)
            ml.predict(feed)
            ml.predict(feed)
            t0 = time.perf_counter()
            for _ in range(iters):
                ml.predict(feed)
            row.append(f"{(time.perf_counter() - t0) * 1000 / iters:7.1f}")
        except Exception as e:  # noqa: BLE001
            row.append(f"  err:{type(e).__name__[:8]}")
    print(" | ".join(row))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm-dir", default=str(HERE / "build" / "lm"))
    p.add_argument("--codec-dir", default=str(HERE / "build" / "codec"))
    p.add_argument("--units", default="all,ane,gpu,cpu")
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    units = args.units.split(",")
    print("model | " + " | ".join(f"{u:>7}" for u in units) + "   (ms per call, warm)")
    for path in sorted(Path(args.lm_dir).glob("*.mlpackage")):
        bench(path, units, args.iters)
    for path in sorted(Path(args.codec_dir).glob("*.mlpackage")):
        seq = None
        if "CodecDecoder" in path.name:
            seq = 57  # frames (4.6 s)
        if "CodecEncoder" in path.name:
            seq = 99 * 3840  # samples (7.9 s prompt)
        bench(path, units, args.iters if seq is None else 3, seq)


if __name__ == "__main__":
    main()
