"""Convert the per-frame streaming codec decoder to CoreML and verify against full decode.

    MossNano-CodecStep-{tag}.mlpackage   codes [16,1,1] + frame_index + 24 KV caches → audio [1,2,3840] + caches
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.codec_coreml import MossCodecStepDecoder  # noqa: E402

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
CU = {"all": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE}


def snr_db(got: np.ndarray, want: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(want**2) / max(np.sum((got - want) ** 2), 1e-20)))


def run_torch(step: MossCodecStepDecoder, codes: torch.Tensor) -> torch.Tensor:
    caches = [torch.zeros(shape) for _, shape in step.cache_shapes()]
    chunks = []
    with torch.no_grad():
        for t in range(codes.shape[-1]):
            out = step(codes[:, :, t : t + 1].to(torch.int32), torch.tensor([t], dtype=torch.int32), *caches)
            chunks.append(out[0])
            caches = list(out[1:])
    return torch.cat(chunks, dim=-1)


def run_coreml(ml, step: MossCodecStepDecoder, codes: np.ndarray) -> tuple[np.ndarray, float]:
    names = [n for n, _ in step.cache_shapes()]
    caches = {n: np.zeros(shape, np.float32) for n, shape in step.cache_shapes()}
    chunks = []
    t_total = 0.0
    for t in range(codes.shape[-1]):
        feed = {"codes": codes[:, :, t : t + 1].astype(np.int32), "frame_index": np.array([t], np.int32), **caches}
        t0 = time.perf_counter()
        out = ml.predict(feed)
        t_total += time.perf_counter() - t0
        chunks.append(out["audio"])
        caches = {n: out[f"{n}_out"] for n in names}
    return np.concatenate(chunks, axis=-1), t_total / codes.shape[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(HERE / "build" / "codec"))
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--tokens", default=str(HERE / "build" / "ref_audio_token_ids.npy"))
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--compute-units", default="all,ane,gpu,cpu")
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "fp16" if args.fp16 else "fp32"

    from transformers import AutoModel

    print(f"[0] loading {CODEC_REPO}")
    codec = AutoModel.from_pretrained(CODEC_REPO, trust_remote_code=True).eval()
    codes = torch.from_numpy(np.load(args.tokens)).long().T[:, None, :]
    with torch.no_grad():
        want = codec.decode(codes, return_dict=True).audio.numpy()

    step = MossCodecStepDecoder(codec).eval()
    print("      stages:", [(s["n"], s["context"], s["layers"]) for s in step.stage_specs])
    print(f"      cache tensors: {len(step.cache_shapes())}  "
          f"elements: {sum(int(np.prod(sh)) for _, sh in step.cache_shapes())/1e6:.2f}M")
    t0 = time.perf_counter()
    got = run_torch(step, codes).numpy()
    print(f"[1] torch step decoder vs upstream full decode: SNR={snr_db(got, want):.1f} dB  "
          f"({(time.perf_counter()-t0)*1000/codes.shape[-1]:.1f} ms/frame fp32 cpu)")
    if args.skip_convert:
        return

    example = [codes[:, :, :1].to(torch.int32), torch.tensor([0], dtype=torch.int32)]
    example += [torch.zeros(shape) for _, shape in step.cache_shapes()]
    with torch.no_grad():
        traced = torch.jit.trace(step, tuple(example), strict=False)
    inputs = [
        ct.TensorType(name="codes", shape=(16, 1, 1), dtype=np.int32),
        ct.TensorType(name="frame_index", shape=(1,), dtype=np.int32),
    ] + [ct.TensorType(name=n, shape=shape, dtype=np.float32) for n, shape in step.cache_shapes()]
    outputs = [ct.TensorType(name="audio", dtype=np.float32)] + [
        ct.TensorType(name=f"{n}_out", dtype=np.float32) for n, _ in step.cache_shapes()
    ]
    prec = ct.precision.FLOAT32 if not args.fp16 else ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in {"softmax"}
    )
    t0 = time.perf_counter()
    ml = ct.convert(
        traced, inputs=inputs, outputs=outputs, compute_precision=prec,
        minimum_deployment_target=ct.target.macOS14, convert_to="mlprogram",
    )
    path = out_dir / f"MossNano-CodecStep-{tag}.mlpackage"
    ml.save(str(path))
    print(f"[2] saved {path.name} ({time.perf_counter()-t0:.0f}s)")
    for cu in args.compute_units.split(","):
        m = ct.models.MLModel(str(path), compute_units=CU[cu])
        pred, per_frame = run_coreml(m, step, codes.numpy())
        print(f"      coreml[{cu}]: SNR={snr_db(pred, want):.1f} dB  {per_frame*1000:.2f} ms/frame  "
              f"(frame budget 80 ms → {80/(per_frame*1000):.0f}x RT)")
    print("[done]")


if __name__ == "__main__":
    main()
