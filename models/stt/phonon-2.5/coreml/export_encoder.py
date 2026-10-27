# Adapted from FluidInference/mobius PR #111 (e35c1953), Apache-2.0.
#!/usr/bin/env python3
"""Phase 1: export the Phonon-2 encoder as an fp16 mlprogram (iOS18) and inventory its weight consts.

The fp16 mlpackage is the parity reference; the const inventory drives the five-value LUT pass (fivevalue_encoder.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import coremltools as ct  # noqa: E402
import soundfile as sf  # noqa: E402

from wrappers import EncoderWrapper, ExportSettings, PreprocessorWrapper, _coreml_convert  # noqa: E402
from components import load_components  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--target", default="ios18", choices=["ios17", "ios18"])
    ap.add_argument("--name", default="phonon25", help="Tag for output files")
    ap.add_argument("--audio", type=Path, required=True, help="Public 16 kHz reference audio")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    tag = args.name + ("" if args.target == "ios18" else "_ios17")
    target = ct.target.iOS18 if args.target == "ios18" else ct.target.iOS17

    torch.set_num_threads(4)
    model = load_components(args.container, args.container.parent / "config.json")

    sr = int(model.cfg.preprocessor.sample_rate)
    max_samples = int(round(15.0 * sr))
    data, file_sr = sf.read(str(args.audio), dtype="float32")
    assert file_sr == sr
    if data.ndim > 1:
        data = data[:, 0]
    data = np.pad(data, (0, max(0, max_samples - data.size)))[:max_samples]
    audio = torch.from_numpy(data)[None]
    audio_len = torch.tensor([max_samples], dtype=torch.int32)

    pre = PreprocessorWrapper(model.preprocessor.eval())
    enc = EncoderWrapper(model.encoder.eval())
    with torch.inference_mode():
        mel, mel_len = pre(audio, audio_len)
        mel_len = mel_len.to(torch.int32)
        enc_out, enc_len = enc(mel, mel_len)
    mel, mel_len, enc_out = mel.clone(), mel_len.clone(), enc_out.clone()
    np.savez(args.work / f"encoder_ref_{tag}.npz", mel=mel.numpy(), mel_length=mel_len.numpy(),
             encoder=enc_out.numpy(), encoder_length=enc_len.numpy())
    print("mel", tuple(mel.shape), "encoder", tuple(enc_out.shape), "len", enc_len.tolist())

    traced = torch.jit.trace(enc, (mel, mel_len), strict=False).eval()
    settings = ExportSettings(
        output_dir=args.work, compute_units=ct.ComputeUnit.CPU_ONLY, deployment_target=target,
        compute_precision=None, max_audio_seconds=15.0, max_symbol_steps=1,
    )
    t0 = time.time()
    mlmodel = _coreml_convert(
        traced,
        [ct.TensorType(name="mel", shape=tuple(mel.shape), dtype=np.float32),
         ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32)],
        [ct.TensorType(name="encoder", dtype=np.float32), ct.TensorType(name="encoder_length", dtype=np.int32)],
        settings,
    )
    print(f"convert took {time.time() - t0:.0f}s")
    out = args.work / f"encoder_fp16_{tag}.mlpackage"
    mlmodel.short_description = f"{tag} encoder fp16 (15 s window)"
    mlmodel.author = "Fluid Inference"
    mlmodel.save(str(out))

    prog = mlmodel._mil_program
    inv = []
    for op in prog.functions["main"].operations:
        if op.op_type != "const":
            continue
        val = op.outputs[0].val
        if val is None or np.size(val) < 100_000:
            continue
        children = sorted({c.op_type for c in op.outputs[0].child_ops})
        inv.append({"name": op.name, "shape": list(np.shape(val)), "dtype": str(val.dtype), "children": children})
    (args.work / f"encoder_consts_{tag}.json").write_text(json.dumps(inv, indent=1))
    print(f"{len(inv)} large consts; first few:")
    for e in inv[:12]:
        print("  ", e)
    print("saved", out)


if __name__ == "__main__":
    main()
