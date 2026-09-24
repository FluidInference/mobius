#!/usr/bin/env python3
"""Phase 1: export the parakeet-redux encoder as an fp16 mlprogram (iOS18) and inventory its weight consts.

The fp16 mlpackage is the parity reference; the const inventory drives the ternary LUT pass.
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
V3_DIR = HERE.parent.parent / "parakeet-tdt-v3-0.6b" / "coreml"
sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(HERE))

import coremltools as ct  # noqa: E402
import nemo.collections.asr as nemo_asr  # noqa: E402
import soundfile as sf  # noqa: E402

from individual_components import EncoderWrapper, ExportSettings, PreprocessorWrapper, _coreml_convert  # noqa: E402
from redux_weights import load_into_nemo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, default=Path("~/Documents/parakeet-redux-work/hf").expanduser())
    ap.add_argument("--work", type=Path, default=Path("~/Documents/parakeet-redux-work").expanduser())
    ap.add_argument("--stock", action="store_true", help="Export the stock nvidia v3 encoder instead of redux")
    ap.add_argument("--target", default="ios18", choices=["ios17", "ios18"])
    ap.add_argument("--name", default="redux", help="Tag for output files (e.g. ultra)")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    tag = ("stock" if args.stock else args.name) + ("" if args.target == "ios18" else "_ios17")
    target = ct.target.iOS18 if args.target == "ios18" else ct.target.iOS17

    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
    model.eval()
    if not args.stock:
        load_into_nemo(model, args.hf_dir)

    sr = int(model.cfg.preprocessor.sample_rate)
    max_samples = int(round(15.0 * sr))
    data, file_sr = sf.read(str(V3_DIR / "audio" / "yc_first_minute_16k_15s.wav"), dtype="float32")
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
    mlmodel.short_description = f"parakeet-{tag} encoder fp16 (15 s window)"
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
