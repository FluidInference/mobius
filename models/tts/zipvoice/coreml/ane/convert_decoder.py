"""Convert AneFmDecoder to CoreML (iOS17 fp16, fixed S=1024).

Same I/O contract as the original FmDecoder (t, x, text_condition,
speech_condition, guidance_scale, padding_mask -> v), so coreml/parity.py
feeds and swift/RssBench.swift work unchanged. Saves
build/coreml-ane/AneFmDecoder.mlpackage, copies TextEncoder.mlpackage from
build/coreml, and compiles both to .mlmodelc (the decoder as
FmDecoder.mlmodelc so rss_bench finds it).

Run: .venv/bin/python -m coreml.ane.convert_decoder
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

import coremltools as ct
import numpy as np
import torch

from coreml.ane.decoder import AneFmDecoder, AneFmDecoderIO
from coreml.convert_coreml import FEAT_DIM, load_model, patch_coremltools_int

SEQ_LEN = 1024
SRC_DIR = Path("build/coreml")


def compile_to(mlpackage: Path, dest: Path):
    compiled = ct.models.utils.compile_model(str(mlpackage))
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(compiled, dest)
    print(f"compiled {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="build/coreml-ane")
    parser.add_argument(
        "--deployment-target", default="iOS17", choices=["iOS17", "iOS18"],
        help="iOS18 enables grouped-channel palettization downstream",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_coremltools_int()
    model, _ = load_model()

    ane = AneFmDecoderIO(AneFmDecoder(model.fm_decoder, SEQ_LEN)).eval()
    for sl, err in ane.core.basis_errs.items():
        print(f"pos basis S={sl}: rel_max_err={err:.2e}")

    t = torch.tensor([0.5])
    x = torch.randn(1, SEQ_LEN, FEAT_DIM)
    text = torch.randn(1, SEQ_LEN, FEAT_DIM)
    speech = torch.randn(1, SEQ_LEN, FEAT_DIM)
    g = torch.tensor([3.0])
    mask = torch.zeros(1, SEQ_LEN)
    mask[0, SEQ_LEN - 124 :] = 1.0

    with torch.no_grad():
        ref = ane(t, x, text, speech, g, mask)
        traced = torch.jit.trace(ane, (t, x, text, speech, g, mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="t", shape=(1,), dtype=np.float32),
            ct.TensorType(name="x", shape=(1, SEQ_LEN, FEAT_DIM), dtype=np.float32),
            ct.TensorType(
                name="text_condition", shape=(1, SEQ_LEN, FEAT_DIM), dtype=np.float32
            ),
            ct.TensorType(
                name="speech_condition", shape=(1, SEQ_LEN, FEAT_DIM), dtype=np.float32
            ),
            ct.TensorType(name="guidance_scale", shape=(1,), dtype=np.float32),
            ct.TensorType(name="padding_mask", shape=(1, SEQ_LEN), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="v", dtype=np.float32)],
        minimum_deployment_target=getattr(ct.target, args.deployment_target),
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    pkg = out_dir / "AneFmDecoder.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    mlmodel.save(str(pkg))
    print(f"saved {pkg}")

    # Sanity: fp16 CoreML (CPU) vs torch fp32 on the trace inputs.
    cm = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = cm.predict(
        {
            "t": t.numpy(),
            "x": x.numpy(),
            "text_condition": text.numpy(),
            "speech_condition": speech.numpy(),
            "guidance_scale": g.numpy(),
            "padding_mask": mask.numpy(),
        }
    )["v"]
    n = SEQ_LEN - 124
    a, b = ref.numpy()[:, :n].ravel(), out[:, :n].ravel()
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"CoreML fp16 (CPU) vs torch fp32: cos={c:.6f} max_abs={np.abs(a - b).max():.4f}")

    # TextEncoder + compiled models.
    te_dst = out_dir / "TextEncoder.mlpackage"
    if not te_dst.exists():
        shutil.copytree(SRC_DIR / "TextEncoder.mlpackage", te_dst)
        print(f"copied {te_dst}")
    compile_to(pkg, out_dir / "FmDecoder.mlmodelc")
    compile_to(te_dst, out_dir / "TextEncoder.mlmodelc")


if __name__ == "__main__":
    main()
