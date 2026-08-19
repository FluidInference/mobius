"""CoreML benchmark: original Zipformer2EncoderLayer vs AneZipformerLayer.

Converts both to CoreML (iOS17, fp16, mlprogram, fixed S=1024), times each
under CPU_AND_NE and CPU_AND_GPU (3 warmup + 10 timed), checks the ANE
layer's fp16 output against the torch fp32 reference, and runs
coreml-cli --fallback on the compiled ANE package.

Run: .venv/bin/python -m coreml.ane.bench
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import coremltools as ct
import numpy as np
import torch
from torch import Tensor, nn

from coreml.ane.layer import AneZipformerLayer, ane_to_tbc, tbc_to_ane
from coreml.convert_coreml import (
    load_model,
    patch_coremltools_int,
    patch_simple_downsample,
)

SEQ_LEN = 1024
EMBED_DIM = 512
OUT_DIR = Path("build/coreml/ane_trial")
COREML_CLI = Path("/Users/hanweng/Documents/mobius-zipvoice/tools/coreml-cli")


class OrigLayerWrapper(nn.Module):
    """(S, 1, C) seq-first original layer with frozen pos_emb constant."""

    def __init__(self, layer, pos_emb: Tensor):
        super().__init__()
        self.layer = layer
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, src: Tensor, time_emb: Tensor) -> Tensor:
        return self.layer(src, self.pos_emb, time_emb=time_emb)


def convert(traced, inputs, name):
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="out", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    path = OUT_DIR / f"{name}.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    print(f"saved {path}")
    return path


def bench(path, feeds, units, warmup=3, runs=10):
    model = ct.models.MLModel(str(path), compute_units=units)
    for _ in range(warmup):
        out = model.predict(feeds)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = model.predict(feeds)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), out["out"]


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fallback_check(mlpackage: Path):
    compiled = ct.models.utils.compile_model(str(mlpackage))
    dest = mlpackage.with_suffix(".mlmodelc")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(compiled, dest)
    print(f"\n--- coreml-cli --fallback: {dest.name} ---")
    try:
        res = subprocess.run(
            ["uv", "run", "coreml-cli", str(dest.resolve()), "--fallback"],
            cwd=COREML_CLI,
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr[-2000:])
    except Exception as e:  # noqa: BLE001 — report and continue
        print(f"coreml-cli failed: {e}")


def main():
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patch_coremltools_int()
    patch_simple_downsample()
    model, _ = load_model()
    model.eval()

    enc0 = model.fm_decoder.encoders[0]
    layer = enc0.layers[0] if hasattr(enc0, "layers") else enc0.encoder.layers[0]
    with torch.no_grad():
        pos_emb = enc0.encoder_pos(torch.randn(SEQ_LEN, 1, EMBED_DIM)).detach()

    src = torch.randn(SEQ_LEN, 1, EMBED_DIM)
    time_emb = torch.randn(1, EMBED_DIM)

    # torch fp32 reference
    with torch.no_grad():
        ref = layer(src, pos_emb, time_emb=time_emb).numpy()

    # --- original layer, seq-first ---
    orig = OrigLayerWrapper(layer, pos_emb).eval()
    with torch.no_grad():
        traced_orig = torch.jit.trace(orig, (src, time_emb))
    orig_path = convert(
        traced_orig,
        [
            ct.TensorType(name="src", shape=(SEQ_LEN, 1, EMBED_DIM), dtype=np.float32),
            ct.TensorType(name="time_emb", shape=(1, EMBED_DIM), dtype=np.float32),
        ],
        "OrigLayer",
    )

    # --- ANE-canonical layer ---
    ane = AneZipformerLayer(layer, pos_emb, SEQ_LEN).eval()
    x_ane = tbc_to_ane(src)
    t_ane = time_emb.reshape(1, EMBED_DIM, 1, 1)
    with torch.no_grad():
        traced_ane = torch.jit.trace(ane, (x_ane, t_ane))
    ane_path = convert(
        traced_ane,
        [
            ct.TensorType(name="x", shape=(1, EMBED_DIM, 1, SEQ_LEN), dtype=np.float32),
            ct.TensorType(name="time_emb", shape=(1, EMBED_DIM, 1, 1), dtype=np.float32),
        ],
        "AneLayer",
    )

    orig_feeds = {"src": src.numpy(), "time_emb": time_emb.numpy()}
    ane_feeds = {"x": x_ane.numpy(), "time_emb": t_ane.numpy()}

    print(f"\n{'model':<12} {'units':<12} {'mean ms':>10}")
    results = {}
    for name, path, feeds in (
        ("orig", orig_path, orig_feeds),
        ("ane", ane_path, ane_feeds),
    ):
        for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_AND_GPU):
            ms, out = bench(path, feeds, units)
            results[(name, units.name)] = (ms, out)
            print(f"{name:<12} {units.name:<12} {ms:>10.2f}")

    # fp16 CoreML (ANE) vs torch fp32 accuracy
    ane_ne_out = results[("ane", "CPU_AND_NE")][1]  # (1, C, 1, S)
    ane_out_tbc = ane_to_tbc(torch.from_numpy(ane_ne_out)).numpy()
    orig_ne_out = results[("orig", "CPU_AND_NE")][1]
    print(f"\nane CoreML fp16 (NE) vs torch fp32: cos={cosine(ane_out_tbc, ref):.6f}, "
          f"max_abs={np.abs(ane_out_tbc - ref).max():.4f}")
    print(f"orig CoreML fp16 (NE) vs torch fp32: cos={cosine(orig_ne_out, ref):.6f}")

    fallback_check(ane_path)
    fallback_check(orig_path)


if __name__ == "__main__":
    main()
