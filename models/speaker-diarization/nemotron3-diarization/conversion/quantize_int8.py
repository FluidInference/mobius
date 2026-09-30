"""Weight-only int8 quantization of the Nemotron 3 diarizer mlpackages.

Same recipe as the multilingual Nemotron ASR encoder (per-channel linear-symmetric
int8 weights, activations stay fp16, ANE residency preserved).

Usage: uv run python quantize_int8.py [--variants low offline ...]
"""

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as ctc

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(config.VARIANTS))
    ap.add_argument("--build-dir", default="build")
    args = ap.parse_args()

    build = Path(args.build_dir)
    for name in args.variants:
        src = build / f"Nemotron3Diarizer_{name}.mlpackage"
        dst = build / f"Nemotron3Diarizer_{name}_int8.mlpackage"
        mlmodel = ct.models.MLModel(str(src))
        cfg = ctc.OptimizationConfig(
            global_config=ctc.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        )
        quantized = ctc.linear_quantize_weights(mlmodel, config=cfg)
        if dst.exists():
            shutil.rmtree(dst)
        quantized.save(str(dst))

        def du(p):
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6

        print(f"{name}: {du(src):.1f} MB -> {du(dst):.1f} MB")


if __name__ == "__main__":
    main()
