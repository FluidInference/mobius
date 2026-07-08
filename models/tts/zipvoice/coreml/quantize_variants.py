"""Weight-compress the FmDecoder: int8 linear + 4-bit palettization variants.

TextEncoder stays fp16 (8.3 MB, irrelevant). Output dirs are drop-in
replacements for build/coreml so parity.py / benchmarks run unchanged.

Usage:
    .venv/bin/python -m coreml.quantize_variants --source-dir build/coreml
"""

import argparse
import shutil
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto


def size_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="build/coreml")
    args = parser.parse_args()

    src = Path(args.source_dir)
    fm = ct.models.MLModel(str(src / "FmDecoder.mlpackage"))

    variants = {}

    # int8: linear symmetric, per-channel (iOS16+)
    cfg8 = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    )
    variants["int8"] = cto.linear_quantize_weights(fm, cfg8)

    # 6-bit: k-means palettization, per-tensor (iOS16+). int4 was evaluated and
    # scrapped - see trials.md quantization matrix (per-tensor: -2.1dB; grouped
    # needs iOS18 and still ~2x the 6-bit error; nothing gained on the ANE path).
    cfg6 = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(nbits=6, mode="kmeans"))
    variants["6bit"] = cto.palettize_weights(fm, cfg6)

    for name, model in variants.items():
        out = src.parent / f"{src.name}-{name}"
        out.mkdir(parents=True, exist_ok=True)
        dst_te = out / "TextEncoder.mlpackage"
        if not dst_te.exists():
            shutil.copytree(src / "TextEncoder.mlpackage", dst_te)
        dst_fm = out / "FmDecoder.mlpackage"
        if dst_fm.exists():
            shutil.rmtree(dst_fm)
        model.save(str(dst_fm))
        print(f"{name}: {dst_fm}  {size_mb(dst_fm):.1f} MB (fp16 was {size_mb(src / 'FmDecoder.mlpackage'):.1f} MB)")


if __name__ == "__main__":
    main()
