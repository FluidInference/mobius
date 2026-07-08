"""int4 grouped-channel palettization (kmeans, group_size=16) for the FmDecoder.

Requires an iOS18-target source package (grouped-channel LUTs are an iOS18
feature); the iOS17 packages fall back to per-tensor, which loses 2.1 dB.
TextEncoder stays fp16. Output dir is a drop-in replacement for build/coreml
so parity.py / benchmarks run unchanged.

Usage:
    .venv/bin/python -m coreml.quantize_int4_grouped \
        --source-dir build/coreml-ios18 --out-dir build/coreml-int4g
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
    parser.add_argument("--source-dir", default="build/coreml-ios18")
    parser.add_argument("--out-dir", default="build/coreml-int4g")
    parser.add_argument("--decoder", default="FmDecoder.mlpackage")
    parser.add_argument("--group-size", type=int, default=16)
    args = parser.parse_args()

    src = Path(args.source_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fm = ct.models.MLModel(str(src / args.decoder))
    cfg = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            nbits=4, mode="kmeans", granularity="per_grouped_channel", group_size=args.group_size
        )
    )
    quantized = cto.palettize_weights(fm, cfg)

    dst_fm = out / args.decoder
    if dst_fm.exists():
        shutil.rmtree(dst_fm)
    quantized.save(str(dst_fm))

    dst_te = out / "TextEncoder.mlpackage"
    if not dst_te.exists():
        shutil.copytree(src / "TextEncoder.mlpackage", dst_te)

    print(f"{dst_fm}  {size_mb(dst_fm):.1f} MB (source fp16 {size_mb(src / args.decoder):.1f} MB)")


if __name__ == "__main__":
    main()
