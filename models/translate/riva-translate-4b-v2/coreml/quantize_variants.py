"""Produce alternative compressed decoder variants to find the fastest GPU dequant path.

Variants:
  int8   — linear symmetric, per-channel (classic fast path)
  pal4   — 4-bit palettization LUT, per-grouped-channel(16), uniform mode
  int4c  — linear symmetric int4, per-channel (no per-block scales)

Usage:
    uv run quantize_variants.py --model-dir ./out --variants int8,pal4
"""

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "coremltools>=8.0",
#     "numpy<2",
# ]
# ///

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./out")
    parser.add_argument("--variants", default="int8,pal4,int4c")
    args = parser.parse_args()

    import coremltools as ct
    import coremltools.optimize as cto

    model_dir = Path(args.model_dir)
    src = model_dir / "riva4b_decoder_stateful.mlpackage"

    def cfg_int8():
        return cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_channel"
            )
        )

    def cfg_pal4():
        return cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(
                mode="uniform", nbits=4, granularity="per_grouped_channel", group_size=16
            )
        )

    def cfg_int4c():
        return cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int4", granularity="per_channel"
            )
        )

    makers = {"int8": cfg_int8, "pal4": cfg_pal4, "int4c": cfg_int4c}

    for name in args.variants.split(","):
        name = name.strip()
        dst = model_dir / f"riva4b_decoder_stateful_{name}.mlpackage"
        if dst.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"Quantizing variant {name}...")
        t0 = time.time()
        model = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        if name == "pal4":
            quantized = cto.coreml.palettize_weights(model, makers[name]())
        else:
            quantized = cto.coreml.linear_quantize_weights(model, makers[name]())
        quantized.save(str(dst))
        print(f"  saved {dst.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
