"""Post-training int4 (linear, per-block) quantization of the Riva-4B decoder.

Data-free linear_quantize_weights is used instead of k-means palettization
because k-means over 4B params takes hours on CPU; per-block linear int4 is
minutes and typically sufficient to judge feasibility.

Usage:
    uv run quantize_int4.py --model-dir ./out
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
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()

    import coremltools as ct
    import coremltools.optimize as cto

    model_dir = Path(args.model_dir)

    config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=args.block_size,
        )
    )

    for name in ("riva4b_decoder_stateful", "riva4b_lm_head"):
        src = model_dir / f"{name}.mlpackage"
        dst = model_dir / f"{name}_int4.mlpackage"
        print(f"Quantizing {src.name} (block_size={args.block_size})...")
        t0 = time.time()
        model = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        quantized = cto.coreml.linear_quantize_weights(model, config)
        quantized.save(str(dst))
        print(f"  saved {dst.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
