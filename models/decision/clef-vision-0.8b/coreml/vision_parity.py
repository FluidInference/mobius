"""Vision tower parity on image files: trace-friendly VisionTower (and optionally the Core ML package) vs HF.

    uv sync --extra check
    uv run python vision_parity.py --merged <student>/merged/backbone --images a.jpg b.png [--package build/coreml/Vision_P784/VisionTower_fp32.mlpackage]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vision_export import load_vision, prepare_inputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--images", type=Path, nargs="+", required=True)
    ap.add_argument("--package", type=Path, help="Core ML vision package to check as well")
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--min-pixels", type=int, default=128 * 128)
    args = ap.parse_args()
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    tower, cfg, pos_table = load_vision(args.merged, 784)
    hf = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.merged), dtype=torch.float32).eval().model.visual
    processor = AutoProcessor.from_pretrained(str(args.merged)).image_processor
    processor.size = {"shortest_edge": args.min_pixels, "longest_edge": args.max_pixels}
    processor.min_pixels, processor.max_pixels = args.min_pixels, args.max_pixels
    package = None
    if args.package:
        import coremltools as ct

        package = ct.models.MLModel(str(args.package), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    worst = worst_pkg = 0.0
    for path in args.images:
        enc = processor(images=[Image.open(path).convert("RGB")], return_tensors="pt")
        pixel_values, grid = enc["pixel_values"], enc["image_grid_thw"][0]
        n = int(grid.prod())
        with torch.no_grad():
            ref = hf(pixel_values.float(), grid_thw=grid.reshape(1, 3)).pooler_output
            inputs = prepare_inputs(cfg, pos_table, pixel_values, grid, 784)
            ours = tower(*inputs[:-1])[: n // 4]
        diff = (ours - ref).abs().max().item()
        worst = max(worst, diff)
        line = f"{path.name:30s} grid {grid.tolist()} patches {n:4d} torch-module vs HF {diff:.2e}"
        if package is not None:
            feed = {k: v.numpy().astype(np.float32) for k, v in zip(("patches", "pos_embeds", "cos", "sin", "mask"), inputs[:-1])}
            out = package.predict(feed)["tokens"][: n // 4]
            pdiff = float(np.abs(out - ref.numpy()).max())
            worst_pkg = max(worst_pkg, pdiff)
            line += f"  Core ML vs HF {pdiff:.2e}"
        print(line, flush=True)
    print(f"WORST torch-module {worst:.3e}" + (f"  Core ML {worst_pkg:.3e}" if package is not None else ""),
          "PASS" if worst < 2e-3 and worst_pkg < 2e-3 else "CHECK")


if __name__ == "__main__":
    main()
