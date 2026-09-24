"""Convert the merged Cua-S1-4B multimodal vision tower to Core ML for a patch budget.

Output: build/multimodal/vision/CuaS1Vision_P<N>.mlpackage (+ pos_embed_table.f16, vision_config.json)
  patches [N, 768], pos_embed [N, 1024], cos/sin [N, 64], key_mask [1, N]  ->  image_embeds [N/4, 2560]
See qwen35_vision.VisionTower for the host-side contract.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from qwen35_vision import VisionTower

BASE = "Qwen/Qwen3.5-4B"


def precision(name: str, fp32_ops: set[str]):
    if name == "fp16":
        return ct.precision.FLOAT16
    if name == "fp32":
        return ct.precision.FLOAT32
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in fp32_ops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-patches", type=int, default=4096)
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument("--precision", choices=["fp16", "fp32", "mixed"], default="fp16")
    ap.add_argument("--fp32-ops", default="add,layer_norm", help="op types kept fp32 for --precision mixed")
    ap.add_argument("--tag", default="", help="output suffix override")
    args = ap.parse_args()
    N = args.max_patches
    vcfg = json.loads(Path(hf_hub_download(BASE, "config.json")).read_text())["vision_config"]
    tower = VisionTower(vcfg, N).eval()
    with safe_open(str(args.build / "merged-multimodal" / "vision.safetensors"), "pt") as f:
        table = tower.load_merged({k: f.get_tensor(k) for k in f.keys()})
    out_dir = args.build / "multimodal" / "vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.half().numpy().tofile(out_dir / "pos_embed_table.f16")
    (out_dir / "vision_config.json").write_text(json.dumps(vcfg, indent=2))

    t0 = time.time()
    example = (torch.zeros(N, 768), torch.zeros(N, 1024), torch.zeros(N, 64), torch.zeros(N, 64), torch.zeros(1, N))
    with torch.no_grad():
        traced = torch.jit.trace(tower, example, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType("patches", shape=(N, 768), dtype=np.float16),
            ct.TensorType("pos_embed", shape=(N, 1024), dtype=np.float16),
            ct.TensorType("cos", shape=(N, 64), dtype=np.float16),
            ct.TensorType("sin", shape=(N, 64), dtype=np.float16),
            ct.TensorType("key_mask", shape=(1, N), dtype=np.float16),
        ],
        outputs=[ct.TensorType("image_embeds", dtype=np.float16)],
        compute_precision=precision(args.precision, set(args.fp32_ops.split(","))),
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.short_description = (
        f"Cua-S1-4B-0.2 multimodal Qwen3.5 vision tower + merger, up to {N} patches ({N // 4} image tokens)"
    )
    suffix = f"-{args.tag}" if args.tag else ("" if args.precision == "fp16" else f"-{args.precision}")
    out = out_dir / f"CuaS1Vision_P{N}{suffix}.mlpackage"
    mlmodel.save(str(out))
    print(f"{out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
