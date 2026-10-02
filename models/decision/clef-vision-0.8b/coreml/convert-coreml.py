"""Convert a merged clef-vision student to the three Core ML packages FluidUse's ClefVisionManager loads.

    uv run python convert-coreml.py --merged <student>/merged/backbone --student <student>/best --out build/coreml

Inputs: the merged (LoRA folded) Qwen3.5-0.8B backbone directory (HF layout, model.safetensors + config.json) and the
student directory holding joint_head.safetensors + joint_head_config.json. Produces Vision_P784 (fp32), LM_L{512,1024,
2048} (fp16) and Head_L{512,1024,2048}_Q16_O64 (fp32), embeddings.f16 and Vision_P784/pos_embed.f32.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("build/coreml"))
    ap.add_argument("--buckets", default="512,1024,2048")
    args = ap.parse_args()
    args.merged, args.student, args.out = args.merged.resolve(), args.student.resolve(), args.out.resolve()  # subprocesses cwd differs
    here = Path(__file__).parent
    run = lambda *cmd: subprocess.run([sys.executable, *cmd], check=True, cwd=here)  # noqa: E731
    run("vision_export.py", "--merged", str(args.merged), "--patches", "784", "--precision", "fp32", "--out", str(args.out))
    for L in args.buckets.split(","):
        run("lm_export.py", "--merged", str(args.merged), "--length", L, "--precision", "fp16", "--out", str(args.out))
        run("head_export.py", "--student", str(args.student), "--length", L, "--max-q", "16", "--max-o", "64",
            "--precision", "fp32", "--out", str(args.out))
    print("done:", sorted(p.name for p in args.out.iterdir()))  # vision_export.py also wrote Vision_P784/pos_embed.f32


if __name__ == "__main__":
    main()
