"""Fold the Kev LoRA into its Qwen3.5 base with Kev's own loader (fp32, the path his published numbers use) and save
the merged text backbone, the pointer head and the calibration temperature for export.

    cd <kev repo> && uv run python <this dir>/merge_reference.py --run jaredpalmer/kev-0.8b --out <this dir>/build/merged
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from kev.checkpoint import Checkpoint, LoadOptions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ck = Checkpoint(args.run)
    tok, model = ck.load("cpu", LoadOptions(backend="torch", dtype=torch.float32, merge=True))
    model.eval()
    lm = model.lm
    state = {k: v.detach().float().contiguous() for k, v in lm.state_dict().items()}
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.out / "text.safetensors"))
    head = {"q.weight": model.head.q.weight, "q.bias": model.head.q.bias, "k.weight": model.head.k.weight,
            "k.bias": model.head.k.bias}
    save_file({k: v.detach().float().contiguous() for k, v in head.items()}, str(args.out / "head.safetensors"))
    config = lm.config.to_dict()
    meta = {"run": args.run, "resolved": str(ck.path), "lm_class": type(lm).__name__, "temperature": float(model.head.temperature),
            "pointer_scale": float(model.head.scale), "pad_id": int(model.pad_id), "text_config": config,
            "special_tokens": {name: tok.convert_tokens_to_ids(t) for name, t in
                               zip(("state", "q", "opt", "close_opt", "decide"),
                                   ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"))}}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({k: meta[k] for k in ("lm_class", "temperature", "pointer_scale", "pad_id", "special_tokens")}, indent=2))
    print("keys:", len(state), list(state)[:4], "...")
    print("layer_types:", config.get("layer_types"))


if __name__ == "__main__":
    main()
