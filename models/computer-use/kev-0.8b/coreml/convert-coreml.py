"""Convert the merged Kev row (Qwen3.5 backbone + pointer head) to one fixed-length Core ML package.

Outputs (build/L<length>_K<options>/):
  KevRow.mlpackage   hidden [1,L,D], cos/sin [L,R], decide_onehot [1,L], option_onehot [K,L], option_mask [K]
                     -> logits [K] (calibration temperature applied), probabilities [K]
  embeddings.f16     token embedding table [vocab, D] fp16, row-major (host gather)
  config.json        shapes, special token ids, pad id, temperature, source run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from kev_export import load_kev_row, row_inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, default=Path("build/merged"))
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-options", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--build", type=Path, default=Path("build"))
    args = ap.parse_args()
    L, K = args.length, args.max_options
    row, cfg, meta, embed = load_kev_row(args.merged, L, K, args.chunk_size)
    out = args.build / f"L{L}_K{K}"
    out.mkdir(parents=True, exist_ok=True)
    emb_path = out / "embeddings.f16"
    if not emb_path.exists():
        embed.to(torch.float16).numpy().tofile(emb_path)
    special = meta["special_tokens"]
    ids = [special["state"], 100, 200, special["q"], 300, special["opt"], 400, special["close_opt"], special["opt"], 500,
           special["close_opt"], special["decide"]]
    example = row_inputs(cfg, embed, ids, list(range(len(ids))), len(ids) - 1, [7, 10], L, K, meta["pad_id"])
    start = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(row, example, check_trace=False)
    shapes = [(1, L, cfg.hidden_size), (L, cfg.rotary_dim), (L, cfg.rotary_dim), (1, L), (K, L), (K,)]
    names = ["hidden", "cos", "sin", "decide_onehot", "option_onehot", "option_mask"]
    model = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in zip(names, shapes)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32), ct.TensorType(name="probabilities", dtype=np.float32)],
    )
    model.short_description = "Kev question row: Qwen3.5 backbone + pointer head, one causal row, prefill only"
    model.author = "Jared Palmer (Kev, Apache-2.0); Fluid Inference (Core ML conversion)"
    model.license = "Apache-2.0"
    model.user_defined_metadata.update({"source_run": meta["run"], "length": str(L), "max_options": str(K),
                                        "temperature": str(meta["temperature"])})
    package = out / f"KevRow_{args.precision}.mlpackage"
    model.save(str(package))
    config = {"source_run": meta["run"], "length": L, "max_options": K, "hidden_size": cfg.hidden_size,
              "rotary_dim": cfg.rotary_dim, "vocab_size": int(embed.shape[0]), "pad_id": meta["pad_id"],
              "special_tokens": special, "temperature": meta["temperature"], "precision": args.precision}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"saved {package} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
