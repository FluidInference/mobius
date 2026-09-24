"""Convert the merged Cua-S1-4B text decoder to Core ML, split into layer parts.

Outputs (build/<modality>/L<len>/):
  CuaS1Decoder_part{i}.mlpackage   hidden [1,L,2560] fp16, cos/sin [L,64] -> hidden
  last part additionally takes last_onehot [1,L] and returns letter_logits [1,26]
  embeddings.f16                   tied embedding table, row-major [vocab, 2560] fp16 (host gather)
  config.json                      shapes, letter token ids, layer split
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

from qwen35_export import DecoderChunk, TextConfig

BASE = "Qwen/Qwen3.5-4B"


def load_state(path: Path, keys_prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(str(path), "pt") as f:
        for key in f.keys():
            if key.startswith(keys_prefixes):
                out[key] = f.get_tensor(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "multimodal"], default="text")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--only-part", type=int, default=-1)
    ap.add_argument("--build", type=Path, default=Path("build"))
    args = ap.parse_args()

    raw_cfg = json.loads(Path(hf_hub_download(BASE, "config.json")).read_text())["text_config"]
    cfg = TextConfig(raw_cfg)
    merged = args.build / f"merged-{args.modality}"
    letters = json.loads((merged / "letters.json").read_text())
    out_dir = args.build / args.modality / f"L{args.length}"
    out_dir.mkdir(parents=True, exist_ok=True)

    L, D, R = args.length, cfg.hidden_size, cfg.rotary_dim
    per = cfg.num_layers // args.parts
    bounds = [(i * per, (i + 1) * per if i + 1 < args.parts else cfg.num_layers) for i in range(args.parts)]

    emb_path = out_dir.parent / "embeddings.f16"
    if not emb_path.exists():
        emb = load_state(merged / "text.safetensors", ("embed_tokens.",))["embed_tokens.weight"]
        emb.half().numpy().tofile(emb_path)
        print(f"wrote {emb_path} {tuple(emb.shape)}")

    for part, (start, end) in enumerate(bounds):
        if args.only_part >= 0 and part != args.only_part:
            continue
        with_head = part == len(bounds) - 1
        prefixes = tuple(f"layers.{i}." for i in range(start, end))
        if with_head:
            prefixes += ("norm.", "embed_tokens.")
        state = load_state(merged / "text.safetensors", prefixes)
        letter_rows = state.pop("embed_tokens.weight")[letters["token_ids"]].float() if with_head else None
        model = DecoderChunk(cfg, start, end, L, args.chunk_size, with_head, len(letters["token_ids"]))
        model.load_merged(state, letter_rows)
        model.eval()
        del state

        ex = [torch.zeros(1, L, D), torch.zeros(L, R), torch.zeros(L, R)]
        inputs = [
            ct.TensorType("hidden", shape=(1, L, D), dtype=np.float16),
            ct.TensorType("cos", shape=(L, R), dtype=np.float16),
            ct.TensorType("sin", shape=(L, R), dtype=np.float16),
        ]
        if with_head:
            ex.append(torch.zeros(1, L))
            inputs.append(ct.TensorType("last_onehot", shape=(1, L), dtype=np.float16))
        outputs = [ct.TensorType("letter_logits" if with_head else "hidden_out", dtype=np.float16)]

        t0 = time.time()
        with torch.no_grad():
            traced = torch.jit.trace(model, tuple(ex), check_trace=False)
        del model
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.ALL,
        )
        mlmodel.short_description = (
            f"Cua-S1-4B-0.2 ({args.modality}) Qwen3.5 decoder layers {start}-{end - 1}, prefill L={L}"
        )
        path = out_dir / f"CuaS1Decoder_part{part}.mlpackage"
        mlmodel.save(str(path))
        print(f"part {part} layers {start}-{end - 1} -> {path} ({time.time() - t0:.0f}s)", flush=True)
        del traced, mlmodel

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "modality": args.modality,
                "seq_len": L,
                "hidden_size": D,
                "rotary_dim": R,
                "rope_theta": cfg.rope_theta,
                "mrope_section": cfg.mrope_section,
                "vocab_size": raw_cfg["vocab_size"],
                "delta_chunk_size": args.chunk_size,
                "parts": [{"layers": [s, e]} for s, e in bounds],
                "letters": letters["letters"],
                "letter_token_ids": letters["token_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
