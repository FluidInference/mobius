"""Export embedding tables required by Swift for LM-input construction.

Swift needs two lookup tables to build the `lm_input_embeds` tensor fed into the
LLM-Prefill CoreML model:

    1. Qwen2 text embedding  : [151936, 896]  from  llm.pt
       key = "llm.model.model.embed_tokens.weight"
    2. Speech embedding      : [6761,   896]  from  llm.pt
       key = "speech_embedding.weight"

Both get shipped alongside the mlpackages so the on-device wrapper can:

    lm_input = concat([
        speech_embedding[sos_id],            # sos
        text_embedding[prompt_ids + text_ids],
        speech_embedding[task_id],           # task id
        speech_embedding[prompt_speech_ids], # prompt speech tokens
    ])

We emit both a safetensors file (Swift-friendly, mmap-able via swift-transformers
or its own parser) and a small JSON metadata file describing the layout.

Usage:
    uv run python export-embeddings.py --output-dir ./build/embeddings
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


HERE = Path(__file__).parent
LLM_PT = HERE / "cosyvoice3_dl" / "llm.pt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(HERE / "build" / "embeddings"))
    p.add_argument("--fp16", action="store_true",
                   help="Save as float16 (halves file size; ~2% absolute MAE for LLM input)")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading {LLM_PT.name}")
    sd = torch.load(str(LLM_PT), map_location="cpu", weights_only=False)
    text_w = sd["llm.model.model.embed_tokens.weight"].contiguous()
    sp_w = sd["speech_embedding.weight"].contiguous()
    print(f"      text_embedding  : {tuple(text_w.shape)} {text_w.dtype}")
    print(f"      speech_embedding: {tuple(sp_w.shape)} {sp_w.dtype}")

    dtype = torch.float16 if args.fp16 else torch.float32
    text_w = text_w.to(dtype)
    sp_w = sp_w.to(dtype)

    tag = "fp16" if args.fp16 else "fp32"
    st_path = out_dir / f"embeddings-{tag}.safetensors"
    print(f"[2/3] Saving safetensors → {st_path.name}")
    save_file(
        {
            "text_embedding":   text_w,
            "speech_embedding": sp_w,
        },
        str(st_path),
        metadata={
            "text_vocab_size":  str(text_w.shape[0]),
            "speech_vocab":     str(sp_w.shape[0]),
            "hidden_dim":       str(text_w.shape[1]),
            "sos_id":           "6561",
            "task_id":          "6563",
            "eos_id_start":     "6561",
            "eos_id_end":       "6761",
            "endofprompt_id":   "151646",
            "dtype":            tag,
        },
    )

    meta_path = out_dir / f"embeddings-{tag}.json"
    print(f"[3/3] Writing metadata → {meta_path.name}")
    with open(meta_path, "w") as f:
        json.dump({
            "files": {
                "safetensors": st_path.name,
            },
            "tensors": {
                "text_embedding": {
                    "shape": list(text_w.shape),
                    "dtype": tag,
                    "description": "Qwen2 model.embed_tokens.weight",
                },
                "speech_embedding": {
                    "shape": list(sp_w.shape),
                    "dtype": tag,
                    "description": "CosyVoice3 LLM speech_embedding.weight "
                                   "(tokens 0..6560 are speech; 6561=sos/eos; 6563=task_id)",
                },
            },
            "usage": (
                "lm_input = concat(["
                "speech_embedding[sos=6561], "
                "text_embedding[prompt_ids + tts_ids], "
                "speech_embedding[task=6563], "
                "speech_embedding[prompt_speech_ids]"
                "], dim=1)"
            ),
            "stop_tokens": [6561, 6762],
            "sos": 6561,
            "task_id": 6563,
            "endofprompt": 151646,
        }, f, indent=2)

    bytes_total = st_path.stat().st_size
    print(f"\nsaved: {st_path}  ({bytes_total / 1024 / 1024:.1f} MB)")
    print(f"       {meta_path}")


if __name__ == "__main__":
    main()
