"""Export host-side runtime tables for the Nano Swift port.

  * tables.safetensors — text_emb (50276×768), speech_emb (6563×768).
    No positional tables: GPT2's wpe is applied in-graph by the T3 packages.
  * voice-default.safetensors — precomputed built-in voice conditioning:
    T3 cond embeds (1×376×768, output of T3CondEnc + speech_emb on conds.pt)
    and the S3Gen ref dict (prompt_token, prompt_feat, embedding).

Voice cloning from a reference wav needs VoiceEncoder + S3TokenizerV2 +
CAMPPlus (not converted in this trial); voices prepared with `--ref-wav`
offline ship as tensors files the Swift runtime can load.

Usage:
    uv run python export-tables-nano.py --out-dir build/tables-nano [--ref-wav voice.wav]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_model():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.nano_ckpt import load_nano

    return load_nano()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("build/tables-nano"))
    ap.add_argument("--ref-wav", type=Path, default=None,
                    help="optional reference wav → voice-<name>.safetensors")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    t3 = model.t3
    dt = torch.float16 if args.fp16 else torch.float32

    tables = {
        "text_emb": t3.text_emb.weight.detach().to(dt).contiguous(),
        "speech_emb": t3.speech_emb.weight.detach().to(dt).contiguous(),
    }
    p = args.out_dir / "tables.safetensors"
    save_file(tables, str(p))
    print(f"saved {p} ({sum(v.numel() * v.element_size() for v in tables.values()) / 1e6:.1f} MB)")

    if args.ref_wav is not None:
        model.prepare_conditionals(str(args.ref_wav))
        name = args.ref_wav.stem
    else:
        name = "default"
    with torch.no_grad():
        cond_emb = t3.prepare_conditioning(model.conds.t3)  # (1, 376, 768)
    ref = model.conds.gen
    voice = {
        "t3_cond_emb": cond_emb.detach().to(dt).contiguous(),
        "prompt_token": ref["prompt_token"].detach().to(torch.int32).contiguous(),
        "prompt_feat": ref["prompt_feat"].detach().to(dt).contiguous(),
        "embedding": ref["embedding"].detach().to(dt).contiguous(),
    }
    p = args.out_dir / f"voice-{name}.safetensors"
    save_file(voice, str(p))
    print(f"saved {p} (cond {tuple(cond_emb.shape)}, "
          f"prompt {tuple(ref['prompt_token'].shape)})")


if __name__ == "__main__":
    main()
