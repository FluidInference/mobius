"""Export host-side runtime tables for the Swift port.

  * tables.safetensors — text_emb (2454×1024), speech_emb (8194×1024),
    text_pos_emb, speech_pos_emb (learned positional tables).
  * voice-default.safetensors — precomputed built-in voice conditioning:
    T3 cond embeds (1×34×1024, output of T3CondEnc on conds.pt) and the
    S3Gen ref dict (prompt_token, prompt_feat, embedding).

Voice cloning from a reference wav needs VoiceEncoder + S3TokenizerV2 +
CAMPPlus (not converted in this trial); any voice prepared with this script's
`--ref-wav` offline path ships as a tensors file the Swift runtime can load.

Usage:
    uv run python export-tables.py --out-dir build/tables [--ref-wav voice.wav]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("build/tables"))
    ap.add_argument("--ref-wav", type=Path, default=None,
                    help="optional reference wav → voice-<name>.safetensors")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from verify.e2e_coreml import load_model

    model = load_model()
    t3 = model.t3
    dt = torch.float16 if args.fp16 else torch.float32

    tables = {
        "text_emb": t3.text_emb.weight.detach().to(dt),
        "speech_emb": t3.speech_emb.weight.detach().to(dt),
        "text_pos_emb": t3.text_pos_emb.emb.weight.detach().to(dt),
        "speech_pos_emb": t3.speech_pos_emb.emb.weight.detach().to(dt),
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
        cond_emb = t3.prepare_conditioning(model.conds.t3)  # (1, 34, 1024)
    ref = model.conds.gen
    voice = {
        "t3_cond_emb": cond_emb.detach().to(dt).contiguous(),
        "prompt_token": ref["prompt_token"].detach().to(torch.int32).contiguous(),
        "prompt_feat": ref["prompt_feat"].detach().to(dt).contiguous(),
        "embedding": ref["embedding"].detach().to(dt).contiguous(),
    }
    p = args.out_dir / f"voice-{name}.safetensors"
    save_file(voice, str(p))
    print(f"saved {p}")


if __name__ == "__main__":
    main()
