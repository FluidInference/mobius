"""Generate the PyTorch reference for NeuTTS-2E parity checks.

Runs the upstream pipeline (transformers backbone + neucodec decode, no
watermark) with a fixed seed and saves everything the CoreML side needs to
compare against:

    build/ref/prompt_ids.json      — full prompt token ids
    build/ref/gen_ids.json         — generated token ids (speech tokens + EOS)
    build/ref/ref_codes.json       — NeuCodec code indices actually decoded
    build/ref/ref.wav              — 24 kHz reference audio (bf16 backbone, fp32 codec)

Usage:
    uv run python gen-pytorch-ref.py [--text ...] [--speaker emily] [--emotion happy]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompt import MAX_CONTEXT, SAMPLE_RATE, build_prompt_ids, extract_speech_codes

BACKBONE_REPO = "neuphonic/neutts-2e"
CODEC_REPO = "neuphonic/neucodec"

DEFAULT_TEXT = "I can't believe it's finally here! The whole team worked so hard on this."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--speaker", default="emily")
    p.add_argument("--emotion", default="happy")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--output-dir", default="build/ref")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading backbone {BACKBONE_REPO} (bf16, cpu)...")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_REPO)
    backbone = AutoModelForCausalLM.from_pretrained(BACKBONE_REPO, dtype=torch.bfloat16)
    backbone.eval()

    # Sanity: speech token ids must be contiguous (extract_speech_codes relies on it).
    s0 = tokenizer.convert_tokens_to_ids("<|speech_0|>")
    assert tokenizer.convert_tokens_to_ids("<|speech_1|>") == s0 + 1
    assert tokenizer.convert_tokens_to_ids("<|speech_65535|>") == s0 + 65_535

    prompt_ids = build_prompt_ids(tokenizer, args.text, args.speaker, args.emotion)
    print(f"      prompt length: {len(prompt_ids)} tokens")

    print(f"[2/4] Generating (seed={args.seed}, temp={args.temperature}, top_k={args.top_k})...")
    torch.manual_seed(args.seed)
    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0)
    with torch.no_grad():
        output = backbone.generate(
            prompt_tensor,
            max_length=MAX_CONTEXT,
            eos_token_id=speech_end_id,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            use_cache=True,
            min_new_tokens=50,
        )
    gen_ids = output[0, len(prompt_ids):].tolist()
    codes = extract_speech_codes(tokenizer, gen_ids)
    print(f"      generated {len(gen_ids)} tokens -> {len(codes)} speech codes "
          f"({len(codes) / 50.0:.2f}s audio)")

    print(f"[3/4] Decoding with {CODEC_REPO} (fp32, cpu)...")
    from neucodec import NeuCodec

    codec = NeuCodec.from_pretrained(CODEC_REPO)
    codec.eval()
    with torch.no_grad():
        codes_t = torch.tensor(codes, dtype=torch.long)[None, None, :]
        wav = codec.decode_code(codes_t).numpy()[0, 0, :]

    print(f"[4/4] Saving to {out_dir}/ ...")
    (out_dir / "prompt_ids.json").write_text(json.dumps(prompt_ids))
    (out_dir / "gen_ids.json").write_text(json.dumps(gen_ids))
    (out_dir / "ref_codes.json").write_text(json.dumps(codes))
    (out_dir / "meta.json").write_text(json.dumps(vars(args)))
    sf.write(out_dir / "ref.wav", wav.astype(np.float32), SAMPLE_RATE)
    print(f"      ref.wav: {len(wav) / SAMPLE_RATE:.2f}s, peak {np.abs(wav).max():.3f}")


if __name__ == "__main__":
    main()
