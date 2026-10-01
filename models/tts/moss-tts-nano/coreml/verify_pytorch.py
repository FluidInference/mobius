"""PyTorch reference synthesis for MOSS-TTS-Nano (voice clone mode).

Produces the baseline WAV that CoreML outputs are compared against.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog near the riverbank.")
    parser.add_argument("--prompt-audio", default=str(HERE / "assets" / "en_2.wav"))
    parser.add_argument("--output", default=str(HERE / "build" / "ref_pytorch.wav"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nq", type=int, default=None)
    parser.add_argument("--do-sample", type=int, default=1)
    parser.add_argument("--max-new-frames", type=int, default=375)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(TTS_REPO, trust_remote_code=True, dtype=torch.float32)
    model.config.attn_implementation = "eager"
    model.config.local_transformer_attn_implementation = "eager"
    for module in model.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = "eager"
    model.eval()
    print(f"load: {time.perf_counter() - t0:.2f}s  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    t1 = time.perf_counter()
    result = model.inference(
        text=args.text,
        output_audio_path=args.output,
        mode="voice_clone",
        prompt_audio_path=args.prompt_audio,
        audio_tokenizer_type="moss-audio-tokenizer-nano",
        audio_tokenizer_pretrained_name_or_path=CODEC_REPO,
        device=args.device,
        nq=args.nq,
        do_sample=bool(args.do_sample),
        max_new_frames=args.max_new_frames,
        voice_clone_max_text_tokens=0,
    )
    dt = time.perf_counter() - t1
    keys = {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in result.items() if k != "waveform"}
    print("result:", keys)
    import numpy as np

    out_dir = Path(args.output).parent
    stem = Path(args.output).stem
    np.save(out_dir / f"{stem}_audio_token_ids.npy", np.asarray(result["audio_token_ids"]))
    np.save(out_dir / f"{stem}_prompt_audio_token_ids.npy", np.asarray(result["reference_audio_token_ids"]))
    import soundfile as sf

    audio, sr = sf.read(args.output)
    dur = audio.shape[0] / sr
    print(f"synth: {dt:.2f}s  audio={dur:.2f}s @ {sr}Hz shape={audio.shape}  RTFx={dur/dt:.2f}")


if __name__ == "__main__":
    main()
