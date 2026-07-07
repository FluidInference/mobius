"""Synthesize sample wavs with the CoreML pipeline (torch vocoder) and
transcribe them with Whisper to sanity-check intelligibility.

Usage:
    .venv/bin/python -m coreml.synth_samples --out-dir build/samples
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

from coreml.convert_coreml import MAX_FRAMES, MAX_TOKENS, load_model
from coreml.parity import expand_text_condition
from scripts.reference_infer import load_models_cpu_torch
from zipvoice.models.modules.solver import get_time_steps

TEXTS = [
    "The quick brown fox jumps over the lazy dog, and honestly, it felt great.",
    "FluidAudio runs speech models locally on Apple silicon, no cloud required.",
    "In the last seven days, we've signed the same number of contracts as we signed in the whole of Q4.",
    "Voice cloning at forty eight kilohertz, from a five second reference clip.",
]


def synth(text, tokenizer, te, fm, prompt_tokens, prompt_features, prompt_len, prompt_rms, vocos, seed=42, speed=1.3):
    text_tokens = tokenizer.texts_to_token_ids([text])[0]
    cat = prompt_tokens + text_tokens
    S = len(cat)
    assert S + 1 <= MAX_TOKENS, f"{S + 1} tokens > {MAX_TOKENS}"

    tok_in = np.zeros((1, MAX_TOKENS), dtype=np.int32)
    tok_in[0, :S] = cat
    tmask = np.zeros((1, MAX_TOKENS), dtype=np.float32)
    tmask[0, S:] = 1.0
    embeds = te.predict({"tokens": tok_in, "padding_mask": tmask})["token_embeds"]
    embeds = torch.from_numpy(embeds[:, : S + 1, :].astype(np.float32))

    # speed: generate() default is 1.0 * 1.3
    features_len = prompt_len + int(np.ceil(prompt_len / len(prompt_tokens) * len(text_tokens) / speed))
    assert features_len <= MAX_FRAMES, f"{features_len} frames > {MAX_FRAMES}"

    text_cond = expand_text_condition(embeds, S, features_len)
    speech_cond = torch.nn.functional.pad(prompt_features, (0, 0, 0, features_len - prompt_features.size(1)))

    pad = lambda z: torch.nn.functional.pad(z, (0, 0, 0, MAX_FRAMES - z.size(1))).numpy().astype(np.float32)
    fmask = np.zeros((1, MAX_FRAMES), dtype=np.float32)
    fmask[0, features_len:] = 1.0

    timesteps = get_time_steps(num_step=4, t_shift=0.5)
    torch.manual_seed(seed)
    x = torch.randn(1, features_len, 100)
    text_np, speech_np = pad(text_cond), pad(speech_cond)
    for step in range(4):
        t_cur, t_next = float(timesteps[step]), float(timesteps[step + 1])
        v = fm.predict({
            "t": np.array([t_cur], dtype=np.float32), "x": pad(x),
            "text_condition": text_np, "speech_condition": speech_np,
            "guidance_scale": np.array([3.0], dtype=np.float32), "padding_mask": fmask,
        })["v"]
        v = torch.from_numpy(v[:, :features_len, :].astype(np.float32))
        x1p, x0p = x + (1.0 - t_cur) * v, x - t_cur * v
        x = (1.0 - t_next) * x0p + t_next * x1p if step < 3 else x1p

    with torch.no_grad():
        mel = x[:, prompt_len:features_len, :].permute(0, 2, 1) / 0.1
        wav = vocos.decode(mel).squeeze(1).clamp(-1, 1).numpy().squeeze()
    if prompt_rms < 0.1:
        wav = wav * (prompt_rms / 0.1)
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", default="build/oracle")
    parser.add_argument("--coreml-dir", default="build/coreml")
    parser.add_argument("--out-dir", default="build/samples")
    args = parser.parse_args()

    oracle = Path(args.oracle_dir)
    meta = json.loads((oracle / "meta.json").read_text())
    prompt_features = torch.from_numpy(np.load(oracle / "prompt_features.npy"))
    prompt_tokens = np.load(oracle / "prompt_tokens.npy").tolist()

    _, _, vocos, tokenizer, _ = load_models_cpu_torch()
    vocos.freq_range = 12000
    vocos.return_48k = True

    te = ct.models.MLModel(str(Path(args.coreml_dir) / "TextEncoder.mlpackage"))
    fm = ct.models.MLModel(str(Path(args.coreml_dir) / "FmDecoder.mlpackage"))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(TEXTS):
        wav = synth(text, tokenizer, te, fm, prompt_tokens, prompt_features,
                    meta["prompt_features_lens"], meta["prompt_rms"], vocos)
        sf.write(out / f"sample_{i}.wav", wav, 48000)
        print(f"sample_{i}.wav ({len(wav)/48000:.2f}s): {text}")


if __name__ == "__main__":
    main()
