"""Dump fixtures for the FluidAudio Swift LuxTts backend.

Produces small JSON + raw-binary fixtures that pin every host-side algorithm
the Swift port implements:

  - EmiliaTokenizer output (prompt + 2 test texts): token strings + ids
  - VocosFbank prompt mel (24 kHz, n_fft 1024, hop 256, 100 mels, x0.1 scale)
  - duration expansion (prepare_avg_tokens_durations / get_tokens_index)
  - anchor-Euler solver timesteps (t_shift 0.5, 4 steps) + a mini numeric
    solver trajectory
  - an end-to-end CoreML (gpu graphs + Vocoder555) synthesis stat block for
    TEXTS[0] (duration + RMS gates; exact waveform match is not expected)

Usage:
    .venv/bin/python -m coreml.dump_swift_fixtures \
        --prompt-audio build/wavs/07_prompt_clip.wav \
        --out-dir <FluidAudio>/Tests/FluidAudioTests/TTS/LuxTts/Resources
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import coremltools as ct
import librosa
import numpy as np
import soundfile as sf
import torch

from zipvoice.models.modules.solver import get_time_steps
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.common import get_tokens_index, prepare_avg_tokens_durations
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm

TEXTS = [
    "The quick brown fox jumps over the lazy dog, and honestly, it felt great.",
    "FluidAudio runs speech models locally on Apple silicon, no cloud required.",
]

MAX_TOKENS = 256
MAX_FRAMES = 1024
FEAT_DIM = 100
TARGET_RMS = 0.1
FEAT_SCALE = 0.1
GUIDANCE_SCALE = 3.0
NUM_STEPS = 4
T_SHIFT = 0.5
SPEED = 1.0  # NOT upstream's hidden 1.3 (clips sentence onsets)
HOP_48K = 512  # 256 at 24 kHz -> 512 at 48 kHz


def expansion_fixture(tokens_len: int, features_len: int):
    durations = prepare_avg_tokens_durations(
        torch.tensor([features_len]), torch.tensor([tokens_len])
    )
    index = get_tokens_index(durations, features_len)[0].tolist()
    return {
        "tokens_len": tokens_len,
        "features_len": features_len,
        "avg_token_duration": int(durations[0][0]),
        "tokens_index_len": len(index),
        "tokens_index_first20": index[:20],
        "tokens_index_last20": index[-20:],
        "tokens_index": index,
    }


def features_len_for(prompt_len: int, prompt_tokens: int, text_tokens: int, speed: float) -> int:
    return prompt_len + int(np.ceil(prompt_len / prompt_tokens * text_tokens / speed))


def synth_coreml(te, fm, voc555, prompt_ids, text_ids, prompt_features, prompt_len, seed):
    """CoreML gpu-path synthesis, mirroring what the Swift host does."""
    cat = prompt_ids + text_ids
    S = len(cat)
    assert S + 1 <= MAX_TOKENS

    tok = np.zeros((1, MAX_TOKENS), dtype=np.int32)
    tok[0, :S] = cat
    tmask = np.zeros((1, MAX_TOKENS), dtype=np.float32)
    tmask[0, S:] = 1.0
    embeds = te.predict({"tokens": tok, "padding_mask": tmask})["token_embeds"]
    embeds = torch.from_numpy(embeds[:, : S + 1, :].astype(np.float32))

    features_len = features_len_for(prompt_len, len(prompt_ids), len(text_ids), SPEED)
    assert features_len <= MAX_FRAMES

    durations = prepare_avg_tokens_durations(
        torch.tensor([features_len]), torch.tensor([S])
    )
    index = get_tokens_index(durations, features_len)
    text_cond = torch.gather(
        embeds, dim=1, index=index.unsqueeze(-1).expand(1, features_len, FEAT_DIM)
    )
    speech_cond = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, features_len - prompt_features.size(1))
    )

    pad = lambda z: torch.nn.functional.pad(
        z, (0, 0, 0, MAX_FRAMES - z.size(1))
    ).numpy().astype(np.float32)
    fmask = np.zeros((1, MAX_FRAMES), dtype=np.float32)
    fmask[0, features_len:] = 1.0

    timesteps = get_time_steps(num_step=NUM_STEPS, t_shift=T_SHIFT)
    torch.manual_seed(seed)
    x = torch.randn(1, features_len, FEAT_DIM)
    text_np, speech_np = pad(text_cond), pad(speech_cond)
    for step in range(NUM_STEPS):
        t_cur, t_next = float(timesteps[step]), float(timesteps[step + 1])
        v = fm.predict({
            "t": np.array([t_cur], dtype=np.float32),
            "x": pad(x),
            "text_condition": text_np,
            "speech_condition": speech_np,
            "guidance_scale": np.array([GUIDANCE_SCALE], dtype=np.float32),
            "padding_mask": fmask,
        })["v"]
        v = torch.from_numpy(v[:, :features_len, :].astype(np.float32))
        x1p, x0p = x + (1.0 - t_cur) * v, x - t_cur * v
        x = (1.0 - t_next) * x0p + t_next * x1p if step < NUM_STEPS - 1 else x1p

    gen = features_len - prompt_len
    mel = (x[0, prompt_len:features_len, :].numpy().T / FEAT_SCALE).astype(np.float32)
    mel_padded = np.full((1, FEAT_DIM, 555), math.log(1e-7), dtype=np.float32)
    mel_padded[0, :, :gen] = mel
    wav = voc555.predict({"mel": mel_padded})["audio"].squeeze().astype(np.float32)
    wav = np.clip(wav[: (gen - 1) * HOP_48K], -1.0, 1.0)
    return wav, features_len, gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-audio", default="build/wavs/07_prompt_clip.wav")
    parser.add_argument("--staging-dir", default="build/hf-staging")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(args.staging_dir)

    tokenizer = EmiliaTokenizer(token_file=str(staging / "tokens.txt"))
    # The Swift tokenizer tests parse the same token table.
    shutil.copy(staging / "tokens.txt", out / "tokens.txt")

    # ---- prompt: load 24k, transcribe, rms-norm, mel ----
    wav24, _ = librosa.load(args.prompt_audio, sr=24000, duration=5)
    wav16, _ = librosa.load(args.prompt_audio, sr=16000, duration=5)

    from transformers import pipeline

    transcriber = pipeline(
        "automatic-speech-recognition", model="openai/whisper-tiny", device="cpu"
    )
    prompt_text = transcriber(wav16)["text"].strip()
    print(f"[prompt] transcript: {prompt_text}")

    prompt_wav = torch.from_numpy(wav24).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, TARGET_RMS)
    prompt_rms = float(prompt_rms)

    fbank = VocosFbank()
    prompt_features = fbank.extract(prompt_wav, sampling_rate=24000).unsqueeze(0) * FEAT_SCALE
    prompt_len = prompt_features.size(1)
    print(f"[prompt] rms={prompt_rms:.5f} frames={prompt_len}")

    # Binary fixtures: post-rms-norm 24k waveform + mel (row-major T x 100, f32le)
    prompt_wav_np = prompt_wav[0].numpy().astype(np.float32)
    prompt_wav_np.tofile(out / "prompt_24k_f32le.bin")
    mel_np = prompt_features[0].numpy().astype(np.float32)
    mel_np.tofile(out / "prompt_mel_f32le.bin")

    # ---- tokens ----
    prompt_tokens = tokenizer.texts_to_tokens([prompt_text])[0]
    prompt_ids = tokenizer.tokens_to_token_ids([prompt_tokens])[0]
    texts_entries = []
    for text in TEXTS:
        toks = tokenizer.texts_to_tokens([text])[0]
        ids = tokenizer.tokens_to_token_ids([toks])[0]
        S = len(prompt_ids) + len(ids)
        features_len = features_len_for(prompt_len, len(prompt_ids), len(ids), SPEED)
        texts_entries.append({
            "text": text,
            "tokens": toks,
            "phoneme_string": "".join(toks),
            "token_ids": ids,
            "cat_tokens_len": S,
            "features_len_speed1": features_len,
            "expansion": expansion_fixture(S, features_len),
        })

    # ---- solver fixtures ----
    timesteps = get_time_steps(num_step=NUM_STEPS, t_shift=T_SHIFT).tolist()
    rng = np.random.RandomState(0)
    x = rng.randn(8)
    vs = rng.randn(NUM_STEPS, 8)
    x_solver = {"x0": x.tolist(), "v_steps": vs.tolist()}
    for step in range(NUM_STEPS):
        t_cur, t_next = timesteps[step], timesteps[step + 1]
        x1p = x + (1.0 - t_cur) * vs[step]
        x0p = x - t_cur * vs[step]
        x = x1p if step == NUM_STEPS - 1 else (1.0 - t_next) * x0p + t_next * x1p
    x_solver["x_final"] = x.tolist()

    # ---- e2e CoreML (gpu graphs + Vocoder555) for TEXTS[0] ----
    cu = ct.ComputeUnit.CPU_AND_GPU
    te = ct.models.CompiledMLModel(str(staging / "gpu/TextEncoder.mlmodelc"), compute_units=cu)
    fm = ct.models.CompiledMLModel(str(staging / "gpu/FmDecoder.mlmodelc"), compute_units=cu)
    voc = ct.models.CompiledMLModel(
        str(staging / "vocoder/Vocoder555.mlmodelc"), compute_units=cu
    )
    wav, features_len, gen = synth_coreml(
        te, fm, voc, prompt_ids, texts_entries[0]["token_ids"],
        prompt_features, prompt_len, args.seed,
    )
    if prompt_rms < TARGET_RMS:
        wav = wav * (prompt_rms / TARGET_RMS)
    rms = float(np.sqrt(np.mean(np.square(wav))))
    sf.write(out / "e2e_python_coreml_48k.wav", wav, 48000)
    print(f"[e2e] features_len={features_len} gen={gen} "
          f"samples={len(wav)} ({len(wav)/48000:.3f}s) rms={rms:.5f}")

    fixtures = {
        "prompt": {
            "audio_source": str(Path(args.prompt_audio).name),
            "transcript": prompt_text,
            "rms_pre_norm": prompt_rms,
            "target_rms": TARGET_RMS,
            "wav_24k_samples": int(prompt_wav_np.shape[0]),
            "mel_frames": prompt_len,
            "mel_dim": FEAT_DIM,
            "mel_first3_frames": mel_np[:3].tolist(),
            "mel_mean": float(mel_np.mean()),
            "mel_std": float(mel_np.std()),
            "tokens": prompt_tokens,
            "phoneme_string": "".join(prompt_tokens),
            "token_ids": prompt_ids,
        },
        "texts": texts_entries,
        "solver": {
            "num_steps": NUM_STEPS,
            "t_shift": T_SHIFT,
            "timesteps": timesteps,
            "mini_trajectory": x_solver,
        },
        "e2e": {
            "text_index": 0,
            "seed": args.seed,
            "speed": SPEED,
            "guidance_scale": GUIDANCE_SCALE,
            "features_len": features_len,
            "gen_frames": gen,
            "wav_samples": int(len(wav)),
            "wav_seconds": float(len(wav) / 48000.0),
            "rms": rms,
            "wav_file": "e2e_python_coreml_48k.wav",
        },
        "constants": {
            "max_tokens": MAX_TOKENS,
            "max_frames": MAX_FRAMES,
            "feat_dim": FEAT_DIM,
            "feat_scale": FEAT_SCALE,
            "sample_rate_mel": 24000,
            "sample_rate_out": 48000,
            "n_fft": 1024,
            "hop_length": 256,
        },
    }
    (out / "luxtts_fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=1))
    print(f"wrote {out / 'luxtts_fixtures.json'}")


if __name__ == "__main__":
    main()
