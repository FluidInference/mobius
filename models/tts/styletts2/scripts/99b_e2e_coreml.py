"""End-to-end StyleTTS2 inference using ONLY CoreML packages for the
hot path. Times each stage and the full wall, computes RTFx.

PyTorch is still used for:
  - phonemizer (espeak, no CoreML equivalent needed)
  - ref_s extraction (style_encoder/predictor_encoder — small CNN heads,
    not on the hot path; could be exported but currently aren't)

Everything downstream of `bert_dur` and `style` (text_predictor, diffusion,
f0n_energy, decoder) runs through CoreML mlpackages.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _styletts2_lib import (  # noqa: E402
    COREML_DIR,
    DEFAULT_CHECKPOINT,
    LibriTTSConfig,
    load_inference_modules,
    register_coreml_op_shims,
)

register_coreml_op_shims()

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."

TOKEN_BUCKETS = (32, 64, 128, 256, 512)
MEL_BUCKETS = (256, 512, 1024, 2048, 4096)


def pick_bucket(n, buckets):
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"value {n} exceeds largest bucket {buckets[-1]}")


def phonemize(text):
    import phonemizer
    from nltk.tokenize import word_tokenize
    from text_utils import TextCleaner

    backend = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    ps = backend.phonemize([text.strip()])
    ps = " ".join(word_tokenize(ps[0]))
    cleaner = TextCleaner()
    tokens = cleaner(ps)
    tokens.insert(0, 0)
    return torch.LongTensor(tokens).unsqueeze(0)


def compute_ref_s(modules, wav_path):
    import librosa
    import torchaudio

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
    mean, std = -4.0, 4.0
    wave, sr = librosa.load(str(wav_path), sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    wav_t = torch.from_numpy(audio).float()
    mel = to_mel(wav_t)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std

    with torch.no_grad():
        ref_s = modules["style_encoder"](mel.unsqueeze(1))
        ref_p = modules["predictor_encoder"](mel.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def karras_sigmas(num_steps, sigma_min=0.0001, sigma_max=3.0, rho=9.0):
    rho_inv = 1.0 / rho
    ramp = np.linspace(0, 1, num_steps).astype(np.float64)
    min_inv = sigma_min ** rho_inv
    max_inv = sigma_max ** rho_inv
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return np.concatenate([sigmas, [0.0]]).astype(np.float32)


def adpm2_sample(ds_predict, noise, sigmas, embedding, features):
    """ADPM2 sampler over the diffusion_step model. embedding_scale=1."""
    x = noise.astype(np.float32) * float(sigmas[0])
    for i in range(len(sigmas) - 1):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        if s_next == 0.0:
            denoised = ds_predict(x, s, embedding, features)
            d = (x - denoised) / s
            x = x + d * (s_next - s)
            continue
        s_mid = float(np.exp((np.log(s) + np.log(s_next)) / 2))
        denoised = ds_predict(x, s, embedding, features)
        d = (x - denoised) / s
        x_mid = x + d * (s_mid - s)
        denoised_mid = ds_predict(x_mid, s_mid, embedding, features)
        d_mid = (x_mid - denoised_mid) / s_mid
        x = x + d_mid * (s_next - s)
    return x  # (1, 1, 256)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--reference-wav", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--coreml-dir", type=Path, default=COREML_DIR)
    ap.add_argument("--out", type=Path, default=Path("/tmp/styletts2-e2e/coreml.wav"))
    ap.add_argument("--diffusion-steps", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    timings = {}
    t_total = time.time()

    print("[e2e] loading PyTorch helpers (style/predictor encoders only)…")
    t0 = time.time()
    modules, cfg = load_inference_modules(args.checkpoint)
    timings["load_pytorch"] = time.time() - t0

    print("[e2e] phonemize + ref_s …")
    t0 = time.time()
    tokens = phonemize(args.text)
    T_tok = tokens.shape[-1]
    tok_bucket = pick_bucket(T_tok, TOKEN_BUCKETS)
    ref_s = compute_ref_s(modules, args.reference_wav)
    timings["frontend"] = time.time() - t0
    print(f"[e2e]   T_tok={T_tok}  tok_bucket={tok_bucket}  ref_s={tuple(ref_s.shape)}")

    import coremltools as ct

    def load(name):
        return ct.models.MLModel(str(args.coreml_dir / name))

    print(f"[e2e] loading CoreML packages (T_tok bucket={tok_bucket})…")
    t0 = time.time()
    tp = load(f"styletts2_text_predictor_{tok_bucket}.mlpackage")
    ds = load(f"styletts2_diffusion_step_{tok_bucket}.mlpackage")
    fn = load("styletts2_f0n_energy.mlpackage")
    timings["load_text+diff+f0n"] = time.time() - t0

    # ---- Stage A: text_predictor ----
    t0 = time.time()
    tokens_pad = np.zeros((1, tok_bucket), dtype=np.int32)
    tokens_pad[0, :T_tok] = tokens.numpy().astype(np.int32)[0]
    style_half = ref_s[:, :128].numpy().astype(np.float32)
    A = tp.predict({"tokens": tokens_pad, "style": style_half})
    timings["A_text_predictor"] = time.time() - t0

    t_en = A["t_en"][..., :T_tok]                        # (1, 512, T_tok)
    d_full = A["d"][:, :T_tok, :]                        # (1, T_tok, 640)
    pred_dur_log = A["pred_dur_log"][:, :T_tok, :]       # (1, T_tok, 50)
    bert_dur = A["bert_dur"][:, :T_tok, :]               # (1, T_tok, 768)

    pred_dur = (
        torch.round(torch.sigmoid(torch.from_numpy(pred_dur_log)).sum(dim=-1).squeeze())
        .clamp(min=1)
        .long()
        .numpy()
    )
    T_mel = int(pred_dur.sum())
    mel_bucket = pick_bucket(T_mel, MEL_BUCKETS)
    print(f"[e2e]   T_mel={T_mel}  mel_bucket={mel_bucket}")

    # ---- Stage B: diffusion sampler ----
    t0 = time.time()
    np.random.seed(args.seed)
    noise = np.random.randn(1, 1, 256).astype(np.float32)
    bert_pad = np.zeros((1, tok_bucket, bert_dur.shape[-1]), dtype=np.float32)
    bert_pad[:, :T_tok, :] = bert_dur
    features = ref_s.numpy().astype(np.float32)
    sigmas = karras_sigmas(args.diffusion_steps)

    def ds_predict(x, s, emb, feats):
        return ds.predict({
            "x_noisy": x.astype(np.float32),
            "sigma": np.array([s], dtype=np.float32),
            "embedding": emb.astype(np.float32),
            "features": feats.astype(np.float32),
        })["denoised"]

    s_pred = adpm2_sample(ds_predict, noise, sigmas, bert_pad, features).squeeze(1)
    timings["B_diffusion"] = time.time() - t0

    s = s_pred[:, 128:]
    ref = s_pred[:, :128]
    ref = args.alpha * ref + (1 - args.alpha) * ref_s[:, :128].numpy()
    s = args.beta * s + (1 - args.beta) * ref_s[:, 128:].numpy()

    # ---- Build alignment, compute en/asr ----
    t0 = time.time()
    aln = np.zeros((T_tok, T_mel), dtype=np.float32)
    c = 0
    for i in range(T_tok):
        n = int(pred_dur[i])
        aln[i, c:c + n] = 1.0
        c += n
    aln = aln[None]                                       # (1, T_tok, T_mel)

    # d_full is (1, T_tok, 640); transpose → (1, 640, T_tok); @ aln → (1, 640, T_mel)
    en = np.matmul(d_full.transpose(0, 2, 1), aln)        # (1, 640, T_mel)
    asr = np.matmul(t_en, aln)                            # (1, 512, T_mel)

    # hifigan shift-right by 1 frame
    en_s = np.zeros_like(en)
    en_s[:, :, 0] = en[:, :, 0]
    en_s[:, :, 1:] = en[:, :, :-1]
    asr_s = np.zeros_like(asr)
    asr_s[:, :, 0] = asr[:, :, 0]
    asr_s[:, :, 1:] = asr[:, :, :-1]
    en, asr = en_s, asr_s
    timings["align_build"] = time.time() - t0

    # ---- Stage C: f0n_energy ----
    t0 = time.time()
    en_pad = np.zeros((1, 640, mel_bucket), dtype=np.float32)
    en_pad[:, :, :T_mel] = en
    C = fn.predict({"en": en_pad, "s": s.astype(np.float32)})
    F0 = C["F0"][:, :2 * T_mel]
    N = C["N"][:, :2 * T_mel]
    timings["C_f0n_energy"] = time.time() - t0

    # ---- Stage D: decoder bucket ----
    t0 = time.time()
    decoder = load(f"styletts2_decoder_{mel_bucket}.mlpackage")
    timings["D_load_decoder"] = time.time() - t0

    t0 = time.time()
    asr_pad = np.zeros((1, 512, mel_bucket), dtype=np.float32)
    asr_pad[:, :, :T_mel] = asr
    F0_pad = np.zeros((1, 2 * mel_bucket), dtype=np.float32)
    F0_pad[:, :F0.shape[1]] = F0
    N_pad = np.zeros((1, 2 * mel_bucket), dtype=np.float32)
    N_pad[:, :N.shape[1]] = N
    D = decoder.predict({
        "asr": asr_pad,
        "F0_curve": F0_pad,
        "N": N_pad,
        "s": ref.astype(np.float32),
    })
    timings["D_decode"] = time.time() - t0

    wav = D["waveform"].squeeze()[: T_mel * 600]
    if wav.shape[-1] > 50:
        wav = wav[..., :-50]
    sf.write(str(args.out), wav, 24000)

    timings["total"] = time.time() - t_total
    audio_dur = len(wav) / 24000.0
    inference_time = sum(
        timings[k] for k in (
            "A_text_predictor", "B_diffusion", "align_build",
            "C_f0n_energy", "D_decode",
        )
    )

    print()
    print(f"[e2e] wrote {args.out} ({audio_dur:.2f}s)")
    print()
    print("[e2e] === Timings ===")
    for k, v in timings.items():
        print(f"  {k:24s}  {v*1000:8.1f} ms")
    print()
    print(f"  inference (A+B+align+C+D_decode): {inference_time*1000:8.1f} ms")
    print(f"  audio duration:                   {audio_dur*1000:8.1f} ms")
    print(f"  RTFx (audio/inference):           {audio_dur/inference_time:8.2f}×")


if __name__ == "__main__":
    main()
