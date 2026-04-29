"""End-to-end StyleTTS2 inference + parity vs PyTorch reference.

Pipeline (mirrors `Demo/Inference_LibriTTS.ipynb`):
  1. Phonemize text → tokens.
  2. ref_s ← style_encoder + predictor_encoder on reference WAV (24 kHz mel).
  3. text_predictor (Package A): tokens, ref_s.first_half → t_en, d_en, d, dur, fixed_emb.
  4. Diffusion sampler (5 ADPM2 steps) using diffusion_step (Package B), CFG with
     embedding_scale=1 collapses to a single forward per step.
  5. Build alignment from rounded predicted durations.
  6. en = d.transpose @ alignment;  asr = t_en @ alignment;
     hifigan-shift both right by 1 frame (matches notebook).
  7. f0n_energy (Package C): en, s → F0, N (at 2× T_mel).
  8. decoder bucket (Package D): asr, F0, N, ref → waveform.

For each stage we run both PyTorch and CoreML and report
  cosine_sim, max_abs_delta
side-by-side. Both pipelines write a WAV; outputs land in --out-dir.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _styletts2_lib import (  # noqa: E402
    COREML_DIR,
    DEFAULT_CHECKPOINT,
    LibriTTSConfig,
    load_inference_modules,
    register_coreml_op_shims,
)

register_coreml_op_shims()

DEFAULT_TEXT = (
    "StyleTTS 2 is a text to speech model that leverages style diffusion and "
    "adversarial training with large speech language models to achieve human "
    "level text to speech synthesis."
)

TOKEN_BUCKETS = (32, 64, 128, 256, 512)
MEL_BUCKETS = (256, 512, 1024, 2048, 4096)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n == 0:
        return float("nan")
    return float(a @ b / n)


def report(name: str, ref: np.ndarray, hyp: np.ndarray) -> None:
    if ref.shape != hyp.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} hyp={hyp.shape}")
        return
    print(
        f"  {name}: cos={cos_sim(ref, hyp):.6f}  "
        f"max|Δ|={np.max(np.abs(ref - hyp)):.4e}  "
        f"shape={tuple(ref.shape)}"
    )


def pick_bucket(n: int, buckets) -> int:
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"value {n} exceeds largest bucket {buckets[-1]}")


def phonemize_to_tokens(text: str):
    """Mirror notebook cell 16: espeak phonemize → word_tokenize → TextCleaner."""
    import phonemizer
    from nltk.tokenize import word_tokenize
    from text_utils import TextCleaner  # vendor/StyleTTS2

    backend = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )
    ps = backend.phonemize([text.strip()])
    ps = " ".join(word_tokenize(ps[0]))
    cleaner = TextCleaner()
    tokens = cleaner(ps)
    tokens.insert(0, 0)
    return torch.LongTensor(tokens).unsqueeze(0)  # (1, T_tok)


def compute_ref_s(modules, wav_path: Path) -> torch.Tensor:
    """Mirror notebook `compute_style`."""
    import librosa
    import torchaudio

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300
    )
    mean, std = -4.0, 4.0

    wave, sr = librosa.load(str(wav_path), sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    wav_t = torch.from_numpy(audio).float()
    mel = to_mel(wav_t)
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std  # (1, 80, T)

    with torch.no_grad():
        ref_s = modules["style_encoder"](mel.unsqueeze(1))         # (1, 128)
        ref_p = modules["predictor_encoder"](mel.unsqueeze(1))     # (1, 128)
    return torch.cat([ref_s, ref_p], dim=1)                         # (1, 256)


# --- PyTorch reference --------------------------------------------------------


@torch.no_grad()
def pytorch_inference(modules, cfg: LibriTTSConfig, tokens: torch.Tensor,
                       ref_s: torch.Tensor, *, alpha=0.3, beta=0.7,
                       diffusion_steps=5, embedding_scale=1, seed=0):
    """Mirror notebook `inference()` on CPU. Returns dict of intermediates + wav."""
    from Modules.diffusion.sampler import (
        ADPM2Sampler,
        DiffusionSampler,
        KarrasSchedule,
    )

    torch.manual_seed(seed)
    device = "cpu"
    text_encoder = modules["text_encoder"]
    bert = modules["bert"]
    bert_encoder = modules["bert_encoder"]
    predictor = modules["predictor"]
    decoder = modules["decoder"]
    diffusion = modules["diffusion"]

    T_tok = tokens.shape[-1]
    input_lengths = torch.LongTensor([T_tok])
    text_mask = torch.zeros(1, T_tok, dtype=torch.bool)  # all valid

    t_en = text_encoder(tokens, input_lengths, text_mask)              # (1, h, T_tok)
    bert_dur = bert(tokens, attention_mask=(~text_mask).int())          # (1, T_tok, bert)
    d_en = bert_encoder(bert_dur).transpose(-1, -2)                     # (1, h, T_tok)

    sampler = DiffusionSampler(
        diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )
    noise = torch.randn(1, 1, 256, device=device)
    s_pred = sampler(
        noise=noise,
        embedding=bert_dur,
        embedding_scale=embedding_scale,
        features=ref_s,
        num_steps=diffusion_steps,
    ).squeeze(1)  # (1, 256)

    s = s_pred[:, 128:]
    ref = s_pred[:, :128]
    ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
    s = beta * s + (1 - beta) * ref_s[:, 128:]

    d = predictor.text_encoder(d_en, s, input_lengths, text_mask)       # (1, T, h+s)
    x, _ = predictor.lstm(d)
    duration = predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(dim=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()

    T_total = int(pred_dur.sum().item())
    aln = torch.zeros(T_tok, T_total)
    c = 0
    for i in range(T_tok):
        n = int(pred_dur[i].item())
        aln[i, c:c + n] = 1.0
        c += n
    aln = aln.unsqueeze(0)                                              # (1, T_tok, T_mel)

    en = d.transpose(-1, -2) @ aln                                      # (1, h+s, T_mel)
    asr = t_en @ aln                                                    # (1, h, T_mel)
    # hifigan shift-right by 1 frame
    en_shift = torch.zeros_like(en)
    en_shift[:, :, 0] = en[:, :, 0]
    en_shift[:, :, 1:] = en[:, :, :-1]
    asr_shift = torch.zeros_like(asr)
    asr_shift[:, :, 0] = asr[:, :, 0]
    asr_shift[:, :, 1:] = asr[:, :, :-1]
    en, asr = en_shift, asr_shift

    F0_pred, N_pred = predictor.F0Ntrain(en, s)                         # (1, 2*T_mel)

    out = decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))     # (1, 1, T_mel*600)
    wav = out.squeeze().cpu().numpy()
    if wav.shape[-1] > 50:
        wav = wav[..., :-50]
    return {
        "t_en": t_en.numpy(),
        "d_en": d_en.numpy(),
        "d": d.numpy(),
        "pred_dur": pred_dur.numpy(),
        "s": s.numpy(),
        "ref": ref.numpy(),
        # Raw ADPM2 sampler output before alpha/beta blending. Used by Stage B
        # parity to compare CoreML sampler output against the same quantity
        # in PyTorch (rather than against the post-blend `s` / `ref`).
        "s_pred_raw": s_pred.numpy(),
        "asr": asr.numpy(),
        "en": en.numpy(),
        "F0": F0_pred.numpy(),
        "N": N_pred.numpy(),
        "wav": wav,
        "T_tok": T_tok,
        "T_mel": T_total,
        "bert_dur": bert_dur.numpy(),
        "fixed_emb": diffusion.unet.fixed_embedding(bert_dur).numpy(),
        "noise": noise.numpy(),
    }


# --- CoreML side --------------------------------------------------------------


def coreml_inference(coreml_dir: Path, ref, cfg: LibriTTSConfig):
    """Run each stage through CoreML using PyTorch intermediates as input.

    Note: this does NOT re-run the diffusion sampler in CoreML (the Python
    sampler loop is what we ship; we just need the per-step package to match
    PyTorch numerically). For style we reuse the PyTorch-sampled `s_pred`.
    """
    import coremltools as ct

    out = {}
    print("[parity] loading CoreML packages …")
    tp = ct.models.MLModel(str(coreml_dir / "styletts2_text_predictor.mlpackage"))
    ds = ct.models.MLModel(str(coreml_dir / "styletts2_diffusion_step.mlpackage"))
    fn = ct.models.MLModel(str(coreml_dir / "styletts2_f0n_energy.mlpackage"))

    # === Stage A: text_predictor ===
    T_tok = ref["T_tok"]
    tok_bucket = pick_bucket(T_tok, TOKEN_BUCKETS)
    tokens_pad = np.zeros((1, tok_bucket), dtype=np.int32)
    # we stored tokens via the input torch tensor — recompute from t_en presence
    # (we don't have raw tokens here; receive separately in driver below)
    out["_tok_bucket"] = tok_bucket
    out["_text_predictor"] = tp
    out["_diffusion_step"] = ds
    out["_f0n_energy"] = fn
    return out


def run_text_predictor(tp, tokens: torch.Tensor, ref_s: torch.Tensor, tok_bucket: int):
    T_tok = tokens.shape[-1]
    tokens_pad = np.zeros((1, tok_bucket), dtype=np.int32)
    tokens_pad[0, :T_tok] = tokens.numpy().astype(np.int32)[0]
    style_half = ref_s[:, :128].numpy().astype(np.float32)
    feed = {"tokens": tokens_pad, "style": style_half}
    pred = tp.predict(feed)
    return pred  # dict of arrays at full bucket length


def run_diffusion_step(ds, x_noisy: np.ndarray, sigma: float, embedding: np.ndarray,
                       features: np.ndarray):
    feed = {
        "x_noisy": x_noisy.astype(np.float32),
        "sigma": np.array([sigma], dtype=np.float32),
        "embedding": embedding.astype(np.float32),
        "features": features.astype(np.float32),
    }
    return ds.predict(feed)


def run_f0n_energy(fn, en: np.ndarray, s: np.ndarray, mel_bucket: int):
    en_pad = np.zeros((1, en.shape[1], mel_bucket), dtype=np.float32)
    T = en.shape[2]
    en_pad[:, :, :T] = en
    feed = {"en": en_pad, "s": s.astype(np.float32)}
    pred = fn.predict(feed)
    return pred  # F0/N at (1, 2*mel_bucket)


def run_decoder(coreml_dir: Path, asr: np.ndarray, F0: np.ndarray, N: np.ndarray,
                ref: np.ndarray, mel_bucket: int):
    import coremltools as ct
    pkg = coreml_dir / f"styletts2_decoder_{mel_bucket}.mlpackage"
    print(f"[parity] loading {pkg.name} …")
    decoder = ct.models.MLModel(str(pkg))

    asr_pad = np.zeros((1, asr.shape[1], mel_bucket), dtype=np.float32)
    asr_pad[:, :, :asr.shape[2]] = asr
    F0_pad = np.zeros((1, 2 * mel_bucket), dtype=np.float32)
    F0_pad[:, :F0.shape[1]] = F0
    N_pad = np.zeros((1, 2 * mel_bucket), dtype=np.float32)
    N_pad[:, :N.shape[1]] = N
    feed = {
        "asr": asr_pad,
        "F0_curve": F0_pad,
        "N": N_pad,
        "s": ref.astype(np.float32),
    }
    return decoder.predict(feed)


# --- Diffusion sampler reimplemented over the CoreML step ---------------------
#
# Mirrors `Modules.diffusion.sampler.{KarrasSchedule, ADPM2Sampler}` paired with
# `KDiffusion.denoise_fn` (which is what we wrapped as Package B). With
# embedding_scale=1, CFG collapses to a single forward → exactly one denoise
# call per step, matching the wrapper.


def karras_sigmas(num_steps: int, sigma_min=0.0001, sigma_max=3.0, rho=9.0):
    rho_inv = 1.0 / rho
    ramp = np.linspace(0, 1, num_steps).astype(np.float64)
    min_inv = sigma_min ** rho_inv
    max_inv = sigma_max ** rho_inv
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return np.concatenate([sigmas, [0.0]]).astype(np.float32)


def adpm2_sample_coreml(ds, noise: np.ndarray, sigmas: np.ndarray,
                        embedding: np.ndarray, features: np.ndarray):
    """ADPM2 sampler over the CoreML denoise step. embedding_scale=1."""
    x = noise.astype(np.float32) * float(sigmas[0])
    for i in range(len(sigmas) - 1):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        # midpoint sigma per ADPM2: geometric mid in log space
        if s_next == 0.0:
            denoised = run_diffusion_step(ds, x, s, embedding, features)["denoised"]
            d = (x - denoised) / s
            x = x + d * (s_next - s)
            continue
        s_mid = float(np.exp((np.log(s) + np.log(s_next)) / 2))
        denoised = run_diffusion_step(ds, x, s, embedding, features)["denoised"]
        d = (x - denoised) / s
        # half-step Euler to midpoint
        x_mid = x + d * (s_mid - s)
        denoised_mid = run_diffusion_step(ds, x_mid, s_mid, embedding, features)["denoised"]
        d_mid = (x_mid - denoised_mid) / s_mid
        x = x + d_mid * (s_next - s)
    return x  # (1, 1, 256)


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--reference-wav", type=Path, required=True,
                        help="WAV used to extract ref_s (≥4 s, 24 kHz preferred).")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--coreml-dir", type=Path, default=COREML_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/styletts2-parity"))
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[parity] loading PyTorch modules …")
    modules, cfg = load_inference_modules(args.checkpoint)

    print(f"[parity] computing ref_s from {args.reference_wav}")
    ref_s = compute_ref_s(modules, args.reference_wav)
    print(f"[parity]   ref_s shape={tuple(ref_s.shape)}")

    print("[parity] phonemizing …")
    tokens = phonemize_to_tokens(args.text)
    T_tok = tokens.shape[-1]
    print(f"[parity]   T_tok={T_tok}")
    tok_bucket = pick_bucket(T_tok, TOKEN_BUCKETS)
    print(f"[parity]   token bucket={tok_bucket}")

    print("[parity] running PyTorch reference …")
    t0 = time.time()
    ref = pytorch_inference(
        modules, cfg, tokens, ref_s,
        diffusion_steps=args.diffusion_steps, seed=args.seed,
    )
    pt_time = time.time() - t0
    T_mel = ref["T_mel"]
    mel_bucket = pick_bucket(T_mel, MEL_BUCKETS)
    print(f"[parity]   PyTorch done in {pt_time:.2f}s  T_mel={T_mel}  bucket={mel_bucket}")

    pt_wav_path = args.out_dir / "pytorch.wav"
    sf.write(pt_wav_path, ref["wav"], 24000)
    print(f"[parity]   wrote {pt_wav_path}  ({len(ref['wav']) / 24000:.2f}s)")

    # === CoreML stages (each is best-effort; failures are reported, not fatal) ===
    import coremltools as ct

    def try_load(name: str):
        try:
            return ct.models.MLModel(str(args.coreml_dir / name))
        except Exception as e:  # noqa: BLE001
            print(f"[parity] FAILED to load {name}: {e}")
            return None

    print("[parity] loading CoreML packages …")
    # text_predictor and diffusion_step are now per-bucket (T_tok). Fall back
    # to the legacy single-file names if per-bucket files aren't present.
    tp_per_bucket = f"styletts2_text_predictor_{tok_bucket}.mlpackage"
    tp = try_load(tp_per_bucket) or try_load("styletts2_text_predictor.mlpackage")
    ds_per_bucket = f"styletts2_diffusion_step_{tok_bucket}.mlpackage"
    ds = try_load(ds_per_bucket) or try_load("styletts2_diffusion_step.mlpackage")
    fn = try_load("styletts2_f0n_energy.mlpackage")

    # --- Stage A: text_predictor ---
    if tp is not None:
        print(f"[parity] === Stage A: text_predictor (bucket={tok_bucket}) ===")
        try:
            t0 = time.time()
            A = run_text_predictor(tp, tokens, ref_s, tok_bucket)
            print(f"[parity]   coreml: {time.time() - t0:.2f}s")
            for key, ref_arr, slicer in [
                ("t_en", ref["t_en"], lambda a: a[..., :T_tok]),
                ("d_en", ref["d_en"], lambda a: a[..., :T_tok]),
                ("d   ", ref["d"], lambda a: a[:, :T_tok, :]),
                ("fixed_emb", ref["fixed_emb"][:, :T_tok, :],
                 lambda a: a[:, :T_tok, :]),
            ]:
                k = key.strip()
                if k not in A and k != "fixed_emb":
                    print(f"  {key}: missing from CoreML output (keys={list(A)})")
                    continue
                pred = A.get(k if k != "fixed_emb" else "fixed_embedding")
                if pred is None:
                    print(f"  {key}: missing")
                    continue
                report(key, ref_arr, slicer(pred))
        except Exception as e:  # noqa: BLE001
            print(f"[parity] Stage A failed: {e}")

    # --- Stage B: diffusion sampler over CoreML step ---
    if ds is not None:
        print(f"[parity] === Stage B: diffusion ({args.diffusion_steps} steps) ===")
        try:
            sigmas = karras_sigmas(args.diffusion_steps)
            # per-bucket diffusion_step needs embedding padded to tok_bucket.
            bert_dur = ref["bert_dur"]
            bert_dim = bert_dur.shape[-1]
            embedding = np.zeros((1, tok_bucket, bert_dim), dtype=np.float32)
            embedding[:, :T_tok, :] = bert_dur
            features = ref_s.numpy().astype(np.float32)
            t0 = time.time()
            s_pred_coreml = adpm2_sample_coreml(
                ds, ref["noise"], sigmas, embedding, features
            )
            print(f"[parity]   coreml: {time.time() - t0:.2f}s")
            s_pred_coreml = s_pred_coreml.squeeze(1)
            # Compare raw sampler outputs only (pre-blend). The blended
            # `s` / `ref` mix in the reference style and would conflate
            # sampler parity with blending arithmetic.
            pt_s_pred = ref["s_pred_raw"]
            report("s_pred", pt_s_pred, s_pred_coreml)
            print(f"  s_pred_coreml:  mean={s_pred_coreml.mean():.4f} std={s_pred_coreml.std():.4f}")
            print(f"  s_pred_pytorch: mean={pt_s_pred.mean():.4f} std={pt_s_pred.std():.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"[parity] Stage B failed: {e}")

    # --- Stage C: F0Ntrain ---
    if fn is not None:
        print(f"[parity] === Stage C: f0n_energy (bucket={mel_bucket}) ===")
        try:
            t0 = time.time()
            C = run_f0n_energy(fn, ref["en"], ref["s"], mel_bucket)
            print(f"[parity]   coreml: {time.time() - t0:.2f}s")
            coreml_F0 = C["F0"][:, :2 * T_mel]
            coreml_N = C["N"][:, :2 * T_mel]
            report("F0", ref["F0"], coreml_F0)
            report("N ", ref["N"], coreml_N)
        except Exception as e:  # noqa: BLE001
            print(f"[parity] Stage C failed: {e}")

    # --- Stage D: decoder bucket ---
    print(f"[parity] === Stage D: decoder (bucket={mel_bucket}) ===")
    try:
        t0 = time.time()
        D = run_decoder(args.coreml_dir, ref["asr"], ref["F0"], ref["N"],
                        ref["ref"], mel_bucket)
        print(f"[parity]   coreml: {time.time() - t0:.2f}s")
        coreml_wav = D["waveform"].squeeze()[: T_mel * 600]
        if coreml_wav.shape[-1] > 50:
            coreml_wav = coreml_wav[..., :-50]
        report("waveform", ref["wav"], coreml_wav)
        cm_wav_path = args.out_dir / "coreml.wav"
        sf.write(cm_wav_path, coreml_wav, 24000)
        print(f"[parity]   wrote {cm_wav_path}  ({len(coreml_wav) / 24000:.2f}s)")
    except Exception as e:  # noqa: BLE001
        print(f"[parity] Stage D failed: {e}")

    print("[parity] done.")


if __name__ == "__main__":
    main()
