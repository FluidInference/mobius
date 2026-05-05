"""End-to-end validation: PyTorch reference WAV vs CoreML 7-graph WAV.

Reports:
  - log-mel cosine similarity (target ≥ 0.99 to ship)
  - audio dB stats: RMS dB, peak dB, dB difference between PyTorch and CoreML
  - WAV files written to --out-dir for manual A/B listening

Usage:
    uv run python 99_e2e_validate.py \
        --phrase "Hello world." \
        --coreml-dir ../../coreml/build/ane \
        --out-dir /tmp/styletts2-ane-validate

Note: this script does **not** run a real diffusion sampler — it pulls a
fixed reference style vector and skips the diffusion stage by using the
existing legacy 4-graph reference for the style code-path validation, and
uses the new 7-graph CoreML pipeline for everything else. The diffusion
stage is parity-checked separately by `04_export_diffusion_step.py
--trace-only` against the legacy `02_export_diffusion_step.py`.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _styletts2_ane_lib import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MAX_T_A,
    UPSAMPLE_SCALE,
    AlignmentTraceable,
    NoiseTraceable,
    PLBertTraceable,
    PostBertTraceable,
    ProsodyTraceable,
    VocoderTraceable,
    install_sinegen_v2_constfold_fix,
    load_modules_for_ane,
)


def _to_float32_audio(wav: np.ndarray) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=tuple(range(wav.ndim - 1)))
    return wav.astype(np.float32, copy=False)


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    import wave

    pcm = np.clip(wav, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm16.tobytes())


def db(x: np.ndarray, eps: float = 1e-10) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + eps)
    return 20.0 * math.log10(rms)


def peak_db(x: np.ndarray, eps: float = 1e-10) -> float:
    p = float(np.max(np.abs(x.astype(np.float64))) + eps)
    return 20.0 * math.log10(p)


def log_mel(audio: np.ndarray, sr: int, n_mels: int = 80, n_fft: int = 2048, hop: int = 300) -> np.ndarray:
    # Lightweight log-mel without librosa/torchaudio so the validator stays
    # dependency-light. Reuse the LibriTTS hop/win conventions.
    import torch.nn.functional as F

    a = torch.from_numpy(np.ascontiguousarray(audio)).flatten().unsqueeze(0)  # (1, T)
    win = torch.hann_window(n_fft)
    a = F.pad(a, (n_fft // 2, n_fft // 2), mode="reflect")  # (1, T+pad)
    spec = torch.stft(
        a,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=win,
        center=False,
        return_complex=True,
    )
    mag = spec.abs()
    # Mel filterbank.
    f_min, f_max = 0.0, sr / 2.0
    mel = torch.linspace(0.0, 2595 * math.log10(1 + f_max / 700.0), n_mels + 2)
    hz = 700 * (10 ** (mel / 2595) - 1)
    bins = torch.floor((n_fft + 1) * hz / sr).to(torch.long).clamp(0, n_fft // 2)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1].item(), bins[m].item(), bins[m + 1].item()
        if c > l:
            for k in range(l, c):
                fb[m - 1, k] = (k - l) / max(1, c - l)
        if r > c:
            for k in range(c, r):
                fb[m - 1, k] = (r - k) / max(1, r - c)
    mel_spec = (fb @ mag.squeeze(0)).log1p()
    return mel_spec.numpy()


def cosine_2d(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[-1], b.shape[-1])
    a = a[..., :n].reshape(-1)
    b = b[..., :n].reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def synth_pytorch_reference(modules, cfg, ref_s: np.ndarray, tokens: np.ndarray, n_diff_steps: int = 5):
    """Synthesize a PyTorch-only reference using the legacy `_styletts2_lib`
    traceables wired together with a tiny ADPM2 sampler. This is the parity
    target for the CoreML pipeline.

    Returns: (audio_fp32, F0, N, en, asr) numpy arrays.
    """
    from _styletts2_lib import (  # noqa: E402  (legacy lib; pythonpath set by ane lib)
        DiffusionDenoiseTraceable,
        F0NEnergyTraceable,
        HifiGanDecoderTraceable,
        TextPredictorTraceable,
    )

    style = torch.from_numpy(ref_s).float().unsqueeze(0)        # (1, 256)
    s_pros = style[:, : cfg.style_dim]
    s_acoustic = style[:, cfg.style_dim:]
    tok = torch.from_numpy(tokens).long().unsqueeze(0)           # (1, T_tok)

    text_pred = TextPredictorTraceable(modules).eval()
    diff_step = DiffusionDenoiseTraceable(modules, sigma_data=cfg.diffusion_sigma_data).eval()
    f0n = F0NEnergyTraceable(modules).eval()
    decoder = HifiGanDecoderTraceable(modules["decoder"]).eval()

    with torch.no_grad():
        t_en, _d_en, d, _pred_dur_log, fixed_emb, bert_dur = text_pred(tok, s_pros)

        # Trivial duration: round(sigmoid(pred_dur_log).sum(-1)). For a
        # validator we just use the PyTorch path's pred_dur_log directly.
        pred_dur_log = _pred_dur_log
        pred_dur = torch.sigmoid(pred_dur_log).sum(-1).round().clamp(min=1).long()  # (1, T_tok)

        # Diffusion: run the existing model-agnostic ADPM2 here for reference.
        # We use a fixed-noise seed for determinism.
        torch.manual_seed(0)
        x = torch.randn(1, 1, cfg.style_dim * 2)
        # Karras sigmas (5 steps).
        sigmas = torch.linspace(1.0, 0.0001, n_diff_steps + 1)
        for i in range(n_diff_steps):
            sigma = sigmas[i].view(1)
            d_pred = diff_step(x, sigma, bert_dur, style)
            sigma_next = sigmas[i + 1].view(1, 1, 1)
            x = d_pred + (x - d_pred) * (sigma_next / sigma.view(1, 1, 1))
        ref_s_pred = x.squeeze(1)                                # (1, 256)

        # Replace style with the predicted style (StyleTTS2 inference reality).
        # Decoder consumes only the acoustic half (ref_s[:, style_dim:]); ProsodyPredictor.F0Ntrain
        # consumes only the prosody half (ref_s[:, :style_dim]).
        s_pred_pros = ref_s_pred[:, : cfg.style_dim]
        s_pred_acoustic = ref_s_pred[:, cfg.style_dim:]

        # Alignment (simple repeat_interleave for the reference).
        T_a = int(pred_dur.sum().item())
        align = torch.zeros(1, tok.shape[1], T_a)
        idx = 0
        for t in range(tok.shape[1]):
            dur = int(pred_dur[0, t].item())
            align[0, t, idx:idx + dur] = 1.0
            idx += dur
        en = d.transpose(-1, -2) @ align                          # (1, h+s, T_a)
        asr = t_en @ align                                        # (1, h, T_a)

        F0, N = f0n(en, s_pred_pros)

        audio = decoder(asr, F0, N, s_pred_acoustic)              # (1, T_audio)
    return (
        audio.squeeze(0).numpy().astype(np.float32),
        F0.squeeze(0).numpy().astype(np.float32),
        N.squeeze(0).numpy().astype(np.float32),
        en.squeeze(0).numpy().astype(np.float32),
        asr.squeeze(0).numpy().astype(np.float32),
        ref_s_pred.squeeze(0).numpy().astype(np.float32),
        pred_dur.squeeze(0).numpy().astype(np.float32),
        bert_dur.numpy().astype(np.float32),
    )


def _load_mlmodel(coreml_dir: Path, name: str, compute_units: str = "cpu_and_gpu"):
    """Load a .mlpackage by name. Returns the MLModel or None if missing.

    Defaults to ``cpu_and_gpu`` because the Vocoder graph (Stage 7) fails ANE
    compile per the converter warning (`MILCompilerForANE error: failed to
    compile ANE model using ANEF`); forcing CPU+GPU avoids the compile stall.
    """
    import coremltools as ct

    pkg = coreml_dir / f"styletts2_ane_{name}.mlpackage"
    if not pkg.exists():
        return None
    cu_map = {
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }
    return ct.models.MLModel(str(pkg), compute_units=cu_map[compute_units])


def synth_coreml_pipeline(
    modules,
    cfg,
    ref_s: np.ndarray,
    tokens: np.ndarray,
    *,
    coreml_dir: Path,
    ref_s_pred: np.ndarray | None = None,
    pred_dur_override: np.ndarray | None = None,
    bert_dur_override: np.ndarray | None = None,
    use_mlpackage: bool = True,
    compute_units: str = "cpu_and_gpu",
):
    """Synthesize using compiled .mlpackage bundles for the 7-graph pipeline.

    When ``use_mlpackage`` is True and all 6 .mlpackages exist (1,2,3,5,6,7;
    diffusion is run by the PyTorch reference and its output is shared via
    ``ref_s_pred``), the pipeline loads each via coremltools and executes
    ``predict``. Falls back to PyTorch traceable wrappers per-stage when a
    package is missing.

    Args:
        ref_s: raw style vector [256], used by PostBert (text encoder + duration).
        ref_s_pred: post-diffusion style vector [256], used by Prosody + Vocoder.
                    If None, ref_s is used for all stages.
        pred_dur_override: shared per-token durations (T_tok,). When provided,
                    we skip the CoreML PostBert pred_dur_log → round() step and
                    use these durations for Alignment. This isolates parity to
                    graph computations rather than stochastic fp16 round()
                    flips at duration boundaries.
    """
    style_raw = torch.from_numpy(ref_s).float().unsqueeze(0)            # (1, 256)
    style_pred_np = ref_s_pred if ref_s_pred is not None else ref_s
    style_pred = torch.from_numpy(style_pred_np).float().unsqueeze(0)   # (1, 256)
    tok = torch.from_numpy(tokens).long().unsqueeze(0)
    s_pros_pred = style_pred[:, : cfg.style_dim]
    s_acoustic_pred = style_pred[:, cfg.style_dim:]

    # Try to load all 6 mlpackages up front so we report a single status line.
    mlmodels: dict = {}
    if use_mlpackage:
        try:
            import coremltools as ct  # noqa: F401
            for name in ("plbert", "postbert", "alignment", "prosody", "noise", "vocoder"):
                print(f"[99-ane]   loading mlpackage {name!r} (compute_units={compute_units}) …", flush=True)
                m = _load_mlmodel(coreml_dir, name, compute_units=compute_units)
                if m is None:
                    print(f"[99-ane]     {name!r} missing; will fall back to PyTorch for that stage.", flush=True)
                else:
                    mlmodels[name] = m
                    print(f"[99-ane]     {name!r} loaded.", flush=True)
            print(f"[99-ane]   loaded {len(mlmodels)} of 6 mlpackages from {coreml_dir}", flush=True)
        except Exception as e:
            print(f"[99-ane] coremltools unavailable: {e}; using all-Python wrappers.", flush=True)
            mlmodels = {}

    # PyTorch traceables for fallback.
    plbert_pt = PLBertTraceable(modules).eval()
    postbert_pt = PostBertTraceable(modules, cfg).eval()
    align_pt = AlignmentTraceable(max_T_a=MAX_T_A).eval()
    prosody_pt = ProsodyTraceable(modules).eval()

    T_tok = int(tokens.shape[-1])
    h_plus_s = cfg.hidden_dim + cfg.style_dim
    bert_hidden = modules["bert"].config.hidden_size

    with torch.no_grad():
        # ----- Stage 1: PLBert -----
        print("[99-ane] Stage 1: PLBert.predict …", flush=True)
        if bert_dur_override is not None:
            bert_dur = torch.from_numpy(bert_dur_override).float()
            print("[99-ane]   PLBert SKIPPED (using bert_dur_override).", flush=True)
        elif "plbert" in mlmodels:
            out = mlmodels["plbert"].predict({
                "tokens": tokens.astype(np.int32).reshape(1, T_tok),
            })
            bert_dur_np = next(iter(out.values())).astype(np.float32)
            bert_dur = torch.from_numpy(bert_dur_np)                    # (1, T_tok, 768)
            print("[99-ane]   PLBert done.", flush=True)
        else:
            bert_dur = plbert_pt(tok)
            print("[99-ane]   PLBert done (PyTorch fallback).", flush=True)

        # ----- Stage 2: PostBert -----
        print("[99-ane] Stage 2: PostBert.predict …", flush=True)
        if "postbert" in mlmodels:
            out = mlmodels["postbert"].predict({
                "bert_dur": bert_dur.numpy().astype(np.float32),
                "tokens": tokens.astype(np.int32).reshape(1, T_tok),
                "style": style_raw.numpy().astype(np.float32),
            })
            t_en = torch.from_numpy(out["t_en"].astype(np.float32))
            d = torch.from_numpy(out["d"].astype(np.float32))
            pred_dur_log = torch.from_numpy(out["pred_dur_log"].astype(np.float32))
        else:
            t_en, d, pred_dur_log, _fixed_emb = postbert_pt(bert_dur, tok, style_raw)
        print("[99-ane]   PostBert done.", flush=True)

        if pred_dur_override is not None:
            pred_dur = torch.from_numpy(pred_dur_override).float().unsqueeze(0)  # (1, T_tok)
        else:
            pred_dur = torch.sigmoid(pred_dur_log).sum(-1).round().clamp(min=1).float()  # (1, T_tok)

        # ----- Stage 3: Alignment -----
        print("[99-ane] Stage 3: Alignment.predict …", flush=True)
        if "alignment" in mlmodels:
            out = mlmodels["alignment"].predict({
                "pred_dur": pred_dur.numpy().astype(np.float32),
                "d": d.numpy().astype(np.float32),
                "t_en": t_en.numpy().astype(np.float32),
            })
            en = torch.from_numpy(out["en"].astype(np.float32))         # (1, 640, MAX_T_A)
            asr = torch.from_numpy(out["asr"].astype(np.float32))       # (1, 512, MAX_T_A)
        else:
            en, asr = align_pt(pred_dur, d, t_en)
        print("[99-ane]   Alignment done.", flush=True)

        T_a = int(pred_dur.sum().item())
        # Stage 5 (Prosody) is fixed at MAX_T_A. The alignment graph already
        # zero-padded `en` out to MAX_T_A; if we used the PT fallback we still
        # padded inside align_pt (its output is MAX_T_A). So pass `en` as-is.
        en_padded = en  # (1, 640, MAX_T_A)
        asr_full = asr  # (1, 512, MAX_T_A)

        # ----- Stage 5: Prosody -----
        print("[99-ane] Stage 5: Prosody.predict …", flush=True)
        if "prosody" in mlmodels:
            out = mlmodels["prosody"].predict({
                "en": en_padded.numpy().astype(np.float32),
                "s": s_pros_pred.numpy().astype(np.float32),
            })
            F0_full = torch.from_numpy(out["F0"].astype(np.float32))    # (1, MAX_T_A*2)
            N_full = torch.from_numpy(out["N"].astype(np.float32))      # (1, MAX_T_A*2)
        else:
            F0_full, N_full = prosody_pt(en_padded, s_pros_pred)
        print("[99-ane]   Prosody done.", flush=True)

        # Slice Prosody outputs back to T_a*2 for downstream Noise+Vocoder
        # (which run at the actual T_a we'll synthesize). We ALSO need to
        # rebuild a full-MAX_T_A padded version for Stages 6/7 since they're
        # static at MAX_T_A*2.
        F0_act = F0_full[:, : T_a * 2]
        N_act = N_full[:, : T_a * 2]

        # Re-install SineGen const-fold patch for the actual T_a; both PyTorch
        # NoiseTraceable fallback and the .mlpackage Stage 6 (which baked
        # T_a=MAX_T_A) need the same length to line up. Stage 6 .mlpackage is
        # static at MAX_T_A so we pad F0 there.
        install_sinegen_v2_constfold_fix(t_mel=T_a)

        # ----- Stage 6: Noise -----
        print("[99-ane] Stage 6: Noise.predict …", flush=True)
        if "noise" in mlmodels:
            # Static at MAX_T_A — pad F0 to MAX_T_A*2.
            F0_padded = F0_full  # already at MAX_T_A*2 from Stage 5 mlpackage
            if F0_padded.shape[-1] != MAX_T_A * 2:
                F0_padded = torch.zeros(1, MAX_T_A * 2, dtype=torch.float32)
                F0_padded[:, : F0_act.shape[-1]] = F0_act
            out = mlmodels["noise"].predict({
                "F0_curve": F0_padded.numpy().astype(np.float32),
            })
            sine_full = torch.from_numpy(out["sine_waves"].astype(np.float32))
            # sine_waves is (1, MAX_T_A*2*UPSAMPLE_SCALE, harm+1).
            sine_act = sine_full[:, : T_a * 2 * UPSAMPLE_SCALE, :]
        else:
            noise_pt = NoiseTraceable(modules["decoder"]).eval()
            sine_act, _uv = noise_pt(F0_act)
        print("[99-ane]   Noise done.", flush=True)

        # ----- Stage 7: Vocoder -----
        print("[99-ane] Stage 7: Vocoder.predict …", flush=True)
        if "vocoder" in mlmodels:
            # RangeDim on T_a — pass active inputs directly. HiFi-GAN's stack
            # of conv-transpose ups + noise_convs is sensitive to input length
            # at the active/zero boundary; running zero-padded at MAX_T_A and
            # slicing produces +20 dB edge-effect leakage. Active-T_a call
            # avoids that entirely.
            asr_act = asr_full[:, :, :T_a].contiguous()
            out = mlmodels["vocoder"].predict({
                "asr": asr_act.numpy().astype(np.float32),
                "F0_curve": F0_act.numpy().astype(np.float32),
                "N": N_act.numpy().astype(np.float32),
                "s": s_acoustic_pred.numpy().astype(np.float32),
                "sine_waves": sine_act.numpy().astype(np.float32),
            })
            audio = torch.from_numpy(out["audio"].astype(np.float32))
        else:
            vocoder_pt = VocoderTraceable(modules["decoder"]).eval()
            asr_slice = asr_full[:, :, :T_a]
            audio = vocoder_pt(asr_slice, F0_act, N_act, s_acoustic_pred, sine_act)
        print("[99-ane]   Vocoder done.", flush=True)

    return (
        audio.squeeze(0).numpy().astype(np.float32),
        F0_act.squeeze(0).numpy().astype(np.float32),
        N_act.squeeze(0).numpy().astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--coreml-dir", type=Path,
        default=THIS_DIR.parent.parent / "coreml" / "build" / "ane",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/styletts2-ane-validate"))
    parser.add_argument("--phrase", type=str, default="Hello world.")
    parser.add_argument(
        "--ref-s",
        type=Path,
        default=None,
        help="Optional .bin file with [256] fp32 ref_s (default: zeros).",
    )
    parser.add_argument("--n-diff-steps", type=int, default=5)
    parser.add_argument(
        "--no-mlpackage",
        action="store_true",
        help="Skip CoreML .mlpackage loading; use PyTorch traceables for the CoreML pipeline path.",
    )
    parser.add_argument(
        "--compute-units",
        type=str,
        default="cpu_and_gpu",
        choices=("cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"),
        help="Compute units for mlpackage load (default cpu_and_gpu — Vocoder fails ANE compile).",
    )
    parser.add_argument(
        "--skip-plbert",
        action="store_true",
        help="Diagnostic: replace CoreML PLBert output with PyTorch reference bert_dur "
             "to isolate whether downstream graphs alone hit ≥0.99 cos.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[99-ane] loading modules …")
    modules, cfg = load_modules_for_ane(args.checkpoint)

    # Tokens: simplest viable path — call the same upstream phonemizer the
    # legacy 99b_e2e_coreml.py uses. We import lazily to avoid a hard dep.
    try:
        sys.path.insert(0, str(THIS_DIR.parent))
        from _styletts2_lib import VENDOR_DIR  # noqa: E402

        sys.path.insert(0, str(VENDOR_DIR))
        from text_utils import TextCleaner  # type: ignore

        # Phonemize via espeak-ng (same as upstream) then map to ids.
        from phonemizer.backend import EspeakBackend  # type: ignore

        backend = EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
        phonemes = backend.phonemize([args.phrase])[0]
        cleaner = TextCleaner()
        tokens = np.asarray(cleaner(phonemes), dtype=np.int64)
    except Exception as e:
        print(f"[99-ane] tokenizer fallback (espeak missing? {e}); using ascii bytes.")
        tokens = np.frombuffer(args.phrase.lower().encode("utf-8"), dtype=np.uint8).astype(np.int64)
        tokens = np.clip(tokens, 0, cfg.n_token - 1)

    print(f"[99-ane] tokens: shape={tokens.shape}")

    if args.ref_s is not None:
        ref_s = np.fromfile(args.ref_s, dtype=np.float32)
        assert ref_s.shape == (cfg.style_dim * 2,), (
            f"ref_s shape {ref_s.shape} != ({cfg.style_dim * 2},)"
        )
    else:
        ref_s = np.zeros(cfg.style_dim * 2, dtype=np.float32)

    print("[99-ane] synthesizing PyTorch reference …")
    pt_audio, pt_F0, pt_N, _en, _asr, ref_s_pred, pred_dur_ref, bert_dur_ref = synth_pytorch_reference(
        modules, cfg, ref_s, tokens, n_diff_steps=args.n_diff_steps
    )

    # Share BOTH the diffusion-predicted style and the per-token durations
    # with the CoreML path's downstream stages. This mirrors the StyleTTS2
    # inference flow exactly and isolates the parity check to the graph
    # computations rather than stochastic fp16 `round()` flips on
    # `sigmoid(pred_dur_log).sum(-1)` (a 0.0001 shift can flip a token's
    # duration ±1 and time-misalign the entire downstream audio).
    print("[99-ane] synthesizing CoreML 7-graph pipeline "
          "(raw ref_s + shared pred_dur for PostBert/Alignment, diffusion-predicted style for Prosody+Vocoder) …")
    cm_audio, cm_F0, cm_N = synth_coreml_pipeline(
        modules,
        cfg,
        ref_s,
        tokens,
        coreml_dir=args.coreml_dir,
        ref_s_pred=ref_s_pred,
        pred_dur_override=pred_dur_ref,
        bert_dur_override=bert_dur_ref if args.skip_plbert else None,
        use_mlpackage=not args.no_mlpackage,
        compute_units=args.compute_units,
    )

    # Trim both to the shortest length for a fair comparison.
    n_aud = min(pt_audio.shape[-1], cm_audio.shape[-1])
    n_t = min(pt_F0.shape[-1], cm_F0.shape[-1])
    pt_aud_t, cm_aud_t = pt_audio[..., :n_aud], cm_audio[..., :n_aud]

    # Audio-domain dB metrics (the user's explicit ask).
    pt_rms_db = db(pt_aud_t)
    cm_rms_db = db(cm_aud_t)
    pt_peak_db = peak_db(pt_aud_t)
    cm_peak_db = peak_db(cm_aud_t)
    diff = cm_aud_t - pt_aud_t
    diff_rms_db = db(diff)
    snr_db = pt_rms_db - diff_rms_db

    # Log-mel cosine for the parity gate.
    pt_lm = log_mel(pt_aud_t, cfg.sample_rate, n_mels=cfg.n_mels, n_fft=cfg.n_fft, hop=cfg.hop_length)
    cm_lm = log_mel(cm_aud_t, cfg.sample_rate, n_mels=cfg.n_mels, n_fft=cfg.n_fft, hop=cfg.hop_length)
    cos = cosine_2d(pt_lm, cm_lm)

    # F0/N parity (post-Prosody).
    f0_cos = cosine_2d(pt_F0[..., :n_t], cm_F0[..., :n_t])
    n_cos = cosine_2d(pt_N[..., :n_t], cm_N[..., :n_t])

    pt_path = args.out_dir / "pytorch_ref.wav"
    cm_path = args.out_dir / "coreml_ane.wav"
    _write_wav(pt_path, pt_aud_t, cfg.sample_rate)
    _write_wav(cm_path, cm_aud_t, cfg.sample_rate)

    print()
    print("=== Parity report ===")
    print(f"  Phrase:       {args.phrase!r}")
    print(f"  T_tok:        {tokens.shape[-1]}")
    print(f"  T_audio:      {n_aud}  ({n_aud / cfg.sample_rate:.3f} s @ {cfg.sample_rate} Hz)")
    print(f"  PyTorch RMS:  {pt_rms_db:7.2f} dBFS    peak: {pt_peak_db:7.2f} dBFS    → {pt_path}")
    print(f"  CoreML  RMS:  {cm_rms_db:7.2f} dBFS    peak: {cm_peak_db:7.2f} dBFS    → {cm_path}")
    print(f"  Δ RMS:        {cm_rms_db - pt_rms_db:+.2f} dB")
    print(f"  Diff RMS:     {diff_rms_db:7.2f} dBFS    SNR: {snr_db:7.2f} dB")
    print(f"  log-mel cos:  {cos:.4f}     (target ≥ 0.99)")
    print(f"  F0  cos:      {f0_cos:.4f}")
    print(f"  N   cos:      {n_cos:.4f}")

    if cos < 0.99:
        print()
        print("[99-ane] FAIL — log-mel cosine below 0.99.")
        sys.exit(1)
    print()
    print("[99-ane] OK.")


if __name__ == "__main__":
    main()
