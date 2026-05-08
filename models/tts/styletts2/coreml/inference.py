"""End-to-end CoreML inference for StyleTTS2 LibriTTS.

Drives all 8 CoreML `.mlpackage` stages (text_encoder, bert, ref_encoder,
diffusion_unet, duration_predictor, f0n_predictor, har_source, decoder)
and writes a 24 kHz WAV. The Python side keeps:

    * phonemizer + tokenizer        (CPU-only by definition)
    * Karras sigma schedule         (5 floats, trivial)
    * ADPM2 step loop               (5 steps × 2 dispatches per step;
                                     each dispatch runs CoreML UNet)
    * alignment matrix construction (data-dependent shape)

Every neural-net stage runs in CoreML.

Usage:

    cd models/tts/styletts2
    uv run python coreml/inference.py \
        --text "StyleTTS 2 is a text to speech model." \
        --output out_coreml.wav

Shape strategy:

  * `text_encoder`, `duration_predictor`, `f0n_predictor`, `har_source`,
    `decoder` are converted with `ct.RangeDim` on the variable axis
    (token T, frame F, or 2*frame F0; har at 600*frame). All five run
    at native length. The decoder's `T_FRAME` / `F0_LEN` / `HAR_LEN`
    RangeDims are independent in CoreML's symbolic shape inference, but
    coremltools propagates trace defaults consistently so that runtime
    shape checks (e.g. ios18.add broadcast in noise_convs/ups sums)
    pass when the caller feeds inputs with the matching ratio.
  * `bert` and `diffusion_unet` keep a fixed token axis of 57. HF Albert
    and the cross-attention diffusion U-Net both produce shape ops that
    coremltools' MLProgram backend rejects under RangeDim
    ("data-dependent shapes were disabled"). Tokens are padded to 57 for
    these two stages; BERT respects `attention_mask` so contamination at
    real positions is bounded.

Sentences must phonemize to ≤ 57 tokens until BERT / diffusion get
RangeDim support.
"""

from __future__ import annotations

import argparse
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_HERE = Path(__file__).resolve().parent.parent  # models/tts/styletts2
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import coremltools as ct  # noqa: E402

from coreml._runtime import HERE, ensure_nltk  # noqa: E402

PACKAGES_DIR = HERE / "coreml" / "packages"


# ---------- Per-stage compute placement (Kokoro-ANE-style mixed pattern) ----------
#
# Picked from a sweep over {fp16, fp32} × {ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE}
# minimising warm predict latency (tools/profile_stages.py). Notes:
#   * bert / ref_encoder / diffusion_unet / f0n_predictor → ANE: graphs are
#     ANE-friendly (Albert, small CNN, attention U-Net, conv stack); fp16 ANE
#     warm is 3-4x faster than CPU.
#   * text_encoder / duration_predictor → CPU_ONLY: tiny LSTMs that don't
#     benefit from ANE and have <1 s load there.
#   * har_source → CPU_AND_GPU: trivial graph, GPU is free + fast warm.
#   * decoder → CPU_ONLY: HiFi-GAN ANE compile *fails*
#     ("MILCompilerForANE error: ANECCompile() FAILED"); CPU_ONLY skips the
#     compile attempt entirely (1.6 s load vs ~38 s for ALL) and Accelerate
#     fp16 SIMD keeps warm predict ~300 ms.
_STAGE_COMPUTE: dict[str, ct.ComputeUnit] = {
    "text_encoder":       ct.ComputeUnit.CPU_ONLY,
    "bert":               ct.ComputeUnit.CPU_AND_NE,
    "ref_encoder":        ct.ComputeUnit.CPU_AND_NE,
    "diffusion_unet":     ct.ComputeUnit.CPU_AND_NE,
    "duration_predictor": ct.ComputeUnit.CPU_ONLY,
    "f0n_predictor":      ct.ComputeUnit.CPU_AND_NE,
    "har_source":         ct.ComputeUnit.CPU_AND_GPU,
    "decoder":            ct.ComputeUnit.CPU_ONLY,
}


# ---------- Per-stage precision (Kokoro-ANE-style mixed pattern) ----------
#
# Default: fp16 everywhere except the two audio-domain stages that compute
# sin(2π × cumsum(f0)). At fp16, cumsum over 74 400 samples accumulates
# ~10 bits of error and produces audible phase drift in the tail of the
# clip (clean first ~2 s, distorted afterwards). har_source and decoder
# therefore stay fp32.
#
# `bert` was tried in fp32 to skip the ~3 s fp16-Albert ANE compile on
# fresh process startup. Quality A/B is identical, but for long-running
# servers fp16 wins on warm (~20 ms vs ~25 ms × however many utterances).
# Default favours warm-path speed; flip to fp32 with `--fp32 bert` if cold
# start matters (CLI / one-shot invocations).
_STAGE_PRECISION: dict[str, str] = {
    "text_encoder":       "fp16",
    "bert":               "fp16",
    "ref_encoder":        "fp16",
    "diffusion_unet":     "fp16",
    "duration_predictor": "fp16",
    "f0n_predictor":      "fp16",
    # har_source / decoder kept in fp32: both compute sin(2π × cumsum(f0))
    # in the audio domain (har_source explicitly, decoder inside SineGen).
    # fp16 cumsum loses ~10 bits of precision after a few thousand samples,
    # producing audible phase drift that gets worse the further into the
    # clip you go (clean first ~2 s, distorted tail).
    "har_source":         "fp32",
    "decoder":            "fp32",
}


# ---------- CoreML helpers ----------


def _load_stage(
    stage: str,
    *,
    precision: str | None = None,
    compute_units: ct.ComputeUnit | None = None,
) -> ct.models.MLModel:
    """Load `<stage>.mlpackage` or `<stage>_fp16.mlpackage` per the manifests.

    `precision=None` consults `_STAGE_PRECISION`; pass `"fp16"`/`"fp32"` to
    override. `compute_units=None` consults `_STAGE_COMPUTE`.
    """
    prec = precision if precision is not None else _STAGE_PRECISION[stage]
    if prec not in ("fp16", "fp32"):
        raise ValueError(f"precision must be fp16 or fp32, got {prec!r}")
    suffix = "_fp16" if prec == "fp16" else ""
    pkg = PACKAGES_DIR / f"{stage}{suffix}.mlpackage"
    if not pkg.exists():
        raise FileNotFoundError(f"missing {pkg} — run coreml/convert.py first")
    cu = compute_units if compute_units is not None else _STAGE_COMPUTE[stage]
    return ct.models.MLModel(str(pkg), compute_units=cu)


def _spec_outputs_in_order(mlmodel: ct.models.MLModel) -> list[str]:
    return [o.name for o in mlmodel.get_spec().description.output]


def _predict(mlmodel: ct.models.MLModel, feed: dict) -> list[np.ndarray]:
    out = mlmodel.predict(feed)
    return [np.asarray(out[name]) for name in _spec_outputs_in_order(mlmodel)]


# ---------- Karras schedule + ADPM2 step (CPU side) ----------


def _karras_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: float) -> torch.Tensor:
    rho_inv = 1.0 / rho
    steps = torch.arange(num_steps, dtype=torch.float32)
    sigmas = (
        sigma_max ** rho_inv
        + (steps / (num_steps - 1)) * (sigma_min ** rho_inv - sigma_max ** rho_inv)
    ) ** rho
    return torch.cat([sigmas, torch.zeros(1)])  # F.pad(..., value=0.0)


def _adpm2_get_sigmas(sigma: float, sigma_next: float, rho: float = 1.0):
    sigma_up = sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
    sigma_down = sqrt(max(sigma_next ** 2 - sigma_up ** 2, 0.0))
    sigma_mid = ((sigma ** (1 / rho) + sigma_down ** (1 / rho)) / 2) ** rho
    return sigma_up, sigma_down, sigma_mid


def _denoise_via_coreml(
    unet: ct.models.MLModel,
    x_noisy: np.ndarray,
    sigma: float,
    embedding: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    feed = {
        "x_noisy": x_noisy.astype(np.float32),
        "sigma": np.array([sigma], dtype=np.float32),
        "embedding": embedding.astype(np.float32),
        "features": features.astype(np.float32),
    }
    return _predict(unet, feed)[0]


def _adpm2_sample(
    unet: ct.models.MLModel,
    noise: np.ndarray,
    embedding: np.ndarray,
    features: np.ndarray,
    *,
    num_steps: int,
    sigma_min: float = 0.0001,
    sigma_max: float = 3.0,
    rho_schedule: float = 9.0,
    rho_sampler: float = 1.0,
) -> np.ndarray:
    sigmas = _karras_sigmas(num_steps, sigma_min, sigma_max, rho_schedule).numpy()
    x = (sigmas[0] * noise).astype(np.float32)
    for i in range(num_steps - 1):
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        sigma_up, sigma_down, sigma_mid = _adpm2_get_sigmas(sigma, sigma_next, rho_sampler)

        d = (x - _denoise_via_coreml(unet, x, sigma, embedding, features)) / sigma
        x_mid = x + d * (sigma_mid - sigma)
        d_mid = (x_mid - _denoise_via_coreml(unet, x_mid, sigma_mid, embedding, features)) / sigma_mid
        x = x + d_mid * (sigma_down - sigma)

        # Stochastic mid-step (matches ADPM2Sampler.step). Use torch RNG
        # so this is reproducible under torch.manual_seed().
        x = x + (torch.randn(*x.shape).numpy() * sigma_up).astype(np.float32)
    return x  # [1, 1, 256]


# ---------- Alignment + hifigan asr shift ----------


def _build_pred_aln_trg(pred_dur: torch.Tensor, n_tokens: int) -> torch.Tensor:
    total = int(pred_dur.sum().item())
    aln = torch.zeros(n_tokens, total)
    c = 0
    for i in range(n_tokens):
        d = int(pred_dur[i].item())
        aln[i, c : c + d] = 1
        c += d
    return aln


def _hifigan_shift(t: torch.Tensor) -> torch.Tensor:
    # Mirrors run_inference lines 230–238 / pipeline/stages.py:198–210.
    out = torch.zeros_like(t)
    out[:, :, 0] = t[:, :, 0]
    out[:, :, 1:] = t[:, :, 0:-1]
    return out


# ---------- Reference style (mel → ref_s via CoreML ref_encoder) ----------


def _compute_mel_4d(reference_path: str) -> torch.Tensor:
    import librosa

    import run_inference  # type: ignore

    wave, sr = librosa.load(reference_path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    preprocess = run_inference.make_preprocess()
    mel = preprocess(audio)              # [1, 80, T_mel]
    return mel.unsqueeze(1)              # [1, 1, 80, T_mel]


# ---------- Main ----------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="StyleTTS 2 is a text to speech model.",
        help="Any sentence that phonemizes to ≤ 57 tokens.",
    )
    parser.add_argument(
        "--reference",
        default=str(HERE / "reference_audio" / "696_92939_000016_000006.wav"),
    )
    parser.add_argument("--output", default=str(HERE / "out_coreml.wav"))
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--diffusion-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fp32",
        nargs="*",
        default=None,
        metavar="STAGE",
        help=(
            "Override `_STAGE_PRECISION` to fp32 for the listed stages. "
            "`--fp32` with no args flips all stages to fp32; "
            "`--fp32 decoder diffusion_unet` flips only those."
        ),
    )
    parser.add_argument(
        "--fp16",
        nargs="*",
        default=None,
        metavar="STAGE",
        help=(
            "Override `_STAGE_PRECISION` to fp16 for the listed stages. "
            "Mirror of `--fp32`."
        ),
    )
    args = parser.parse_args()

    # ------ Load eager artefacts that stay on Python ------
    import phonemizer  # noqa: E402

    import run_inference  # type: ignore  # noqa: E402
    from text_utils import TextCleaner  # type: ignore  # noqa: E402

    ensure_nltk()

    cleaner = TextCleaner()
    espeak = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )

    # We still load StyleTTS2 itself for `model_params.decoder.type`
    # lookup (controls the hifigan asr-shift). Everything else is
    # dispatched to CoreML below.
    print("Loading eager StyleTTS2 (used only for params lookup)…")
    t0 = time.perf_counter()
    _, eager_params = run_inference.load_styletts2(
        Path(HERE / "checkpoints" / "LibriTTS"), "cpu"
    )
    print(f"  eager load: {time.perf_counter() - t0:.2f}s")

    # Resolve per-stage precision (manifest + CLI overrides)
    precision = dict(_STAGE_PRECISION)
    for flag_val, target_prec, flag_name in (
        (args.fp32, "fp32", "--fp32"),
        (args.fp16, "fp16", "--fp16"),
    ):
        if flag_val is None:
            continue
        targets = list(_STAGE_PRECISION.keys()) if not flag_val else flag_val
        unknown = [s for s in targets if s not in precision]
        if unknown:
            raise ValueError(
                f"unknown stage(s) for {flag_name}: {unknown}; valid: {list(precision)}"
            )
        for s in targets:
            precision[s] = target_prec

    # ------ Load all CoreML stages (per-stage compute + precision; see
    #         _STAGE_COMPUTE / _STAGE_PRECISION manifests at top of file) ------
    print("\nLoading CoreML stages…")
    print("  precision: " + ", ".join(f"{k}={v}" for k, v in precision.items()))
    t0 = time.perf_counter()
    text_encoder = _load_stage("text_encoder", precision=precision["text_encoder"])
    bert = _load_stage("bert", precision=precision["bert"])
    ref_encoder = _load_stage("ref_encoder", precision=precision["ref_encoder"])
    diffusion_unet = _load_stage("diffusion_unet", precision=precision["diffusion_unet"])
    duration_predictor = _load_stage("duration_predictor", precision=precision["duration_predictor"])
    f0n_predictor = _load_stage("f0n_predictor", precision=precision["f0n_predictor"])
    har_source_model = _load_stage("har_source", precision=precision["har_source"])
    decoder = _load_stage("decoder", precision=precision["decoder"])
    print(f"  coreml load: {time.perf_counter() - t0:.2f}s")

    # ------ Stage 1: phonemize + tokenize (Python) ------
    from nltk.tokenize import word_tokenize

    # bert / diffusion_unet keep a fixed token axis (see module docstring).
    BERT_TOKENS = 57
    HOP = 300  # 24 kHz × 12.5 ms

    text = args.text.strip()
    ps = espeak.phonemize([text])
    ps = " ".join(word_tokenize(ps[0]))
    token_ids = cleaner(ps)
    token_ids.insert(0, 0)

    real_n = len(token_ids)
    if real_n > BERT_TOKENS:
        raise ValueError(
            f"text produced {real_n} tokens, exceeds bert/diffusion fixed axis "
            f"of {BERT_TOKENS}; shorten the sentence or RangeDim those stages."
        )

    # Native-length tokens for the RangeDim stages.
    tokens_native = torch.LongTensor(token_ids).unsqueeze(0)        # [1, real_n]
    input_lengths = torch.LongTensor([real_n])
    text_mask_native = run_inference.length_to_mask(input_lengths)  # [1, real_n], all False

    # Padded tokens for the fixed-axis stages (bert + diffusion_unet).
    tokens_padded_ids = list(token_ids) + [0] * (BERT_TOKENS - real_n)
    tokens_padded = torch.LongTensor(tokens_padded_ids).unsqueeze(0)
    pad_cols = BERT_TOKENS - real_n
    text_mask_padded = torch.cat(
        [
            text_mask_native,
            torch.ones(1, pad_cols, dtype=text_mask_native.dtype),
        ],
        dim=-1,
    ) if pad_cols > 0 else text_mask_native
    print(f"\nText:    {text!r}")
    print(f"Tokens:  {real_n} (bert/diffusion padded to {BERT_TOKENS})")

    # ------ Stage 2: text_encoder (CoreML, RangeDim T) ------
    t0 = time.perf_counter()
    feed = {
        "tokens": tokens_native.numpy().astype(np.int32),
        "input_lengths": input_lengths.numpy().astype(np.int32),
        "text_mask": text_mask_native.numpy().astype(np.float32),
    }
    (t_en_np,) = _predict(text_encoder, feed)
    print(f"text_encoder:       {time.perf_counter() - t0:.3f}s  out={t_en_np.shape}")

    # ------ Stage 3: bert + bert_encoder (CoreML, fixed T=57) ------
    t0 = time.perf_counter()
    feed = {
        "tokens": tokens_padded.numpy().astype(np.int32),
        "attention_mask": (~text_mask_padded).int().numpy().astype(np.int32),
    }
    bert_dur_np, d_en_np_padded = _predict(bert, feed)
    print(
        f"bert+encoder:       {time.perf_counter() - t0:.3f}s  "
        f"bert_dur={bert_dur_np.shape}  d_en={d_en_np_padded.shape}"
    )
    # d_en is fed to duration_predictor (RangeDim T). Slice padded
    # positions off so the LSTM only sees real-token features.
    d_en_np = d_en_np_padded[:, :, :real_n].astype(np.float32)

    # ------ Stage 4: ref_encoder (CoreML, uses reference mel) ------
    mel_4d = _compute_mel_4d(args.reference)
    t0 = time.perf_counter()
    (ref_s_np,) = _predict(ref_encoder, {"mel": mel_4d.numpy().astype(np.float32)})
    print(f"ref_encoder:        {time.perf_counter() - t0:.3f}s  ref_s={ref_s_np.shape}")
    ref_s = torch.from_numpy(ref_s_np).float()  # [1, 256]

    # ------ Stage 5: ADPM2 sample (CoreML UNet) + alpha/beta blend ------
    # Seed *right* before the first RNG draw to match the trace-time
    # captures in `_runtime.build_runtime` (which calls `seed_everything`
    # immediately before `run_pipeline`). This way the predicted
    # durations come out to 147 frames — the shape the stages were
    # traced with — until we promote token/frame axes to RangeDim.
    run_inference.seed_everything(args.seed)
    t0 = time.perf_counter()
    noise = torch.randn(1, 256).unsqueeze(1).numpy().astype(np.float32)  # [1, 1, 256]
    s_pred_np = _adpm2_sample(
        diffusion_unet,
        noise=noise,
        embedding=bert_dur_np,
        features=ref_s_np,
        num_steps=args.diffusion_steps,
    )
    s_pred = torch.from_numpy(s_pred_np).squeeze(1)  # [1, 256]
    s_diff = s_pred[:, 128:]
    ref_diff = s_pred[:, :128]
    ref = args.alpha * ref_diff + (1.0 - args.alpha) * ref_s[:, :128]
    s = args.beta * s_diff + (1.0 - args.beta) * ref_s[:, 128:]
    print(
        f"adpm2 sampler:      {time.perf_counter() - t0:.3f}s  "
        f"s_pred={tuple(s_pred.shape)}  ref={tuple(ref.shape)}  s={tuple(s.shape)}  "
        f"({args.diffusion_steps} steps × 2 dispatches)"
    )

    # ------ Stage 6: duration_predictor (CoreML, RangeDim T) → alignment ------
    t0 = time.perf_counter()
    feed = {
        "d_en": d_en_np,
        "s": s.numpy().astype(np.float32),
        "text_mask": text_mask_native.float().numpy().astype(np.float32),
    }
    d_np, duration_logits_np = _predict(duration_predictor, feed)
    duration = torch.sigmoid(torch.from_numpy(duration_logits_np)).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    real_frames = int(pred_dur.sum().item())
    pred_aln_trg = _build_pred_aln_trg(pred_dur, real_n)  # [real_n, real_frames]
    print(
        f"duration_predictor: {time.perf_counter() - t0:.3f}s  "
        f"pred_aln_trg={tuple(pred_aln_trg.shape)}  frames={real_frames}"
    )

    # ------ Stage 7: en/asr build + f0n_predictor (CoreML, RangeDim F) ------
    d = torch.from_numpy(d_np).float()                   # [1, real_n, 640]
    t_en = torch.from_numpy(t_en_np).float()             # [1, 512, real_n]
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)  # [1, 640, real_frames]
    asr = t_en @ pred_aln_trg.unsqueeze(0)                # [1, 512, real_frames]
    if eager_params.decoder.type == "hifigan":
        en = _hifigan_shift(en)
        asr = _hifigan_shift(asr)

    t0 = time.perf_counter()
    feed = {
        "en": en.numpy().astype(np.float32),
        "s": s.numpy().astype(np.float32),
    }
    f0_pred_np, n_pred_np = _predict(f0n_predictor, feed)
    print(
        f"f0n_predictor:      {time.perf_counter() - t0:.3f}s  "
        f"f0={f0_pred_np.shape}  n={n_pred_np.shape}"
    )

    # ------ Stage 8: har_source (CoreML, RangeDim F0_LEN) ------
    t0 = time.perf_counter()
    (har_np,) = _predict(har_source_model, {"f0": f0_pred_np.astype(np.float32)})
    print(f"har_source:         {time.perf_counter() - t0:.3f}s  har={har_np.shape}")

    # ------ Stage 9: decoder (CoreML, RangeDim T_FRAME / F0_LEN / HAR_LEN) ------
    ref_in = ref.squeeze().unsqueeze(0).numpy().astype(np.float32)  # [1, 128]
    t0 = time.perf_counter()
    feed = {
        "asr": asr.numpy().astype(np.float32),
        "f0": f0_pred_np.astype(np.float32),
        "n": n_pred_np.astype(np.float32),
        "ref": ref_in,
        "har_source": har_np.astype(np.float32),
    }
    (audio_np,) = _predict(decoder, feed)
    print(f"decoder:            {time.perf_counter() - t0:.3f}s  audio={audio_np.shape}")

    # Mirror run_inference's tail trim of 50 samples (no other trim
    # needed: decoder runs at native frame length now).
    waveform = np.squeeze(audio_np)[..., :-50].astype(np.float32)
    out_path = Path(args.output)
    sf.write(str(out_path), waveform, 24000)
    duration_s = waveform.shape[-1] / 24000.0
    print(f"\nWrote {out_path}  ({duration_s:.2f}s @ 24 kHz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
