"""End-to-end CoreML inference for StyleTTS2 LibriTTS.

Drives all 7 converted `.mlpackage` stages and writes a 24 kHz WAV.
The Python side keeps:

    * phonemizer + tokenizer        (CPU-only by definition)
    * Karras sigma schedule         (5 floats, trivial)
    * ADPM2 step loop               (5 steps × 2 dispatches per step;
                                     each dispatch runs CoreML UNet)
    * alignment matrix construction (data-dependent shape)
    * `precompute_har_source(...)`  (SineGen + SourceModuleHnNSF on CPU,
                                     see trials.md Stage 7)

Everything else lives in CoreML.

Usage:

    cd models/tts/styletts2
    uv run python coreml/inference.py \
        --text "StyleTTS 2 is a text to speech model." \
        --output out_coreml.wav

Shape strategy:

  * `text_encoder`, `duration_predictor`, `f0n_predictor` are converted
    with `ct.RangeDim` on the variable axis (token T or frame F) and
    run at native length. This avoids LSTM bidirectional contamination
    on the token axis (their wrappers drop `pack_padded_sequence` for
    trace).
  * `bert` and `diffusion_unet` keep a fixed token axis of 57. HF Albert
    and the cross-attention diffusion U-Net both produce shape ops that
    coremltools' MLProgram backend rejects under RangeDim
    ("data-dependent shapes were disabled"). Tokens are padded to 57 for
    these two stages; BERT respects `attention_mask` so contamination at
    real positions is bounded.
  * `decoder` (HiFi-GAN) runs **eager PyTorch** at native frame length.
    Two reasons it isn't CoreML:
      - Frame-axis zero-padding fundamentally contaminates real frames
        through the snake-activation residual stack and AdaIN convs
        (verified at "core employees" / "core email" output drift).
      - With `T_FRAME` promoted to `ct.RangeDim`, the decoder MIL graph
        has internal shape constants baked at trace time that the CPU
        MLProgram backend rejects at runtime ("Invalid blob shape:
        Data-dependent shapes were disabled: input_13 [1, ?, ?] vs
        [1, 57, ?]"). The trace shape leaks through `torch.zeros_like`
        / shape-derived constants in the source-filter / AdaIN paths.
    Running the decoder eagerly at native frames gives bit-identical
    HiFi-GAN output and avoids the ~14ms HiFi-GAN x300-upsample CoreML
    speedup; that's an acceptable trade-off until the wrapper can be
    rewritten free of shape-baked constants.

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


# ---------- CoreML helpers ----------


def _load_stage(stage: str, compute_units: ct.ComputeUnit) -> ct.models.MLModel:
    pkg = PACKAGES_DIR / f"{stage}.mlpackage"
    if not pkg.exists():
        raise FileNotFoundError(f"missing {pkg} — run coreml/convert.py first")
    return ct.models.MLModel(str(pkg), compute_units=compute_units)


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
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    args = parser.parse_args()

    cu_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    cu = cu_map[args.compute_units]

    # ------ Load eager artefacts that stay on Python ------
    import phonemizer  # noqa: E402

    import run_inference  # type: ignore  # noqa: E402
    from text_utils import TextCleaner  # type: ignore  # noqa: E402

    from coreml.wrappers import precompute_har_source  # noqa: E402

    ensure_nltk()

    cleaner = TextCleaner()
    espeak = phonemizer.backend.EspeakBackend(
        language="en-us", preserve_punctuation=True, with_stress=True
    )

    # We still load StyleTTS2 itself, but only for `precompute_har_source`
    # (CPU-side SineGen) and `model_params.decoder.type == 'hifigan'` lookup.
    # Everything else is dispatched to CoreML below.
    print("Loading eager StyleTTS2 (used only for SineGen precompute)…")
    t0 = time.perf_counter()
    eager_model, eager_params = run_inference.load_styletts2(
        Path(HERE / "checkpoints" / "LibriTTS"), "cpu"
    )
    print(f"  eager load: {time.perf_counter() - t0:.2f}s")

    # ------ Load all CoreML stages ------
    print(f"\nLoading CoreML stages (compute_units={args.compute_units})…")
    t0 = time.perf_counter()
    text_encoder = _load_stage("text_encoder", cu)
    bert = _load_stage("bert", cu)
    ref_encoder = _load_stage("ref_encoder", cu)
    diffusion_unet = _load_stage("diffusion_unet", cu)
    duration_predictor = _load_stage("duration_predictor", cu)
    f0n_predictor = _load_stage("f0n_predictor", cu)
    print(f"  coreml load: {time.perf_counter() - t0:.2f}s")

    # Decoder is eager (see module docstring). Apply the same
    # weight_norm / dropout / generator patches DecoderWrapper applies,
    # so the eager decoder here matches the reference output exactly.
    from coreml.wrappers import (  # noqa: E402
        _patch_generator_use_har,
        _remove_weight_norm_recursive,
        _strip_dropout,
    )

    eager_decoder = eager_model.decoder
    _strip_dropout(eager_decoder)
    _remove_weight_norm_recursive(eager_decoder)
    _patch_generator_use_har(eager_decoder.generator)
    eager_decoder.eval()

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

    # ------ Stage 8: precompute har_source (CPU) + decoder (eager) ------
    f0_pred = torch.from_numpy(f0_pred_np).float()
    n_pred = torch.from_numpy(n_pred_np).float()

    t0 = time.perf_counter()
    har = precompute_har_source(eager_decoder, f0_pred)
    print(f"har_source (CPU):   {time.perf_counter() - t0:.3f}s  har={tuple(har.shape)}")

    ref_in = ref.squeeze().unsqueeze(0)  # [1, 128]
    t0 = time.perf_counter()
    # Inline DecoderWrapper.forward at native frame length. The eager
    # decoder produces bit-identical HiFi-GAN output without the CoreML
    # frame-axis padding contamination or RangeDim shape errors.
    with torch.no_grad():
        F0 = eager_decoder.F0_conv(f0_pred.unsqueeze(1))
        N = eager_decoder.N_conv(n_pred.unsqueeze(1))
        x_dec = torch.cat([asr, F0, N], dim=1)
        x_dec = eager_decoder.encode(x_dec, ref_in)
        asr_res = eager_decoder.asr_res(asr)
        res = True
        for block in eager_decoder.decode:
            if res:
                x_dec = torch.cat([x_dec, asr_res, F0, N], dim=1)
            x_dec = block(x_dec, ref_in)
            if block.upsample_type != "none":
                res = False
        # Patched Generator.forward signature: (x, s, har_source, f0).
        audio_t = eager_decoder.generator(x_dec, ref_in, har, f0_pred)
    audio_np = audio_t.cpu().numpy()
    print(f"decoder (eager):    {time.perf_counter() - t0:.3f}s  audio={audio_np.shape}")

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
