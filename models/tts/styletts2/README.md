# StyleTTS2 — CoreML Conversion

[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) (LibriTTS multi-speaker
checkpoint) ported to CoreML for on-device inference on Apple Silicon
(macOS 14+ / iOS 17+).

**Status:** all four stages converted, optimized, and validated end-to-end.
Vendored upstream is in `vendor/StyleTTS2/` (gitignored). Shared loader and
traceable wrappers in `scripts/_styletts2_lib.py`. See `coreml/PRECISION.md`
for the precision/quantization recipe and `coreml/TRIALS.md` for the
chronological conversion log.

## Headline numbers

- **RTFx:** 4.32× warm (M-series Mac, 5-step ADPM2 sampler)
- **On-disk size:** ~1.4 GB (decoder is fp32; see PHASE6_FP16_DECODER.md for
  why fp16 produces robotic audio and isn't shipped)
- **Log-mel cosine vs PyTorch fp32:** 0.9687
- **Voice-clone fidelity (ECAPA-TDNN cos to ref):** 0.18 — at the model's
  architectural ceiling (PyTorch fp32 itself is 0.29; see TRIALS.md Phase 5)

## Run

```bash
cd models/tts/styletts2
uv sync
uv run python scripts/00_fetch_weights.py
uv run python scripts/01_export_text_predictor.py
uv run python scripts/02_export_diffusion_step.py
uv run python scripts/03_export_f0n_energy.py
uv run python scripts/04_export_decoder.py
# `99c_e2e_optimized.py` needs a reference WAV (any 24 kHz speech clip; the
# style encoder is robust to ~5 s of speech). Output goes to
# /tmp/styletts2-e2e/coreml_int8_diff512.wav by default.
uv run python scripts/99c_e2e_optimized.py --reference-wav <path/to/ref.wav>
```

Each export script accepts `--trace-only` to validate the PyTorch trace
without invoking `coremltools.convert` (useful for fast iteration).

## Target

- Initial checkpoint: **LibriTTS multi-speaker** (`yl4579/StyleTTS2-LibriTTS`,
  `Models/LibriTTS/epochs_2nd_00020.pth`). HiFi-GAN decoder — avoids
  `torch.stft` / complex tensors. Multi-speaker via reference clip → `ref_s`.
- Follow-up (not in initial scope): LJSpeech (iSTFTNet decoder; would need a
  Swift-side iSTFT).

## CoreML model split

| Package                                  | CU        | Precision        | Buckets                        | Inputs                                                   | Called      |
|------------------------------------------|-----------|------------------|--------------------------------|----------------------------------------------------------|-------------|
| `styletts2_text_predictor_{B}.mlpackage` | ANE       | fp16             | `B ∈ {32, 64, 128, 256, 512}` | `tokens (1, T_tok)`                                      | 1× per utt  |
| `styletts2_diffusion_step_512.mlpackage` | CPU+GPU   | fp16             | 1 (512)                        | `x`, `sigma`, `embedding (bert_dur)`, `features (ref_s)` | ~5× per utt |
| `styletts2_f0n_energy.mlpackage`         | ANE       | fp16             | dynamic                        | `en (1, 512, T_mel)`, `s (1, 128)`                       | 1× per utt  |
| `styletts2_decoder_{M}.mlpackage`        | CPU+GPU   | **fp32**         | `M ∈ {256, 512, 1024, 2048, 4096}` | `asr`, `F0`, `N`, `ref (1, 128)`                  | 1× per utt  |

The diffusion sampler loop (ADPM2 + Karras schedule + CFG) lives in Swift and
calls `styletts2_diffusion_step_512.mlpackage` `num_steps` (default 5) times
per utterance. The hard-alignment matrix (cumsum of predicted durations →
one-hot → matmul) also lives in Swift.

**Why only one diffusion bucket?** Empirically every `bert_dur` we observed
fit in the 512 bucket and the cost ladder is non-linear (B=32: 66 ms/step,
B=512: 152 ms/step), so the 4 smaller buckets were dead weight (192 MB).
See TRIALS.md Phase 4.

**Why fp16 across text_predictor / diffusion / f0n and fp32 only on the
decoder?** Selective int8 PTQ on text_predictor was tried and dropped
(see `coreml/PRECISION.md` — ANE has no int8 GEMM, so the only payoff
is ~3 MB of bandwidth per bucket; not worth the per-bucket validation
cost). The decoder must be fp32 because SineGen accumulates phase to
~4000 magnitude and fp16 collapses the sine output to a few discrete
values (see `coreml/PHASE6_FP16_DECODER.md`).

## Phonemizer

espeak-ng IPA + stress (same family as Kokoro / PocketTTS). The 178-token
vocabulary in `text_utils.TextCleaner` differs from Kokoro and is exported as
JSON in `coreml/constants/`.

## Layout

```
models/tts/styletts2/
├── README.md              # this file
├── PLAN.md                # original design + risks + open questions
├── LICENSE_REVIEW.md      # upstream weight licensing concerns
├── pyproject.toml         # conversion env (uv-managed)
├── .gitignore             # excludes vendor/, checkpoints/, coreml/*.mlpackage/
├── coreml/
│   ├── PRECISION.md       # mixed-precision recipe + per-stage rationale
│   ├── TRIALS.md          # chronological conversion log
│   └── *.mlpackage        # generated artifacts (gitignored)
├── vendor/StyleTTS2/      # upstream repo, cloned locally, gitignored
└── scripts/
    ├── _styletts2_lib.py             # shared loader + traceable wrappers
    ├── 00_fetch_weights.py           # download .pth, strip discriminators
    ├── 01_export_text_predictor.py   # → text+predictor mlpackage  (ANE)
    ├── 02_export_diffusion_step.py   # → diffusion-step mlpackage  (CPU+GPU)
    ├── 03_export_f0n_energy.py       # → F0/N mlpackage            (ANE)
    ├── 04_export_decoder.py          # → HiFi-GAN decoder mlpackage (CPU+GPU)
    ├── 99_parity_check.py            # PyTorch ↔ CoreML cosine check (per stage)
    ├── 99b_e2e_coreml.py             # baseline e2e driver (no quantization)
    ├── 99c_e2e_optimized.py          # optimized e2e: fp16 TP + diff B=512 + warmup
    └── optimize/
        ├── quantize_text_predictor_int8.py  # historical: int8 PTQ recipe (not shipped)
        └── measure_diffusion_buckets.py     # per-bucket warm timing
```

## License

The yl4579 LibriTTS weights ship with non-permissive consent / disclosure
terms in the upstream README. Treat any converted artifacts as **local-only**.
See `LICENSE_REVIEW.md` before any redistribution.
