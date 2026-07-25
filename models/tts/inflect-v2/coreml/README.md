# Inflect v2 (Micro / Nano) → CoreML

Conversion of [owensong/Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2) (9.36M params)
and [owensong/Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2) (3.97M params) —
ultra-tiny VITS-family end-to-end TTS, 24 kHz mono, Apache-2.0.

## Pipeline split

The VITS inference graph is split into two fixed-shape, fully deterministic CoreML models.
Everything stochastic or dynamically shaped runs on the host:

```
text ── espeak frontend ── tokens (interspersed with blanks, pad to T_text=256)
                              │
              ┌───────────────▼────────────────┐
              │ encoder.mlpackage              │  tokens + x_mask
              │ TextEncoder + DurationPredictor│  → m_p, logs_p, logw   [1, C, 256]
              └───────────────┬────────────────┘
                              │ host: w = ceil(exp(logw))·length_scale
                              │ host: repeat-expand m_p/logs_p to frames
                              │ host: z_p = m_p + randn·exp(logs_p)·noise_scale
              ┌───────────────▼────────────────┐
              │ synthesizer.mlpackage          │  z_p + y_mask  [1, C, T_frames]
              │ reverse coupling flow + HiFiGAN│  → waveform    [1, 1, T_frames·256]
              └────────────────────────────────┘
                              │ host: trim to y_len·256 samples
```

- `use_sdp: false` in both configs → deterministic duration predictor, no noise inside either model.
- All masking is host-provided (`x_mask`, `y_mask`), so padded positions stay zero through the
  masked encoder/flow. Only the unmasked HiFiGAN decoder sees pad bleed-through, limited to its
  receptive field at the valid/pad boundary (audio parity below includes this effect).
- Config-driven: Micro and Nano share runtime code; only channel widths differ.

## Commands

```bash
uv sync
# both variants, fp16, T_text=256 tokens / T_frames=1024 frames (10.9 s audio budget)
uv run python convert-coreml.py
uv run python compare-models.py --variant micro
uv run python compare-models.py --variant nano
```

## Parity (fp16, M5 Pro, macOS 26.6, seed 0, "The quick brown fox…")

| stage | metric | Micro | Nano |
|---|---|---|---|
| encoder m_p | max_abs / rel_l2 | 0.0032 / 0.0005 | 0.0034 / 0.0006 |
| encoder logw | max_abs / rel_l2 | 0.0018 / 0.0008 | 0.0020 / 0.0008 |
| durations | frame diff vs torch | 0 | 0 |
| audio (same z_p) | corr / rel_l2 | 0.99991 / 0.0136 | 0.99996 / 0.0096 |

## Profiling (coreml-cli, warm predicts, Micro)

| model | shape | best unit | predict | CPU+NE split |
|---|---|---|---|---|
| encoder | T_text=256 | any (~1 ms) | 1.2 ms | 26% ANE |
| synthesizer | T_frames=1024 (10.9 s) | GPU | 33.4 ms (~330× RT) | 17% ANE / 83% CPU |
| synthesizer | T_frames=256 (2.7 s) | GPU | 9.9 ms (~275× RT) | **77% ANE** / 23% CPU |

### ANE findings

- At `T_frames=1024` the waveform-rate tensors hit the ANE tensor-width limit
  (`W ≤ 65536`; final stages are W=131072/262144) → almost everything falls back.
- At `T_frames=256` (2.7 s bucket, final W=65536 exactly at the limit) residency jumps to 77%.
  Remaining fallbacks: 8 dilated resblock convs ("space-to-batch exceeds L2 DMA buffer") and
  2 convs at W=65538 (same-pad overshoot past the limit).
- GPU is fastest at every size; ANE only matters for power. Full-ANE would need the
  kokoro-ane treatment (reshape waveform rate to 2D, split dilated convs) — not pursued here.

## Trace notes (future VITS conversions)

- `attentions.MultiHeadAttention` relative-position helpers do Python `max()` over traced sizes →
  coremltools `aten::int` crash. Patched at conversion time to coerce lengths with `int()`
  (static shapes make them constants). See `_static_relative_attention`.
- `commons.fused_add_tanh_sigmoid_multiply` indexes an `IntTensor` channel count →
  `intimplicit` not implemented. Patched to a Python-int slice. See `_patch_fused_activation`.
- Weight norm must be removed (`dec` + flow WN layers) before tracing, matching the
  reference `optimize_for_inference`.

## Host responsibilities (Swift port checklist)

1. espeak-ng en-us phonemization with stress (same frontend family as existing G2P work),
   keithito symbol table, blank interspersal (`add_blank: true`).
2. Duration expansion: `repeat` each token column `ceil(exp(logw)/speed)` times.
3. `z_p = m_p + N(0,1)·exp(logs_p)·noise_scale` (default noise_scale 0.667).
4. Trim output to `y_len·256` samples; optional 5 ms edge fade + inter-chunk pauses
   (see upstream `inference.py`).

## MiniMax-English benchmark vs FluidAudio TTS backends

`benchmark-minimax.py` runs the FluidAudio `tts-benchmark` methodology (100
MiniMax-English phrases, warm one-shot latency, agg RTFx) over the CoreML
pipeline: persistent espeak G2P → t512 encoder → host expansion/noise →
smallest fitting synthesizer bucket {256, 512, 1024, 2048 frames}. WER/CER
scored with the identical Parakeet TDT + TextNormalizer path via
`fluidaudio tts-asr-verify --score-only` (added for this purpose).

M5 Pro, macOS 26.6, `ComputeUnit.ALL` (routes to GPU), fp16, one bucket set
preloaded (cold start = first compile of all buckets). FluidAudio rows from
`Documentation/TTS/Benchmarks.md` (Kokoro re-baselined same day, same host):

| Backend | Weights | Synth p50/p95 | Agg RTFx | WER | CER |
|---|---|---|---|---|---|
| **Inflect Micro CoreML** | 19 MB fp16 (9.4M) | **25.7 / 35.5 ms** | **245×** | 1.43% | 0.44% |
| **Inflect Nano CoreML** | 8 MB fp16 (4.0M) | **13.2 / 16.9 ms** | **460×** | 1.92% | 0.57% |
| Kokoro ANE (en) | ~330 MB (82M) | 241 / 305 ms | 31× | 0.68% | 0.15% |
| PocketTTS (en, streaming) | ~330 MB | 933 / 1233 ms (TTFT 26 ms) | 6.5× | 0.51% | 0.08% |
| Supertonic-3 (int4) | ~100 MB | 81 / 120 ms | 94× | 1.02% | 0.31% |
| StyleTTS2 (M2 numbers) | ~670 MB | 1574 / 3088 ms | 4.6× | 9.4% | 4.1% |

Notes:
- Synthesizer cost is bucket-quantized and ~linear in frames (micro ≈
  0.033 ms/frame on GPU). The first run used coarse buckets {256, 512, 1024,
  2048}; 75% of phrases (p50 = 621 frames) overpaid in f1024 (p50 37 ms /
  192×). The shipped set {256, 384, 512, 640, 768, 896, 1024, 2048} tracks
  the corpus distribution — at 128-frame granularity, padding overhead is
  within ~5% of exact-shape synthesis, so dynamic shapes buy nothing more.
- fp16 model I/O and `CompiledMLModel` dispatch were A/B'd at f768: both
  within noise (~25.3 ms either way) — the GPU is compute-bound, casts and
  dispatch don't matter. Kept fp32 I/O.
- Upstream claims 6.28× realtime on 4-thread CPU for Micro; the CoreML GPU
  pipeline is ~30× that.
- The upstream Python frontend costs ~450 ms/phrase constructing a fresh
  espeak backend per `phonemize()` call; the runner uses one persistent
  `EspeakBackend` (byte-identical phonemes, ~0.4 ms) — this matches what an
  in-process Swift port would do. First runs with the stock frontend read
  p50 457 ms / RTFx 13.8× — that artifact was 85% of the latency.
- Peak RSS (Micro 628 MB / Nano 587 MB) is the whole Python process
  (interpreter + numpy + 4 synthesizer buckets) — not comparable to the
  Swift harness numbers.
- Quality-per-byte is the headline: Nano is 1/40th the weights of Kokoro at
  ~3× the (still very low) WER; Micro at 19 MB beats Supertonic-3's WER.

## Assets

`checkpoints/` (downloaded via `hf download`, not committed) and `build/` outputs are local-only.
