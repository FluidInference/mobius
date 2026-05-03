# 7-stage chain architecture

Kokoro is a single PyTorch graph; for ANE-friendly CoreML deployment laishere
split it into 7 mlpackages so each stage can be sized, quantized, and assigned
to a compute unit independently. The split also lets the dual-output vocoder
trick (see below) be applied cleanly.

```
text  ─G2P→  IPA  ─encode→  input_ids  (int32 [1, T_enc])
                                                │
   ┌────────────────────────────────────────────┘
   │
   ▼
[1] Albert         input_ids, attention_mask         → bert_dur
                   ALBERT-mini text encoder. fp16+int8pal. CPU+ANE.

[2] PostAlbert     bert_dur, input_ids, style_s,     → duration, d, t_en
                   speed, attention_mask
                   Duration predictor head + style mixing.

   ── duration → np.round → pred_dur (int32) ──→  T_a = sum(pred_dur)

[3] Alignment      pred_dur, d, t_en                 → en, asr
                   Builds the [T_enc → T_a] alignment matrix and applies it.
                   T_a varies per sentence (read from tensor shape post-call).

[4] Prosody        en, style_s                       → F0, N
                   F0 + noise envelope predictor. fp16+int8pal. ALL.

[5] Noise          F0_curve, style_timbre            → x_source_0, x_source_1
                   Source-filter excitation (NOT vocoder body).
                   fp32+int8pal because the sin/exp ops break ANE's fp16 range.

[6] Vocoder        asr, F0_curve, N_pred, x_source_0,→ anchor, x_pre
                   x_source_1, style_timbre
                   The HiFi-GAN body up to (but excluding) the final iSTFT
                   conv_post. fp16+int8pal. CPU+ANE.

   ── DISCARD anchor; keep x_pre only ──

[7] Tail           x_pre                             → audio (fp32 [1, N])
                   Final fp32 conv_post + iSTFT. ALL.
```

## Dual-output vocoder trick

The HiFi-GAN body terminates in a `conv_post → exp → sin → iSTFT` block whose
`exp/sin` operations have a dynamic range that ANE's fp16 cannot represent
without artifacts.

**The trick** — split it cleanly across two models:

* `KokoroVocoder` runs the body in fp16 on ANE up to and including
  `conv_post`, then **emits two outputs**:
   * `anchor` — a static-shape sentinel CoreML needs to keep the trace happy
   * `x_pre` — the pre-iSTFT [1, 128, T_pre] tensor consumed by Tail
* `KokoroTail` runs the final `exp/sin/iSTFT` in fp32 on CPU/GPU.

The host code (Swift / Python) **discards `anchor`** and converts `x_pre` to
fp32 before calling Tail.

## fp16 ↔ fp32 boundaries

The host code converts numpy/Core ML tensors at three boundaries:

| Boundary | Direction | Rationale |
|---|---|---|
| Prosody → Noise   | fp16 → fp32 | Noise net needs fp32 sin/exp range |
| Noise → Vocoder   | fp32 → fp16 | Vocoder is fp16 ANE |
| Vocoder → Tail    | fp16 → fp32 | Tail is the fp32 iSTFT bookend |

In Swift this is done via `Accelerate.vImageConvert_PlanarFtoPlanar16F` (and
inverse). In Python it's `np.array(...).astype(np.float16/float32)`.

## Voice pack layout

`af_heart.bin` is `[510, 256]` flat fp32:

* row index = `min(max(T_enc - 1, 0), 509)`  — voice changes with utterance length
* cols `[0:128]` = `style_timbre`     — fed into Noise + Vocoder
* cols `[128:256]` = `style_s`        — fed into PostAlbert + Prosody

## Token format

```
input_ids = [BOS=0, vocab.lookup(c) for c in phonemes if c in vocab, EOS=0]
attention_mask = [1] * T_enc
```

Hard cap: `T_enc ≤ 512` (ALBERT context window). The benchmark caps phonemes
to 510 to leave room for BOS/EOS.

## Compute unit choice — why not all ANE?

Conversion experiments showed each stage's optimal compute unit was different:

* `Prosody`, `Noise`, `Tail` — `compute_units=ALL` won (ANE rejected the
  sin/exp ops or the dynamic shapes in Noise; ANE doesn't run fp32 paths in
  Tail). Letting CoreML scheduler pick is fastest.
* Everything else — `CPU_AND_NE` was strictly faster than `ALL` because the
  scheduler kept moving small ops to GPU and paying the dispatch tax.

This per-stage compute-unit assignment is **mirrored exactly** in the Swift
`KokoroLaiModelStore`.
