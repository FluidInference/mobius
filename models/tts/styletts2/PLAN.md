# StyleTTS2 → CoreML Conversion Plan

> **Status note (May 2026):** This plan documents the original 4-graph
> CoreML port that powered the legacy `Sources/FluidAudio/TTS/StyleTTS2/`
> backend. The backend has since been frozen — see
> `coreml/TRIALS.md` Phase 4 (Trial 35). New work targets the 7-graph
> StyleTTS2-ANE re-cut (`Sources/FluidAudio/TTS/StyleTTS2Ane/`).
> This file is preserved as historical reference — none of the
> per-stage shapes, bucket bounds, or stage splits below apply to
> the ANE re-cut, which mirrors Kokoro-ANE's Albert / PostAlbert /
> Alignment / DiffusionStep / Prosody / Noise / Vocoder layout.

## 1. Inference flow we are reproducing

Source: `Demo/Inference_LibriTTS.ipynb` in [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2).

```
text
  └─ phonemizer (espeak-ng en-us, with_stress) ─┐
                                                ▼
                                        TextCleaner → tokens (1, T_tok)
                                                │
        ┌───────────────────────────────────────┼──────────────────────────────┐
        ▼                                       ▼                              ▼
  TextEncoder(tokens)                   PL-BERT(tokens)                 StyleEncoder(ref_mel)
   → t_en (1,512,T_tok)                  → bert_dur (1,T_tok,768)         → ref_s (1,256)
                                        bert_encoder(bert_dur)
                                          → d_en (1,512,T_tok)

  s_pred = ADPM2_sample(noise=(1,1,256),
                        embedding=bert_dur,
                        features=ref_s,
                        steps=5, cfg_scale=1.0)
  s_pred = α·s_pred + (1-α)·ref_s
  ref, s = s_pred.split([128,128])      # ref → decoder; s → predictor

  d = predictor.text_encoder(d_en, s, lengths)
  x, _ = predictor.lstm(d)
  pred_dur = sigmoid(predictor.duration_proj(x)).sum(-1).round().clamp(min=1)

  pred_aln_trg = one_hot_alignment(pred_dur)             # (T_tok, T_mel)
  en  = d.transpose(-1,-2) @ pred_aln_trg                # (1,512,T_mel)
  asr = t_en              @ pred_aln_trg                 # (1,512,T_mel)

  F0, N = predictor.F0Ntrain(en, s)                      # (1,T_mel) each
  wav   = decoder(asr, F0, N, ref)                       # (1, T_mel*300)  @ 24 kHz
```

## 2. Split into CoreML packages

| # | Package                      | Compute units | Why                                                     |
|---|------------------------------|---------------|---------------------------------------------------------|
| A | `styletts2_text_predictor`   | CPU + GPU     | BiLSTM (TextEncoder + DurationEncoder) — ANE rejects    |
| B | `styletts2_diffusion_step`   | CPU + GPU     | Single UNet step. Sampler loop runs in Swift            |
| C | `styletts2_f0n_energy`       | ANE           | AdainResBlk1d stack — pure conv, fixed shapes           |
| D | `styletts2_decoder`          | ANE           | HiFi-GAN (Conv1d + ConvTranspose1d + snake activation)  |

### Bucketing

| Axis  | Buckets                       | Notes                                 |
|-------|-------------------------------|---------------------------------------|
| T_tok | 32, 64, 128, 256, 512         | Pad with PAD; covers ~most utterances |
| T_mel | 256, 512, 1024, 2048, 4096    | ≈3.2 / 6.4 / 12.8 / 25.6 / 51.2 s     |

Use `coremltools.EnumeratedShapes` per Kokoro precedent.

## 3. Swift-side responsibilities

- Phonemization (espeak-ng) — reuse FluidAudio's existing G2P path
- TextCleaner → 178-token mapping (table exported as JSON)
- ADPM2 + Karras sigma schedule + CFG batching, calling Package B 5×
- `pred_aln_trg` build (cumsum + scatter) and matmul with `t_en` / `d`
- Style blending `α·s_pred + (1-α)·ref_s`
- Reference style extraction from a WAV (small CoreML or Accelerate path TBD)

## 4. Conversion blockers + mitigations

| Blocker                                                         | Mitigation                                                            |
|-----------------------------------------------------------------|-----------------------------------------------------------------------|
| Diffusion sampler `for` loop + `**kwargs` plumbing (issue #245) | Convert single denoise step; loop in Swift                            |
| Variable `T_tok` / `T_mel`                                      | EnumeratedShapes buckets                                              |
| `torch.stft` / complex tensors in iSTFTNet                      | Use HiFi-GAN variant (LibriTTS); LJSpeech deferred                    |
| BiLSTM dynamic seq                                              | Pin Package A to CPU+GPU                                              |
| Hard-alignment Python loop                                      | Build matrix in Swift                                                 |
| Discriminators / optimizer state in `.pth` (~600 MB unused)     | Strip in `00_fetch_weights.py`; keep inference modules only           |

## 5. Script order

```
scripts/00_fetch_weights.py          # download .pth, strip to inference modules
scripts/01_export_text_predictor.py  # → coreml/styletts2_text_predictor.mlpackage
scripts/02_export_diffusion_step.py  # → coreml/styletts2_diffusion_step.mlpackage
scripts/03_export_f0n_energy.py      # → coreml/styletts2_f0n_energy.mlpackage
scripts/04_export_decoder.py         # → coreml/styletts2_decoder.mlpackage
scripts/99_parity_check.py           # PyTorch ↔ CoreML cosine on intermediate tensors
```

Each export script: load PyTorch module → wrap with a thin traceable adapter
(no kwargs, fixed dtypes) → `torch.jit.trace` → `coremltools.convert` with
`EnumeratedShapes`.

## 6. Open questions

- [ ] Inference-only param count after stripping the `.pth` (estimated 100–180 M; needs measurement).
- [ ] Whether the HiFi-GAN ConvTranspose1d strides (10/5/3/2) place on ANE; needs empirical test.
- [ ] Whether to ship `StyleEncoder` as its own `.mlpackage` for runtime ref-clip extraction, or pre-compute styles offline.
- [ ] Whether to stay on yl4579 weights (license risk) or pivot to a permissively-licensed StyleTTS2 derivative for any future redistribution.

## 7. Out of scope (this branch)

- LJSpeech / iSTFTNet variant
- Multilingual PL-BERT (papercup-ai/multilingual-pl-bert)
- HuggingFace upload of converted artifacts
- VoiceInk app integration
