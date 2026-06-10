# Supertonic-3 VectorEstimator: fused 8-step denoising loop (L128)

Campaign date: 2026-06-10, M5 Pro / macOS 26.5, system python3.11 +
coremltools 9.0 + torch 2.10. Branch `feat/supertonic-ve-fusion`.

(Results being filled in — placeholder.)

## Verdict (2026-06-10): DECLINED

Fused 8-step graph built and measured (scripts in `coreml/`, logs in `build/`):

| | 8-call loop (shipped) | fused (palette4) |
|---|---|---|
| ANE plan | 94% | **99.5%** (4522 ops) |
| per-chunk | ~30.4 ms (8 × 3.8) | 28.06 ms |
| size | 33 MB | 33.9 MB |
| parity (final latent, max_abs) | — | **1.469 — 30× outside the 0.05 band** |

Why parity fails: the host loop's per-step fp32 IO casts re-quantize the
latent every step, acting as error-containment barriers; the fused graph
runs all 8 steps in fp16 end-to-end and the LSD denoiser's documented
precision sensitivity (PRECISION notes) compounds. Fixing it would mean
fp32 internal latents (kills the ANE win) or per-step internal casts
(re-creates the boundary cost being fused away).

Cost/benefit: ~2–3 ms/chunk (~3% of an 81 ms synth, less in-host per the
EOU compression lesson) against an audible-quality gamble requiring a
Whisper gate to even evaluate. **Declined; candidate closed.** Re-visit
only if a future Core ML exposes fp32 islands inside ANE graphs.
