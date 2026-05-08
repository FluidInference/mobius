# StyleTTS2 → CoreML iteration_2

Production-ready fp32 mlpackages adopting Trials 4 + 6 + 8b from
`coreml/fusions.md`.

## Pipeline (8 stages, 8 dispatches)

```
text_encoder       → CPU_ONLY      fp32   21 MB
bert               → ALL           fp32   23 MB
ref_encoder        → CPU_AND_GPU   fp32  106 MB
fused_diffusion_sampler  → ALL    fp32   94 MB   ← Trial 4 (replaces diffusion_unet × 8)
duration_predictor → CPU_ONLY      fp32   30 MB
fused_f0n_har_source     → CPU_ONLY  fp32  32 MB ← Trial 6 (replaces f0n_predictor + har_source)
decoder_pre        → CPU_AND_NE    fp32  128 MB
decoder_upsample   → CPU_ONLY      fp32   79 MB
```

Total: **514 MB**, 8 mlpackages, 8 dispatches per utterance.

## Performance

Warm latency on M-series Mac, single-process, no other GPU/ANE workloads:

* Pipeline warm: **~480–565 ms** (down from ~1030 ms baseline)
* Stage count: 9 → 8 (Trials 4 + 6)
* Dispatches per utterance: 16 → 8 (−50%)

See `coreml/fusions.md` for full trial history, latency tables, parity
chains, and per-stage placement sweep results.

## Adopted trials

| Trial | Change                                               | Save |
|-------|------------------------------------------------------|------|
| 4     | fused 5-step ADPM2 sampler (8 dispatches → 1)        | −437 ms warm |
| 6     | fused f0n_predictor + har_source                     | −42 ms warm |
| 8b    | bert→ALL, ref_encoder→CPU_AND_GPU, sampler→ALL       | small but stable |

## Skipped / dropped

| Trial | Outcome                                              |
|-------|------------------------------------------------------|
| 5     | har + decoder_upsample fuse — partition tax (+290 ms) |
| 7     | ref_encoder + sampler fuse — partition tax (200 MB graph) |
| 8a    | aggressive `decoder_upsample → ALL` — bimodal 322–759 ms |
| 9     | `_hifigan_shift` fold — sub-1 ms saving, dominated by Trial 8 |

## Usage

Drop `packages/` into `models/tts/styletts2/coreml/` (or symlink) and
run `python -m coreml.inference` from the styletts2 root. The
`_STAGE_COMPUTE` and `_STAGE_PRECISION` manifests in
`coreml/inference.py` are wired to load these by default.

To compare against the legacy 9-package path:

```bash
python -m coreml.inference --no-fused
```
