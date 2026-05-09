# StyleTTS2 → CoreML — current status

End-to-end status of the StyleTTS2 LibriTTS → CoreML port. Detailed
per-iteration notes live in `iteration_2/README.md`,
`iteration_3/README.md`, and `coreml/fusions.md` / `coreml/trials.md`.

## Current production iteration: **iteration_3**

* 8 CoreML stages, 8 dispatches per utterance.
* 7 stages fp16, 1 stage fp32 (`fused_f0n_har_source` — har cumsum drift
  over 88 200 samples).
* 274 MB on disk (down from 514 MB iteration_2 fp32, −47 %).
* Pipeline-stage sum: **460 / 683 / 1110 ms** (cool min/avg/max,
  M-series Mac), down from 782 / 898 / 1075 ms in iteration_2.

### Per-stage manifest (Trials 4 + 6 + 8b placement, iteration_3 precision)

| Stage                       | Precision | Compute      | Warm avg   | Size    |
| --------------------------- | --------- | ------------ | ---------- | ------- |
| text_encoder                | fp16      | CPU_ONLY     |   1.2 ms   | 11.6 MB |
| bert                        | fp16      | ALL          |   8.8 ms   | 11.6 MB |
| ref_encoder                 | fp16      | CPU_AND_GPU  |  12.1 ms   | 52.9 MB |
| fused_diffusion_sampler     | fp16      | ALL          |  16.9 ms   | 47.4 MB |
| duration_predictor          | fp16      | CPU_ONLY     |   2.5 ms   | 14.9 MB |
| **fused_f0n_har_source**    | **fp32**  | CPU_ONLY     |  10.9 ms   | 32.1 MB |
| decoder_pre                 | fp16      | CPU_AND_NE   |   3.9 ms   | 64.1 MB |
| decoder_upsample            | fp16      | CPU_ONLY     | 304.2 ms   | 40.0 MB |
| **Total**                   |           |              | **≈ 360 ms** | **≈ 274.8 MB** |

Warm avg = sum of per-stage `mlmodel.predict` timings on synthetic
inputs (4 iters per stage, M-series Mac). End-to-end Python pipeline
measures higher because of phonemizer + alignment matmul + per-stage
MLModel.predict overhead.

### Notable decisions

* **fused_diffusion_sampler** (Trial 4) collapses the 5-step ADPM2 loop
  (8 diffusion_unet calls) into 1 dispatch. fp16 parity 4.66e-3.
* **fused_f0n_har_source** (Trial 6) collapses f0n_predictor + har_source
  into 1 dispatch. **Stays fp32**: fp16 cumsum drift over the 88 200
  audio-rate samples produces audible second-half phase distortion
  (verified by A/B listening — `sanity_fp16_plus_f0n.wav`).
* **decoder split** (`decoder_pre` + `decoder_upsample`): pre runs ANE
  (AdaIN + 1D conv), upsample stays CPU_ONLY because the HiFi-GAN
  ConvTranspose1d ups stack triggers MILCompilerForANE failures and
  ALL-placement is bimodal under contention (322–759 ms).
* **decoder_upsample** is the dominant cost at ≈ 84 % of pipeline.

## Swift status

The earlier prototype Swift driver (`iter3-bench` + `iter3-tts`) has
been removed from this repo. It only ran the eight neural stages and
side-loaded everything else (phonemizer, ref-mel, alignment matmul,
asr-shift, `s/ref` split, RNG-deterministic noise) from Python `.npy`
fixtures, so it was not a viable integration path on its own. The
production Swift consumer lives in **FluidAudio**, which will reuse
the existing G2P / mel / audio plumbing it already ships for Kokoro
and PocketTTS.

The CoreML packages produced by `coreml/inference.py` are the
deliverable. The canonical artefact set for downstream consumption is
on HuggingFace:

  https://huggingface.co/FluidInference/StyleTTS-2-coreml/tree/main/iteration_3

## Layout

```
models/tts/styletts2/
├── coreml/                      # converters, runtime helpers, fusion notes
│   ├── inference.py             # Python end-to-end pipeline (8 stages)
│   ├── convert.py               # mlpackage builder (per-stage / per-precision)
│   ├── fusions.md, trials.md    # design + benchmark history
│   └── packages/                # ALL precision/stage variants (gitignored)
├── iteration_2/                 # adopted Trials 4+6+8b at all-fp32 (514 MB)
├── iteration_3/                 # mixed precision (274 MB)
│   ├── packages/                # 8 mlpackages (gitignored, on HF)
│   ├── compiled/                # 8 mlmodelc (gitignored, on HF)
│   └── README.md
└── SUMMARY.md                   # this file
```

## What's done

* [x] 9 → 8 stage fusion (Trial 4 sampler, Trial 6 f0n+har).
* [x] Per-stage compute placement matrix (Trial 8b winning combo).
* [x] Mixed-precision sweep (iteration_3): 7 fp16 / 1 fp32, A/B verified.
* [x] iteration_3 mlpackages + mlmodelc compiled.
* [x] Python end-to-end inference (`coreml/inference.py`).
* [x] HuggingFace upload of iteration_3 packages
  ([link](https://huggingface.co/FluidInference/StyleTTS-2-coreml/tree/main/iteration_3)).

## What's outstanding

* [ ] FluidAudio Swift integration owns the eager glue
  (phonemizer / ref-mel / alignment matmul / asr-shift / `s/ref` split).
  Reuse Kokoro or PocketTTS G2P; do not ship espeak. Tracked downstream.
* [ ] Tackle `decoder_upsample` dominant cost (≈ 84 % of total):
  Trial 8 ALL placement is bimodal; explore ConvTranspose1d → ConvTranspose2d
  rewrite or weight palettization.
  * **Trial 10 (done)**: fp32 + fixed shapes (no RangeDim) probed for ANE
    acceptance. ANE *still* refused — `CPU_AND_NE` ran *slower* than
    `CPU_ONLY` (ANE-attempt-then-fallback signature). Fixed shape did
    stabilize `ALL` placement (303 ms spread vs Trial 8's 437 ms
    bimodality), but fp32 cost was ~5–13× over fp16. Verdict: shape was
    *not* the blocker — ConvTranspose1d itself is off-limits to ANE on
    this graph. Details in `coreml/trials.md`.
  * **Trial 10b (done)**: rewrote all 105 1D convs (101 Conv1d + 4
    ConvTranspose1d) to Conv2d analogs (`unsqueeze(H=1) → conv2d →
    squeeze`). Eager bit-equivalent. CoreML latency improved -27 to
    -45 % vs Trial 10 across all placements (fp32: CPU_ONLY 2937 ms,
    ALL 1111 ms). **ANE still partially rejects** — `CPU_AND_NE`
    remained slower than `CPU_ONLY`, so ConvTranspose2d at stride 10 /
    256→512 ch is also off-limits, OR a non-conv op in the subgraph is
    the structural blocker. Trial 10b at fp32 still 3.7× slower than
    fp16 baseline, so not promoted; rewrite *is* sound.
  * Next: Trial 10c (fp16 + Conv2d) — expected ~210 ms CPU_ONLY,
    finally beats baseline. Or Trial 10d (drill into MILCompilerForANE
    log to identify the structural blocker). Or accept and pursue
    Vocos/iSTFT vocoder swap.
* [ ] int8 linear weight quantization on the surviving fp16 stages
  (deferred — fp16 already pays for itself; int8 needs A/B that hasn't been done).
  int8 palettization is closed: every attempt failed on these graphs.
* [ ] FluidAudio integration as a TTS backend.
* RangeDim on the token axis for BERT + diffusion is **closed as dead end**
  (HF Albert + cross-attention force MLProgram into "data-dependent shapes
  were disabled"). Token-axis bucketing is the production answer
  (`bert_fp16_t{64,128,256}` + sampler buckets in `iteration_3`).

## Reference numbers

iteration_2 fp32 (Trials 4+6+8b adoption, all stages fp32):
```
text_encoder            CPU_ONLY      ~2  ms
bert                    ALL           ~9  ms
ref_encoder             CPU_AND_GPU  ~13  ms
fused_diffusion_sampler ALL          ~18  ms
duration_predictor      CPU_ONLY      ~3  ms
fused_f0n_har_source    CPU_ONLY     ~12  ms
decoder_pre             CPU_AND_NE   ~35  ms
decoder_upsample        CPU_ONLY    ~614 ms
                                    ~706 ms total
```

iteration_3 fp16/fp32 mix (current):
```
                                   ~360 ms total  (sum of per-stage warm avg)
                                   ~683 ms total  (Python end-to-end avg)
```
