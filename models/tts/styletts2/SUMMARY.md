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

Warm avg = `iter3-bench` synthetic-input timed loop (Swift, 4 iters
per stage). End-to-end Python pipeline measures higher because of
phonemizer + alignment matmul + per-stage MLModel.predict overhead.

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

`iteration_3/swift/` ships two SwiftPM executables.

### iter3-bench (Swift, synthetic inputs)

Loads each `.mlmodelc` with the documented placement, runs warmup +
4 timed predicts on synthesised inputs (shape resolved from each
model's input description), reports load + min/avg/max latency. No
audio output. Used to baseline per-stage cost without leaving Swift.

### iter3-tts (Swift, real audio via side-loaded fixtures)

`dump_intermediates.py` monkey-patches `coreml.inference._load_stage`
+ `_predict` to dump every stage's inputs and outputs as `.npy` files
plus a `manifest.json` describing shape/dtype/order. Swift binary then
loads each fixture, runs predict on the corresponding `.mlmodelc`, and
writes a 24 kHz mono int16 WAV.

* NPY v1/v2/v3 reader (`<f4`, `<i4`, C-contiguous).
* float16 → float32 conversion at the audio output stage.
* RIFF/WAVE writer mirroring `inference.py`'s 50-sample tail trim.
* Parity vs `fixtures_python.wav`: cosine sim 1.000000, max|Δ| ≈ 3e-5.

This proves all 8 mlmodelc stages produce bit-equivalent audio to
Python when run from Swift. The eager glue (phonemizer + ref-mel +
alignment matmul + asr-shift + s/ref split) is still in Python.

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
│   ├── packages/                # 8 mlpackages (gitignored)
│   ├── compiled/                # 8 mlmodelc (gitignored)
│   ├── swift/                   # SwiftPM with iter3-bench + iter3-tts
│   └── README.md
└── SUMMARY.md                   # this file
```

## What's done

* [x] 9 → 8 stage fusion (Trial 4 sampler, Trial 6 f0n+har).
* [x] Per-stage compute placement matrix (Trial 8b winning combo).
* [x] Mixed-precision sweep (iteration_3): 7 fp16 / 1 fp32, A/B verified.
* [x] iteration_3 mlpackages + mlmodelc compiled.
* [x] Python end-to-end inference (`coreml/inference.py`).
* [x] Swift bench (`iter3-bench`) on synthetic inputs.
* [x] Swift side-loaded TTS (`iter3-tts`) writing bit-equivalent WAV.
* [x] HuggingFace upload of fp32 packages (iteration_2 lineage).

## What's outstanding

* [ ] Port eager glue to Swift to drop the side-load:
  * phonemizer (espeak + TextCleaner)
  * reference-audio mel extraction
  * alignment matmul `d_en @ pred_aln_trg` and asr-shift
  * `s/ref` split off `fused_diffusion_sampler` output
* [ ] Tackle `decoder_upsample` dominant cost (≈ 84 % of total):
  Trial 8 ALL placement is bimodal; explore ConvTranspose1d → ConvTranspose2d
  rewrite or weight palettization.
* [ ] int8 weight palettization on the surviving fp16 stages
  (deferred — fp16 already pays for itself; int8 needs A/B that hasn't been done).
* [ ] FluidAudio integration as a TTS backend.
* [ ] BERT + diffusion RangeDim on token axis (currently fixed at 57 tokens).

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
                                   ~360 ms total  (Swift bench warm)
                                   ~683 ms total  (Python end-to-end avg)
```
