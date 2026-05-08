# iter3-bench (Swift)

Self-contained Swift CLI that loads each iteration_3 `.mlmodelc` with
the placement recommended by `_STAGE_COMPUTE` in
`coreml/inference.py`, runs four warm predictions per stage on
synthesised inputs (shape resolved from each model's own description),
and reports load + warm latency.

## Build & run

```bash
# 1. Compile mlpackages (one-time)
DST=../compiled
SRC=../packages
mkdir -p "$DST"
for pkg in "$SRC"/*.mlpackage; do
  xcrun coremlcompiler compile "$pkg" "$DST"
done

# 2. Build & run the Swift bench
swift build -c release
.build/release/iter3-bench --compiled ../compiled
```

## Sample output (M-series Mac)

```
  [text_encoder              | CPU_ONLY   ] load=33ms    warm: min=1.1   avg=1.2   max=1.5   ms
  [bert                      | ALL        ] load=607ms   warm: min=6.6   avg=8.8   max=12.7  ms
  [ref_encoder               | CPU_AND_GPU] load=236ms   warm: min=11.1  avg=12.1  max=14.0  ms
  [fused_diffusion_sampler   | ALL        ] load=1394ms  warm: min=14.2  avg=16.9  max=23.9  ms
  [duration_predictor        | CPU_ONLY   ] load=123ms   warm: min=2.4   avg=2.5   max=2.7   ms
  [fused_f0n_har_source      | CPU_ONLY   ] load=189ms   warm: min=10.7  avg=10.9  max=11.2  ms
  [decoder_pre               | CPU_AND_NE ] load=1461ms  warm: min=3.8   avg=3.9   max=4.0   ms
  [decoder_upsample          | CPU_ONLY   ] load=1022ms  warm: min=278.0 avg=304.2 max=375.8 ms
```

Pipeline-stage sum ≈ 360 ms (synthetic inputs).

## Scope

`iter3-bench` is a **scaffolding** sanity check — it proves all 8
mlmodelc stages load and predict in Swift with the documented
placement, and gives a baseline for per-stage cost without leaving
Swift. It does **not** produce audio (synthetic random inputs).

## `iter3-tts` (side-loaded audio)

A second target wires the same 8 stages into a real-audio path by
side-loading the Python eager glue (phonemizer, ref-mel extraction,
alignment matmul + asr-shift, s/ref split) as on-disk fixtures.

```bash
# 1. Dump every stage's input + output as .npy fixtures
cd ..                                            # styletts2 root
uv run python iteration_3/swift/dump_intermediates.py \
    --text "StyleTTS 2 is a text to speech model." \
    --reference reference_audio/696_92939_000016_000006.wav

# 2. Run the Swift consumer
cd iteration_3/swift
swift build -c release
.build/release/iter3-tts \
    --compiled ../compiled \
    --fixtures fixtures \
    --output fixtures_swift.wav
```

Sample output (warm):

```
  [text_encoder              | CPU_ONLY   ] load=37ms  predict=3.0ms
  [bert                      | ALL        ] load=160ms predict=137.5ms
  [ref_encoder               | CPU_AND_GPU] load=39ms  predict=66.8ms
  [fused_diffusion_sampler   | ALL        ] load=71ms  predict=149.8ms
  [duration_predictor        | CPU_ONLY   ] load=16ms  predict=3.9ms
  [fused_f0n_har_source      | CPU_ONLY   ] load=20ms  predict=14.7ms
  [decoder_pre               | CPU_AND_NE ] load=41ms  predict=6.1ms
  [decoder_upsample          | CPU_ONLY   ] load=70ms  predict=289.4ms
Pipeline total: 1148ms
Wrote fixtures_swift.wav (3.67s @ 24000 Hz)
```

Parity vs `fixtures_python.wav`: cosine similarity 1.000000,
max|Δ| ≈ 3×10⁻⁵ (within int16 PCM quantization).

To extend to a fully-Swift end-to-end pipeline, port the eager glue
to Swift:

* phoneme tokenisation (espeak + TextCleaner)
* reference-audio mel extraction (for `ref_encoder.mel`)
* alignment matmul `d_en @ pred_aln_trg` and asr-shift to build the
  predictor input `en` for `fused_f0n_har_source`
* `s`/`ref` split off `fused_diffusion_sampler.var_6189`
