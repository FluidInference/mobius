# Paradee-8M → Core ML

[Paradee-8M](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2) (Sahil Mahendrakar, Apache-2.0) is Kokoro-82M
distilled into ~8M parameters, 24 kHz English. Paper: [arXiv 2610.06817](https://arxiv.org/abs/2610.06817)
(describes v1.0). Two releases are converted here:

| Release | Voices | Decoder | Scripts |
|---|---|---|---|
| [v1.2](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2) | `af_heart`, `bf_isabella`, `am_puck`, `bm_fable` | v1.1's frame-rate source-filter head | `ref12.py`, `convert12.py`, `quantize12.py`, `evaluate12.py` |
| [v1.0](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0) | `af_heart` | iSTFTNet generator + phase-lock filter | `paradee_ref.py`, `convert.py`, `quantize.py`, `verify.py`, `evaluate.py` |

Published: [FluidInference/paradee-8m-coreml](https://huggingface.co/FluidInference/paradee-8m-coreml)
(`v1.2/int8`, `v1.2/fp32`; v1.0 in `int8/`, `fp32/`). Runtime: FluidAudio `ParadeeManager`.

## Layout

Both releases use the same two models, with duration expansion on the host. v1.2 adds a `voice` input to both.

```
phonemes (misaki; British misaki for bf_/bm_) -> ids [1,T] (0 pad at both ends, T <= 512)
ParadeeText      input_ids [1,T] int32, voice [1] int32 (v1.2)
                 -> duration [1,T] (pre-round), d [1,224,T], asr_tok [1,512,T]
host             n = max(1, round(duration / speed)); en = repeat(d, n); asr = repeat(asr_tok, n)    (F = sum n)
                 noise = N(0,1) [1,1,600F]
ParadeeAcoustic  en [1,224,F], asr [1,512,F], noise [1,1,600F], voice [1] int32 (v1.2) -> audio [1,600F] @ 24 kHz
```

- `T` is a RangeDim 3…512; `F` is a RangeDim 1…4000 (100 s at speed 1).
- `voice` is the row of each half's style table (text side [4, 32], decoder [4, 16]), in `config.json` order:
  0 `af_heart`, 1 `bf_isabella`, 2 `am_puck`, 3 `bm_fable`.
- The noise is an input (upstream draws it in-graph), so both graphs are deterministic and testable against PyTorch.
- Compute units: `CPU_ONLY` or `CPU_AND_NE`. **`ALL` / `CPU_AND_GPU` abort** in MPSGraph
  (`GPURNNOps.mm:1565: failed assertion 'JIT not supported'`) on the LSTMs, the same assert Kokoro's prosody stage hits.

| Build | Text | Acoustic | Total |
|---|---:|---:|---:|
| v1.2 int8 (DFT bases and warp matrices kept fp32) | 3.1 MB | 8.0 MB | 11 MB |
| v1.2 fp32 | 12 MB | 22 MB | 34 MB |
| v1.0 int8 (DFT bases kept fp32) | 3.1 MB | 9.1 MB | 12 MB |
| v1.0 fp32 | 12 MB | 22 MB | 34 MB |

## Run

```bash
uv sync && uv pip install --reinstall coremltools==9.0   # the macOS platform wheel
git clone https://github.com/sahilmahendrakar/paradee upstream && git -C upstream checkout aae3cc4

# v1.2
uv run hf download sahilmahendrakar/Paradee-8M-v1.2 --local-dir weights-v1.2
uv run python ref12.py              # PyTorch reference vs the released fp32 ONNX (noise as an input), every voice
uv run python convert12.py          # build-v1.2/ + ref.npz
uv run python quantize12.py         # build-v1.2-int8/
uv run python evaluate12.py         # [--dir build-v1.2-int8 --wav-dir out/wav-v1.2-int8] [--cu CPU_AND_NE]
uv run python asr_check.py out/wav-v1.2 coreml,onnx,onnx8

# British lexicon for FluidAudio's English frontend (gb_lexicon_cache.json)
uv run hf download FluidInference/kokoro-82m-coreml us_gold.json us_silver.json gb_gold.json gb_silver.json \
    us_lexicon_cache.json --local-dir lexicon
uv run python build_lexicon_cache.py lexicon --lang gb

# v1.0
uv run hf download sahilmahendrakar/Paradee-8M-v1.0 --local-dir weights
uv run python convert.py            # build/ + build/ref.npz
uv run python quantize.py           # build-int8/
uv run python verify.py acoustic CPU_ONLY
uv run python evaluate.py           # [--dir build-int8] [--cu CPU_AND_NE]; writes out/wav*/
uv run python asr_check.py out/wav coreml,onnx
```

Compile for shipping with `xcrun coremlcompiler compile <name>.mlpackage <dir>`.

## v1.2

### What changed upstream

v1.1 replaced the end of v1.0's decoder (Kokoro's upsampling iSTFTNet generator plus a phase-lock filter) with a
head that runs at the trunk's 80 frames a second: ConvNeXt blocks predict, per STFT bin (n_fft 600, hop 150), a gain
and phase for a clean 64-harmonic sine bank built from F0, gains for white noise and for noise pulsing with the
pitch, a per-frame pulse shift, and a short-window (150/75) burst-noise path; one iSTFT makes the audio. v1.2 is the
same graph with a 4-row style table in each half.

### Conversion

- **Voice tables.** `style` is `[4, 32]` (text side) and `[4, 16]` (decoder); `index_select` on the `voice` input.
  `ref12.py` loads them with `strict=False` and asserts that `style` is the only key it fills itself.
- **STFT / iSTFT without complex ops or strided convolutions.** Frames come from reshaping the reflect-padded
  signal into hop-sized blocks and concatenating 4 shifted slices (n_fft = 4 hop; the burst path uses 2), then one
  GEMM with the `[n, 2K]` DFT basis. The inverse is a GEMM followed by a shifted overlap-add, divided by the
  overlap-added squared window. No gathers.
- **Harmonic phase in int32.** Upstream accumulates cycles per frame in float64. Core ML has no float64, and an fp32
  running sum reaches ~4e3 cycles on a 28 s input, where one ulp (5e-4 cycles) is 0.03 cycles on harmonic 64. The
  per-frame advance, wrapped to [0, 1), is rounded to 1/2^17 cycle and summed in int32, which is exact for the
  8000-frame maximum (8000 · 2^17 < 2^31), then wrapped.
- **Closed-form sine bank.** `sum_{k<=M} sin(k t) = sin(M t/2) sin((M+1) t/2) / sin(t/2)`, with `M` the frame's
  harmonic count below 11 kHz, instead of a `[64, samples]` tensor (615 MB at 100 s). Where `|sin(t/2)| < 1e-4` it
  uses the first-order limit `M(M+1)/2 · sin(t)`.
- The per-sample phase (`vhead.sine_bank` interpolates between frame centres with clamped gathers) is built as two
  `[frames, 150]` halves from the current and previous frame, so it needs no index ops.
- int8: weight-only per-channel linear quantization of every weight ≥ 1024 elements except the head's DFT bases and
  its two frequency-warp matrices (96 KB together), which stay fp32 (`quantize12.py` finds them by shape).

### Results (M5 Pro, macOS 27)

40 cases: 10 inputs (the README line, upstream's 5 sample sentences, a numbers/abbreviation line, a 28 s
paragraph, one sentence at speed 0.8 and 1.3) × 4 voices, misaki phonemes (British for `bf_`/`bm_`).

- `ref12.py`: the PyTorch reference equals the released fp32 ONNX with the same noise, 72–88 dB SNR, same lengths,
  for every voice.
- fp32 Core ML vs PyTorch, same noise: **0 duration rounding differences** in all 40 cases; waveform SNR 34–65 dB
  (lowest on the 28 s paragraph); log-mel L1 against the released fp32 ONNX ≤ 0.012.
- Whisper large-v3-turbo over the 40 clips: Core ML fp32 **3.39 %** = ONNX fp32 3.39 %; Core ML int8 **3.28 %** =
  upstream `paradee_int8.onnx` 3.28 %. Per voice, every build is within one word of the others.
- int8 rounds 0–7 durations per input differently from fp32 (as v1.0's int8 does); where every duration matches,
  log-mel L1 to fp32 is 0.11–0.22.
- Speed, text + host + acoustic, `CPU_ONLY`, 320 s of audio: **RTFx 207 (fp32), 202 (int8)**, against 150 for v1.0.

### British phonemes

The British voices were trained on misaki's British G2P (`en.G2P(british=True)`). For FluidAudio,
`build_lexicon_cache.py` turns misaki's `gb_gold.json` + `gb_silver.json` into `gb_lexicon_cache.json`, in the format
of the `us_lexicon_cache.json` FluidAudio already ships (silver then gold, heteronyms take `DEFAULT`, keys
lower-cased last-wins plus a case-sensitive map). The script first rebuilds the US cache from `us_gold`/`us_silver`
and checks it equals the shipped file.

Words missing from the lexicon go through FluidAudio's BART G2P, which only produces American phonemes. Instead of
misaki's eSpeak `en-gb` fallback, FluidAudio rewrites them: US-only symbols (`O`→`Q`, `ᵻ`→`ɪ`, flaps → `t`),
non-rhotic `ɹ` before a non-vowel, `æ`→`a`, and `ɑ`→`ɒ` unless long. Measured on the 153,873 words in both misaki
lexicons, this takes the mean edit distance to the British entry from 1.59 to 0.78 phonemes.

### Trials, errors, lessons (v1.2)

1. `clamp_min(60)` with an int bound fails in the torch frontend (`assert x.dtype == y.dtype` in `clamp_min`); use
   float bounds everywhere.
2. Reading a size from a traced tensor (`fr.shape[1] // hop`) turns it into graph ops, and the `aten::Int` that
   feeds `F.pad` then fails with `only 0-dimensional arrays can be converted to Python scalars`. Pass static sizes
   as Python ints.
3. The released int8 ONNX cannot be compared sample by sample: its rounded durations differ from fp32, so the noise
   length differs. Compare transcripts instead.

## v1.0

Sources: HF weights `f662642d` (`pytorch/text_side.pt`, `pytorch/decoder.pt`).

- The source noise is an input. One channel is enough: kokoro adds noise to each of the 9 harmonics and then mixes
  them with `l_linear`, so `W @ (a * n) == a * ||W|| * z` with `z ~ N(0,1)` (checked: wrapper vs upstream torch
  5.9e-7).
- kokoro's random initial harmonic phases are dropped. They only touch sample 0, and the 1/300 linear downsample in
  `SineGen._f02sine` never reads it, so they are a no-op upstream too.

### Results (M5 Pro, macOS 27)

10 cases: the README quick-start line, the model card's 5 held-out sentences, the Space's "Zzyzx" line, a 490-token
paragraph (28.6 s), and sentence 2 at speed 0.8 and 1.3.

- Durations: 0 rounding mismatches in all 10 cases; lengths identical.
- Waveform SNR vs torch: 45.6 dB (3 s) down to 19.7 dB (28.6 s). The drop is harmonic-source phase: `cumsum` of
  frame phase in fp32 times 300 reaches ~4e4 rad, and Core ML and torch accumulate in a different order. It is the
  same size as torch's own fp32 error, and it is phase only:
- Log-mel L1 Core ML vs ONNX = ONNX vs ONNX (two runs, different noise) in every case (0.123 vs 0.123 … 0.145 vs 0.144).
- Whisper large-v3-turbo: Core ML and ONNX transcripts are identical, WER 3.81 % each. All misses are text
  normalization or G2P ("Ia", "favourite", "Zzyzx", "Paradee").
- Speed, total 93.2 s of audio, text + host + acoustic: **RTFx 150 (CPU_ONLY), 156 (CPU_AND_NE)** vs upstream ONNX
  fp32 on one onnxruntime thread 20.4.
- int8 (`build-int8`): log-mel L1 vs fp32 ONNX 0.18–0.20 where the durations still match, against 0.21–0.31 for the
  upstream `paradee_int8.onnx`; whisper WER 3.81 % vs upstream int8 2.97 % (one word on ~236). RTFx 133–143.

### Trials, errors, lessons (v1.0)

1. **`torch.remainder(x, 1.0)` converts to all zeros** (coremltools 9.0, fp32). The harmonic phase was 0 everywhere,
   audio corr 0.16. `x - torch.floor(x)` is exact. Found by exposing F0 / N / phase / source / STFT as outputs
   (`debug_acoustic.py`): F0 and N matched to 1e-4, phase was 0.
2. `AdainResBlk1d.forward` multiplies by `torch.rsqrt(torch.tensor(2))`, an int tensor that Core ML's rsqrt rejects.
   Patched to a float constant (`convert.py`; `convert12.py` imports the patch).
3. **The phase-lock filter's 1024-tap strided STFT convs ran 740 ms on CPU_ONLY** for 3 s of audio (and the whole
   model 4.4 s on ALL), while the rest of the graph took 17.6 ms (ablation in `exp_ablate.py`). Rewriting the
   band STFT as one gather into frames + a [1024, 512] GEMM, and the iSTFT as GEMM + a 4-way shifted overlap-add
   (`N_FFT = 4 * HOP`), took it to 19.8 ms with the same output. The generator's own 20-point STFT stays a conv.
4. Gather indices from a dynamic `arange` come out fp32 in MIL; `.long()` them explicitly.
5. **Quantizing the DFT bases ruins the filter.** A plain int8 pass gave log-mel 0.22 (floor 0.12) and SNR 5 dB.
   The three `linear` ops whose weights have a 1024 dim are excluded by name in `quantize.py`; upstream likewise
   regenerates its bases at load time instead of storing them quantized.
6. Static-shape conversion of the acoustic model, or of the prosody predictor alone, fails in the torch frontend
   (`_cast ... only 0-dimensional arrays can be converted`); the RangeDim conversion of the full graph works.
7. Not tried: fp16 / ANE. At 150× real time on CPU there is little to gain, and Kokoro's fp16 prosody on CPU/ANE
   corrupted F0 at the start of longer utterances.
8. **misaki's output form matters.** A frontend that reads the misaki lexicon raw gets flap `ɾ` and glottal stop
   `ʔ`; misaki's `G2P.__call__` rewrites them to `T` / `t` for Kokoro v1.0, and Paradee never saw `ɾ`/`ʔ` in
   training ("kittens" → "kins", "satellite … patterns" garbled without the rewrite).
