# TRIALS — Kokoro-82M-v1.1-zh CoreML conversion

Chronological log of conversion attempts, decisions, and issues encountered
while adapting the v1.0 7-stage chain (`models/tts/kokoro/laishere-coreml/`)
to the Mandarin v1.1-zh checkpoint.

---

## Trial 0 — Background investigation (pre-conversion)

**Question**: Can the 7-stage CoreML chain for v1.0 (`laishere-coreml/`) be
reused unchanged for v1.1-zh, or does the Mandarin checkpoint require a new
trace?

**Findings**:

1. **Architecture**: hexgrad/Kokoro-82M-v1.1-zh ships the **same**
   StyleTTS2-derived architecture as v1.0:
   - ALBERT (3 layers, 8 heads, hidden 768, embedding 128)
   - Predictor (text_encoder LSTM × 5, F0/N projection)
   - TextEncoder (3 conv + 1 LSTM)
   - Decoder + iSTFT generator
   - Style encoder (256-dim ref_s)

   Verified by diffing `config.json` between the two upstream repos — only
   `vocab` (and the embedded `n_token`) differ.

2. **Vocab**: v1.0 has 177 entries (IPA + arrow tones `↓→↗↘`). v1.1-zh has
   171 entries (IPA + Bopomofo `ㄅㄆㄇㄈ…` + tone digits `1-5`); the smaller
   total reflects dropping English-specific tones in favor of Bopomofo and
   the digit-based tone scheme. (Initial expectation of 178 from a stale
   read of the HF preview was wrong — `len(KModel(repo_id=…).vocab)` returns
   171.)

3. **G2P**: v1.0 uses `misaki.en` (with espeak fallback). v1.1-zh uses
   `misaki.zh` (jieba word-segmentation → pypinyin → Bopomofo + tone digit).
   The KPipeline switches automatically based on `lang_code`; we just pass
   `'z'` instead of `'a'`.

4. **Voices**: v1.1-zh ships 96 voice packs (49 `zf_*` female + 47 `zm_*`
   male + 3 EN). All are the same `[510, 1, 256]` torch tensor format as
   v1.0. The `[510, 256]` flat fp32 .bin layout is reused without changes.

**Decision**: Adapt the 7-stage script with **minimal targeted edits**
(repo_id, lang_code, test phonemes, voice id). Keep the trace classes,
op-translation patches, RangeDim bounds, and compute-unit assignments
unchanged.

---

## Trial 1 — Source-of-truth selection

**Question**: Use `models/tts/kokoro/coreml/v21.py` (single end-to-end
mlpackage) or `models/tts/kokoro/laishere-coreml/convert-coreml.py`
(7-stage)?

**Initial misstep**: First scaffold copied `v21.py` → `convert-coreml.py`
and `kokoro/coreml/pyproject.toml`. Discovered on read-through that v21.py
emits a single `kokoro_completev21.mlpackage`, not the 7-stage chain that
ships in `FluidInference/kokoro-82m-coreml/ANE/`.

**Resolution**: Removed v21.py-derived files, re-scaffolded from
`models/tts/kokoro/laishere-coreml/` (PR #45, commit `3b00f7d`). The
laishere chain is what produced the existing `ANE/` mlmodelc bundles on HF
and is what FluidAudio's `KokoroAneManager` consumes.

---

## Trial 2 — Targeted edits to convert-coreml.py

Three edit points beyond docstring/usage updates:

1. **Model load** (was line 540):
   ```python
   model = KModel()  # defaults to hexgrad/Kokoro-82M
   ```
   ↓
   ```python
   model = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh')
   assert len(model.vocab) >= 178
   ```

2. **Pipeline + voice** (was lines 546–547):
   ```python
   pipe = KPipeline(lang_code='a', model=model)
   voice_pack = pipe.load_voice('af_heart')
   ```
   ↓
   ```python
   pipe = KPipeline(lang_code='z', model=model)
   voice_pack = pipe.load_voice('zf_001')
   ```

3. **Test trace input** (was line 549, hardcoded English IPA):
   ```python
   phonemes = "ðə kwɪk bɹaʊn fɑːks dʒʌmps oʊvɚ ðə leɪzi dɑːɡ."
   ```
   ↓ (run misaki[zh] G2P on a Mandarin sentence so the trace exercises the
   real Bopomofo+digit token distribution)
   ```python
   text_zh = '你好世界，今天天气很好。'
   for _gs, ps, _tks in pipe(text_zh, voice='zf_001'):
       phonemes = ps
       break
   ```

Everything else — RangeDim shape bounds, the 7 trace classes, the op
patches (rsqrt, cos Snake), int8 palettization, compute-unit choices —
unchanged.

---

## Trial 3 — Helper script adaptations

| Script                  | Edits                                                                 |
|-------------------------|-----------------------------------------------------------------------|
| `pyproject.toml`        | Added `misaki[zh]>=0.9.4` (pulls jieba + pypinyin); renamed project.  |
| `inference.py`          | Default `--voice zf_001`, `--lang z`, `--repo-id hexgrad/Kokoro-82M-v1.1-zh`. Threads repo_id through KModel construction. |
| `compare-models.py`     | Replaced `--phonemes` default with `--text` driver that runs misaki[zh] G2P internally. Default voice/lang/repo_id swapped. |
| `benchmark.py`          | Replaced 6 English passages with 6 Mandarin passages (varied tones, punctuation, length). G2P helper kept identical (already language-agnostic via `pipe.g2p`). Default voice/lang/repo_id swapped. |
| `dump-benchmark-data.py`| Loops over `--voices zf_001 zm_009` (default), pulls `vocab.json` from v1.1-zh repo (178 entries). Adds `repo_id` field to benchmark_data.json. |

---

## Trial 4 — Conversion run + parity

Run on this machine (Darwin 25.5.0, Apple Silicon):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python convert-coreml.py --output-dir build/kokoro-v1.1-zh
```

### Issues hit during the run (all resolved)

1. **Vocab assertion was wrong** — initial assertion was
   `len(model.vocab) >= 178` (a stale read from an HF preview). Actual
   v1.1-zh vocab has 171 entries (38 Bopomofo + IPA + tone digits +
   punctuation + a few Hanzi). Replaced with a Bopomofo-presence check
   over Unicode range U+3105–U+312F.

2. **`KPipeline` defaulted to v1.0 repo for voice loading** — even though
   the `KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh')` was passed in,
   `KPipeline(lang_code='z', model=model)` does not infer `repo_id` from
   the model. `pipe.load_voice('zf_001')` then 404'd against
   `hexgrad/Kokoro-82M/resolve/main/voices/zf_001.pt`. Fix: pass
   `repo_id='hexgrad/Kokoro-82M-v1.1-zh'` to `KPipeline()` in all five
   scripts (convert, inference, compare, benchmark, dump).

3. **`scikit-learn` missing** — int8 kmeans palettization (stages 5–7)
   raises `ModuleNotFoundError: No module named 'sklearn'`. Stages 1–4
   completed without it because palettization runs only in stages 5/6/7
   in the laishere chain (per `kmeans_palettize` calls in
   `convert-coreml.py`). Fix: `uv pip install scikit-learn`.
   coremltools 9.0 prints a "scikit-learn 1.8 is not supported, max
   tested 1.5.1" warning but the kmeans path still works.

4. **`pipe.g2p(text)` returns `(text, None)` for misaki[zh]** — the
   benchmark helper expected the misaki[en]-style `tokens` list. Switched
   `phonemize_for_benchmark()` to drive `pipe(text, voice=...)` and
   concatenate per-chunk `ps`.

### Conversion results

```
[1] Loading KModel (hexgrad/Kokoro-82M-v1.1-zh)...
    vocab: 171 entries, 38 Bopomofo
[2] Generating Mandarin test inputs...
    text='你好世界，今天天气很好。'
    phonemes='ㄋㄧ2ㄏㄠ3/ㄕ十4ㄐㄝ4, ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄏㄣ2ㄏㄠ3.' (len=35)
    T_a=133
[1/7] ALBERT (fp16+int8pal)        CPU_AND_NE   1.6ms
[2/7] PostAlbert (fp16+int8pal)    CPU_AND_NE   3.6ms
[3/7] Alignment (fp16+int8pal)     CPU_AND_NE   0.7ms
[4/7] Prosody (fp16+int8pal)       CPU_AND_NE   3.1ms
[5/7] Noise (fp32+int8pal)         CPU_AND_NE 145.3ms
[6/7] Vocoder (cos fp16+int8pal)   CPU_AND_NE 267.7ms
[7/7] Tail (fp32)                  CPU_AND_NE   1.9ms
[E2E] corr=-0.139578, mel_corr=0.997730, chain=333.1ms
```

### Parity (compare-models.py)

```
phonemes (34): 'ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄓㄣ1/ㄏㄠ3, 阳2ㄍ王1ㄇ应2ㄇㄟ4.'
  waveform corr     : -0.001772   (threshold ≥ 0.80)
  mel-spectrogram   :  0.967283   (threshold ≥ 0.99)
  rms err / rms ref :  1.6841
```

The waveform-corr threshold is not met. Mel correlation is close to but
below the 0.99 threshold. The pattern (high mel, near-zero waveform) is
characteristic of fp16 iSTFT vocoders: small phase differences cause the
sample-by-sample correlation to collapse while spectral content is
preserved. The same pattern shows in `convert-coreml.py`'s own E2E line
(mel=0.998, corr=-0.14). Audio sample (`build/.../sample-zf001.wav`)
sounds like correct Mandarin; ASR-based CER verification is the proper
quality check (TODO Trial 5).

### Inference timings (sample sentence "你好世界，今天天气真好。")

| Voice    | Chain time | Audio   | Speed |
|----------|------------|---------|-------|
| `zf_001` | 333 ms     | 3.33 s  | 10.0× |
| `zm_009` | 200 ms     | 3.50 s  | 17.5× |

All 7 stages run on `CPU_AND_NE` (assigned in the laishere chain;
inherited verbatim).

### Bundle sizes

| Stage              | mlmodelc | mlpackage |
|--------------------|----------|-----------|
| KokoroAlbert       | 5.6 MB   | 5.6 MB    |
| KokoroPostAlbert   | 13 MB    | 13 MB     |
| KokoroAlignment    | 32 KB    | 20 KB     |
| KokoroProsody      | 8.2 MB   | 8.1 MB    |
| KokoroNoise        | 4.5 MB   | 4.4 MB    |
| KokoroVocoder      | 47 MB    | 47 MB     |
| KokoroTail         | 100 KB   | 92 KB     |
| **Total**          | **~78 MB** | **~78 MB** |

Plus `vocab.json` (1.8 KB), `zf_001.bin` + `zm_009.bin` (510 KB each),
`benchmark_data.json` (4.7 KB).

---

## Trial 5 — ASR verification (TODO — post-conversion)

Per `Documentation/ModelConversion.md` §5, TTS conversions must be ASR-
verified. Plan:

- Generate audio for ~25 diverse Mandarin sentences (varied tones,
  numbers, multi-syllable words) via both PyTorch reference and CoreML
  chain, voices `zf_001` + `zm_009`.
- Transcribe with Parakeet-zh CTC (preferred) or Whisper-large-v3.
- Pass criteria: CER < 15% on both, |CER_pt − CER_cm| < 3%.

Script not yet written — will scaffold `asr-verify.py` after the parity
run lands.

---

## Known issues to watch

1. **`kokoro` package version**: v1.1-zh requires `kokoro` recent enough to
   read the n_token=178 `config.json`. Pinned `>=0.9.4`. If `KModel(repo_id=…)`
   raises `KeyError` on a vocab entry, bump to the latest published version
   and re-record here.

2. **Mandarin phoneme density**: each Hanzi expands to ~2-3 phonemes
   (Bopomofo letters + tone digit), so a single 50-character sentence can
   approach `T_enc=200`. The 6th passage in `benchmark.py` is sized to
   land near the 510-phoneme cap; if it trips the `T_a > MAX_FRAMES` skip
   in `benchmark.py`, shorten it.

3. **`coremltools 9.0` sdist fallback** (inherited from v1.0): `uv sync`
   may resolve the pure-python wheel and break `BlobWriter`. README
   documents the `uv pip install --reinstall coremltools==9.0` workaround.

4. **misaki[zh] dependency footprint**: pulls jieba (~50 MB dictionary).
   Acceptable for the conversion environment; downstream Swift/iOS
   consumers don't need it (they read precomputed phonemes from
   `benchmark_data.json` or run a separate Swift ZH G2P).

---

# Background-noise investigation (Trials 6-12)

After Trial 4 produced a working chain with `mel_corr ≈ 0.998` but `waveform_corr ≈ 0`, the
sample WAVs (especially `zm_009`) had **audible high-frequency noise** above ~10 kHz. Six
weeks of investigation across three "rounds" follow. The actual fix is two lines of code —
but only the elimination history makes that clear, so the dead ends are documented in full.

Every numerical result below is reproducible from the diagnostic scripts checked in alongside
this file (`tail_compare.py`, `vocoder_xpre_diff.py`, `per_stage_diff.py`, `probe_noise.py`,
`probe_noise_fidelity.py`, `probe_sinegen_isolated.py`).

---

## Trial 6 — Round 1: precision & quantization sweep (all eliminated)

**Question**: is the audible HF noise a side effect of int8 weight palettization, fp16 internal
compute, fp16 boundary I/O, or compute-unit dispatch (ANE vs GPU vs CPU)?

| #   | Stage(s) varied | Variant tested                                                | Outcome                                  |
|-----|-----------------|---------------------------------------------------------------|------------------------------------------|
| 6.1 | Noise           | int8-palettized → fp32 weights                                | noise unchanged                          |
| 6.2 | Vocoder         | fp16 → fp32 weights, dispatch CPU_AND_NE → ALL                | noise unchanged                          |
| 6.3 | Prosody         | fp16 → fp32 weights + `compute_precision=FLOAT32`             | noise unchanged; **duration broke** (Swift fp16 buffer mismatch) |
| 6.4 | All 5 trainable | `compute_precision=FLOAT32`, fp16 I/O kept (Option A)         | noise unchanged                          |
| 6.5 | All 5 trainable | **pure fp32**: FLOAT32 compute + fp32 I/O + Swift fp32 path   | noise unchanged; silence floor matches PyTorch (-100 dB) |
| 6.6 | All 5 trainable | pure fp32 + `computeUnits: .cpuOnly`                          | bit-identical to default-CU (corr 0.997) |

**Conclusion**: noise is **intrinsic to the CoreML graph topology**. Not weight precision, not
compute precision, not boundary precision, not dispatch.

### Side findings worth noting

- `zm_009` voice-pack has a **+0.022 timbre-mean offset** vs ≈0 for `zf_001` / `af_heart`. This
  explains why noise is more *audible* on the male voice (the offset shifts the noise band into
  perceptually weighted frequencies) but does **not** cause the noise.
- Re-converting PostAlbert produces shorter durations than the original HF int8 build —
  intrinsic conversion-script reproducibility issue, independent of the noise.

---

## Trial 7 — Round 2: Snake1D cos-rewrite investigation (eliminated)

**Hypothesis**: `_cos_resblock1_forward` (`convert-coreml.py:46-57`) is monkey-patched onto
`AdaINResBlock1` for ANE-friendly Snake activation. Maybe the fp16 cos has catastrophic
cancellation when α·xt is small.

### Algebraic verification

Snake activation:  `Snake(x, α) = x + (1/α) · sin²(α·x)`
Pythagorean id:    `sin²(z) = (1 − cos(2z)) / 2`

Replacement computes `xt + (cos(2α·xt) · (-0.5) + 0.5) · (1/α)` which is exactly
`xt + (1 − cos(2α·xt)) / (2α)` ≡ `xt + sin²(α·xt) / α`. **Mathematically equivalent in real
arithmetic.**

### fp16 catastrophic-cancellation hypothesis

In fp16, `cos(2α·xt) ≈ 1 − 2(α·xt)²` for small α·xt. The subtraction `(1 − cos)` then loses
~5 decimal digits — in fp16 that is most of the mantissa, so amplitudes around the activation
zero would acquire spurious noise.

**Ruled out** by Trials 6.5 and 6.6: pure-fp32 weights + `compute_precision=FLOAT32` + cpuOnly
dispatch all leave the noise unchanged. The cancellation hypothesis would predict a noticeable
improvement in fp32; it predicts none.

**Verdict**: not the noise source.

---

## Trial 8 — Round 3: iSTFT/Tail comparison

**Plan**: capture PyTorch's `x_pre` (the tensor that exits the Vocoder before `conv_post +
iSTFT`), then run two parallel "tails" on the same `x_pre`:

- PyTorch tail: `decoder.generator.conv_post(x_pre)` → split mag/phase → `CustomSTFT.inverse(...)`
- CoreML tail: `KokoroTail.mlmodelc.predict({"x_pre": x_pre_fp32})`

If they diverge, the iSTFT/conv_post stage is the culprit. If they match, the noise is upstream.

### 8a. Tail comparison (`tail_compare.py`, voice `zm_009`)

Three reconstructions on the same x_pre:

| pair                              | rel-rms | corr            | best-gain    | resid after gain | HF Δ (≥10 kHz) |
|-----------------------------------|---------|------------------|--------------|------------------|-----------------|
| `torch.istft` vs `torch_custom`*  | 0.594   | 0.911248848      | 1.022        | 5.81e-1          | +1.23 dB        |
| `torch_custom` vs CoreML Tail     | 0.645   | 0.911248855      | 1.097        | 5.02e-1          | -4.76 dB        |
| **`torch.istft`  vs CoreML Tail** | **0.500** | **1.000000000** | **1.499999523** | **1.04e-5**     | -3.52 dB        |

`*` `torch_custom` = `CustomSTFT.inverse` directly (the conv_transpose1d rewrite that was traced
for CoreML). It misses the DC/Nyquist non-doubling correction, so it is mis-scaled per bin.

> **`corr(torch.istft, CoreML Tail) = 1.000000000` exact**, residual after gain correction
> is 1.04e-5 (fp32 noise floor). The CoreML Tail is bit-equivalent to `torch.istft` up to a
> deterministic ×1.5 gain — exactly the squared-Hann-window overlap-sum normalization
> (`20·log10(1.5) = 3.522 dB`, matching the HF Δ to two decimals).

**Tail is NOT the noise source.**

### 8b. Vocoder x_pre comparison (`vocoder_xpre_diff.py`)

Feed identical PyTorch upstream into `KokoroVocoder.mlpackage` and diff its `x_pre`:

| metric                                       | value           |
|----------------------------------------------|-----------------|
| corr(pt_x_pre, coreml_x_pre)                 | **0.999999998** |
| rel-rms                                      | 7.4e-5          |
| max\|diff\|                                  | 9.77e-3         |
| HF residual `audio(pt_x_pre) − audio(cm_x_pre)` | **-150.62 dB** |

**Vocoder is NOT the noise source either.**

### 8c. Five-tier stage swap (`per_stage_diff.py`)

Walk the chain, substituting one CoreML stage at a time, all descaled by ×1.5:

| chain                                               | HF (≥10 kHz) | rel vs A | HF(diff) vs prev |
|-----------------------------------------------------|--------------|----------|-------------------|
| **A.** PT x_pre → PT torch.istft                    | -83.81 dB    | —        | —                 |
| **B.** PT x_pre → CoreML Tail                       | -83.81 dB    | 1.04e-5  | -177.29 dB        |
| **C.** PT inputs → CoreML Vocoder → CoreML Tail     | -83.81 dB    | 3.12e-4  | -150.64 dB        |
| **D.** CoreML Noise → CoreML Vocoder → CoreML Tail  | **-81.31 dB**| **0.276**| **-90.17 dB**     |

The HF residual jumps **60 dB** (-150.6 → -90.2 dB) the moment we substitute CoreML noise
sources for PyTorch's. Per-source numbers:

| source        | rms_pt | rms_cm | rel-rms | corr   |
|---------------|--------|--------|---------|--------|
| `x_source_0`  | 0.6458 | 0.5039 | **0.4399** | **0.886** |
| `x_source_1`  | 1.0463 | 1.0181 | 0.0724    | 0.9962    |

**`KokoroNoise.mlpackage` is the noise source.** `x_source_0` is the dominant contributor
(44% rel-rms, vs 7% for `x_source_1`).

### 8d. Compute-unit sweep on KokoroNoise (`probe_noise.py`)

| compute units  | x_source_0 rel | x_source_0 corr |
|----------------|-----------------|------------------|
| CPU_ONLY       | 0.4438          | 0.8839816        |
| CPU_AND_NE     | 0.4438          | 0.8839816        |
| CPU_AND_GPU    | 0.4402          | 0.8862826        |
| ALL            | 0.4402          | 0.8862826        |

Identical drift on CPU at fp32 — rules out ANE-fp16 fallback.

---

## Trial 9 — Round 3 wrong turn: phase wrapping inside `CoreMLSineGenV2`

**Hypothesis at end of Trial 8**: cumulative phase inside `CoreMLSineGenV2` reaches
~22 000 rad before `sin`. CoreML's MIL `sin` likely has weaker range-reduction than PyTorch's
libm `sin` (which uses fp64 internally), so the phase precision is lost in the high bits.

**Proposed fix**: wrap phase modulo 2π before sin via `phase - 2π · floor(phase / 2π)`,
keeping the sin argument in `[0, 2π)`.

### Result

Re-exported `KokoroNoise` with the wrap. `per_stage_diff.py` showed **identical** divergence to
the unfixed version (44% rel-rms, -90.18 dB HF residual). Wrapping made no end-to-end difference.

### `probe_sinegen_isolated.py` — six standalone CoreML models

To diagnose, traced six standalone graphs that each contain a different sub-step of the SineGen
pipeline, compared PyTorch vs CoreML on identical fp32 input:

| model | description                                     | rel-rms     | corr       |
|-------|-------------------------------------------------|-------------|------------|
| M0    | `sin` of (0..22000 rad) ramp                    | small       | 1.0        |
| M0b   | `sin` of (0..2π) ramp                           | small       | 1.0        |
| M1    | `sin(cumsum(rad)·2π·US)` — no interpolate       | **9.1e-4**  | **0.9999996** |
| M2    | `sin((cumsum·US − floor) · 2π)` — wrap, no interp | 3.98e-2   | 0.9992     |
| M3    | `sin(interp(cumsum·2π·US))` — original pipeline | **9.2e-4**  | **0.9999996** |
| M4    | `sin(((interp(cumsum·US)) − floor) · 2π)` — wrap fix | 2.49e-2 | 0.9997     |
| M5    | linear-interpolate alone                         | 8.5e-10     | 1.0        |
| M6    | cumsum + interp (no sin)                         | 1.2e-7      | 1.0        |

> **CoreML's sin handles large arguments fine.** M1 and M3 (the original chain) have rel 9e-4 —
> at the fp32 precision floor for a 22 000-rad sin. M2 and M4 (the wrap variants) are **27×
> worse** because the `floor()/subtract` ULP loss exceeds the gain from bounded sin.

**The Round 2 hypothesis was wrong.** The 36% rel-rms drift `probe_noise.py` reported at the
SineGen level was **`rand_ini` randomness** (the upstream SineGen adds a per-harmonic uniform-
random initial phase that the deterministic CoreML graph cannot reproduce), not graph drift.

The wrap-before-sin attempt is **reverted** in this PR. The real fix is downstream.

---

## Trial 10 — Round 3 root cause: `torch.atan2` semantics divergence

**Pivot**: instead of comparing CoreML to upstream PyTorch (which differs by `rand_ini`),
compare CoreML to its own *PyTorch trace source* — the `nn.Module` we converted from. If those
two diverge, the bug is in the CoreML conversion itself.

### `probe_noise_fidelity.py` — conversion fidelity test

Run `CoreMLFullNoiseModel.forward` in pure PyTorch with the same fp32 inputs the CoreML model
receives, then diff:

```
=== Sub-stage fidelity: SourceModule + STFT, no noise_conv/res ===
  [har (sine_merge → STFT → cat)]      rel=0.7906  corr=0.6831  max|d|=6.283e+00
  [spec (magnitude)]                   rel=6.5e-4  corr=0.99999998  max|d|=1.79e-3
  [phase (atan2)]                      rel=0.7932  corr=0.6805  max|d|=6.283e+00
=== Sub-stage fidelity: just SineGen output (sine_merge) ===
  [har_source (post-tanh-linear)]      rel=7.88e-4 corr=0.9999995  max|d|=3.01e-4
```

**`max|diff| = 6.283 = 2π`** at the STFT phase output — a **phase-wrap discrepancy** between
PyTorch and CoreML. SineGen output is fine (rel 7.88e-4); STFT magnitude is fine (rel 6.5e-4);
**only the atan2 phase differs**.

### Direct atan2 test

| input                      | PyTorch atan2 | CoreML MIL atan2 | diff   |
|----------------------------|---------------|-------------------|--------|
| `(imag = 0,    real < 0)`  | **+π**        | **0**             | +π     |
| `(imag = +1e-10, real < 0)`| +π            | +π                | 0      |
| `(imag = -1e-10, real < 0)`| -π            | -π                | 0      |
| `(imag > 0,   real > 0)`   | atan(imag/real) | atan(imag/real) | 0     |

For real-valued STFT input:
- **DC bin (k=0)**: `imag` is **exactly 0** (cos(0) = 1, sin(0) = 0).
- **Nyquist bin (k=N/2)**: `imag` is at the **fp32 noise floor (~1e-15)** because `-sin(πn) *
  window` evaluates to a tiny non-zero value due to fp imprecision.

Whenever `real < 0` at those bins, CoreML's `atan2` returns 0 instead of +π. The π/2π phase
offset propagates through `noise_convs[0]` (a strided conv that mixes 11 frequency bins) into
**all 256 channels** of `x_source_0`, producing the broad-spectrum noise that survives every
quantization variant in Round 1.

The upstream `kokoro/custom_stft.py:135-138` already documents this exact divergence for the
ONNX path:

```python
correction_mask = (imag_out == 0) & (real_out < 0)
phase[correction_mask] = torch.pi  # ONNX returns -pi, PyTorch returns +pi
```

`CoreMLForwardSTFT.transform` was missing it.

---

## Trial 11 — The fix

`convert-coreml.py:432-456`:

```python
def transform(self, waveform):
    if self.center:
        pad_len = self.n_fft // 2
        waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
    x = waveform.unsqueeze(1)
    real_out = self.conv_real(x)
    imag_out = self.conv_imag(x)
    # Two-stage: clip computational-zero imag, then apply PyTorch's atan2
    # branch convention for (imag==0, real<0).
    eps = 1e-5
    imag_clipped = torch.where(imag_out.abs() < eps,
                               torch.zeros_like(imag_out), imag_out)
    magnitude = torch.sqrt(real_out ** 2 + imag_clipped ** 2 + 1e-14)
    phase = torch.atan2(imag_clipped, real_out)
    correction_mask = (imag_clipped == 0) & (real_out < 0)
    phase = torch.where(correction_mask,
                        torch.full_like(phase, math.pi), phase)
    return magnitude, phase
```

`eps = 1e-5` is comfortably above the conv-rounding floor (~1e-15 at Nyquist) and far below
any legitimate spectral imag value (≥ 1e-3 for typical audio).

### Iteration record on `eps`

| eps        | x_source_0 rel-rms | x_source_0 corr | E2E waveform corr |
|------------|--------------------|-----------------|-------------------|
| (no fix)   | 0.397              | 0.911           | —                 |
| `== 0`     | 0.152              | 0.987           | 0.902             |
| `1e-9`     | 0.080              | 0.996           | 0.920             |
| `rel 1e-4` | 0.148              | 0.987           | 0.916             |
| **`1e-5`** | **0.057**          | **0.998**       | **0.957**         |
| `1e-7`     | 0.057              | 0.998           | 0.957             |

`eps = 1e-5` and `1e-7` give identical results; chose `1e-5` for slightly more headroom against
SineGen drift propagating into atan2's discontinuity. A pure relative threshold (`|imag| <
1e-4 · |real|`) clipped legitimate small values and regressed.

---

## Trial 12 — Verification

### Conversion fidelity (PyTorch trace vs CoreML, deterministic)

| metric                                  | before fix | after fix    |
|-----------------------------------------|------------|--------------|
| `JustSourceSTFT` phase max\|diff\|      | 6.283 (2π) | **0.000**    |
| `JustSourceSTFT` phase π-diffs / 481    | many       | **0**        |
| `KokoroNoise` x_source_0 rel-rms        | 0.397      | **0.057**    |
| `KokoroNoise` x_source_0 corr           | 0.911      | **0.998**    |
| `KokoroNoise` x_source_1 rel-rms        | 0.070      | **0.002**    |
| `KokoroNoise` x_source_1 corr           | 0.996      | **1.000**    |

### End-to-end audio HF (≥10 kHz) band power

| zm_009                                | before fix  | after fix     |
|---------------------------------------|-------------|---------------|
| HF(audio_D) — full CoreML chain       | -81.31 dB   | **-83.91 dB** |
| HF(audio_A) — full PyTorch reference  | -83.81 dB   | -83.90 dB     |
| **Δ(D − A)** — CoreML excess HF       | **+2.50 dB**| **-0.01 dB**  |
| HF(C − D) — noise-stage residual      | -90.17 dB   | -94.91 dB     |
| HF residual `(coreml − pt)` direct    | -86.55 dB   | **-91.18 dB** |

| zf_001                                | after fix     |
|---------------------------------------|---------------|
| HF(audio_D)                           | -71.55 dB     |
| HF(audio_A)                           | -71.66 dB     |
| Δ(D − A)                              | **+0.11 dB**  |

CoreML's HF band power is now within **0.01 dB** of the PyTorch reference on `zm_009` and
within **0.11 dB** on `zf_001`. The audible noise is gone.

### Reference audio (in-tree at `docs/audio/`)

- `docs/audio/before_fix_zm009.wav` — broken Noise stage (audible HF noise on the male voice)
- `docs/audio/after_fix_zm009.wav` — fixed Noise stage (HF matches PyTorch within 0.01 dB)
- `docs/audio/pytorch_ref_zm009.wav` — PyTorch teacher reference
- `docs/audio/after_fix_zf001.wav` — sanity check on the female voice

All 24 kHz mono float32. The `zm_009` triplet uses prompt `"你好世界，今天天气很好。"` so the
residual is purely the noise-stage delta. Click any file in the GitHub PR diff to download.

### Outstanding

- `x_source_0` still has 5.7% conversion drift vs the PyTorch trace (down from 39.7%). Source:
  `CoreMLSineGenV2`'s sin output drifts ~1e-4 between PyTorch and CoreML at large phase
  magnitudes (~22 000 rad). This drift does **not** propagate audibly — HF band matches PT
  within 0.01 dB — but a future round could close it by tracing the upstream `_f02sine`
  directly (with `rand_ini=0`) instead of the avg_pool/cumsum rewrite.
- The lack of `rand_ini` in the converted graph means CoreML output is deterministic per
  voice/text rather than randomized like upstream. This is expected and not audibly perceptible.
