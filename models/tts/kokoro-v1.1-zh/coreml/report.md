# Kokoro v1.1-zh — Background Noise Investigation Report

## Status

Background noise (more audible on `zm_009`) persists in CoreML output across
**every precision and dispatch configuration** tried so far. Conclusion: the
noise is **intrinsic to the CoreML graph topology**, not to weight quantization
(int8 → fp16 → fp32), boundary precision (fp16 ↔ fp32 MLMultiArray), internal
compute precision (FLOAT16 vs FLOAT32), or compute-unit dispatch (ANE vs GPU
vs CPU-only).

PyTorch ground truth is consistently 3–4 dB cleaner in the >10 kHz band.
Silence-frame floors match (≈ −100 dB), so the issue is speech-frame only.

---

## Experiments performed (all eliminated)

| # | Stage    | Variant tested                                              | Outcome              |
|---|----------|-------------------------------------------------------------|----------------------|
| 1 | Noise    | int8-palettize → fp32 weights                               | Noise unchanged      |
| 2 | Vocoder  | fp16 weights → fp32 weights, CPU_AND_NE → ALL               | Noise unchanged      |
| 3 | Prosody  | fp16 → fp32 weights + FLOAT32 compute                       | Noise unchanged; duration broke (Swift fp16 buffer mismatch) |
| 4 | All 5    | FLOAT32 compute, fp16 I/O kept (Option A)                   | Noise unchanged      |
| 5 | All 5    | **Pure fp32**: FLOAT32 compute + fp32 I/O + Swift fp32 path | Noise unchanged; silence floor matches PyTorch (-100 dB) |
| 6 | All 5    | Pure fp32 + `computeUnits: .cpuOnly`                        | Bit-identical to default-CU (corr +0.997) |

Side findings:
- `zm_009` voice-pack has +0.022 timbre-mean offset vs ≈0 for `zf_001` /
  `af_heart` — explains why noise is more audible on the male voice but does
  not cause it.
- Re-converting PostAlbert produces shorter durations than the original HF
  int8 build — intrinsic conversion-script reproducibility issue (independent
  of the noise).

---

## Top suspect investigated this round: `_cos_resblock1_forward`

Located at `scripts/convert-coreml.py:46–57`, monkey-patched onto `AdaINResBlock1`
inside the Vocoder's decoder generator (`scripts/convert-coreml.py:816`). Comment
claims a `sin² → cos` identity rewrite for ANE speed.

### Side-by-side diff

**Upstream `kokoro.istftnet.AdaINResBlock1.forward`**
(`.venv/lib/python3.11/site-packages/kokoro/istftnet.py:68–77`)

```python
def forward(self, x, s):
    for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2,
                                      self.adain1, self.adain2,
                                      self.alpha1, self.alpha2):
        xt = n1(x, s)
        xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)   # Snake1D
        xt = c1(xt)
        xt = n2(xt, s)
        xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)   # Snake1D
        xt = c2(xt)
        x = xt + x
    return x
```

**Replacement `_cos_resblock1_forward`** (`scripts/convert-coreml.py:46–57`)

```python
def _cos_resblock1_forward(self, x, s):
    for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2,
                                      self.adain1, self.adain2,
                                      self.alpha1, self.alpha2):
        xt = n1(x, s)
        cv = torch.cos(xt * (a1 * 2))
        xt = xt + (cv * (-0.5) + 0.5) * (1.0 / a1)
        xt = c1(xt)
        xt = n2(xt, s)
        cv = torch.cos(xt * (a2 * 2))
        xt = xt + (cv * (-0.5) + 0.5) * (1.0 / a2)
        xt = c2(xt)
        x = xt + x
    return x
```

### Algebraic verification

Snake activation: `Snake(x, α) = x + (1/α) · sin²(α·x)`.
Pythagorean identity: `sin²(z) = (1 − cos(2z)) / 2`.
Substituting `z = α·x`:

```
Snake(x, α) = x + (1/α) · (1 − cos(2αx)) / 2
            = x + ( ((-0.5)·cos(2αx)) + 0.5 ) · (1/α)
```

Replacement computes exactly this:
- `cv = cos(xt · (a · 2))` ≡ `cos(2α·xt)` ✓
- `cv·(-0.5) + 0.5` ≡ `(1 − cv)/2` ≡ `(1 − cos(2αx))/2` ✓
- `... · (1/a)` ≡ multiply by `(1/α)` ✓
- `xt + ...` ≡ residual add ✓

**Conclusion: numerically equivalent in real-arithmetic. It is a faithful
identity rewrite, not a topology change.**

### Floating-point caveat (ruled out by experiments)

The cosine form has a *latent* catastrophic-cancellation risk in fp16 when
`α·xt` is small: `cos(2αxt) ≈ 1 − 2(αxt)²`, so the subtraction
`(cv·(-0.5) + 0.5)` produces a tiny number from two near-1 magnitudes
(losing ~5 decimal digits). In fp16 (~3-decimal-digit mantissa) this would
manifest as noise at low signal levels — exactly the symptom reported.

However, experiments #5 and #6 (pure fp32 weights + FLOAT32 compute, plus
cpuOnly dispatch where fp32 is real fp32) **did not eliminate the noise**.
This excludes the cancellation hypothesis as the dominant cause.

---

## Verdict

`_cos_resblock1_forward` is a mathematically faithful port of the upstream
Snake1D activation. **Not the noise source.**

Per the agreed plan, escalate to the next suspect: **iSTFT comparison**.

---

## Next step (escalation): iSTFT/Tail comparison

Goal: determine whether the residual noise originates in
`Tail.mlmodelc` (CoreMLCustomSTFT iSTFT) or upstream in the Vocoder.

Procedure:

1. Run PyTorch ground-truth pipeline up to and including
   `decoder.generator` *minus* the final `conv_post + iSTFT` — i.e. capture
   the same `x_pre` tensor that CoreML's `Vocoder.mlpackage` returns.
2. Save `x_pre` as `.npy`.
3. Two parallel reconstructions on the same `x_pre`:
   - **PyTorch tail**: `decoder.generator.conv_post(x_pre)` →
     split mag/phase → `CustomSTFT.inverse(...)`.
   - **CoreML tail**: `Tail.mlmodelc.predict({"x_pre": x_pre_fp32})`.
4. Diff per-sample (RMS / max-abs) and visually compare spectrograms in the
   high-frequency band where the noise is most audible.

Branching rules:
- If the two tail outputs **differ measurably** in the noise band → the
  iSTFT/conv_post stage is the culprit. Inspect `CoreMLCustomSTFT` vs
  `kokoro.custom_stft.CustomSTFT` for any windowing or normalization drift
  during conversion.
- If they **match** to fp32 precision → the noise is upstream in the
  Vocoder graph. Next probes: per-resblock activation diff (instrument
  Vocoder forward pass, dump intermediate tensors at each upsample stage),
  then snake-activation MIL-op decomposition (the
  `cos → mul → add → mul` pattern may be fused suboptimally).

---

## Worktree state at end of previous round

- `Sources/FluidAudioCLI/Commands/TTSCommand.swift`: `computeUnits: .cpuOnly`
  debug flag **removed** (line ~926 reverted to default-CU init).
- `Sources/FluidAudio/TTS/KokoroAne/Pipeline/KokoroAneSynthesizer.swift`:
  pure-fp32 path (rebuild32 / float32Array everywhere; rebuild16 helper
  removed).
- `~/.cache/fluidaudio/Models/kokoro-82m-coreml/ANE-zh/`: all 5 trainable
  stages currently pure fp32. `.int8.mlmodelc.bak` and `.fp16.mlmodelc.bak`
  preserved next to each `.mlmodelc` for rollback.
- `scripts/convert-coreml.py`: `--no-palettize` flag added; all 5 stages set to
  `compute_precision=ct.precision.FLOAT32` with fp32 I/O dtypes; Vocoder
  dispatch changed `CPU_AND_NE → ALL`. `_cos_resblock1_forward` left in
  place pending iSTFT escalation result.

---

# Round 2 — iSTFT/Tail comparison + per-stage localisation

## TL;DR

The noise is **not** in the Tail and **not** in the Vocoder. It is in
**`KokoroNoise.mlpackage`**: the converted CoreML `noise` stage produces a
`x_source_0` tensor that diverges from PyTorch by **44% rel-rms (corr 0.884)**.
Swapping in PyTorch noise sources turns the residual HF-band power from
−90.2 dB (audible noise) down to −150.6 dB (fp32 noise floor) —
a 60 dB reduction.

The divergence is **independent of compute-unit dispatch** (CPU_ONLY,
CPU_AND_NE, CPU_AND_GPU, ALL all return identical drift to within 0.04
percentage points). It is also **not** the missing `rand_ini` random initial
phase — disabling rand_ini in the PyTorch reference still leaves the same
44% gap. The defect is in how `CoreMLSineGenV2` (`scripts/convert-coreml.py:369-397`)
or its STFT chain compiles to CoreML MIL ops at fp32.

---

## Stage-by-stage results

Test setup: `zm_009` voice, prompt `"你好世界，今天天气很好。"` (35 phonemes,
T_a = 139, x_pre shape `(1, 128, 16681)`, audio length 83 400 samples = 3.475 s).
All CoreML stages loaded with `compute_units=ALL`, fp32 I/O.

### Step 1 — iSTFT/Tail comparison (`scripts/tail_compare.py`)

Three reconstructions on the **same** PyTorch x_pre:

| pair                              | rel-rms | corr        | best-gain | resid after gain | HF Δ (≥10 kHz) |
|-----------------------------------|---------|-------------|-----------|------------------|----------------|
| torch.istft  vs torch_custom\*    | 5.94e-1 | 0.911248848 | 1.022     | 5.81e-1          | +1.23 dB       |
| torch_custom vs CoreML Tail       | 6.45e-1 | 0.911248855 | 1.097     | 5.02e-1          | −4.76 dB       |
| **torch.istft vs CoreML Tail**    | 5.00e-1 | **1.000000000** | **1.499999523** | **1.04e-5**     | −3.52 dB       |

`*` `torch_custom` = `CustomSTFT.inverse` directly (the conv_transpose1d
rewrite that was traced for CoreML). It drops the DC/Nyquist non-doubling
correction that `CoreMLCustomSTFT` adds, so its iSTFT is mis-scaled per
bin.

> **`corr(torch.istft, CoreML Tail) = 1.000000000` exact, residual after
> the 1.5x scale correction is 1.04e-5 (fp32 noise floor).** The CoreML
> tail is bit-equivalent to `torch.istft` up to a deterministic ×1.5 gain.
> The 1.5 factor is the squared-Hann-window overlap-sum normalisation that
> `torch.istft` divides by but `CoreMLCustomSTFT` does not (with periodic
> Hann at hop = n_fft/4 the sum of squared windows over each output sample
> is exactly 1.5 — `20·log10(1.5) = 3.522 dB`, matching the HF Δ to two
> decimals).

**Tail is NOT the noise source.**

### Step 2 — Vocoder x_pre comparison (`scripts/vocoder_xpre_diff.py`)

Feed the same PyTorch (asr / F0 / N / style_timbre / noise_sources) into
`KokoroVocoder.mlpackage` and compare its `x_pre` against the PyTorch x_pre:

| metric                           | value         |
|----------------------------------|---------------|
| `rms(pt_x_pre)`                  | 2.331597      |
| `rms(coreml_x_pre)`              | 2.331528      |
| ratio                            | 1.0000        |
| rel-rms (pt vs coreml)           | **7.42e-5**   |
| corr                             | **0.999999998** |
| max\|diff\|                      | 9.77e-3       |
| best-gain g(pt → cm)             | 0.999971      |
| per-frame median rel             | 9.07e-5       |
| per-frame p95 rel                | 2.60e-4       |
| per-frame max rel                | 8.13e-4       |

Audio reconstructed by feeding both x_pre tensors into `KokoroTail`:

| metric                                    | value     |
|-------------------------------------------|-----------|
| rel-rms `audio(pt_x_pre)` vs `audio(cm_x_pre)` | 3.07e-4 |
| HF (≥10 kHz) Δ                            | **+0.00 dB** |
| HF residual `audio_pt − audio_cm`         | **−150.62 dB** |

> **The CoreML Vocoder is bit-equivalent to PyTorch to fp32 precision when
> given the same upstream inputs. HF residual is at the fp32 noise floor.**

**Vocoder is NOT the noise source.**

### Step 3 — Per-stage chain swap (`scripts/per_stage_diff.py`)

Five reconstructions, each substituting one more CoreML stage, all
descaled by ×1.5 to land on the same magnitude:

| chain                                                      | HF (≥10 kHz) | rel vs A | HF(diff) vs prev |
|------------------------------------------------------------|--------------|----------|------------------|
| **A.** PT x_pre → PT torch.istft                           | −83.81 dB    | —        | —                |
| **B.** PT x_pre → CoreML Tail                              | −83.81 dB    | 1.04e-5  | −177.29 dB       |
| **C.** PT inputs → CoreML Vocoder → CoreML Tail            | −83.81 dB    | 3.12e-4  | −150.64 dB       |
| **D.** CoreML Noise → CoreML Vocoder → CoreML Tail         | **−81.31 dB**| **0.276**| **−90.17 dB**    |

Noise sources themselves:

| source        | rms_pt | rms_cm | rel-rms | corr   |
|---------------|--------|--------|---------|--------|
| `x_source_0`  | 0.6458 | 0.5039 | **0.4399** | **0.886** |
| `x_source_1`  | 1.0463 | 1.0181 | 0.0724    | 0.9962    |

> The HF residual jumps from **−150.6 dB → −90.2 dB (60 dB)** the moment
> we substitute the CoreML noise sources for PyTorch's. `x_source_0` is
> the dominant contributor (44% rel-rms, vs 7% for `x_source_1`).

### Step 4 — Compute-unit sweep on KokoroNoise (`scripts/probe_noise.py`)

| compute units  | x_source_0 rel | x_source_0 corr | x_source_1 rel | x_source_1 corr |
|----------------|----------------|------------------|----------------|------------------|
| CPU_ONLY       | 0.4438         | 0.8839816        | 0.0724         | 0.9962213        |
| CPU_AND_NE     | 0.4438         | 0.8839816        | 0.0724         | 0.9962213        |
| CPU_AND_GPU    | 0.4402         | 0.8862826        | 0.0724         | 0.9962221        |
| ALL            | 0.4402         | 0.8862826        | 0.0724         | 0.9962221        |

> **Identical drift on all four dispatches, including pure CPU at fp32.**
> This rules out ANE-fp16 fall-back as the cause — the CoreML graph
> itself is wrong (or at least diverges from PyTorch's same-formulation
> graph) at fp32.

### Step 5 — Per-channel pattern on `x_source_0`

x_source_0 has 256 channels; per-channel rel-rms (CoreML CPU_ONLY vs
PyTorch with rand_ini):

| stat               | value |
|--------------------|-------|
| median             | 0.494 |
| p95                | 1.43  |
| max                | 1.89  |
| best 3 channels    | 0.139, 0.167, 0.185 |
| worst 5 channels   | 1.89, 1.87, 1.79, 1.79, 1.77 |

> Broad-spectrum drift — every channel is wrong by ~50% on average, with
> long-tail outliers up to 188%. Not a localised op problem.

### Step 6 — Sub-stage probe inside the noise pipeline (`scripts/probe_noise.py` step 3)

Run the upstream PyTorch SineGen vs the **CoreMLSineGenV2 formulation
re-implemented in pure PyTorch** (no CoreML), tracking divergence at each
intermediate:

| intermediate                                  | rel-rms | corr   |
|-----------------------------------------------|---------|--------|
| SineGen sin output (with vs without rand_ini) | 0.365   | 0.931  |
| har_source (after `tanh∘linear`)              | 0.317   | 0.927  |
| STFT magnitude                                | 0.257   | 0.967  |
| STFT phase                                    | 1.103   | **0.371** |
| `noise_convs[0]` output                       | 0.459   | 0.876  |
| `noise_res[0]` output (= x_source_0)          | 0.273   | 0.955  |
| `noise_convs[1]` output                       | 0.123   | 0.992  |
| `noise_res[1]` output (= x_source_1)          | 0.018   | 0.9997 |

### Step 7 — Isolating `rand_ini` vs CoreML conversion

Disable `rand_ini` in the PyTorch reference (deterministic upstream) and
compare to CoreML noise output:

| comparison                                   | x_source_0 rel | x_source_0 corr |
|----------------------------------------------|----------------|------------------|
| PT (with rand_ini) vs CoreML CPU             | 0.4438         | 0.884            |
| PT (no rand_ini)   vs CoreML CPU             | **0.4435**     | **0.884**        |
| PT (with rand_ini) vs PT (no rand_ini)       | 0.2283         | 0.968            |

> **Removing `rand_ini` from the reference closes only ~0% of the gap.**
> The 44% divergence is *not* explained by the missing random initial
> phase. It is a real numerical drift between `CoreMLSineGenV2`'s CoreML
> compilation and PyTorch's same-formulation execution.

Pure-PyTorch sanity check (both SineGen formulations, no CoreML, no
rand_ini, single test waveform):

| comparison                                                          | rel-rms | corr      |
|---------------------------------------------------------------------|---------|-----------|
| upstream `_f02sine` (no rand_ini)  vs  `CoreMLSineGenV2` (avg_pool) | 1.31e-3 | 0.9999991 |
| both formulations with `align_corners=True`                         | 1.40    | 0.024     |

> The two formulations are **algorithmically equivalent in pure PyTorch
> (rel 1e-3, corr 0.9999991)**. The ×300 difference between this 0.13%
> baseline and the 44% drift seen against the converted CoreML model
> implies the drift comes in during `coremltools` conversion (or CoreML
> runtime) of the avg_pool1d → cumsum → multiply → interpolate → sin
> chain — most likely the `sin` of cumulative phase, which reaches
> magnitudes around ±39 000 rad (~6 200 cycles). At those magnitudes
> fp32's 7-decimal-digit precision provides only ~6e-4 rad of resolution,
> below sin's slope-of-1 sensitivity threshold.

---

## Verdict

`KokoroNoise.mlpackage` is the noise source. The defect is in the conversion
of `CoreMLSineGenV2` (`scripts/convert-coreml.py:369-397`) — most likely the
chained `cumsum → multiply-by-300 → linear-interpolate → sin` reaching
phase magnitudes ~39 000 rad where fp32 precision is insufficient. CoreML
and PyTorch handle this regime differently, even at FLOAT32 compute, even
on CPU_ONLY.

The original Snake1D / Vocoder / iSTFT investigations cleared all the
wrong suspects. The Tail and Vocoder are bit-equivalent to PyTorch (corr
1.000000000 and 0.999999998 respectively).

---

## Next-step proposals (in order of confidence)

1. **Phase wrap-around inside SineGen.** Reformulate `CoreMLSineGenV2` so
   the phase passed to `sin` is bounded. Concretely, do the `* 2π`
   multiplication *and the modulo `2π`* before the linear-interpolate-up
   step, then sin only the bounded value. The catch is preserving the
   wrap-around continuity across the interpolation — likely needs the
   sin/cos representation pair so wrap is safe (interpolate `cos(phase)`
   and `sin(phase)` separately, then atan2 if needed). This is the
   highest-confidence fix because it directly addresses the precision
   regime where PyTorch and CoreML disagree.

2. **Skip the avg_pool1d trick entirely.** The upstream formulation
   already runs cumsum at the downsampled rate (it does
   `interpolate(rad_values, 1/upsample_scale, mode='linear')` first).
   Trace upstream `_f02sine` directly with `rand_ini` zeroed for
   determinism, instead of the custom rewrite. The upstream formulation
   has run successfully in production for the v1.0 English checkpoint
   (Pax's repo) and is what the network was trained against, so it's the
   safest semantics.

3. **Test on `zf_001` to confirm the fix lands across voices.** Per
   round 1 the male voice has worse audible noise but the female voice
   has the same 44% noise-graph drift; both should benefit equally.

4. **Re-run end-to-end with the fixed Noise stage.** Use
   `scripts/compare-models.py` and listen to `zm_009` and `zf_001` outputs at
   the >10 kHz band. Expect HF residual to drop from −90 dB toward the
   −150 dB floor that the Tail and Vocoder already achieve.

## Diagnostic scripts added (in `models/tts/kokoro-v1.1-zh/coreml/`)

- `scripts/tail_compare.py` — three-way iSTFT comparison (torch.istft, CustomSTFT,
  CoreML Tail) on the same x_pre, with HF-band power deltas and audio
  WAV residuals.
- `scripts/vocoder_xpre_diff.py` — feeds PyTorch upstream into CoreML Vocoder and
  diffs x_pre per-channel + per-frame.
- `scripts/per_stage_diff.py` — five-tier stage-swap chain (full PT → ... →
  full CoreML) with HF-band tracking.
- `scripts/probe_noise.py` — CU sweep, per-channel divergence, sub-stage probe of
  the noise pipeline in pure PyTorch (formulation vs upstream).

---

# Round 3 — Fix landed

## TL;DR

**Root cause was misdiagnosed in Round 2.** It is *not* the `sin` of large
arguments inside `CoreMLSineGenV2`. Both the original cumsum→×US→interp→sin
chain and a wrap-before-sin variant trace to identical CoreML graphs (rel
9.2 × 10⁻⁴, corr 0.9999996). Wrapping the phase actually *worsened* the
result by 27× (rel 2.5 × 10⁻²) because dividing-by-2π and floor-subtracting
introduces its own ULP loss.

The real culprit is **`torch.atan2` semantics at the (imag = 0, real < 0)
boundary in `CoreMLForwardSTFT`** (`scripts/convert-coreml.py:432-441`):

- PyTorch's `atan2(0, -1) = +π`; CoreML's MIL `atan2` returns `0`.
- For real-valued STFT input, the **DC bin always has imag exactly 0**, and
  the Nyquist bin has imag at the fp32 floor (~1e-15) — both bins land in
  this bad branch whenever real < 0.
- The 2π phase offset propagates through `noise_convs[0]` (a strided
  conv that mixes 11 frequency bins) into all 256 channels of `x_source_0`,
  appearing as broad-spectrum noise in the audible >10 kHz band.

The upstream `kokoro/custom_stft.py:135-138` already documents this
specific divergence ("In this case, PyTorch returns pi, ONNX returns -pi")
and applies a correction. `CoreMLForwardSTFT` was missing it.

## Fix

Two-line patch to `CoreMLForwardSTFT.transform`
(`scripts/convert-coreml.py:432-456`):

```python
eps = 1e-5
imag_clipped = torch.where(imag_out.abs() < eps,
                           torch.zeros_like(imag_out), imag_out)
magnitude = torch.sqrt(real_out ** 2 + imag_clipped ** 2 + 1e-14)
phase = torch.atan2(imag_clipped, real_out)
correction_mask = (imag_clipped == 0) & (real_out < 0)
phase = torch.where(correction_mask,
                    torch.full_like(phase, math.pi), phase)
```

`eps = 1e-5` clips both the exact-zero (DC) and the fp-noise-floor
(Nyquist ~1e-15) cases without affecting any legitimate spectral imag value
(which are ≥ 1e-3 for typical audio).

## Verification

### Standalone STFT conversion fidelity (`scripts/probe_noise_fidelity.py`,
`scripts/probe_sinegen_isolated.py`)

| Step                                    | Before fix | After fix    |
|-----------------------------------------|------------|--------------|
| `JustSourceSTFT` phase max\|diff\|      | 6.283 (2π) | **0.000**    |
| `JustSourceSTFT` phase π-diffs / 481    | many       | **0**        |
| `KokoroNoise` x_source_0 rel-rms        | 0.397      | **0.057**    |
| `KokoroNoise` x_source_0 corr           | 0.911      | **0.998**    |
| `KokoroNoise` x_source_1 rel-rms        | 0.070      | **0.002**    |
| `KokoroNoise` x_source_1 corr           | 0.996      | **1.000**    |

`x_source_0` rel-rms dropped 7×, corr from 0.91 → 0.998 against the
deterministic PyTorch trace (`CoreMLFullNoiseModel.forward` run in
PyTorch).

### End-to-end audio HF band (`scripts/per_stage_diff.py`)

| Stage                               | Before fix  | After fix    |
|-------------------------------------|-------------|--------------|
| **`zm_009`**                        |             |              |
| HF(audio_D) — full CoreML chain     | −81.31 dB   | **−83.91 dB** |
| HF(audio_A) — full PyTorch          | −83.81 dB   | −83.90 dB    |
| Δ(D − A) — CoreML excess            | **+2.50 dB**| **−0.01 dB** |
| HF(C − D) — noise-stage residual    | −90.17 dB   | **−94.91 dB**|
| **`zf_001`**                        |             |              |
| HF(audio_D)                         | (untested)  | −71.55 dB    |
| HF(audio_A)                         | (untested)  | −71.66 dB    |
| Δ(D − A)                            | —           | +0.11 dB     |

> CoreML's HF band power is now within 0.01 dB of the PyTorch reference on
> `zm_009`, and within 0.1 dB on `zf_001`. The audible noise is gone.

### Direct file comparison (`scripts/_audio_compare.py`)

| Metric                         | Before fix    | After fix      |
|--------------------------------|---------------|----------------|
| HF (≥10 kHz) power, CoreML     | −81.37 dB     | **−83.96 dB**  |
| HF residual `HF(coreml − pt)`  | −86.55 dB     | **−91.18 dB**  |
| Improvement                    | —             | **−4.62 dB**   |
| 10–12 kHz band specifically    | −81.32 dB     | −83.90 dB      |
| 8–10 kHz band                  | −75.71 dB     | −78.49 dB      |

WAVs at `build/audio_compare_zm009/`:
- `before_fix_zm009.wav` — broken Noise (audible noise band)
- `after_fix_zm009.wav` — fixed Noise (matches PyTorch HF level)
- `pytorch_ref_zm009.wav` — PyTorch teacher reference
- `after_fix_zf001.wav` — sanity check on female voice

## Why Round 2 misdiagnosed

The substage probe in Round 2 (`scripts/probe_noise.py` step 3) ran the upstream
`SineGen` (with `rand_ini`) against the `CoreMLSineGenV2` formulation in
pure PyTorch. That comparison correctly showed 36% rel-rms drift at the
SineGen level, but the drift came from **`rand_ini`** alone — both
formulations were algorithmically equivalent in pure PyTorch (corr
0.9999991 with rand_ini disabled). When CoreML drift was the same 44% as
the rand_ini-driven PyTorch drift, the size match looked confirmatory and
fingered the SineGen sin precision. It was the wrong suspect.

The right next probe was conversion fidelity — *trace then run the same
nn.Module in PyTorch and CoreML*. The `JustSourceSTFT` standalone test
(`scripts/probe_noise_fidelity.py`) immediately exposed `phase rel = 0.79, max|d|
= 2π`, pointing at `atan2`.

## Diagnostic scripts added in Round 3

- `scripts/probe_sinegen_isolated.py` — five standalone CoreML models (sin alone;
  cumsum+sin; cumsum+wrap+sin; cumsum+interp+sin; cumsum+interp+wrap+sin;
  interpolate alone; phase-only) to find which sub-op of `CoreMLSineGenV2`
  diverges. Verdict: **none of them** — the SineGen pipeline is bit-clean.
- `scripts/probe_noise_fidelity.py` — runs `CoreMLFullNoiseModel.forward` in
  PyTorch with the patched `CoreMLForwardSTFT`, then converts and runs in
  CoreML. Tracks fidelity at `har_source` (post-tanh-linear), `spec`,
  `phase`, `har`, `x_source_0`, `x_source_1`.
- `scripts/_audio_compare.py` — generates PyTorch teacher reference and computes
  per-band power deltas vs before-fix and after-fix CoreML WAVs.

## Worktree state

- `scripts/convert-coreml.py:432-456` — `CoreMLForwardSTFT.transform` now clips
  near-zero imag and applies the +π correction.
- `scripts/convert-coreml.py:369-397` — `CoreMLSineGenV2` reverted to original
  (the wrap-before-sin attempt was a dead end).
- `build/ANE-zh/KokoroNoise.mlpackage` and `.mlmodelc` re-exported with
  the fix (compute_precision = FLOAT32, no palettization).
- `build/ANE-zh/KokoroNoise.mlpackage.bak.before_phasewrap` and
  `.mlmodelc.bak.before_phasewrap` preserve the broken version for A/B.
- The other six stages are unchanged from Round 2.

## Outstanding (small)

- `x_source_0` still has 5.7% conversion drift vs the PyTorch trace (down
  from 39.7%). Source: `CoreMLSineGenV2` sin output drifts ~1e-4 between
  PyTorch and CoreML at large phase magnitudes (~22 000 rad). This drift
  doesn't propagate audibly — HF power matches PyTorch reference within
  0.01 dB — but a future round could close it by adapting the upstream
  `_f02sine` (rand_ini=0) directly instead of the avg_pool/cumsum
  rewrite.
- The lack of `rand_ini` in the converted graph means CoreML output is
  deterministic per voice/text rather than randomized like the upstream
  PyTorch pipeline. This is expected and not audibly perceptible.
