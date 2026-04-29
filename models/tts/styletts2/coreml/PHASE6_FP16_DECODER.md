# Phase 6 — Robotic CoreML decoder: SineGen op-translation bugs in coremltools

**Status:** ✅ fp32 fixed and verified end-to-end. fp16 weights produce
robotic audio and are **not** shipped. v2 export with constant-folded fracs
index produces clean fp32 audio in the full pipeline
(`22_coreml_fixed_v2_full.wav`). All five mel buckets {256, 512, 1024, 2048,
4096} ship as fp32 `styletts2_decoder_<T_mel>.mlpackage` with PT/CoreML rms
parity ratio in 0.998–1.000 and max abs diff ≤ 3.3e-2 on random-tensor inputs.

This phase supersedes the earlier hypothesis that the leak was stochastic
SineGen ops baked at trace time. That theory was tested
(`SineGenDeterministicSt2` + zero `rand_ini` + `sin(x*100)` noise replacement)
and **did not fix the audio**. The actual bug is in how `coremltools` lowers
three specific PyTorch ops that live inside `SineGen._f02sine`.

---

## Symptom (unchanged from prior iteration)

PyTorch fp32 reference: clean speech.

CoreML pipeline: robotic / metallic / flattened prosody. Subjective character
described by user as "robotic and shit". All precision and compute-unit
combinations tested in the prior iteration produced the same character.

---

## Investigation timeline

### Disproved hypotheses

| Hypothesis | Test | Result |
|---|---|---|
| fp16 accumulator drift | `compute_precision=FLOAT32` re-export (07_swap_dec_fp32) | still robotic |
| Stochastic SineGen baked by trace | Deterministic shim (zero rand_ini, `sin(x*100)` noise) | still robotic |
| `weight_norm` reparameterization unfusable | Strip all 120 weight_norms before trace (`16_coreml_det_nown`, `17_…_fp32`) | still robotic, rms unchanged at 0.0483 |

### Decoder-internal bisect

`/tmp/styletts2_decoder_layer_bisect.py` exposes 10 intermediate activations
across the Decoder + Generator graph and converts the multi-output trace as
fp32 / CPU_AND_GPU. Result:

```
stage                 PT_rms      CML_rms      rms_ratio
d0_after_encode       1.272792    1.272792     1.0000   ✓
d1_pre_gen            1.803718    1.803718     1.0000   ✓
g0_har_source         0.018665    1.000000    54.7680   ✗ first divergent stage
g1..g4_after_up_i     contaminated downstream
g7_tanh_wav           0.121695    0.082073     0.6744
```

`g0_har_source` is the output of `SourceModuleHnNSF` (which wraps SineGen).
That's where the bug lives.

### SineGen-internal bisect

`/tmp/styletts2_sinegen_only_bisect.py` exposes 11 taps inside
`SourceModuleHnNSF.forward` / `SineGen._f02sine`. Result:

```
tap                   PT_rms      CML_rms      verdict
t0_fn                 ✓ identical
t1_rad_mod            0.239625    0.000000     BUG  ← (fn / sr) % 1 → all zeros
t2_rad_down           NaN
t3_phase              NaN
t4_phase_up           NaN
... downstream NaN
```

Three separate `coremltools` translation bugs in `_styletts2_lib.py`'s existing
`_f02sine_align_corners` shim:

1. **`(x % 1)` → `aten::remainder`** lowers to all-zeros in CoreML's MIL
   backend. Verified by tap test: PT side computes 0.2396 rms, CML side 0.000.
2. **`F.interpolate(scale_factor=1/300, mode="linear", align_corners=True)`**
   downsample produces NaN regardless of `scale_factor=` vs `size=`. Verified
   by `/tmp/styletts2_size_arg_fix_test.py`.
3. **`F.interpolate(scale_factor=300, mode="linear", align_corners=True)`**
   upsample also produces NaN. Verified by `/tmp/styletts2_slice_fix_test.py`
   (which fixes #1 and #2 but t4_phase_up still NaN).

### Three-part fix verified at SineGen-only level

| Bug | Replacement | Test script | Result |
|---|---|---|---|
| `(x % 1)` | `x - torch.floor(x)` | `/tmp/styletts2_frac_fix_test.py` | t1 fixed (max diff 7.45e-9), t2 still NaN |
| Downsample `F.interpolate` | `rad_mod[:, ::300, :]` | `/tmp/styletts2_slice_fix_test.py` | t1, t2, t3 clean; t4 still NaN |
| Upsample `F.interpolate` (nearest-hold) | `phase_scaled.repeat_interleave(300, dim=1)` | `/tmp/styletts2_repeat_interleave_test.py` | all 11 taps clean — but nearest hold creates audible 80 Hz buzz |
| Upsample `F.interpolate` (linear) | manual lerp from primitives | `/tmp/styletts2_manual_lerp_test.py` | all 11 taps clean, t10_har max diff 9.46e-6 |

Manual linear lerp (constructed from CoreML-friendly primitives only):

```python
def manual_linear_lerp_upsample(x_low, scale):
    # held value at each high-rate index
    left = x_low.repeat_interleave(scale, dim=1)
    # next value (last held at the end)
    last = x_low[:, -1:, :]
    x_shift = torch.cat([x_low[:, 1:, :], last], dim=1)
    right = x_shift.repeat_interleave(scale, dim=1)
    # fractional weights at full-rate index i: (i % scale) / scale
    fracs = torch.arange(T_AUDIO, dtype=torch.float32) % scale
    fracs = (fracs / float(scale)).view(1, T_AUDIO, 1)
    return left * (1.0 - fracs) + right * fracs
```

Standalone source-tap mlpackage built with these three substitutions
(`coreml/styletts2_source_tap_lerp_fp32.mlpackage`) achieves rms_ratio = 1.0000
across all 11 taps and t10_har max abs diff 9.46e-6 vs PT.

---

## Integrated-decoder regression

Applying the same three-part fix as a monkey-patch to `SineGen._f02sine` in
`/tmp/styletts2_export_decoder_fixed.py`, then re-tracing the full Decoder +
Generator stack and converting, yields:

```
rms PT-eager(full-fix):  0.0551   ✓ correct (matches paper baseline)
rms CoreML-fixed:        0.0483   ✗ matches the original broken baseline
max |PT-CoreML|:         6.4142e-01
```

Re-running the layer bisect with `install_full_fix()` applied
(`/tmp/styletts2_verify_fixed_har.py`) confirms the patch reaches the trace
on the PT side but **not** the converted CoreML graph:

```
g0_har_source   PT_rms=0.018665   CML_rms=1.000000   ratio=53.58
```

PT side reads the correct 0.0187 (= the fix is captured by trace). CoreML
output reads 1.0 (saturated tanh — same as the original baseline before any
fix).

### Why the fix works standalone but regresses in the integrated graph

Comparing the standalone working export (`manual_lerp_test`) against the
integrated failing export (`install_full_fix`), the only structural difference
in the lerp index path is how the fracs tensor is built:

| Path | fracs construction |
|---|---|
| Standalone (works) | `torch.arange(T_AUDIO_const, dtype=torch.float32) % scale_const` — both args are Python ints, `arange` and `%` constant-fold during trace |
| Integrated (broken) | `torch.arange(T_DOWN * self.upsample_scale, …) % self.upsample_scale` — `T_DOWN = phase_scaled.shape[1]` is a SymInt; `arange` becomes dynamic; `%` lowers to a runtime `aten::remainder` op |

`aten::remainder` is exactly the op coremltools mistranslates (proved by the
SineGen tap bisect — that's why we needed `x - floor(x)` for the rad_values
modulo in the first place). Building fracs from a SymInt-driven `arange`
re-introduces the same broken op pattern in the lerp index path, even though
the modulo on `f0/sr` is now correctly `x - floor(x)`.

---

## Reproduction artifacts (this phase)

All `/tmp` scripts, in time order:

| Script | Purpose | Outcome |
|---|---|---|
| `/tmp/styletts2_no_weight_norm.py` | Strip 120 weight_norms before trace | still robotic |
| `/tmp/styletts2_decoder_layer_bisect.py` | 10-output Decoder+Generator bisect | localized to g0_har_source |
| `/tmp/styletts2_sinegen_only_bisect.py` | 11-tap SineGen bisect | localized to t1_rad_mod (`% 1`) |
| `/tmp/styletts2_frac_fix_test.py` | Test `x - floor(x)` for rad_mod | t1 fixed; t2 still NaN |
| `/tmp/styletts2_size_arg_fix_test.py` | Test `F.interpolate(size=)` | downsample still NaN |
| `/tmp/styletts2_slice_fix_test.py` | Test stride slice for downsample | t1–t3 clean; t4 NaN |
| `/tmp/styletts2_repeat_interleave_test.py` | Test repeat_interleave for upsample | all taps clean (nearest hold, audible buzz) |
| `/tmp/styletts2_manual_lerp_test.py` | Test manual linear lerp upsample | all taps clean (true lerp) |
| `/tmp/styletts2_export_decoder_fixed.py` | Apply full fix in integrated decoder export | regression: CoreML still rms 0.0483 |
| `/tmp/styletts2_verify_fixed_har.py` | Re-run layer bisect post-fix | confirms regression in g0_har_source on CML side only |

Generated `.mlpackage` artifacts under `coreml/`:

- `styletts2_decoder_256_layers_fp32.mlpackage` — instrumented decoder bisect
- `styletts2_source_tap_fp32.mlpackage` — baseline SineGen tap export
- `styletts2_source_tap_frac_fp32.mlpackage` — frac-only fix
- `styletts2_source_tap_size_fp32.mlpackage` — `size=` arg variant
- `styletts2_source_tap_slice_fp32.mlpackage` — slice downsample fix
- `styletts2_source_tap_ri_fp32.mlpackage` — repeat_interleave upsample
- `styletts2_source_tap_lerp_fp32.mlpackage` — full standalone fix (working)
- `styletts2_decoder_256_fixed.mlpackage` — full integrated fix (regressed)
- `styletts2_decoder_256_layers_fixed_fp32.mlpackage` — verify-har bisect

Generated wavs under `/tmp/styletts2-bisect/`:

- `00_pt_full.wav` — paper baseline (clean reference)
- `09_pt_dec_unpatched.wav` / `10_pt_dec_patched.wav` / `11_pt_dec_traced.wav` — PT-side decoder variants
- `12`–`14_coreml_*.wav` — original broken CoreML exports across compute units
- `15_coreml_det_fp32.wav` — deterministic shim attempt (failed)
- `16_coreml_det_nown.wav` / `17_coreml_det_nown_fp32.wav` — weight_norm strip
- `18_coreml_fixed.wav` — full SineGen fix attempt (rms 0.0483, regressed)
- `19_pt_fullfix_eager.wav` — PT-eager with full fix applied (rms 0.0551, clean baseline)

---

## v2 fix (verified)

`/tmp/styletts2_export_decoder_fixed_v2_all.py` patches `SineGen._f02sine` and
`SineGen.forward` so the lerp index path constant-folds at trace time. The
key change vs v1 is building the fracs tensor entirely from Python-int
constants captured in a closure, not from a SymInt-driven `arange % scale`:

```python
def install_constfold_fix(t_mel_bucket: int):
    t_audio = t_mel_bucket * 2 * UPSAMPLE_SCALE       # Python int
    fracs_np = (np.arange(t_audio, dtype=np.float32) % UPSAMPLE_SCALE) \
        / float(UPSAMPLE_SCALE)
    fracs_tensor = torch.from_numpy(fracs_np.reshape(1, t_audio, 1)).contiguous()

    def _f02sine_constfold(self, f0_values):
        rad_div = f0_values / self.sampling_rate
        rad_mod = rad_div - torch.floor(rad_div)               # fix #1: frac
        if not self.flag_for_pulse:
            rad_down = rad_mod[:, ::UPSAMPLE_SCALE, :]         # fix #2: stride slice
            phase = torch.cumsum(rad_down, dim=1) * 2 * float(np.pi)
            phase_scaled = phase * self.upsample_scale
            # fix #3: manual linear lerp from CoreML-friendly primitives
            left  = phase_scaled.repeat_interleave(UPSAMPLE_SCALE, dim=1)
            last  = phase_scaled[:, -1:, :]
            right = torch.cat([phase_scaled[:, 1:, :], last], dim=1) \
                    .repeat_interleave(UPSAMPLE_SCALE, dim=1)
            fracs = fracs_tensor.to(phase_scaled.device, phase_scaled.dtype)
            phase_up = left * (1.0 - fracs) + right * fracs
            return torch.sin(phase_up)
        return SineGen._f02sine_original(self, f0_values)
    ...
```

`SineGen.forward` is also rebound so the noise term uses `sin(fn * 100)` (Kokoro
v21 trick) and `rand_ini` is dropped (zero phase). This makes the trace
deterministic and removes the `torch.rand` / `torch.randn_like` constants that
would otherwise be baked into the converted graph.

### Verification

| Bucket | PT_rms (random input) | CML_rms | ratio | max_diff |
|---|---|---|---|---|
| 256 | 0.121737 | 0.121736 | 1.0000 | 1.28e-02 |
| 512 | 0.041764 | 0.041746 | 0.9996 | 5.64e-03 |
| 1024 | 0.018658 | 0.018658 | 1.0000 | 3.00e-03 |
| 2048 | 0.034940 | 0.034895 | 0.9987 | 2.21e-02 |
| 4096 | 0.038203 | 0.038130 | 0.9981 | 3.31e-02 |

End-to-end audio: `/tmp/styletts2_bisect_v2_full.py` runs the full pipeline
(PT TP/diffusion/F0n + v2 CoreML decoder) on the paper canonical reference
`1221-135767-0014.wav` with `"The quick brown fox jumps over the lazy dog."`.
User-confirmed clean output:

- `/tmp/styletts2-bisect/22_coreml_fixed_v2_full.wav` — clean ✓
- `/tmp/styletts2-bisect/23_pt_full_match.wav` — PT reference

`max |PT-CoreML| = 0.539` over 88,750 samples is expected: tiny `rad_values`
precision drift accumulates through a 153,600-sample `cumsum` then `sin`, so
phase diverges at the sample level even when the model semantics are
bit-identical. Audibly indistinguishable from PT.

---

## fp16 — investigated, not shipped

Tried `convert_to="mlprogram"` with default fp16 weight precision to halve
the package size (~50 MB → ~110 MB total across five buckets vs ~250 MB).
**Result: robotic audio.** Not shipped. fp32 remains canonical.

### Audio regression

Same v2 SineGen patch, same full pipeline, only difference is the
mlpackage's weight precision. fp16 sounds clearly robotic (verified by
ear-test on bucket 256 and 512 wavs). RMS energy drops ~10 % vs fp32
(0.0496 vs 0.0551 on bucket 256), consistent with destructive phase
scrambling in the harmonic source.

Diagnosis: inside `SineGen._f02sine`, the cumsum-then-multiply chain
produces `phase_scaled` values up to ~4000 by mid-frame. fp16 precision
at that magnitude is ~4, much larger than the per-sample phase increment
(~0.05 rad). After the manual lerp `left * (1-frac) + right * frac`, the
phase fed to `sin()` is essentially random in fp16. fp32 has no such
issue (precision ~5e-4 at magnitude 4000).

### Two viable fixes (not pursued)

1. **Mixed precision.** `coremltools.transform.FP16ComputePrecision(op_selector=...)`
   to leave bulk weights at fp16 but force the SineGen ops (cumsum,
   multiply, repeat_interleave around the phase chain) to fp32. Kokoro
   v2 ships exactly this split. Estimated ~120 MB per bucket.
2. **Phase wrapping in SineGen.** Wrap `phase_scaled` mod 2π before the
   lerp — sin-equivalent because `sin(phase - 2πN) = sin(phase)`, but
   keeps phase in `[0, 2π)` where fp16 has plenty of resolution. Compute
   the lerp delta directly from `rad_down × 2π × upsample_scale` instead
   of `phase_scaled[i+1] - phase_scaled[i]` to avoid catastrophic
   cancellation. Tested in `/tmp/styletts2_export_decoder_v3_fp16.py` for
   bucket 256: rms recovers to 0.0547 (matches fp32 0.0551 within 1 %),
   numerically clean. Not promoted because shipping fp32 is fine for the
   current size budget.

### Hung first export attempt — useful for future work

When initially trying fp16 with `compute_units=ct.ComputeUnit.ALL`, the
export hung at bucket 1024 for 42+ min at 0 % CPU. Stack sample showed:

```
ct.convert → MLModel(...) ctor → MLE5ProgramLibrary::prepareAndReturnError
  → MILCompilerForANE::Run → _ANEClient compileModel
  → XPC waiting on anecompilerservice (synchronous, never returns)
```

Coremltools instantiates `MLModel` after conversion to verify the program;
when `.ALL` is in effect that constructor synchronously blocks on the ANE
daemon while it AOT-compiles the IR program for the ANE. For larger graphs
(1024+ mel frames) the daemon hangs.

Workaround for any future fp16 / ANE export:

```python
compute_units=ct.ComputeUnit.CPU_AND_GPU,
skip_model_load=True,
```

Bypasses the post-conversion MLModel instantiation. The saved mlpackage is
**not** locked to CPU+GPU — `compute_units` is a runtime hint, callers can
still load with `.ALL` later (paying the first-load ANE compile cost
themselves).

---

## Open work

1. **Decide on the v1 / `_fixed` / `_det` / `_fixed_v2` mlpackages.** The
   canonical export now writes `styletts2_decoder_<T_mel>.mlpackage` (no
   suffix) via the promoted v2 patch. Earlier suffixed packages can be
   archived or deleted once consumers point at the unsuffixed names.
2. **Optional: revisit fp16** if size becomes a constraint. Two known-good
   approaches sketched above (mixed precision via op_selector, or v3
   phase-wrapping in SineGen).
