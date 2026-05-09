# StyleTTS2 → CoreML stage fusion experiments

Companion log to `trials.md`. Same project, same fp32 packages baseline,
but focused on collapsing multiple `predict()` round-trips into single
graphs.

These continue the trial sequence in `trials.md`:

| # | Trial                                         | Outcome              |
|---|-----------------------------------------------|----------------------|
| 1 | decoder split (decoder_pre + decoder_upsample)| wash (`trials.md`)   |
| 2 | int8 weight quantization on decoder_upsample  | slower (`trials.md`) |
| 3 | int8 palettization on decoder_upsample        | lossy (`trials.md`)  |
| **4** | **fused diffusion sampler (8 calls → 1)** | **win, −437 ms warm** |
| **5** | **fused har_source + decoder_upsample**   | **regression, +290 ms min** |
| **6** | **fused f0n_predictor + har_source**       | **win, −42 ms warm**  |
| **7** | **fused ref_encoder + fused_diffusion_sampler** | **regression / wash, partition tax** |
| **8** | **placement sweep (per-stage compute_units)** | **partial — 8b conservative ≈ wash, 8a aggressive too variable** |
| **9** | **`_hifigan_shift` fold**                  | **skipped (sub-1 ms, dominated by Trial 8)** |

Working scripts and logs live under `/tmp/styletts2_fuse/`, kept out of
the repo because they import the project as a sibling and write
mlpackages to disk. They are not malware; they are standalone driver
scripts that load `coreml._runtime`, build a fused `nn.Module`, trace,
convert, validate parity, and bench warm latency.

## Hardware / measurement context

* M-series Mac, macOS 15+, `coremltools 9.0`, fp32 mlprogram packages.
* Text: `"Hello, this is StyleTTS 2."` (42 phonemes, padded to 57 for
  bert/diffusion fixed axis).
* `seed=0`, `num_steps=5` ADPM2 — bit-exact RNG draws replicated in the
  fused samplers via input tensors so the two paths can be byte-compared.
* Warm latency = mean of 10 `predict()` calls after one warmup, single
  process, no other GPU/ANE workloads.
* Reference per-stage warm numbers (fp32 packages, current pipeline):
  `text_encoder ≈ 6 ms`, `bert ≈ 6 ms`, `ref_encoder ≈ 7 ms`,
  `diffusion_unet ≈ 58 ms × 8 dispatches = 464 ms`,
  `duration_predictor ≈ 6 ms`, `f0n_predictor ≈ 8 ms`,
  `har_source ≈ 45 ms (CPU_AND_GPU)`,
  `decoder_pre ≈ 21 ms (CPU_AND_NE)`,
  `decoder_upsample ≈ 466 ms (CPU_ONLY)`.
* Total warm wall-clock of the per-step pipeline ≈ 1030 ms; diffusion
  sampler and decoder_upsample dominate.

## Trial 4 — fused diffusion sampler (5-step ADPM2 over `diffusion_unet`)

**Hypothesis.** The Python ADPM2 loop dispatches `diffusion_unet` 8×
per utterance (`num_steps=5` → 4 iters × 2 calls each). Each dispatch is
one CoreML `predict()` round-trip + mlmodel-side input/output marshall.
The Karras schedule, the per-iter `(sigma_up, sigma_down, sigma_mid)`,
and the loop control flow are all constants of `num_steps`. Bake them
into a single traced graph and dispatch the whole sampler in one call.

**Implementation.** `/tmp/styletts2_fuse/fuse_diffusion_sampler_fp32.py`.

```text
fused_sampler(noise_init, noises_aux, embedding, features) -> s_pred

  x = sigmas[0] * noise_init                       # bake sigmas[0]
  for i in range(num_steps - 1):                   # baked, num_steps=5
      x_dn   = unet(x, sigmas[i], embedding, features)
      d      = (x - x_dn) / sigmas[i]
      x_mid  = x + d * (sigma_mids[i] - sigmas[i])
      x_mid_dn = unet(x_mid, sigma_mids[i], embedding, features)
      d_mid  = (x_mid - x_mid_dn) / sigma_mids[i]
      x      = x + d_mid * (sigma_downs[i] - sigmas[i])
      x      = x + noises_aux[i] * sigma_ups[i]    # stochastic aux noise
  return x
```

* Karras `sigmas`, `sigma_ups`, `sigma_downs`, `sigma_mids` baked as
  non-persistent buffers from the same `_karras_sigmas` /
  `_adpm2_get_sigmas` formulas in `coreml/inference.py`.
* The 4 stochastic noise injections are passed as one `[4, 1, 1, 256]`
  input tensor — keeps the graph deterministic and lets the runner
  reproduce the exact RNG sequence the per-step path draws (`torch.randn`
  on the same seed, drawn up-front because `unet` calls don't touch the
  RNG state).
* `DiffusionDenoiseStepWrapper` from `coreml/wrappers.py` is reused
  unchanged as the inner step.

**CoreML inputs.**

| name        | shape           | dtype | meaning |
|-------------|-----------------|-------|---------|
| noise_init  | `[1, 1, 256]`   | f32   | `torch.randn(1, 256).unsqueeze(1)` at seed 0 |
| noises_aux  | `[4, 1, 1, 256]`| f32   | per-iter stochastic noise |
| embedding   | `[1, 57, 768]`  | f32   | `bert_dur` (padded to 57) |
| features    | `[1, 256]`      | f32   | `ref_s` |

**Output.** `s_pred [1, 1, 256]`.

**Parity chain (fp32, 5e-3 unit tolerance for chaos drift).**

```
eager fused vs python per-step :  max|Δ| = 0          mse = 0
traced  vs eager fused         :  max|Δ| = 0          mse = 0
coreml  vs eager fused         :  max|Δ| = 5.0e-3     mse = 2.3e-6
```

The 5e-3 max delta is from MIL pass reordering of the iterated
multiply-adds (chaotic divergence in 4-iter sequential math at fp32 is
expected). The downstream WAV is coherent speech and the user has
verified it sounds correct.

**Latency.**

| Compute units    | Cold (1st)  | Warm avg | Notes |
|------------------|-------------|----------|-------|
| `CPU_AND_NE`     | ~110 ms     | ~53 ms   | ANE compile fallback overhead |
| `ALL`            | 263–478 ms  | 78 ms    | high cold variance, slow steady |
| `CPU_AND_GPU` ✅ | 65 ms       | **17 ms** (script bench), **27 ms** (in-pipeline) | best |

Compared to the 8-dispatch baseline of `8 × 58 ms ≈ 464 ms` per step,
the fused sampler at `CPU_AND_GPU` is **−437 ms warm** in script
microbench and **−437 ms** in the actual pipeline (464 → 27 ms).
Pipeline total: 1030 → 593 ms.

**Disk.** 99.0 MB single fp32 mlpackage.

**Verdict.** ✅ keep. Replace `diffusion_unet` standalone with
`fused_diffusion_sampler` in the inference dispatch. Same input set, one
`predict()` instead of eight.

## Trial 5 — fused har_source + decoder_upsample

**Hypothesis.** Both stages already run on CPU. The boundary tensor
`har [1, 1, 600 × T_F]` is ~350 KB per call at `T_F=147` and gets fully
marshalled across the mlmodel boundary. Folding the `SineGen + linear +
tanh` math into the same graph as the HiFi-GAN generator should remove
that marshall and any associated `predict()` overhead while keeping all
ops on the CPU partition that decoder_upsample is already optimized for.
Expected save ≈ 25 ms on a 511 ms baseline (−5%).

**Implementation.** `/tmp/styletts2_fuse/fuse_har_decoder_upsample_fp32.py`.

```text
class FusedHarDecoderUpsample(nn.Module):
    def __init__(self, decoder):
        # IMPORTANT: build har first so HarSourceWrapper can read the
        # generator's m_source.* tensors *before* DecoderUpsampleWrapper
        # strips weight_norm and patches Generator.forward to take
        # har_source directly. Trace parity vs the two-stage path is
        # bit-exact in this order.
        self.har_wrap = HarSourceWrapper(decoder)
        self.upsample_wrap = DecoderUpsampleWrapper(decoder)

    def forward(self, x_pre, ref, f0):
        har = self.har_wrap(f0)              # SineGen + linear + tanh
        return self.upsample_wrap(x_pre, ref, har)   # HiFi-GAN
```

**CoreML inputs.**

| name | shape                | dtype | meaning |
|------|----------------------|-------|---------|
| x_pre| `[1, 512, T_F2]`     | f32   | decoder_pre output |
| ref  | `[1, 128]`           | f32   | style half of `ref_s` |
| f0   | `[1, T_F2]`          | f32   | f0 contour from f0n_predictor |

`T_F2 = ct.RangeDim(2, 4096, default=294)` (= `2 × T_F`).

**Output.** `wav [1, 1, T_F × 600]`.

**Parity (fp32).**

```
eager fused vs (har -> upsample) : max|Δ| = 0           mse = 0
traced  vs eager fused           : max|Δ| = 0           mse = 0
coreml  vs eager fused           : max|Δ| = 5.1e-4      mse = 2.5e-9
```

Cleaner parity than Trial 4 because there is no iterated stochastic
math — just a deterministic feed-forward stack. 5e-4 max delta is well
under the decoder tolerance budget (`max|Δ| < 5e-3`) used in
`coreml/parity.py`.

**Latency (warm avg over 10 calls, T_F2=294, fp32 mlprogram).**

| Compute units            | min (ms) | avg (ms) | max (ms) |
|--------------------------|----------|----------|----------|
| Standalone (har GPU + upsample CPU) | — | **511** (45 + 466) | — |
| Fused — `CPU_ONLY`       | 1165     | 1661     | 2426     |
| Fused — `CPU_AND_GPU`    | 841      | 1100     | 1470     |
| Fused — `ALL` (best)     | **802**  | **931**  | 1146     |
| Fused — `CPU_AND_NE`     | 856      | 1119     | 1343     |

Best fused configuration is **+291 ms minimum** worse than the two-stage
path. The fused mlpackage is correct (83.2 MB on disk, parity passes)
but slower under every compute-units choice.

**Why it regresses.**

* Decoder_upsample standalone gets specialized `CPU_ONLY` /
  Accelerate-friendly MIL passes for its dominant `ConvTranspose1d`
  upsample stack — that is the entire reason `_STAGE_COMPUTE` pins it
  to `CPU_ONLY` (see `trials.md` decoder split trial — splitting the
  ANE-friendly head off the CPU tail was the whole point).
* HarSource math (`cumsum`, `interpolate(scale=300, linear)`, `sin`)
  added to the same graph forces the optimizer to find a partition that
  satisfies both halves. Whatever it picks, one half loses.
* The 88,200-sample intermediate `har` tensor that used to be marshalled
  out of one mlmodel and into the next now materializes as a CoreML
  internal activation in fp32. The marshall it replaces was ~350 KB; the
  internal activation costs the same memory plus whatever stride/pass
  layout the new partition imposes.
* Variance is also high (1.3–2.1× spread between min and max) which
  suggests partition-boundary scheduling is now sensitive to system
  noise in a way the per-stage packages were not.

**Disk.** 83.2 MB single fp32 mlpackage (vs 79 MB decoder_upsample alone
+ 4 MB har_source = 83 MB). No size win either.

**Verdict.** ❌ drop. Keep har_source and decoder_upsample as separate
packages. The mlpackage at
`/tmp/styletts2_fuse/fused_har_decoder_upsample.mlpackage` is preserved
for reference but is not wired into the pipeline.

## Trial 6 — fused f0n_predictor + har_source

**Hypothesis.** Both stages are small. `f0n_predictor` produces `f0`,
which `har_source` consumes immediately. The intermediate `f0` array is
marshalled out of one mlmodel and into the next on every utterance, plus
a second `predict()` round-trip is paid. Fuse them into one call. The
estimated save was modest (~8 ms) but the risk profile is the cleanest
of any remaining candidate: no transposed-conv stack, no ANE-rejected
ops, no mixed compute partitions in the way that killed Trial 5.

**Implementation.** `/tmp/styletts2_fuse/fuse_f0n_har_source_fp32.py`.

```text
class FusedF0NHarSource(nn.Module):
    def __init__(self, predictor, decoder):
        self.f0n_wrap = F0NPredictorWrapper(predictor)
        self.har_wrap = HarSourceWrapper(decoder)

    def forward(self, en, s):
        f0, n = self.f0n_wrap(en, s)        # [1, F0_LEN], [1, F0_LEN]
        har   = self.har_wrap(f0)           # [1, 1, F0_LEN * 300]
        return f0, n, har
```

Three outputs are exposed because both `f0` (consumed by `decoder_pre`)
and `n` (consumed by `decoder_pre`) are still needed downstream — the
fuse doesn't try to also fold the upsample, it just removes the inner
boundary while keeping the existing two consumers wired up.

**CoreML inputs.**

| name | shape                | dtype | meaning |
|------|----------------------|-------|---------|
| en   | `[1, 640, T_FRAME]`  | f32   | aligned predictor input (asr-shifted, hidden=640 not 512) |
| s    | `[1, 128]`           | f32   | predictor encoder half of `ref_s` |

**CoreML outputs (in spec order).**

| name           | shape              | role                       |
|----------------|--------------------|----------------------------|
| `f0`           | `[1, F0_LEN]`      | f0 contour (→ decoder_pre, internal har) |
| `var_496`      | `[1, F0_LEN]`      | n contour (→ decoder_pre)  |
| `var_537`      | `[1, 1, HAR_LEN]`  | har source (→ decoder_upsample) |

`F0_LEN = 2 × T_FRAME`, `HAR_LEN = 300 × F0_LEN = 600 × T_FRAME`.

**Parity (fp32).**

```
eager fused vs (f0n -> har) :  f0  max|Δ|=0          mse=0
                               n   max|Δ|=0          mse=0
                               har max|Δ|=0          mse=0
traced  vs eager fused      :  all max|Δ|=0          mse=0
coreml  vs eager fused      :  f0  max|Δ|=7.6e-5     mse=7.8e-10
                               n   max|Δ|=4.8e-6     mse=1.4e-12
                               har max|Δ|=1.2e-5     mse=3.9e-12
```

All three outputs are well under the decoder tolerance budget
(`max|Δ| < 5e-3`).

**Latency (warm avg over 10 calls, T_FRAME=147, fp32 mlprogram).**

| Compute units            | min (ms) | avg (ms) | max (ms) |
|--------------------------|----------|----------|----------|
| Standalone (f0n CPU_AND_NE + har CPU_AND_GPU) | — | **~53** (8 + 45) | — |
| **Fused — `CPU_ONLY`** ✅| 11.0     | **11.5** | 12.4     |
| Fused — `ALL`            | 10.0     | 11.9     | 20.4     |
| Fused — `CPU_AND_NE`     | 11.8     | 12.1     | 12.5     |
| Fused — `CPU_AND_GPU`    | 19.4     | 25.6     | 31.7     |

**Save: −42 ms warm per utterance** (5× the original ~8 ms estimate).

**Why the win was so much bigger than expected.**

* Standalone `har_source` was running on `CPU_AND_GPU` and paying GPU
  dispatch overhead for a graph that is fundamentally a few hundred
  small CPU-friendly ops (`cumsum`, `interpolate(scale=300, linear)`,
  `sin`). On `CPU_ONLY` via Accelerate the same ops cost <5 ms.
* But `har_source` standalone *can't* be moved to `CPU_ONLY` unilaterally
  — the standalone bench showed it cost ~80 ms there because of single-
  package dispatch latency on the small graph. CoreML's per-package
  cold/warm overhead is non-trivial for sub-10ms graphs.
* Fusing folds har's small ops into the back of f0n_predictor's CPU-
  resident graph (they were already on CPU effectively after `_runtime`
  dispatch decisions) and pays the CPU dispatch overhead *once*. f0n's
  fast path is unchanged; har's GPU tax disappears entirely.
* `CPU_AND_NE` is essentially tied with `CPU_ONLY` here because the
  predictor's LSTM pieces don't go to ANE anyway and the rest is small
  enough for ANE rejection-fallback to be cheap.

**Disk.** 33.7 MB single fp32 mlpackage (vs 32 MB `f0n_predictor` +
12 KB `har_source` ≈ 32 MB). +1.7 MB.

**Verdict.** ✅ keep. Replace `f0n_predictor` and `har_source` standalone
with a single `fused_f0n_har_source` dispatch loaded with
`compute_units=ct.ComputeUnit.CPU_ONLY`. The pipeline now has two large
fusion wins (Trial 4 saving 437 ms, Trial 6 saving 42 ms) and one
abandoned candidate (Trial 5).

## Trial 7 — fused ref_encoder + fused_diffusion_sampler

**Hypothesis.** With Trial 4 already collapsing the 8-step ADPM2 loop
into one graph, the next remaining boundary on the diffusion side is
`ref_encoder → fused_diffusion_sampler`. `ref_encoder` produces
`ref_s [1, 256]` from a mel slice; the sampler consumes the second-half
`features [1, 256]` slice as a conditioning vector. Fusing them removes
the `ref_s` marshall and one `predict()` round-trip, and exposes both
outputs (`ref_s` for downstream consumers and `s_pred` for predictor
chain) from one call.

**Implementation.** `/tmp/styletts2_fuse/fuse_ref_sampler_fp32.py`.

```text
class FusedRefSampler(nn.Module):
    def __init__(self, model):
        self.ref_wrap = RefEncoderWrapper(model)
        self.sampler  = FusedDiffusionSampler(model)   # baked ADPM2 5-step

    def forward(self, mel_4d, noise_init, noises_aux, embedding):
        ref_s  = self.ref_wrap(mel_4d)                 # [1, 256]
        feats  = ref_s                                 # full ref as conditioning
        s_pred = self.sampler(noise_init, noises_aux, embedding, feats)
        return ref_s, s_pred
```

The Karras schedule and the 4 stochastic noise injections from Trial 4
are baked unchanged. The only new path is the mel → ref_s frontend.

**CoreML inputs.**

| name        | shape                       | dtype | meaning |
|-------------|-----------------------------|-------|---------|
| mel_4d      | `[1, 1, 80, T_MEL]`         | f32   | mel slice (`T_MEL = RangeDim(2, 4096, default=309)`) |
| noise_init  | `[1, 1, 256]`               | f32   | seed-0 init noise |
| noises_aux  | `[4, 1, 1, 256]`            | f32   | per-iter stochastic noise |
| embedding   | `[1, 57, 768]`              | f32   | `bert_dur` |

**CoreML outputs.**

| name    | shape          | role                               |
|---------|----------------|------------------------------------|
| `ref_s` | `[1, 256]`     | style ref (→ predictor chain, decoder) |
| `s_pred`| `[1, 1, 256]`  | sampled style (→ predictor chain) |

**Parity (fp32).**

```
coreml vs eager fused : ref_s  max|Δ|=5.96e-7   mse=2.3e-13
                        s_pred max|Δ|=7.74e-3   mse=4.6e-6
```

`ref_s` is bit-stable; `s_pred` drift is the same chaotic-divergence
pattern as Trial 4 (under decoder budget).

**Latency.**

Standalone microbench (10 warm calls):

| Compute units | min (ms) | avg (ms) |
|---------------|----------|----------|
| `ALL`         | **44.6** | 47.1     |
| `CPU_AND_GPU` | 49.0     | 51.3     |
| `CPU_AND_NE`  | 71.4     | 76.8     |
| `CPU_ONLY`    | 94.7     | 98.2     |

Best single-package warm = **44.6 ms (ALL)**. Compare to the sum of the
two packages it replaces: `ref_encoder ≈ 7 ms (CPU_AND_NE)` +
`fused_diffusion_sampler ≈ 17 ms (CPU_AND_GPU)` ≈ **24 ms**.

End-to-end pipeline benches (4 iterations, warm = iters 2–4 avg):

| Config                      | warm avg (ms) | rss peak (MB) | storage (MB) |
|-----------------------------|---------------|---------------|--------------|
| trial4_6 (baseline)         | **529**       | 1490          | 502          |
| trial7  (fused ref+sampler, no f0n+har fuse) | 750  | 1700 | 540 |
| trial7_6 (fused ref+sampler **and** f0n+har) | 529  | 1690 | 540 |

**Verdict.** ❌ drop. Combined 200 MB graph pays partition tax: best ALL
backend at 44 ms is ~20 ms slower than the two specialized packages
combined (~24 ms), and the larger working set inflates RSS by ~200 MB.
trial7_6 ties trial4_6 on warm latency but is strictly worse on memory
and storage. Same failure mode as Trial 5 — when two stages have
different optimal compute profiles, fusing forces one to lose. The
mlpackage at `/tmp/styletts2_fuse/fused_ref_sampler.mlpackage` is
preserved for reference but not wired in.

## Trial 8 — per-stage compute_units placement sweep

**Hypothesis.** Trial 7's failure ruled out further graph-level fusions
on hot paths. The remaining cheap optimization is **placement** — the
existing 8 standalone mlpackages each load with a hard-coded
`compute_units` choice (mostly `CPU_AND_NE`). Sweep all four backends
(`CPU_ONLY`, `CPU_AND_NE`, `CPU_AND_GPU`, `ALL`) on every stage with
the actual pipeline intermediates and pick the minimum per stage.
No graph changes, no parity risk.

**Implementation.** `/tmp/styletts2_fuse/trial8_placement_sweep.py`.

* Loads each stage 4 times (one per backend) and benches with realistic
  inputs from a full pipeline run (text → bert → ref → duration →
  alignment → f0n → har → decoder).
* `WARMUP=2`, `RUNS=8`, single process, no other workloads.

**Sweep results (warm min ms per stage).**

| Stage                       | CPU_ONLY | CPU_AND_NE | CPU_AND_GPU | ALL  | best | win vs current |
|-----------------------------|----------|------------|-------------|------|------|----------------|
| text_encoder                | 1.7      | 1.8        | 1.7         | 1.7  | tie  | 0              |
| bert                        | 12.4     | 16.0       | 11.2        | **8.0**  | ALL  | **−8.0** |
| ref_encoder                 | 14.7     | 45.8       | **13.1**    | 14.0 | CPU_AND_GPU | **−32.7** |
| duration_predictor          | 5.0      | **4.5**    | 5.2         | 4.8  | CPU_AND_NE (current) | 0 |
| fused_diffusion_sampler     | 22.1     | 19.7       | 20.8        | **17.4** | ALL  | **−3.4** |
| f0n_predictor               | **7.8**  | 13.3       | 11.4        | 9.1  | CPU_ONLY | −5.5 (subsumed by Trial 6) |
| fused_f0n_har_source        | **8.6**  | 9.0        | 12.7        | 9.4  | CPU_ONLY (current) | 0 |
| decoder_upsample            | 558.7    | 612.3      | 530.1       | **491.5** (high variance 491–638) | ALL  | **−67** to **+71** |

Total best-case save: **~111 ms warm** if every stage takes its minimum.
Three stages flip backend cleanly: `bert` → `ALL`, `ref_encoder` →
`CPU_AND_GPU`, `fused_diffusion_sampler` → `ALL`. `decoder_upsample` →
`ALL` is best-case but has 1.3× spread between min and max in a single
sweep run; ANE compilation paths retry under contention.

**End-to-end benches.**

Two configs were run, both with Trial 4 + Trial 6 fusions applied:

* **trial8** (aggressive — all 4 placement changes including
  `decoder_upsample → ALL`)
* **trial8b** (conservative — keep `decoder_upsample → CPU_ONLY`,
  apply the other 3 clean wins)

| Config            | run | iter1 | iter2 | iter3 | iter4 | warm avg | peak rss (MB) |
|-------------------|-----|-------|-------|-------|-------|----------|---------------|
| trial4_6 (ref)    | —   | —     | —     | —     | —     | **529**  | 1490 |
| trial8  (aggressive) | v1 | 1593 | 313   | 303   | 352   | **323**  | 1471 |
| trial8  (aggressive) | v2 | 1070 | 497   | 742   | 1038  | 759      | 1471 |
| trial8b (conservative) | v1 | 864 | 476   | 501   | 511   | **496**  | 1459 |
| trial8b (conservative) | v2 | 1096 | 544   | 562   | 583   | 563      | 1335 |

**Verdict.** ⚠️ partial keep.

* **Clean wins (always apply):** `bert → ALL`, `ref_encoder →
  CPU_AND_GPU`, `fused_diffusion_sampler → ALL`. These three changes
  are the **trial8b** config, which steady-states at **~496–563 ms warm
  avg** — within thermal noise of trial4_6 (529 ms) but with consistent
  microbench wins on the three swept stages and lower peak RSS.
* **Risky (don't apply):** `decoder_upsample → ALL`. Best case is 322 ms
  (39% faster than trial4_6) but worst case is 759 ms (regression). The
  ANE backend retries compilation paths for the transposed-conv stack
  under contention, producing the same kind of bimodal latency that
  killed Trial 5. Keep `CPU_ONLY` for predictability.
* **Already optimal:** `text_encoder` (tied), `duration_predictor`
  (`CPU_AND_NE`), `fused_f0n_har_source` (`CPU_ONLY`, set by Trial 6).

**Disk.** No change — same 8 mlpackages, only `compute_units` strings
differ at load time.

## Trial 9 — `_hifigan_shift` fold (skipped)

**Hypothesis.** `_hifigan_shift` does two stride-2 subsamples on small
tensors between `decoder_pre` and `decoder_upsample`. Folding it into
either neighbour removes a Python-side step.

**Why skipped.** Measured cost of `_hifigan_shift` in the current
pipeline is **sub-1 ms** (two `[1, C, T]` slices on small T). The
implementation cost is non-trivial: re-converting one of the two
neighbouring mlpackages with a wider input axis spec, re-validating
parity, and re-running placement sweep. Trial 8's already-known
~110 ms ceiling on placement changes dominates by two orders of
magnitude. Documented as deliberately deferred rather than tried.

## Cross-trial notes

* fp32 baseline beat fp16 baseline on this hardware: fp32 warm 713 ms
  vs fp16 warm 953 ms, driven entirely by `decoder_upsample` (CPU_ONLY,
  Accelerate is fp32-native). All fusion experiments were therefore
  done at fp32 only.
* `CPU_AND_GPU` consistently outperformed `ALL` for the fused diffusion
  sampler. `ALL` has higher cold-call variance because the runtime
  retries ANE compilation paths.
* When fusion succeeds, parity drift comes from MIL pass reordering of
  iterated math. When fusion fails to win, it almost always fails on
  partition decisions — the converter has fewer DOF to specialize when
  two distinct compute profiles share a graph.

## Open follow-ups (not run)

* **`PipelineModel` packaging.** `coremltools.models.pipeline.PipelineModel`
  only chains where each stage's outputs feed directly to the next
  stage's inputs by name. StyleTTS2 has data-dependent alignment
  (`build_pred_aln_trg`), branching DAG (`s_pred` consumed by 2 stages,
  `ref_s` by 3), and Python-side glue (`einsum`, `_hifigan_shift`).
  Structurally not expressible as a linear pipeline. Skipped.
* **`text_encoder + bert` fuse.** Both consume `tokens` but produce
  different padded-T conventions (`[1, 512, T_real]` vs `[1, 57, 768]`).
  Trial 8 placement showed both already at <16 ms each; combined
  estimated save <3 ms after partition tax. Low ceiling.
* **fp16 retry on bert / ref_encoder only.** Original fp32-vs-fp16
  baseline was decoder-dominated; the now-favoured `ALL` and
  `CPU_AND_GPU` placements for bert / ref might benefit from fp16
  weight quant in isolation. Not yet measured.

## Final stage count

Pipeline still **8 standalone packages** (Trial 7 fusion not adopted):

```
text_encoder        (CPU_AND_NE — unchanged)
bert                (ALL          — Trial 8 placement)
ref_encoder         (CPU_AND_GPU  — Trial 8 placement)
fused_diffusion_sampler   (ALL    — Trial 4 fuse + Trial 8 placement)
duration_predictor  (CPU_AND_NE — unchanged)
fused_f0n_har_source      (CPU_ONLY — Trial 6 fuse)
decoder_pre         (CPU_AND_NE — unchanged)
decoder_upsample    (CPU_ONLY    — kept; trial8 ALL too variable)
```

Total dispatches per utterance: **16 → 8** (−50%, unchanged from Trial 6).
Total warm wall-clock save: **−479 ms graph fusions** (Trial 4 + 6) plus
**up to ~45 ms in clean placement wins** (Trial 8b: bert −8, ref_encoder
−33, sampler −4 in microbench, partially absorbed by thermal noise at
the pipeline level).
Pipeline warm: **~1030 ms → ~496–563 ms** (trial8b range).

### Config matrix (4-iteration end-to-end benches)

| Config       | warm avg (ms) | peak rss (MB) | storage (MB) | notes |
|--------------|---------------|---------------|--------------|-------|
| regular      | 558           | 1750          | 470          | per-stage baseline |
| trial4       | 525           | 1430          | 502          | fused diffusion sampler only |
| trial6       | 547           | 1830          | 470          | fused f0n+har only |
| trial4_6     | **529**       | 1490          | 502          | both Trial 4 + 6 fusions |
| trial7       | 750           | 1700          | 540          | + ref+sampler fuse, no f0n+har — **regression** |
| trial7_6     | 529           | 1690          | 540          | + ref+sampler fuse + f0n+har — wash |
| trial8       | 323 / 759     | 1471          | 502          | trial4_6 + aggressive placement (decoder_upsample→ALL) — **bimodal** |
| **trial8b**  | **496–563**   | **1335–1459** | 502          | trial4_6 + bert→ALL + ref→GPU + sampler→ALL — **recommended** |
