# Magpie conversion-only optimization options

Impact-ranked menu of remaining levers on the **shipping Magpie 357M
multilingual checkpoint**, after Trials 1 + 3 already shipped (−978 ms,
−161 ms TTFA respectively per [`PERF.md`](PERF.md)) and every other
graph-level lever in `PERF.md` was exhausted.

## Top-line state (current)

**All Tier 1 options closed under the no-retraining + no-hardware-upgrade
constraints. The 830 ms warm TTFA ceiling is the operating point.**
Tier 2 small wins (Options 4, 5, 6) remain available for future PRs;
ceiling movement requires retraining (deferred Trial 8 QAT or NanoCodec
swap with glue training) — see *Recommended next steps if constraints
relax* below.

| # | Option | Tier | Status | Reference |
|---|---|---|---|---|
| 1 | MIL graph rewriting (tail-fp16) | 1 | 🔴 NO-GO | Trial 11 (`PERF.md`) |
| 2 | coremltools 9.x upgrade | 1 | 🔴 NO-GO | Probe 2 (this doc) |
| 3 | Vocoder swap (Vocos / iSTFTNet) | 1 | 🔴 NO-GO | Probe 3 (this doc) |
| 4 | `local_transformer` body/tail split | 2 | ⏸ Available | est. 10–20 ms TTFA |
| 5 | MLPipeline fusion | 2 | ⏸ Parked | Trial 9 low-EV |
| 6 | Per-stage CU re-audit | 2 | ⏸ Available | LT 55.3 % ANE finding |
| 7 | NanoCodec v3 → v4 (size-only) | 3 | 🔴 NO-GO for default flip | v3/v4 re-bench (`PERF.md`) |
| 8 | QAT / retraining | — | ❌ OOS | #60 hard line |

`🔴` blocked under current constraints; `⏸` available, not pursued in this
PR; `❌` outside scope. Each Tier 1 NO-GO has a `Result:` line below with
the verbatim verdict and pointer to the source-of-truth doc section.

## Constraints

- **No retraining.** No NeMo or HuggingFace fine-tunes, no QAT, no
  distillation runs.
- **No hardware upgrade.** Target hardware stays Apple M2 / M3 / M4 +
  iPhone 15 Pro/16 Pro class. iOS 17 deployment target.
- **No quality regression.** End-to-end audio must remain audibly
  indistinguishable from the current `nanocodec_decoder_v3` fp32 path
  (Phase F gold).
- **Conversion-only.** Acceptable changes: `convert_*.py` exporters,
  CoreML ops / MIL passes, mlpackage splits, Swift `MagpieModelStore`
  wiring, dependency upgrades.

## Current state

- **Warm TTFA: ~830 ms on M2** (3-word "Hello from Magpie.", warm path,
  release build, seed 42). See [`PERF.md`](PERF.md) §TL;DR.
- **Implied Swift overhead: ~204 ms (24 %)** of TTFA. CoreML compute
  total is ~626 ms (PERF.md §TTFA breakdown):

  | Stage | Per-call | Calls | Total | ANE % |
  |---|---|---|---|---|
  | text_encoder | 12.4 ms | 1 | 12 ms | 98.1 |
  | decoder_prefill | 17.1 ms | 1 | 17 ms | 93.9 |
  | decoder_step | 15.7 ms | 24 | 377 ms | 97.3 |
  | local_transformer | 1.6 ms | 24 | 38 ms | **55.3**¹ |
  | nanocodec_v3 | 182 ms | 1 | 182 ms | **0 (CPU)** |
  | **Total** | | | **626 ms** | |

  ¹ PERF.md:62 / 153 record 73.9 %; live `coreml-cli --fallback` on the
  shipping `local_transformer.mlmodelc` reports **55.3 % (228 / 412 ops)**.
  PERF.md is stale on this row; mechanical fix is queued for the
  BASELINE_FP32 follow-up.

- **Codec stage dominates.** `nanocodec_v3` at 182 ms / 0 % ANE is
  29 % of the compute total. Anything that lowers the codec wall is
  the only realistic path to lowering the 830 ms TTFA ceiling.

## Tier 1 — could lower the 830 ms TTFA ceiling

### Option 1 — MIL post-conversion graph rewriting 🔴 NO-GO (Trial 11)

**What.** Walk the converted MIL graph after `ct.convert()` and
explicitly insert `cast_to_fp32` / `cast_to_fp16` at module-named
boundaries (e.g. HiFi-GAN stages 1-2 fp16 ANE, stages 3-5 fp32 CPU).
Sidesteps the broken `coremltools.converters.mil…FP16ComputePrecision`
`op_selector` that Phase F.2b fingered as the structural blocker
(`STATUS.md:391-397`: *"`op.scopes` is not populated when
`FP16ComputePrecision`'s `op_selector` runs in this coremltools version
(8.x). … Per-location filtering at the op_selector level is unavailable.
The only path to per-stage precision control is post-conversion MIL
graph rewriting, which is out of scope for this project."*).

**Impact ceiling.** Phase C+ subgraph probe (`STATUS.md:104-119`)
shows nanocodec stages with `T_out ≤ 8 192` land at 98.8-99.0 % ANE;
stages with `T_out ≥ 10 240` are wholly rejected by ANE. A 5-stage
cumulative-upsample HiFi-GAN at `T_in=24` crosses 16 384 mid-stack —
the first 1-2 stages are ANE-recoverable, the last 3 must stay fp32 CPU
(per Phase F.2 audibility envelope). Optimistic codec saving:
**~30-50 ms** off the 182 ms wall. Realistic with boundary cost:
**~15-30 ms TTFA** (~2-4 %).

**Effort.** **High.** Requires writing a coremltools MIL pass that
walks the program, identifies HiFi-GAN-named ops by `op.scopes` (which
*is* populated post-conversion, unlike during op_selector pre-pass),
inserts the precision-cast pairs, validates numerics against the
fp32-only reference. Multi-day implementation + Phase F.2-style numeric
sweep.

**Probe 1 verdict** below determines technical feasibility.

**Result.** Probe 1 confirmed the MIL pass is fully feasible (`op.scopes`
populated post-conversion; `mb.cast` + `replace_uses_of_var_after_op` is
the canonical pattern from `add_fp16_cast`). **Trial 11** then ran the
v1 (smallest possible region: `post_conv` + `out_activation`, 3 ops):
**SNR 38.79 dB on 5 random-token utterances — 9 dB below Phase F.2's
48 dB audibility threshold — and 0 % ANE residency** (planner refused
to take a 3-op fp16 island anchored to a fp32 output). Both halves of
the hypothesis fail simultaneously. v1 is the smallest possible region;
v2/v3 skipped per the stopping rule. The MIL rewrite works; the
audibility envelope rules out every fp16 island that would buy ANE.
See `PERF.md` §"Trial 11 — Tail-fp16 mixed-precision (Option 1) ✗ DEAD".

### Option 2 — coremltools 9.x upgrade 🔴 NO-GO (Probe 2)

**What.** Phase F.2b was run on coremltools 8.x. The repo currently
ships at coremltools 9.0 (per BASELINE_FP32 metadata captures). 9.x's
release notes need checking for fixes to: `op_selector` scope metadata
exposure, `FP16ComputePrecision` boundary-cast insertion, and the MIL
pass infrastructure. **If 9.x already inserts the missing boundary
casts**, Phase F.2's per-location precision sweep can be re-run cleanly
without writing a custom MIL pass.

**Impact ceiling.** Same numeric ceiling as Option 1 — 9.x would just
make Option 1 trivial instead of high-effort. Latency impact identical.

**Effort.** **Low.** `pip install coremltools>=9` + rerun Phase F.2.

**Probe 2 verdict** below determines if 9.x is even released and
whether the relevant fix is in.

**Result.** Probe 2: **9.0 ships byte-identical pass code to 8.3.**
`AbstractQuantizationPass.apply` and `CastTypeQuantization.transform_op`
unchanged; `mil/scope.py` zero commits since 8.1. No upstream issue or
PR addresses either symptom; no in-flight 9.1 fix to wait on. The bug
has not been filed upstream. Option 1's MIL pass remains the only
viable per-location precision route — and Trial 11 closed that. See
"Probe 2 — coremltools 9.x status" below.

### Option 3 — Vocoder swap (Vocos / iSTFTNet) 🔴 NO-GO (Probe 3)

**What.** Replace `nanocodec_decoder_v3` end-to-end with a different
vocoder graph that doesn't carry NanoCodec's structural ANE blockers
(96 Snake instances, 92 transposed-conv upsamples, dilated convs that
hit the `W ≤ 16 384` ceiling per Phase C+). Public ANE-friendly
candidates: **Vocos** (Conv1d + iSTFT, no transposed convs), **iSTFTNet**
(similar pattern). Both can in principle land 80-99 % ANE at fp16.

**Impact ceiling.** Eliminates most of the 182 ms / 29 %-of-TTFA
codec wall. **~100-150 ms TTFA savings** if a compatible checkpoint
runs at fp16 on ANE.

**Effort.** **Zero if a NanoCodec-input pretrained Vocos checkpoint
exists.** **Weeks (training) otherwise** — Magpie emits 8 codebooks ×
2024 levels at 24 frames/24576 samples; Vocos training requires either
a glue-trained projection or a full retrain on Magpie-NanoCodec data.

**Constraint check.** A glue-trained Vocos counts as retraining, which
violates the no-retraining constraint. Only a **pretrained, drop-in
checkpoint** keeps Option 3 in scope.

**Probe 3 verdict** below determines availability.

**Result.** Probe 3: **zero pretrained vocoders consume NeMo NanoCodec
FSQ codes.** The "nano-codec" hits on HF are name-collision distractors
(16 kHz DAC RVQ); `nineninesix-ai/nanocodec-mlx` is just an MLX port of
the same NeMo HiFi-GAN decoder. Vocos's input head is a learned
`codebook_weights` Parameter hard-tied to EnCodec's centroids — any
substitution requires glue training (~500–3 000 A100-hours) which
violates the no-retraining constraint. Mel-intermediate path
(NanoCodec → mel → stock Vocos) also requires training. See "Probe 3 —
Pretrained Vocos / iSTFTNet for NanoCodec" below.

## Tier 2 — small / marginal wins (1-3 % of TTFA)

### Option 4 — `local_transformer` body / tail split

Live `coreml-cli --fallback` on the shipping LT reports 55.3 % ANE
(228 / 412 ops). Of the 184 CPU ops, **104 are "ANE supported but
scheduler chose CPU"** — exactly the laishere-derived "scheduler
silently spills ANE-eligible ops" pattern. Trigger here is the int32
sampling tail (cumsum × 16, topk × 8, equal × 15, int32 cast × 24)
forcing partition decisions across the whole graph.

**Fix.** Split the LT into `body.mlpackage` (fp16 transformer block,
ANE) + `tail.mlpackage` (cumsum/topk/sample, CPU int32). Anchor trick
not needed — the int32 tail is the structural cut. Body could land at
~80 % ANE on its own (104 + 228) / 412 ≈ 80.6 %.

**Impact.** ~0.4-0.8 ms saved per LT call × 24 calls/first-chunk =
**10-20 ms TTFA**, ~1 % of the 1547 ms total.

**Risk.** Partition boundary cost. Trial 4a's lesson (`PERF.md:385-399`):
*"partition boundary cost dominates wall-clock — every sampler block
forces a round-trip through ANE→CPU→ANE."* The second per-iter dispatch
in Option 4 may eat the savings.

**Effort.** Medium. Two new traceable wrappers + Swift wiring update in
`MagpieFusedLocalSampler.swift`. **Worth a quick spike before any
serious commitment.**

**Greenlight cost.** ⏸ Available. Roughly 0.5–1 day to build the spike
(re-trace + split + bench). Greenlight if (a) someone wants the ~1 %
TTFA win on long-input cases, or (b) we're already touching
`MagpieFusedLocalSampler.swift` for another reason and the wiring cost
is amortized. Not pursued in this PR.

### Option 5 — MLPipeline fusion (Trial 9 revisit)

`PERF.md:515` flagged Trial 9 as "low-EV". The intuition was correct —
fusing text_encoder + decoder_prefill into one `MLPipeline` saves at
most 1 dispatch round-trip on the cold path. Order-of-magnitude estimate:
~5-10 ms TTFA. **Bounded by prefill's 24-output KV-cache contract**
which makes the fused stage awkward to package and wires the prefill's
93.9 % ANE outcome through whatever the fused planner picks.

**Impact.** ~5-10 ms (~0.5-1 %).
**Effort.** Low.
**Recommendation.** Park unless a probe surfaces a 50 + ms scheduler
finding that changes the math.

**Greenlight cost.** ⏸ Parked. Greenlight only if a future scheduler
investigation finds a per-dispatch overhead > ~30 ms on cold paths
that MLPipeline fusion could amortize. Not actionable on current
M2 + macOS 26.5 numbers.

### Option 6 — Per-stage compute-unit override audit

`MagpieModelStore.swift` already routes each stage optimally per
`PERF.md:75-86`. Worth re-auditing on macOS 26.5 + iOS 18 / 26 since
CoreML scheduler heuristics evolve and `coreml-cli --fallback` numbers
may drift between OS minor versions.

**Impact ceiling.** 0 ms if current routing is still correct (most
likely); up to 10-20 ms if any stage was silently mis-routed by an OS
heuristic change.
**Effort.** Low. One `coreml-cli` sweep across all 4 CU policies per
shipping mlmodelc on each target OS. Hooks into the BASELINE_FP32
probe harness directly.

**Greenlight cost.** ⏸ Available. ~30 minutes to run + interpret on a
new OS minor. Worth running before each iOS major release as a
regression check. The live sweep also turned up the **LT 55.3 % ANE
finding** (PERF.md:62 / 153 records 73.9 %, stale by ~18 ppt) — that
mechanical fix lands when this re-audit ships. Not pursued in this PR.

## Tier 3 — investigated, ship-blocked

### Option 7 — NanoCodec v3 → v4 (size-only) 🔴 NO-GO for default flip

**What.** v4 is the palette-quantized fp32 NanoCodec variant from
Trial 10a — same audibility envelope as v3, **4× smaller on disk**
(121 MB → 31 MB). #60 Track 1 raised the question of flipping the
`MagpieModelStore` default to v4 if it ships a TTFA win along with
the size advantage.

**Investigation.** User A/B-listened the 5 fixture pairs at
`nanocodec_experiments/results/ab_v3_v4/utt0{1..5}_{v3,v4}.wav` (now
gitignored — re-generate via the harness) and **confirmed v4 is
acoustically transparent vs v3**. Acoustic question settled.

Two follow-up benches (`bench_v3_v4.py` warm-latency, `bench_rss.py`
RSS) settled the perf question:

- **Warm latency:** v4 is **+47 ms / +40 % slower** than v3 at the
  production CU (`.cpuOnly`, v3=116.89 ms, v4=163.62 ms). Cold-load
  delta +84 ms (palette dequant cost). Reconciles with Trial 10a's
  original "v4 slower on every CU on M2" finding.
- **RSS:** at production CU, **v4 has zero RAM savings.** Both add
  ~537 MB to RSS at steady state; v4 actually +1 MB. The 90 MB
  on-disk advantage is purely a download / asset-bundle benefit;
  runtime RSS is identical because palette weights expand to fp32
  at MLModel load.

**Result.** **Don't flip the default.** v4 stays available as a 4×
smaller artifact for download-size-constrained scenarios (HF / iOS
app-bundle), but the consumer accepts ~47 ms-per-call cost
consciously. End-to-end utterance cost: 20 sliding windows × 47 ms ≈
940 ms slower per 8 s utterance. **No FluidAudio Swift change
needed.** See `PERF.md` §"Trial 10a re-bench — v4 confirmed slower
than v3 (post-ABX) ✗ KEEP v3" for full numbers.

## Out of scope (constraint hard-lines)

### Option 8 — QAT / retraining

Two distinct sub-options that share the no-retraining hard line from
mobius #60:

- **QAT / activation-calibrated int8 on `decoder_step`** — Trial 8 in
  `PERF.md:516`, deferred. Multi-day NeMo training + careful EOS
  preservation work; estimated **−400 ms TTFA** if landed.
- **Architecture-level retraining** — LiteMagpie / distilled smaller
  body / pruned attention / alternate codec head. Multi-week training
  pipeline.

Both violate the no-retraining constraint and are gated on that
constraint relaxing. See *Recommended next steps if constraints relax*
below.

## Decision summary

**All Tier 1 options closed.** Probes 2 and 3 closed Options 2 and 3
as NO-GO; Trial 11 (the tail-fp16 follow-up to Probe 1) closed Option 1
as NO-GO. **Tier 3 Option 7 closed** as NO-GO for default flip via the
post-ABX v3/v4 latency + RSS re-bench. The 830 ms warm TTFA ceiling is
the operating point under the no-retraining + no-hardware-upgrade
constraints.

| # | Option | Impact ceiling | Effort | Status |
|---|---|---|---|---|
| 1 | MIL post-conversion graph rewriting (tail-fp16) | ~15-30 ms (~2-4 %) | 1–2 person-days | 🔴 NO-GO (Trial 11) |
| 2 | coremltools 9.x upgrade | same as #1 | Low | 🔴 NO-GO (Probe 2) |
| 3 | Vocoder swap (Vocos / iSTFTNet) | ~100-150 ms (~12-18 %) | 0 if pretrained, weeks otherwise | 🔴 NO-GO (Probe 3) |
| 4 | `local_transformer` body/tail split | ~10-20 ms (~1 %) | Medium | ⏸ Available, not pursued in this PR |
| 5 | MLPipeline fusion | ~5-10 ms (~0.5-1 %) | Low | ⏸ Parked |
| 6 | Compute-unit override audit | 0-20 ms (+ LT 55.3 % stale-doc fix) | Low | ⏸ Roll into BASELINE_FP32 follow-up |
| 7 | NanoCodec v3 → v4 (size-only) | size: 121 MB → 31 MB; latency: −47 ms / call (regression) | n/a | 🔴 NO-GO for default flip; v4 stays available for download-size scenarios |
| 8 | QAT / retraining (Trial 8 + LiteMagpie etc.) | est. −400 ms (QAT) to open-ended (retrain) | Multi-day to multi-week | ❌ Out of scope (#60 hard line) |

## Recommended next steps if constraints relax

The 830 ms ceiling is structural under the **current** constraints
(no retraining, no hardware upgrade, no quality regression). Three
relaxations would reopen ceiling movement, in increasing cost order:

1. **Cheapest — M3/M4 retest.** All numbers in `PERF.md` and
   `BASELINE_FP32.md` are M2 / macOS 26.5. The ANE compute cores
   widened M2 → M3 → M4; the nanocodec dilated-conv `W ≤ 16 384`
   ceiling and the fp16 audibility envelope are graph-level facts
   that don't change with hardware, but per-stage ANE residency,
   warm latency, and the planner's CU choices may shift enough to
   reopen Tier 1 questions. Effort: ~half a day on each new chip
   (rerun the BASELINE_FP32 + Trial 11 + v3/v4 benches via the
   existing harnesses). Loosens the no-hardware-upgrade constraint.

2. **Highest-EV unblock — Trial 8 QAT** (`PERF.md:516`). NeMo
   activation-calibrated int8 on `decoder_step`, with careful EOS
   preservation to dodge Trial 2's runaway-EOS failure mode.
   Estimated **−400 ms TTFA** per the original Trial 8 entry —
   roughly 48 % of the current ceiling. Effort: hours of GPU + a
   few days of code (NeMo training loop + on-device parity sweep).
   Loosens the no-retraining constraint for **one model only**
   (`decoder_step`); the rest of the pipeline stays as-shipped.

3. **Larger lift — Vocoder swap with glue training.** Train a
   NanoCodec-input Vocos / iSTFTNet head from scratch against the
   shipping NeMo NanoCodec encoder's output (Probe 3 estimate:
   ~500–3 000 A100-hours depending on multi-language quality
   target). Gets ANE-residency on the codec stage, eliminates the
   `W ≤ 16 384` ceiling and the fp32-weight-required audibility
   bind. Effort: weeks. Loosens the no-retraining constraint
   broadly. Probably only worth it if QAT ships and there's still
   a nanocodec-shaped hole in the budget.

If none of those relaxations are funded, the next move is **Tier 2
quick spikes** (Options 4 + 6) — small wins under existing
constraints. Each is ~1 day of work for ~1–2 % TTFA, net of
boundary-cost risk for Option 4.

## Cross-references

- `PERF.md` §TL;DR (current TTFA stack, shipped levers) — line 10.
- `PERF.md` Trial 8 (QAT deferred) — line 516.
- `PERF.md` Trial 9 (`MLPipeline` low-EV) — line 515.
- `PERF.md` Trial 10c (NanoCodec mixed-precision DEAD via Phase F.2)
  — line 567.
- `PERF.md` Trial 10a re-bench (v3 vs v4 latency + RSS post-ABX) —
  closes Option 7.
- `PERF.md` Trial 11 (tail-fp16 MIL rewriting) — closes Option 1.
- `BASELINE_FP32.md` — 5-stage fp32-vs-fp16 parity table.
- `PERF.md` §Repro and §Lever ranking — line 642 onward.
- `nanocodec_experiments/results/STATUS.md` Phase F.2 / F.2b — full
  audibility envelope and `op_selector` failure analysis.
- mobius issue **#60** — Magpie multilingual TTS + ANE follow-ups.

## Probes

The Tier 1 verdicts depend on three independent probes (investigation
only, no conversions). Each lands as a separate commit on
`feat/magpie-options-probe` so the trail is clear.

<!-- Probe sections appended below as separate commits. -->

### Probe 2 — coremltools 9.x status

**One-line verdict: NO-GO.** Upgrading 8.x → 9.0 does not fix the
Phase F.2b per-location precision bug. The path post-conversion MIL
rewriting (Option 1) is still the only supported route.

**Release status.** Latest stable: **coremltools 9.0** (2025-11-10).
No 9.0.x patch release. 31 post-9.0 commits on `main` (through
2026-05-07); next stable would be 9.1 — no in-flight fix to wait on.

**Changelog grep.** The 9.0 release body and 9.0b1 (2025-07-28) full
text contain **zero hits** for any of: `op_selector`,
`FP16ComputePrecision`, `mixed precision`, `boundary`,
`cast_to_fp16` / `cast_to_fp32`, `op.scopes`, `TORCHSCRIPT_MODULE_NAME`,
`per-op`, `per-location`. The 9.0 release body verbatim:

> Compare to 8.3.0 (including features from 9.0b1)
> - Added Python 3.13 support.
> - Bug fix related to upsample_bilinear.
> - Fixed the lowering of broadcast_to for symbolic and dynamic shapes.
> - Support for model input/output with int8 dtype.
> - Ability to read and write model state.
> - iOS26/macOS26/watchOS26/tvOS26 deployment targets.
> - AllowLowPrecisionAccumulationOnGPU optimization hint.
> - Support for PyTorch 2.7 and ExecuTorch 0.5.
> - Additional metadata automatically added to converted models.
> - Optimize im2col PyTorch operation.
> - Various other bug fixes, enhancements, clean ups and optimizations.

**Source-diff cross-check** via `gh api compare/8.3...9.0` (35 commits):

- `coremltools/converters/mil/mil/passes/defs/quantization.py` —
  the relevant `apply` and `transform_op` methods on
  `AbstractQuantizationPass` and `CastTypeQuantization` are
  **byte-identical** between 8.3 and 9.0. The only diff is an
  unrelated `add_int16_cast.should_transform_op` branch detecting
  preceding `uint16 → int32` casts.
- `coremltools/converters/mil/mil/scope.py` — **zero commits since
  8.1** (last touch 2024-11-20). Scope-during-pass semantics
  unchanged from 8.x.
- Post-9.0: PR #2669 ("Fix fp16 NaN from out-of-range fp32 tensor
  sentinels") is the only fp16-adjacent fix. It patches
  `add_fp16_cast.fp16_overflow`'s constant-range check for
  `torch.finfo(fp32).min` from HuggingFace `eager` attention masks.
  Different bug.

**Issue tracker.** `gh search issues --repo apple/coremltools` for
each of `"FP16ComputePrecision op_selector"`, `"boundary cast mixed
precision"`, `"op scopes"`, `"per-op precision"`,
`"selective fp16"`, `"skip cast"`, `"kept fp32"`,
`"compute_precision per op"`, `"fp16 fp32 cast op_selector"` →
**all zero hits**. Broader `"FP16ComputePrecision"` returns 9
unrelated issues (slice_by_index mask, LayerNorm crash, NaN on
transformer, AvgPool calc errors, etc.). No upstream report of the
Magpie F.2b symptom. **The bug has not been filed.**

**Implication.**

1. `AbstractQuantizationPass.apply` invokes `self.op_selector(op)`
   at the same call site as 8.3 with the same `op` object — if
   `op.scopes[TORCHSCRIPT_MODULE_NAME]` came back empty in 8.x at
   op_selector time, it comes back empty in 9.0.
2. `CastTypeQuantization.transform_op` is unchanged — whatever the
   runtime boundary-cast behavior was on 8.x, it is the same on 9.0.

There is no in-flight upstream fix to wait on. Option 2's "low-effort
upgrade" path is closed. **Option 1 (write the MIL post-pass) inherits
the full burden** — Probe 1's verdict on technical feasibility
becomes the gating factor.

### Probe 1 — MIL post-conversion graph rewriting

**One-line verdict: NO-GO on the early-stages-fp16 split**;
**conditional GO on a 1–2 person-day tail-fp16 probe** that Phase F.2
did not cover. **The MIL pass itself is fully feasible — the blocker
is numeric, not API.**

**API surface (everything we need is public).**

- `coremltools/converters/mil/mil/passes/`:
  - `pass_registry.py` — `PASS_REGISTRY` singleton, `@register_pass(namespace=…)`
  - `graph_pass.py` — `AbstractGraphPass` (extension point)
  - `helper.py` — `@block_context_manager` (auto-`with block:` for safe insertion)
  - `defs/quantization.py` — `CastTypeQuantization.transform_op` is the
    reference implementation for boundary insertion (`mb.cast` on each
    input, reverse cast on each output, `replace_uses_of_var_after_op`,
    `remove_ops`).
- `coremltools/converters/mil/mil/scope.py` — `ScopeSource.TORCHSCRIPT_MODULE_NAME`,
  `op.scopes: Dict[ScopeSource, List[str]]`. **Scopes ARE populated
  post-conversion** on torch-frontend ops, and `Block._replace_var →
  _copy_scope_info` propagates them to newly-inserted ops automatically.
- The empty-scope behavior Phase F.2b observed almost certainly came from
  `add_fp16_cast` walking ops it had created earlier in the same pass
  (which only carry `COREMLTOOLS_GRAPH_PASS` scopes). For a standalone
  pass that runs on a saved-and-reloaded MIL program, scopes from the
  torch frontend are intact.

**Cast-insertion mechanism** (verbatim from
`mil/passes/defs/quantization.py:273-363`):

1. For each fp32 input var: `casted = mb.cast(x=v, dtype="fp16",
   before_op=op)`.
2. Reconstruct the op with casted inputs: `getattr(mb,
   op.op_type)(**casted_inputs, before_op=op)`.
3. For each output that changed dtype: insert reverse `mb.cast` and
   `block.replace_uses_of_var_after_op(anchor_op=op, old_var=old,
   new_var=cast_back, force_replace=True)`.
4. `block.remove_ops([op])`.

A self-contained ~80-line `PerLocationFP16Cast(AbstractGraphPass)`
sketch is in `/tmp/magpie_probe_1_mil_rewriting.md` (not executed).

**Numeric expectation — why per-location-fp16-on-early-stages is
ruled out by F.2's data without running the pass.**

| Variant (F.2) | SNR vs fp32 | Audible? |
|---|---|---|
| v_full_fp32 | 211 dB | clean ✓ |
| v_convs_fp32 (all convs fp32, all acts fp16) | 48 dB | noisy ✗ |
| v_acts_fp32 (all acts fp32, all convs fp16) | 28 dB | noisy ✗ |
| v_full_fp16 | 27 dB | noisy ✗ |

Per-location-fp16 in early stages is a **strict subset** of both
"convs fp16" *and* "activations fp16" — keeping stages 0–1 at fp16
allows fp16 in *both* op classes inside those stages. F.2 found no
op-class fp16 subset that crosses the audibility threshold. By
sum-of-noise argument, no per-location subset containing the early
stages can either. Best-case SNR bound: 28 dB (the noisier of the
two op-class envelopes), well below v_full_fp32's 211 dB. **No
first-order reason from F.2's data to expect this to clear the
audibility threshold.** Writing the MIL pass to verify a foregone
conclusion is wasted engineering.

**The one configuration F.2 did not cover** is the converse:
**keep stages 0–1 fp32 and let stages 2–4 + post-head go to fp16**.
Noise compounds toward the output, so tail-fp16 *might* be masked
by the absence of further downstream amplification — F.2's data
offers no signal either way. **This is the only conditional-go.**

**Effort breakdown** (if we pursue tail-fp16):

| Step | Cost |
|---|---|
| Write `PerLocationFP16Cast` post-pass | 0.5–1 person-day |
| Wire to existing F.1/F.2 A/B harness (`/tmp/mono_fp16_vs_fp32.py`, `mixed_precision_sweep.py`) | 0.5–1 day |
| **Total to a yes/no audibility verdict** | **1–2 person-days** |

**Even if tail-fp16 sounds clean, the latency win is bounded.**
ANE residency requires *every* op in a placement region to be
fp16-friendly; CPU fallback is triggered by fp32 islands regardless
of where they sit. The heavy convs are in the tail (the part we'd
keep fp16), and the early-stage fp32 ops would still pull the planner
to CPU. The Phase F production decision (full fp32, CPU-only, ~1.3×
RTFx) likely stands.

**Recommendation.** Run the tail-fp16 probe **only if the team has a
concrete ANE-residency target** that makes the 1–2 day spend
worthwhile. Otherwise: file Option 1 under "API feasible, but Phase
F.2's audibility envelope rules out the configurations that would
buy ANE residency", and treat retraining / QAT (Trial 8 deferred) as
the real path. Probe 3's verdict on Vocoder swap availability is now
the deciding factor — if Probe 3 surfaces a drop-in candidate,
Option 1's 1–2 day probe value collapses further.

### Probe 3 — Pretrained Vocos / iSTFTNet for NanoCodec

**One-line verdict: NO-GO. Drop Option 3.** No pretrained Vocos /
iSTFTNet checkpoint consumes NeMo NanoCodec FSQ codes; both viable
routes (codes→audio Vocos, codes→mel→Vocos) require glue training,
which violates the no-retraining constraint.

**Magpie NanoCodec output spec** (verified against
[`nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps`](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps),
arxiv:2508.05835, the model card, and `convert_nanocodec.py:226-232`):

| Field | Value |
|---|---|
| Upstream codec | **NeMo NanoCodec 22 kHz / 1.89 kbps / 21.5 fps** |
| Sample rate | 22 050 Hz |
| Frame rate | 21.533 fps (1 024 samples/frame) |
| Codebooks | 8 |
| Codebook size | 2 024 logical (= 2 016 FSQ codes + 8 reserved) |
| Quantizer | **Grouped FSQ**, levels per dim `[8, 7, 6, 6]` (not RVQ, not VQ-VAE) |
| Embed dim per codebook | 32 |
| Bitrate | 1.89 kbps |
| Decoder | Causal HiFi-GAN; 5 upsamples; dilations 1/3/5; 96 Snake; ConvTranspose1d stack; weight_norm; ~30-40 M params |
| ANE blockers | (a) `ios17.sin` from Snake; (b) `W ≤ 16 384` ceiling on dilated convs at `T_out ≥ 10 240`; (c) full graph `ANECCompile()` fail past `T_out=8 192` |

**Pretrained candidates inventory.** No HF or GitHub Vocos / iSTFTNet
consumes NanoCodec FSQ codes. Highlights from the search:

| Candidate | Input | Compatible? |
|---|---|---|
| `charactr/vocos-mel-24khz`, `BSC-LT/vocos-mel-22khz`, `lucasnewman/vocos-mel-24khz` | mel-spec | No — wrong input representation |
| `charactr/vocos-encodec-24khz` | EnCodec RVQ codes (75 fps, 1024-entry codebooks) | No — codebook geometry hard-tied to EnCodec via learned `codebook_weights` Parameter |
| `taresh18/nano-codec` | own RVQ codec, 16 kHz | **Name-collision distractor** — unrelated to NeMo NanoCodec |
| `nineninesix-ai/nanocodec-mlx` | NeMo NanoCodec | Same HiFi-GAN decoder, just MLX port — same Snake / dilated / transposed-conv topology |
| `Edresson/NanoCodec` GitHub | — | static demo for the prior "Low Frame-rate Speech Codec" paper |
| iSTFTNet variants (Uberduck, reppy4620, patriotyk, ZDisket) | mel-spec | No |

**Why stock Vocos cannot be reused as-is.** From
`gemelo-ai/vocos/vocos/feature_extractors.py`: `EncodecFeatures`
registers `codebook_weights` as a **learned `nn.Parameter`** — the
concatenated VQ centroids of the codec it was trained on — and the
backbone is GAN-trained to denoise that specific embedding manifold.
Substituting NanoCodec FSQ codes (different codebook count, different
levels-per-dim `[8,7,6,6]`, different embed dim 32, different frame
rate 21.5 fps vs EnCodec's 75 fps) means new codebook embeddings AND
the backbone has to re-learn the manifold. Not a drop-in swap.

**Glue-training cost estimate** (if scope ever opened):

- Architecture: 8 learned embedding tables (2 024 × 32) + Vocos
  ConvNeXt backbone (8 layers, dim 512) + ISTFTHead at 22 050 Hz.
- Loss: multi-res STFT + mel L1 + GAN (MPD, MRSTFT) — Vocos paper
  recipe.
- Data: pairs `(NanoCodec_codes, gt_audio)` generated by running the
  NanoCodec encoder over speech (LibriTTS / MLS / etc.); no labels
  beyond raw speech.
- Compute: ~200–500 A100-hours for a single-speaker baseline;
  ~1 000–3 000 A100-hours to actually match production NeMo HiFi-GAN
  quality on Magpie's 5 speakers × 8 languages.
- **Verdict: counts as retraining. Out of scope.**

**Mel-intermediate adjacent path.** `BSC-LT/vocos-mel-22khz`
(Apache-2.0, 22 050 Hz, 80-bin mel, ConvNeXt + ISTFTHead) has the
right SR and a clean ANE-relevant op profile (no Snake, no dilated
convs with W > 16 384, no transposed-conv stack). But it requires a
NanoCodec-codes → 80-bin-mel module sitting in front, which itself
requires training. **The mel-intermediate path is also glue training**,
just lower-cost than a full Vocos retrain.

**Recommended next step.** **Drop Option 3.** The premise — "swap
NanoCodec for an ANE-friendlier vocoder without retraining" — is not
satisfiable with what's public as of 2026-05. Further ANE-recovery
on the codec stage requires either (a) NVIDIA publishing a Vocos head
for NanoCodec (no signal they will), or (b) accepting a glue-training
project comparable in cost to retraining Magpie's vocoder from
scratch. Both violate the no-retraining constraint. Decoder-side
fusion (`STATUS.md` "Phase D fusion") is a more promising next direction
than vocoder replacement.

## Probe summary

| # | Option | Probe verdict |
|---|---|---|
| 1 | MIL post-conversion graph rewriting | API feasible; F.2 numerics rule out the early-stages-fp16 split. **Conditional GO on a 1–2 person-day tail-fp16 probe** that F.2 didn't cover, only if a concrete ANE-residency target justifies the spend. |
| 2 | coremltools 9.x upgrade | **NO-GO.** 9.0 ships the same byte-identical pass code as 8.3; no upstream issue or PR addresses the symptom; no in-flight fix to wait on for 9.1. |
| 3 | Vocoder swap (Vocos / iSTFTNet) | **NO-GO.** No pretrained checkpoint consumes NanoCodec FSQ codes; both viable routes require glue training, violating the no-retraining constraint. |

**Net: all three Tier 1 options are blocked or marginal under the
no-retraining constraint.** The structural 830 ms TTFA ceiling is
likely the operating point unless (a) the team commits to retraining
(Trial 8 QAT, est. −400 ms; or a NanoCodec→Vocos glue train), or
(b) a future coremltools / iOS release inserts boundary casts
correctly. Tier 2 quick spikes (Option 4 LT body/tail split, Option 6
CU-override re-audit) remain low-cost validation work that can land
without crossing the constraint line.
