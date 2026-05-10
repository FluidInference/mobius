# Magpie conversion-only optimization options

Impact-ranked menu of remaining levers on the **shipping Magpie 357M
multilingual checkpoint**, after Trials 1 + 3 already shipped (−978 ms,
−161 ms TTFA respectively per [`PERF.md`](PERF.md)) and every other
graph-level lever in `PERF.md` was exhausted.

## Top-line state (current)

| # | Option | Status | Verdict source |
|---|---|---|---|
| 1 | MIL post-conversion graph rewriting (tail-fp16) | **🔴 NO-GO** | Probe 1 + Trial 11 (`PERF.md`) — v1 SNR 38.79 dB << 48 dB threshold, 0 % ANE |
| 2 | coremltools 9.x upgrade | **🔴 NO-GO** | Probe 2 — pass code byte-identical to 8.3 |
| 3 | Vocoder swap (Vocos / iSTFTNet) | **🔴 NO-GO** | Probe 3 — no NanoCodec-FSQ consumer; glue training out of scope |
| 4 | `local_transformer` body/tail split | ⏸ Available, not pursued in this PR | Tier 2 (1 % win, partition-cost risk) |
| 5 | MLPipeline fusion (Trial 9 revisit) | ⏸ Parked | Tier 2 (~0.5–1 % win) |
| 6 | Per-stage CU override re-audit | ⏸ Roll into BASELINE_FP32 follow-up | Tier 2 (0–20 ms ceiling) |
| 7 | QAT int8 (Trial 8 deferred) | ❌ Out of scope | Constraint: no retraining |
| 8 | Architecture-level retraining | ❌ Out of scope | Constraint: no retraining |

**All three Tier 1 options are now closed.** The tail-fp16 probe
(Trial 11 in `PERF.md`) ran with the smallest possible fp16 island
(post_conv + out_activation, 3 ops) and failed both halves of the
hypothesis: 38.79 dB SNR (vs the 48 dB Phase F.2 audibility threshold)
AND 0 % ANE residency (the planner refused to partition for a 3-op
fp16 island anchored to a fp32 output). Anything larger only adds
more fp16 noise; the stopping rule fired, v2/v3 skipped.

The structural 830 ms warm TTFA ceiling is the operating point under
the no-retraining constraint. Tier 2 options 4, 5, 6 remain available
for follow-up work but are **not being pursued in this PR**.

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

## Tier 3 — out of scope (require retraining / model-level changes)

### Option 7 — QAT / activation-calibrated int8 on `decoder_step`

`PERF.md:516` Trial 8 deferred. Multi-day NeMo training + EOS
preservation work; estimated **−400 ms TTFA**. Violates the
no-retraining constraint.

### Option 8 — Architecture-level retraining

LiteMagpie / distilled smaller body / pruned attention / alternate
codec head. Multi-week training pipeline. Violates the no-retraining
constraint.

## Decision summary

**Originally:** Only Options 1–3 had a shot at lowering the 830 ms
TTFA ceiling. Probes 2 and 3 closed Options 2 and 3 as NO-GO; Trial 11
(the tail-fp16 follow-up to Probe 1) closed Option 1 as NO-GO. All
three Tier 1 options are now exhausted; the 830 ms ceiling is the
operating point under the no-retraining constraint.

| # | Option | Impact ceiling | Effort | Status |
|---|---|---|---|---|
| 1 | MIL post-conversion graph rewriting (tail-fp16) | ~15-30 ms (~2-4 %) | 1–2 person-days | 🔴 NO-GO (Trial 11) |
| 2 | coremltools 9.x upgrade | same as #1 | Low | 🔴 NO-GO (Probe 2) |
| 3 | Vocoder swap (Vocos / iSTFTNet) | ~100-150 ms (~12-18 %) | 0 if pretrained, weeks otherwise | 🔴 NO-GO (Probe 3) |
| 4 | `local_transformer` body/tail split | ~10-20 ms (~1 %) | Medium | ⏸ Available, not pursued in this PR |
| 5 | MLPipeline fusion | ~5-10 ms (~0.5-1 %) | Low | ⏸ Parked |
| 6 | Compute-unit override audit | 0-20 ms | Low | ⏸ Roll into BASELINE_FP32 follow-up |
| 7 | QAT int8 (`decoder_step`) | est. −400 ms | Multi-day | ❌ Out of scope |
| 8 | Architecture-level retraining | open-ended | Multi-week | ❌ Out of scope |

## Cross-references

- `PERF.md` §TL;DR (current TTFA stack, shipped levers) — line 10.
- `PERF.md` Trial 8 (QAT deferred) — line 516.
- `PERF.md` Trial 9 (`MLPipeline` low-EV) — line 515.
- `PERF.md` Trial 10c (NanoCodec mixed-precision DEAD via Phase F.2)
  — line 567.
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
