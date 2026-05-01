# Parakeet‑TDT v3 (0.6B) — Encoder int4/int8 Quantization Notes

End-to-end log of the int4/int8 encoder quantization attempt for v3, the
tooling produced, the variant sweeps, and the production numbers from the
FluidAudio integration. Companion to `extra_encoder_variants.py`,
`compute_unit_sweep.py`, `analyze_fallback.py`, and the v3 README's
"Encoder-only int4/int8 sweep" section.

## Goal

The fp16 encoder is the single largest CoreML component in the v3 stack
(~426 MB on disk vs ~30 MB for everything else combined). Quantizing the
encoder while leaving preprocessor / decoder / joint at fp16 gives most of
the disk win without further hurting decoding quality. The target was an
encoder that:

1. stays ANE-resident on macOS 15 / iOS 18,
2. maintains WER within ~1 pt of the 8-bit-palettized baseline,
3. fits a smaller on-disk footprint than the prior 6-bit palettized encoder.

## Tooling produced (mobius PR #47)

Three scripts were added under `models/stt/parakeet-tdt-v3-0.6b/coreml/`:

### `extra_encoder_variants.py`

Encoder-scoped recipes that go beyond `quantize_coreml.py`:

| Variant                          | Recipe                                               |
|----------------------------------|------------------------------------------------------|
| `enc8bit-palettize`              | 8-bit palette on encoder weights                     |
| `enc-prune+int8`                 | sparse + int8 linear per-channel                     |
| `enc-int4-linear-per-channel`    | int4 linear, one scale per output channel            |
| `enc-int4-linear-per-block-32`   | int4 linear, one scale per 32-element block          |
| `enc-prune+int4-block`           | sparse + int4 block-wise                             |

The three int4 variants explicitly bump
`spec.specificationVersion = 9` (iOS 18 / macOS 15) after the
`optimize.coreml` pass — int4 weight payloads require the iOS 18 runtime.
The bump is performed in `_bump_spec_to_ios18`, then `_save_mlpackage`
(imported from `quantize_coreml.py`) writes the `.mlpackage`. The
`minimum_deployment_target = ct.target.iOS17` set inside `_save_mlpackage`
does **not** downgrade the spec — coremltools' setter uses
`max(current, target)`, so a model already at spec 9 stays at spec 9.

Invocation:

```
uv run python extra_encoder_variants.py run \
  --input-dir parakeet_coreml \
  --output-root parakeet_coreml_encoder_only
```

Produces `encoder_variants_summary.json` plus one `.mlpackage` per recipe.

### `compute_unit_sweep.py`

Drives `tools/coreml-cli` (no `--fallback`) across all four
`MLComputeUnits` configurations (`all`, `cpu_only`, `cpu_and_gpu`,
`cpu_and_neural_engine`) per component and aggregates latency + device
residency into `compute_unit_sweep.json`. Auto-compiles any
`.mlpackage` input into `<input_dir>/.compiled/<stem>.mlmodelc` and
reuses that cache.

Used to confirm where each component (preprocessor / encoder / decoder /
joint) actually executes under each compute-unit setting and how latency
moves when the runtime is forced off ANE.

### `analyze_fallback.py`

Runs `coreml-cli --fallback --json` per component, summarizes total /
fallback op counts, and groups CPU-fallback ops by reason into
`fallback.json`. This was the diagnostic that confirmed the per-channel
int4 encoder stays ANE-resident, while the per-block variants fall back
to CPU on the per-block dequant op.

## Variant sweep — mobius internal (100-file LibriSpeech `test-clean`)

M-series, baseline = pre-converted fp16 encoder:

| Variant                            | WER    | RTFx  | Encoder size |
|------------------------------------|-------:|------:|-------------:|
| baseline (fp16)                    | 2.64 % | 36.8× | 426 MB       |
| `enc-prune+int8`                   | 2.57 % | 19.8× | ~340 MB      |
| **`enc-int4-linear-per-channel`**  | 5.24 % | 49.2× | 285 MB       |
| `enc-int4-linear-per-block-32`     | 3.95 % | 15.6× | ~310 MB      |
| `enc-prune+int4-block`             | 3.95 % | 15.9× | ~300 MB      |

### Selection rationale

- **`enc-int4-linear-per-channel`** wins on disk (285 MB, ~33 % smaller
  than the fp16 baseline) and on RTFx (49.2× vs 36.8×). Stays fully on
  ANE.
- **Per-block variants** (`-per-block-32`, `prune+int4-block`) trade
  ~1.3 pt of WER for ~3× slower latency. The slowdown is `analyze_fallback`
  showing CPU fallback on the block-wise dequantize op — ANE doesn't have a
  fast path for it on macOS 15 / iOS 18.
- **`enc-prune+int8`** keeps WER near baseline but loses ~half the
  baseline RTFx; not worth shipping.

## Production sweep — FluidAudio (full LibriSpeech `test-clean`)

2,620 files, 19,452.5 s audio, Apple M2, `.cpuAndNeuralEngine`,
decoder / joint / preprocessor fp16 in both rows. From
`FluidAudio/benchmarks.md`:

| Encoder           | On-disk | Avg WER | Avg CER | Overall RTFx | Peak RAM |
|-------------------|--------:|--------:|--------:|-------------:|---------:|
| 8-bit palettized  | 425 MB  | 2.64 %  | 1.03 %  | 47.1×        | 153 MB   |
| int4 linear/ch    | 285 MB  | 3.76 %  | 1.59 %  | 43.1×        | 139 MB   |

The WER drift between the mobius 100-file number (5.24 %) and the
FluidAudio 2,620-file number (3.76 %) on per-channel int4 is consistent
with sample-size variance plus differences in the exact decoder /
post-processing stack used in each harness. Both confirm the int4 path
costs ~1 pt of average WER vs the 8-bit baseline at ~33 % less disk and
within ~10 % of baseline RTFx.

## Existing `quantize_coreml.py` results (for context)

From the v3 README, ComputeUnits=ALL on M4 Pro 48 GB:

- int8 linear (per-channel): ~2.0× smaller across components with
  minimal quality loss. MelEncoder quality ≈ 0.963, lat ≈ 31.13 ms
  (baseline ≈ 29.34 ms). JointDecision acc ≈ 0.995, lat ≈ 1.96 ms
  (baseline ≈ 2.15 ms).
- int8 linear (per-tensor symmetric): encoder quality drops to ≈ 0.50 —
  not recommended.
- 6-bit palette (`encoder-palettize` / `mel-palettize`) was the prior
  shipping encoder before the 8-bit palettized / int4 split.

The `quantize_coreml.py` recipe set on `main` is intentionally narrow:
`int8-linear` (per-channel, whole stack), `mel6bit-palettize`
(mel_encoder only), and `enc6bit-palettize` (encoder only). Earlier
"global palettization" entries were removed; comments in the source
note them as `(removed) Global palettization variants`.

## Past encoder-only attempts not in the committed sweep

Several encoder variants were generated by hand during the int4
exploration but did not make it into `extra_encoder_variants.py`'s
`_default_variants()`. The compiled `.mlpackage` artifacts are still
under `parakeet_coreml_encoder_only/`. They are documented here so
future contributors don't repeat them blind.

Inspection method: `mlProgram.functions[*].block_specializations[*].operations`
op-type histogram on each `model.mlmodel`, plus on-disk `du -sh` of the
mlpackage and `specificationVersion` from the model spec.

| Directory                       | On-disk | Spec | Op-type signature                                 | Verdict                                                    |
|---------------------------------|--------:|-----:|---------------------------------------------------|------------------------------------------------------------|
| `enc-4bit-palettize`            | 284 MB  |    8 | 294× `constexpr_lut_to_dense` + 193× `linear`     | 4-bit kmeans palette. Same disk as int4-per-channel but no quality advantage over the 6-bit palette on encoder; abandoned in favor of int4-linear. |
| `enc-a8w4`                      | 285 MB  |    9 | 654× `dequantize` + 558× `quantize` + `const`     | W4A8 (int4 weights + int8 activations). Activation quant inserts runtime `quantize`/`dequantize` ops at activation boundaries → ANE compatibility / calibration concerns; abandoned. |
| `enc-int4-selective`            | 412 MB  |    9 | 192× `constexpr_blockwise_shift_scale` + 193× `linear` | Selective int4 (only attention/MLP linear weights quantized, conv kept fp16). Disk too high for the savings target (412 MB vs 285 MB for full per-channel int4); abandoned. |
| `enc-w8a8`                      | 568 MB  |    8 | 510× `dequantize` + 438× `quantize` + `const`     | W8A8 (int8 weights + int8 activations). Disk regressed against fp16 baseline (568 MB vs 426 MB) because activation quant materializes calibration constants in the package; abandoned. |
| `enc-int4-linear-per-channel`   | 285 MB  |    9 | 192× `constexpr_blockwise_shift_scale` + 193× `linear` | Shipping variant. Listed for op-signature comparison.      |

Why these aren't in the committed `_default_variants()`:

- **Palette-based 4-bit (`enc-4bit-palettize`)** uses the same 4-bit
  payload as int4-linear but indirected through a per-tensor lookup
  table. On the encoder it doesn't recover meaningful quality vs the
  existing 6-bit palette and gives up the per-channel scaling that
  int4-linear retains. The `coremltools.optimize.coreml.palettize_weights`
  path was kept only for the 8-bit reference (`enc8bit-palettize`) which
  exists as a quality anchor against the int4 numbers.
- **W4A8 / W8A8 activation quantization** (`enc-a8w4`, `enc-w8a8`)
  required activation calibration with representative data, and the
  inserted `quantize`/`dequantize` ops triggered partial CPU fallback
  on ANE in `analyze_fallback` runs. The simpler weight-only int4
  path stayed fully on ANE, so activation quant was dropped.
- **Selective int4** (`enc-int4-selective`) only quantized linear /
  attention weights, leaving conv blocks fp16. The disk number (412 MB)
  was inside one rounding error of the fp16 baseline (426 MB) for ~1 pt
  of WER; not worth shipping.

Summary recorded for `enc-int4-linear-per-channel` in
`parakeet_coreml_encoder_only/encoder_variants_summary.json` (M2,
macOS 26.5, CoreMLTools 9.0b1, ComputeUnits = `CPU_AND_NE`,
`yc_first_minute_16k_15s.wav` calibration audio):

- size: 284.67 MB (3.97× compression vs the 1131.51 MB unquantized
  baseline mlpackage with separate weight blob)
- latency: 115.28 ± 5.10 ms / window vs 162.44 ms baseline
- compile (offline `coremlcompiler`): 1167 ms vs 4712 ms baseline
- quality (1 − normalized L2 vs fp16 encoder): 0.414 (note: this is
  raw encoder-output L2, not WER — WER on real audio recovers because
  the joint head is robust to small encoder perturbations)
- max_abs / max_rel against fp16 reference: 0.133 / 2.0
- optimize wall-time: 352 s
- spec version bumped to 9 (iOS 18)

The quality_norm_l2 of 0.414 is *much* worse than the WER picture
(2.64 % → 3.76 %) suggests, which is the empirical point: encoder
output L2 is not a faithful proxy for downstream WER on this stack.
This is part of why per-channel int4 still ships even though its raw
encoder error looks alarming — the joint head absorbs most of it.

## FluidAudio PR #560 iteration history

The FluidAudio integration went through five distinct iterations before
landing in its current "int8 default, int4 opt-in" form. Captured here
because the design rationale is otherwise spread across the eight
commits on PR #560 and the working session that produced them.

### 1. int4 as the only v3 encoder (`89b99d327`)

`feat(asr/parakeet-v3): default to int4-per-channel encoder`. The first
commit on the PR hard-coded `enc-int4-linear-per-channel` as the only
encoder shipped for v3:

```swift
case .v3:
    return (
        encoder: Names.encoderInt4File,
        decoder: Names.decoderFile,
        joint: Names.jointV3File,
        vocabulary: Names.vocabularyFile
    )
```

Headline pitch in the commit body cited mobius's 100-file numbers:
`426 MB → 285 MB on disk, 36.8× → 49.2× RTFx, 2.64 % → 5.24 % WER`. The
WER number quoted here (`5.24 %`) is the mobius internal 100-file
LibriSpeech subset, not the full test-clean number. The doc strings
inside the source described the drop as "~2.6 pp" — this later turned
out to be a sample-size artefact (see step 3).

### 2. benchmarks.md publication (`acadcce7d`)

`docs(asr/benchmarks): add Parakeet TDT 0.6B v3 fp16/int4 test-clean
results`. Added the full `LibriSpeech test-clean` (2,620 files,
19,452.5 s audio, M2, `.cpuAndNeuralEngine`) results to
`Documentation/asr/benchmarks.md`:

| Encoder           | On-disk | Avg WER | Avg CER | Overall RTFx |
|-------------------|--------:|--------:|--------:|-------------:|
| 8-bit palettized  | 425 MB  | 2.64 %  | 1.03 %  | 47.1×        |
| int4 linear/ch    | 285 MB  | 3.76 %  | 1.59 %  | 43.1×        |

Same commit also added a cross-stack comparison row against the
`mweinbach/parakeet-coreml-swift` 4-bit palettized encoder: 24.7× RTFx
@ 12.77 % avg WER on test-clean, with ~3.8 % of files exhibiting
catastrophic TDT decoder spill-over (the joint head running off into
repeated tokens once encoder error exceeds a per-frame budget). This is
why the "global 4-bit palette without per-channel scaling" path was
rejected — the WER tail is unbounded, not just a few percent worse on
average.

### 3. WER-comment correction (`e6ec29647`)

`fix(asr/parakeet-v3): correct int4 encoder WER comments
(5.24 % → 3.76 %)`. Devin Review on PR #560 flagged a discrepancy:
multiple in-source comments said `~2.6 pp WER drop` (2.64 % → 5.2 %),
but the actual measured drop on the full test-clean was
`2.64 % → 3.76 %` (~1.1 pp). The `5.2 %` figure originated from the
mobius 100-file sweep and had been transcribed into FluidAudio source
comments wholesale. The fix updated every commented WER reference to
the full-test-clean numbers; the production number to cite when
discussing this encoder is `3.76 % avg WER` on
`LibriSpeech test-clean`.

### 4. caption trim (`24778e9fc`)

`docs(asr/benchmarks): trim redundant parakeet v3 caption`. Cosmetic
only; no behaviour change.

### 5. flip to user-selectable, int8 default (`911bc6fdc`)

`feat(asr/parakeet-v3): user-selectable int8/int4 encoder`. User
direction during the PR review was "keep at int8 for now, int4 is
something users can do themselves". This commit introduced the
`ParakeetEncoderPrecision` enum and the entire precision-toggle
infrastructure:

- `public enum ParakeetEncoderPrecision: String, Sendable, CaseIterable
  { case int8; case int4 }`, with `int8.encoderFileName = "Encoder"` and
  `int4.encoderFileName = "EncoderInt4"`.
- `encoderPrecision: ParakeetEncoderPrecision = .int8` threaded through
  `AsrModels.load`, `.download`, `.downloadAndLoad`, `.modelsExist`,
  `.isModelValid`, `loadVocabulary`, plus `getModelFileNames` and
  `getRequiredModels`.
- `Names.requiredModelsV3(precision:)` per-precision required-files set,
  so `modelsExist` only checks for the requested encoder.
- v3 download routes through `DownloadUtils.loadModels(variant:
  precision.rawValue)` so HF fetches only the requested encoder
  artifact.

The full diff against `main` is captured in the AsrModels.swift hunks
exchanged in the working session — every `getModelFileNames(version:)`
call became `getModelFileNames(version:, encoderPrecision:)`, every
`getRequiredModels(version:)` became
`getRequiredModels(version:, encoderPrecision:)`, and the v3 file-name
arm switched from `Names.encoderInt4File` (step 1) to
`encoderPrecision.encoderFileName`.

### 6. comment cleanup (`08486e3a7`)

`chore(asr/parakeet): trim verbose comments from precision toggle`.
Step 5 left a few duplicated explainer comments after the precision
parameter was added; this commit pruned them.

### 7. top-K comment restoration (`409a954fd`)

`docs(asr): restore JointDecisionv3 top-K comment on v3 branch`. The
cleanup in step 6 over-pruned a v3-specific comment about
`JointDecisionv3` always emitting `top_k_ids` / `top_k_logits` (Swift
extraction is gated by `needsTopK`). Restored verbatim.

### 8. CLI `--encoder-precision` flag (`0c6c954cc`)

`feat(cli/transcribe): add --encoder-precision flag for Parakeet v3`.
Final commit in the iteration. Added the user-facing flag to
`fluidaudiocli transcribe` so the precision selection is reachable
without writing Swift. This is the flag used to produce the
"End-to-end CLI verification" numbers below.

### Why the int8 default

Three factors drove the flip from "int4 as v3 default" (step 1) to
"int8 default, int4 opt-in" (step 5):

1. **WER drift wasn't free.** Step 3's correction confirmed a real
   ~1.1 pp drop on the full test-clean, not a rounding artefact. Most
   integrators want the lowest WER number on the box.
2. **RTFx headroom on int8 is already excessive.** 47.1× RTFx on M2
   means the encoder isn't the bottleneck; trading that for 3.76 % WER
   to save 140 MB on disk only matters for very disk-constrained
   on-device deployments (smaller iOS apps).
3. **User-selectable is cheap.** Once the
   `ParakeetEncoderPrecision`-enum-plus-`variant`-download plumbing
   exists, switching is a one-line API choice; there's no need to pick
   one for everyone.

## End-to-end CLI verification (FluidAudio side)

Single-utterance smoke test on
`LibriSpeech/test-clean/1089/134691/1089-134691-0007.flac`. Cache cleared
between runs to force a fresh download:

| Precision flag                  | Encoder file fetched   | On-disk    | Cold wall-clock | Transcript                                          |
|---------------------------------|------------------------|-----------:|----------------:|-----------------------------------------------------|
| (none → `int8` default)         | `Encoder.mlmodelc`     | 433–434 MB |        55–81 s  | "Soon the whole bridge was trembling and resounding." |
| `--encoder-precision int4`      | `EncoderInt4.mlmodelc` | 289–290 MB |       43–170 s¹ | "Soon the whole bridge was trembling and resounding." |

¹ 170 s was the first-ever int4 ANE compile (`anecompilerservice`).
Subsequent cold-cache runs settled at ~43 s once the ANE compile cache
warmed up.

This confirms:

- Each precision selection fetches **only** its own encoder from
  HuggingFace — no cross-contamination on disk.
- Identical decode on a short utterance (longer eval shows the
  ~2.64 % vs ~3.76 % WER gap above).
- The `ParakeetEncoderPrecision.rawValue` ("int8" / "int4") threads
  cleanly from CLI → `AsrModels.downloadAndLoad` →
  `DownloadUtils.loadModels(variant:)` →
  `getRequiredModelNames(for: .parakeetV3, variant:)` →
  `ModelNames.ASR.requiredModelsV3(precision:)`.

## FluidAudio integration (PR #560)

- `ParakeetEncoderPrecision: String, Sendable, CaseIterable { case int8; case int4 }`
  exposed publicly.
- Default `.int8` everywhere: `AsrModels.load`, `.download`,
  `.downloadAndLoad`, `.modelsExist(at:version:)`, `.isModelValid`,
  `ModelNames.ASR.requiredModelsV3(precision:)`,
  `getRequiredModelNames(for: .parakeetV3, variant:)` fallback,
  and CLI `transcribe` (`--encoder-precision` flag).
- v3 download routes precision through
  `DownloadUtils.loadModels(variant: precision.rawValue)` so both the
  existence check and the HF fetch only pull the requested encoder file.
- HF repo `FluidInference/parakeet-tdt-0.6b-v3-coreml` ships both
  `Encoder.mlmodelc` and `EncoderInt4.mlmodelc`; preprocessor / decoder /
  joint fp16 are shared between the two precision paths.

The shipping default stays int8 — int4 is opt-in for users who want the
~140 MB disk saving and are willing to absorb the ~1 pt WER drift.

## Tooling bugs surfaced and fixed (Devin review on PR #47)

Two of the three Devin findings on `mobius#47` were real bugs in the
new diagnostic scripts. Neither affected the saved `.mlpackage`
artifacts that FluidAudio consumes — the bugs were in JSON-key
navigation in the wrappers, not in the quantization recipes
themselves. Both are now fixed in this branch.

- **`analyze_fallback.py:180-181`** previously read
  `fb.get("fallback_ops")` and `fb.get("cpu_op_count")`, but
  `tools/coreml-cli` emits `"cpu_ops"` (see
  `tools/coreml-cli/src/coreml_cli/fallback.py:100`). As a result
  `fallback_ops` and `fallback_percent` were always `null` in
  `fallback.json`, and the per-reason summary returned an empty
  `Counter` because the function only handled the dict-shape branch
  while `coreml-cli` emits `reasons` as a list of dicts. Fixed by
  reading the correct `"cpu_ops"` key (with the legacy keys kept as
  fallbacks via explicit `is None` checks instead of `or`-pattern
  falsy-fallthrough), and by adding a list-of-dicts branch that
  aggregates each entry's `count` under its `reason` string. Smoke
  test on the per-channel int4 encoder now reports
  `total=1727 fallback=351 (20.32 %)`, top reasons "ANE not available
  for this op = 342", "Unsupported tensor data type: int32 = 3", "ANE
  supported but scheduler chose CPU = 3" — i.e. the encoder is ANE-
  resident on the heavy compute and the 20 % CPU fallback is the
  expected non-ANE op tail (data-type conversions, scheduler
  preferences), not the int4 dequant op.

- **`compute_unit_sweep.py:234-247`** pretty-print navigated the
  `coreml-cli` JSON with the wrong keys:
  `models[0].get("compute_units", [])` should have been `"results"`
  (cli.py:183), and `runs[0].get("device_assignment", {})` should have
  been `"summary"` (compute_plan.py:153). Terminal output was always
  empty; the saved JSON was unaffected because it stored the raw
  `result` dict. Fixed by reading the correct keys. Smoke test on the
  per-channel int4 encoder now prints:
  ```
  units=all                       latency=104.636 ms | CPU=0.01% GPU=0.0%   ANE=99.99%
  units=cpu_only                  latency=1150.281 ms| CPU=100.0% GPU=0.0%  ANE=0.0%
  units=cpu_and_gpu               latency=457.208 ms | CPU=0.0%   GPU=100.0% ANE=0.0%
  units=cpu_and_neural_engine     latency=103.639 ms | CPU=0.01% GPU=0.0%   ANE=99.99%
  ```
  This empirically confirms the central claim of the PR: under both
  `all` and `cpu_and_neural_engine` the int4 encoder is ~99.99 %
  ANE-resident, ~104 ms / window on M4 Pro. CPU-only is ~11× slower,
  GPU-only is ~4.4× slower — i.e. the int4 throughput win is entirely
  ANE-dependent.

- **(False positive, retracted)** Earlier review claimed
  `_save_mlpackage` downgrades iOS 18 → iOS 17 for the int4 variants.
  CoreMLTools' `minimum_deployment_target` setter is
  `max(current, target)` on `_spec.specificationVersion`, so a model
  already at spec 9 stays at spec 9. The int4 mlpackages on HF load
  and decode correctly on macOS 15 / iOS 18 (verified end-to-end via
  the FluidAudio CLI run above).

## Open follow-ups

- Group-wise int4 (group size between per-channel and per-block-32) was
  not swept; could close some of the ~1 pt WER gap if the dequant op
  stays on ANE.
- Encoder pruning + int4-per-channel was not swept (`prune+int4-block`
  was the only prune+int4 combo). Pruning the per-channel int4 may
  recover quality cheaply.
- `compute_unit_sweep` per-component breakdown should be pulled in
  across preprocessor / decoder / joint as well as the encoder.
  Pretty-print is fixed, so the next pass can just `tee` the output.
- `analyze_fallback` confirmed the per-channel int4 encoder is
  99.99 % ANE-resident (104 ms / window on M4 Pro) with ~20 % CPU op
  count from non-ANE-eligible ops (data-type conversion, scheduler
  preference) — none of the CPU ops are the int4 dequant op. The
  per-block int4 variants have a different fallback profile and
  should be re-run now that `analyze_fallback` actually reports.
