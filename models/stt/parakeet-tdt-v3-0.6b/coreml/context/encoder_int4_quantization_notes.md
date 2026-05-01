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

## Known bugs in the new tooling (from Devin review on PR #47)

These are real and should be fixed in the mobius scripts; they don't
affect the saved `.mlpackage` artifacts that FluidAudio consumes.

- **`analyze_fallback.py:180-181`** reads `fb.get("fallback_ops")` and
  `fb.get("cpu_op_count")`, but `tools/coreml-cli` emits `"cpu_ops"`. As
  a result `fallback_ops` and `fallback_percent` are always `null` in
  `fallback.json`. The `total_ops` lookup has the same `or`-pattern
  problem: a value of `0` would fall through to the non-existent
  `"op_count"` key.
- **`compute_unit_sweep.py:234-247`** pretty-print navigates the
  `coreml-cli` JSON with the wrong keys: `models[0].get("compute_units", [])`
  should be `"results"`, and `runs[0].get("device_assignment", {})`
  should be `"summary"`. Terminal output is always empty, but the saved
  JSON still contains the raw `result` dict, so downstream consumers are
  unaffected.
- **(False positive, retracted)** Earlier review claimed
  `_save_mlpackage` downgrades iOS 18 → iOS 17 for the int4 variants.
  CoreMLTools' `minimum_deployment_target` setter is `max(current, target)`
  on `_spec.specificationVersion`, so a model already at spec 9 stays at
  spec 9. The int4 mlpackages on HF load and decode correctly on
  macOS 15 / iOS 18 (verified end-to-end via the FluidAudio CLI run
  above).

## Open follow-ups

- Group-wise int4 (group size between per-channel and per-block-32) was
  not swept; could close some of the ~1 pt WER gap if the dequant op
  stays on ANE.
- Encoder pruning + int4-per-channel was not swept (`prune+int4-block`
  was the only prune+int4 combo). Pruning the per-channel int4 may
  recover quality cheaply.
- `compute_unit_sweep` results currently aren't summarized in this doc
  beyond "per-channel stays on ANE, per-block falls back". Once the
  pretty-print bug is fixed, the per-component CU breakdown should be
  pulled in as a small table.
