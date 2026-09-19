# CUA-S1-FORMS → Core ML

Converts the real **706,048-parameter** CUA-S1-FORMS checkpoint to a **1.51 MB
FP16 Core ML package**. The complete pinned 196-decision demo passes: PyTorch and
Core ML both get **196/196 correct**, with **196/196 matching selected options**.
The unmodified upstream Cua evaluation harness also reports no wrong actions or
wrong targets on these saved decisions.

The demo check is a bounded local conversion verification. The demo covers three forms and
three PDFs; it does not establish generalization to other forms or successful
execution in a live application. The separate full synthetic test below matches classification accuracy but
fails the original numerical gates; no training was performed.

## Reproduce

Run from this directory on an Apple silicon Mac:

```bash
uv sync --frozen
uv run python assets.py
uv run python convert-coreml.py
uv run python verify.py
uv run python score-report.py
uv run pytest -q
```

For independent Swift/runtime parity, `uv run python export-reference.py` writes
the unmodified PyTorch probabilities for every pinned demo row to
`build/reference-probabilities.json`, including the model revision and dataset
SHA-256. The [FluidAudio integration checks](#swift-integration-checks)
consume this file without requiring PyTorch in the application.

The converted packages, compiled bundles, and reports are available on
[Hugging Face](https://huggingface.co/FluidInference/cua-s1-forms-coreml).
Automatic FluidAudio downloading uses the model repository's `main` branch.

Python **3.11.11**, PyTorch **2.7.0**, and coremltools **9.0** are pinned. Python
3.11 follows the upstream package's minimum version rather than the older Mobius
Python 3.10 default. `uv.lock` pins the remaining dependencies.

`assets.py` verifies SHA-256 hashes and downloads only missing pinned assets:
the safetensors checkpoint, configuration, model/dataset cards, and `demo.jsonl`.
This setup does not download the original pickle checkpoint or the
train/validation/test datasets. The separate synthetic benchmark downloads only
the pinned test split when explicitly run. Existing files with incorrect hashes fail validation.

Outputs:

- `build/cua_s1_forms_fp16_options32.mlpackage` — portable Core ML model for Xcode.
- `build/conversion.json` — configuration, versions, source revisions, and package hashes.
- [reports/verification.json](reports/verification.json) — per-decision parity and timing results.
- [reports/upstream-metrics.json](reports/upstream-metrics.json) — original Cua action/target metrics.
- [reports/ane-fallback.json](reports/ane-fallback.json) — scheduler placement and ANE rejection reasons.
- [reports/ane-profile.json](reports/ane-profile.json) — four-policy compute plans and real-input timing.

The generated model and downloaded data stay under ignored `build/` and
`artifacts/`. Source, lockfiles, notices, tests, and compact JSON reports can be
reviewed independently of the generated assets.

## Verified results

Local run on **Apple M5 Pro, 24 GB, macOS 27.0**, September 19, 2026:

| Check | Core ML `ALL` | Core ML `CPU_AND_NE` |
| --- | ---: | ---: |
| Correct labeled decisions | 196 / 196 | 196 / 196 |
| Selected options matching PyTorch | 196 / 196 | 196 / 196 |
| Maximum absolute probability difference | 0.003099 | 0.002336 |
| Maximum absolute live-logit difference | 0.03654 | 0.07231 |
| Warm model-call median | 1.85 ms | 0.90 ms |
| Warm model-call p95 | 2.49 ms | 0.94 ms |
| Load time in this run | 142 ms | 579 ms |

The separate FP32 export-adapter check has maximum probability difference
**0.00000113** against the unmodified upstream model and its variable-length
collator. This catches preprocessing and padding mistakes before attributing
any difference to FP16 conversion. Six reversed-option-order checks pass on
both Core ML configurations.

The parity gates were selected before conversion: **100% option agreement**,
**no accuracy loss**, **maximum probability error ≤ 0.005**, finite outputs,
valid probability sums, and zero probability assigned to padded options. The
FP32 adapter has a separate maximum probability error gate of 0.0001.

Upstream Cua metrics count **36 fill, 4 check, 6 click, and 150 skip decisions**.
All are correct. The evaluator treats `skip` as abstention, so its 23.47%
coverage reflects 46 actionable decisions, not missing predictions.
`score-report.py` requires every row exactly once before invoking that evaluator.

Timing is exploratory, includes Python-to-Core-ML call overhead, and excludes
document extraction, UI observation, and GUI execution. Encoding is measured
separately (median 0.038 ms). Load measurements may benefit from existing system
caches; they are not a guaranteed first-install cold-start measurement. PyTorch
uses two CPU threads and variable-length inputs; no optimized MPS comparison is
claimed.

## Swift integration checks

The [Swift API reference](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/API.md#decision-scoring)
describes local loading, input limits, and output validation. After the base setup
above, `uv run python export-reference.py` exports probabilities from the unmodified
PyTorch model plus the exact demo SHA-256 and checkpoint revision.

Run from FluidAudio with absolute paths to those generated assets:

```bash
FLUIDAUDIO_CUA_MODEL_PATH=/path/to/coreml/build/cua_s1_forms_fp16_options32.mlpackage \
FLUIDAUDIO_CUA_DEMO_PATH=/path/to/coreml/artifacts/demo.jsonl \
FLUIDAUDIO_CUA_REFERENCE_PATH=/path/to/coreml/build/reference-probabilities.json \
swift test --filter CuaS1Forms
```

Set `FLUIDAUDIO_CUA_COMPILED_PATH` to the real `.mlmodelc` to include the shared-cache
check; see [compilation instructions](#device-placement). The portable-model fixture
also checks loading through Hugging Face cache symlinks and temporary-copy cleanup.
Use the ANE-gather package path to check that variant against the same reference. Integration tests skip if
their asset paths are absent; encoding, limits, rejection, and registry unit tests
still run. Tests do not automatically download models or data. XCTest requires a
full Xcode installation on macOS.

## Full published synthetic test

Evaluated the complete published synthetic [`test.jsonl`](https://huggingface.co/datasets/cua-ai/cua-s1-forms/blob/8273f34778b99ac2e12d9f6e7d57dad99ae20845/test.jsonl):
**24,370 decisions across 1,040 episode seeds**, with no excluded rows or truncated
inputs. The dataset revision is `8273f34778b99ac2e12d9f6e7d57dad99ae20845`; its
SHA-256 is `d63a7e0db195d4d20154a40b2f8dd09ce3bb65487a158c638da5c609d4475e7c`.
The upstream [model card](https://huggingface.co/cua-ai/cua-s1-forms/blob/f54adbf447f4ca6ec259f529ee3f2e3e09f8cc71/README.md) reports 99.95% on an approximately 15,000-row
synthetic test. Our result rounds to that accuracy, but the released file contains
24,370 rows; this does not reconstruct the card's unspecified smaller manifest.

| Model / backend | Correct decisions | Top-1 accuracy | Median call | p95 call |
| --- | ---: | ---: | ---: | ---: |
| Upstream PyTorch / CPU | 24,359 / 24,370 | 99.9549% | 1.787 ms | 3.104 ms |
| Original FP16 Core ML / CPU + ANE | 24,359 / 24,370 | 99.9549% | 1.003 ms | 1.133 ms |
| ANE-gather FP16 Core ML / CPU + ANE | 24,359 / 24,370 | 99.9549% | 1.052 ms | 1.177 ms |

Both Core ML exports select the same option as PyTorch on **all 24,370 rows**.
All three share the same 11 errors: choosing `fill` when the label is `skip`.
The original Cua evaluator counts these as wrong/unsafe actions; this offline
benchmark executes no actions. All 9,802 fill, 816 check, and 1,040 click labels
are correct; skip accuracy is 12,701/12,712. Higher ANE placement is approximately
**4.8% slower** by median here, so the original remains the default.

**Strict numerical conversion parity fails for both exports.** Each has 11 rows
above the original 0.005 absolute probability-error limit, with a maximum error
of **0.0204874**. One further row (zero-based index 19270) has a live-probability
sum of **0.99893665**, which failed the original Swift manager's 0.001
normalization guard despite a correct argmax. The [Swift probability fix](#swift-probability-fix)
now handles that output. These raw conversion reports retain the original scores,
failed gates, tolerances, and model weights; the runtime fix does not establish
raw numerical parity. The earlier 196-row demo passed its numerical gates.

Measured September 19, 2026 on **Apple M5 Pro, 24 GB, macOS 27.0 (26A428)**,
Python 3.11.11, PyTorch 2.7.0, coremltools 9.0. Batch size 1, three warmup rows per
model, one timed pass over the whole split; both Core ML models remain loaded
and alternate AB/BA order by row. PyTorch uses two CPU threads and one inter-op
thread, with the Transformer fast path disabled. Timers cover PyTorch
forward + softmax or synchronous Core ML prediction, excluding encoding,
validation, loading, UI, and network. These compare deployment backends, not
algorithms on equal hardware, and differ from the separate Swift timings at the end of this guide.

Upstream describes this synthetic split as disjoint from training/validation by
form signature; those signatures were not independently re-audited here. No
training, validation inference, test-based tuning, or hosted Jev/API comparison
was performed. This measures supplied-option classification, not unseen real-world
GUI completion or document extraction.

[Full report](reports/synthetic-test.json) · [Pinned test manifest](synthetic-test.lock.json) ·
[Compressed per-row trace](https://huggingface.co/FluidInference/cua-s1-forms-coreml/resolve/62ffd3653cf0edef7222a886e2006503e2367d10/reports/synthetic-test-decisions.jsonl.gz). The report records every failed row,
model and harness hashes, action metrics, raw-score NLL/ECE, and the trace hash.
The trace contains every selected index, correctness, confidence, gold score,
probability sum, probability error, and latency. The published models are unchanged.

Reproduce after the base setup and optional ANE-gather conversion:

```bash
uv run --frozen python convert-coreml.py --optimization ane-gather --output-dir build/ane-gather
uv run --frozen python benchmark-synthetic.py --require-parity \
  --report build/synthetic-test-reproduction.json \
  --trace build/synthetic-test-reproduction.jsonl.gz
uv run --frozen pytest -q tests/test_synthetic_test.py
```

The benchmark verifies `synthetic-test.lock.json` before reading the full split;
all rows must fit the existing tensor limits. Use fresh report/trace paths to
preserve previous runs. `--require-parity` writes the complete results and exits
**1** when the unchanged gates fail, as they did in the recorded run. It retains
finite but imperfect probability sums without normalizing them. Structural,
nonfinite, range, and padding failures still abort. The earlier strict attempt
stopped at row 19270; the published report comes from one subsequent complete
pass with normalization failures retained, not samples combined across attempts.

The focused tests inspect the published dataset and recorded real Core ML output;
they do not run full inference. Dataset-dependent tests skip when the separate
test asset has not been downloaded. No training or validation split is downloaded.

## Input and output contract

One model prediction scores one interface element against all its supplied
options. The full context encoder (two layers), option encoder (one layer),
pooling, normalization, and attention readout retain their original weights.

| Tensor | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `context_ids` | int32 | `[1, 224]` | UTF-8 bytes plus one; zero padding |
| `option_ids` | int32 | `[1, 32, 96]` | One encoded option per row |
| `option_mask` | int32 | `[1, 32]` | One for supplied options, zero for padding |
| `logits` | float32 | `[1, 32]` | Raw option scores; padding is `-10000` |
| `probabilities` | float32 | `[1, 32]` | Softmax across supplied options; padding is zero |

`preprocessing.prepare_inputs` implements the upstream UTF-8 **byte** truncation,
byte-plus-one encoding, and zero padding. Context must be nonempty and each
request must contain 2–32 nonempty options. Empty contexts are rejected to avoid
an entirely masked attention readout. Option overflow raises an error; options
are never silently discarded. Text exceeding 224/96 bytes follows the original
model's truncation policy.

All 196 original demo records fit these bounds without truncation: maximum
context length 183 bytes, option length 89 bytes, and option count 27. For a
larger option capacity, export and verify a separate package:

```bash
uv run python convert-coreml.py --max-options 64 --output-dir build/options64
uv run python verify.py --build-dir build/options64 --report reports/options64.json
```

The larger variant is a reproduction option, not an artifact measured in the
checked-in report. The model scores existing document entities and fixed actions;
the surrounding application still owns document extraction and action execution.

## Device placement

The existing `coreml-cli` reports **149 operations on the Neural Engine**,
**24 on CPU**, and **0 on GPU** with `CPU_AND_NE`. The remaining CPU operations
are integer/mask preparation and embedding gathers. This is mixed execution;
the 86.1% figure is an operation count, not a share of runtime.

The September 19 follow-up profiles the same hash-verified package with the public
`MLComputePlan` API across all four policies. It uses rows **0, 68, 130**: one
real initial input from each form, with 27, 21, and 19 supplied options. Per policy,
it measures the first call, runs two warmup passes, then ten timed passes over the
three rows (30 timed predictions). All 120 timed predictions select the correct
label; output validation also checks finite scores, normalization, and padding.

| Policy | CPU ops | GPU ops | ANE ops | Warm p50 | Warm p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CPU_ONLY` | 173 | 0 | 0 | 1.527 ms | 1.602 ms |
| `CPU_AND_GPU` | 0 | 173 | 0 | 0.929 ms | 2.380 ms |
| `CPU_AND_NE` | 24 | 0 | 149 | 0.929 ms | 0.973 ms |
| `ALL` | 0 | 173 | 0 | 0.912 ms | 1.229 ms |

On this M5 Pro, `ALL` selects the GPU; `.cpuAndNeuralEngine` is needed to obtain
the ANE plan measured here. The ANE path had a **566.8 ms load** and a **1.65 ms
first prediction** in this run. System caches were retained, so these are not
first-install cold-start measurements. The table times synchronous Python model
calls with pre-encoded real inputs; it excludes Swift encoding, UI work, and
animation. Its small manifest differs from the original 196-row parity timing.

Operation placement is the scheduler's preferred-device plan, not a measured
utilization, energy, or per-device runtime trace. Xcode Instruments was unavailable
on this Mac. The fallback report identifies 17 integer-type rejections, five
unresolved mask/comparison dependencies, and two unsupported gather-index types.
No graph or weights were changed to obtain this profile.

Reproduce the bounded profile from this directory:

```bash
uv run --frozen python profile-coreml.py
uv run --frozen pytest -q tests/test_profile_manifest.py
```

The report records input rows, package hashes, OS build, policy order, load/first
call timings, every timed prediction, and every assigned operation. Use `--rows`,
`--warmup-passes`, `--timed-passes`, and `--compute-units` for an explicitly chosen
different protocol; the default does not run the full corpus.

The profiler needs a compiled `.mlmodelc` input because its current `.mlpackage`
loader attempts an unsupported `MLModel.save(...mlmodelc)` call. Compile using
the conversion environment first:

```bash
uv run python - <<'PY'
from pathlib import Path
import coremltools as ct

compiled = Path("build/cua_s1_forms_fp16_options32.mlmodelc")
if not compiled.exists():
    ct.utils.compile_model(
        "build/cua_s1_forms_fp16_options32.mlpackage",
        destination_path=str(compiled),
    )
print(compiled.resolve())
PY
```

From `mobius/tools/coreml-cli`, use a separate Python 3.12 environment because
its existing Python 3.14 environment cannot load coremltools' native bindings:

```bash
UV_PROJECT_ENVIRONMENT=/absolute/path/to/ignored/profiler-venv \
uv run --frozen --python 3.12.13 coreml-cli \
  /absolute/path/to/cua_s1_forms_fp16_options32.mlmodelc \
  --fallback --json --plan-timeout 30
```

This uses the existing profiler's placement/fallback path without timing its
randomly generated inputs. All reported model timing and accuracy use real
demo inputs through `verify.py`.

## Conversion details and provenance

### Optional higher-ANE variant

`--optimization ane-gather` raises ANE placement from **149/173 operations
(86.1%) to 162/165 (98.2%)** on this M5 Pro. CPU operations fall from **24 to 3**:
only the three int32 input casts remain. Both `.cpuAndNeuralEngine` and `.all`
select this ANE plan on the test Mac. The original input/output names, dtypes,
shapes, trained layers, and weights are preserved.

The variant prepares masks using shared float16 byte IDs and rewrites the two
embedding gathers to use unsigned 16-bit indices ([Core ML operation reference](https://apple.github.io/coremltools/_modules/coremltools/converters/mil/mil/ops/defs/iOS17/scatter_gather.html)). IDs 0–256 are exactly
representable in both types. This removes signed-index correction and lets ANE
execute the gathers. The conversion uses the pinned coremltools 9.0 MIL API;
the application continues to use public Core ML inference APIs.

The full 196-row check passes on `ALL` and `CPU_AND_NE`, with **196/196 correct
and matching options** and maximum probability error **0.002336**, below the
unchanged 0.005 tolerance. All **28 regression tests** pass against this variant,
including raw byte-ID boundaries, padding, full capacity, truncation, and reordered
options. The Swift manager independently passes all 196 decisions plus cache and
concurrency checks ([Swift report](reports/swift-ane-validation.json)); the native
demo also passes all three form checks. See [verification](reports/ane-gather-verification.json),
[placement](reports/ane-gather-profile.json), and [fallbacks](reports/ane-gather-fallback.json).

More ANE placement did **not** improve latency here. A matched, same-process ABBA
comparison uses rows 0, 68, and 130, two warmup passes per block, and ten timed
passes per block: 60 timed calls per model, all correct.

| Model | CPU ops | ANE ops | Warm p50 | Warm p95 |
| --- | ---: | ---: | ---: | ---: |
| Default | 24 | 149 | 0.915 ms | 0.968 ms |
| `ane-gather` | 3 | 162 | 0.970 ms | 0.988 ms |

The higher-ANE model is about **6% slower** in this small local comparison, so
the original export remains the default. No energy or CPU-time saving is claimed:
operation counts are scheduler assignments, not runtime shares. The model still
needs host-side byte encoding. [Raw comparison](reports/ane-comparison.json).

Build, verify, and compare the optional artifact without replacing the default:

```bash
uv run --frozen python convert-coreml.py --optimization ane-gather --output-dir build/ane-gather
uv run --frozen python verify.py --build-dir build/ane-gather --report reports/ane-gather-verification.json
CUA_COREML_BUILD_DIR=build/ane-gather uv run --frozen pytest -q
uv run --frozen python profile-coreml.py --build-dir build/ane-gather --report reports/ane-gather-profile.json
uv run --frozen python compare-ane.py
```

The optional HF artifact lives under `ane-gather/`. Load its `.mlpackage` with
the existing Swift manager or pass it to the demo's `--model` argument. Default
FluidAudio downloads and the demo's pinned default package stay unchanged.

### Shared export adaptations

The export adapter replaces upstream Boolean mask slice assignments with
equivalent concatenation, uses a floating-point pooling clamp constant, and
makes only padded output logits finite. PyTorch's fused Transformer inference
fast path is disabled for tracing. No trained layer or real option is removed.
Upstream masked attention constants can emit a float16 cast-overflow warning
during conversion; the regression checks verify finite outputs for supported
inputs, including fully occupied option slots and truncated text.

The regression suite covers the real checkpoint, byte encoding, truncation across a
multibyte boundary, padding, invalid input rejection, option-count changes,
option reordering, complete reference export, and coverage when adapting decisions to the original
evaluation harness. Derived text fixtures are numerical regression cases; they
are excluded from the 196-example benchmark accuracy.

Pinned inputs:

- [Model](https://huggingface.co/cua-ai/cua-s1-forms/tree/f54adbf447f4ca6ec259f529ee3f2e3e09f8cc71):
  `f54adbf447f4ca6ec259f529ee3f2e3e09f8cc71`.
- [Dataset](https://huggingface.co/datasets/cua-ai/cua-s1-forms/tree/8273f34778b99ac2e12d9f6e7d57dad99ae20845):
  `8273f34778b99ac2e12d9f6e7d57dad99ae20845`; `demo.jsonl` in the base asset
  lock and `test.jsonl` in the separate synthetic-test lock.
- [Source and evaluator](https://github.com/trycua/cua/tree/83f142c4290a0f7d9ed545ae8532858c6e4f8145/libs/cua-s1):
  `83f142c4290a0f7d9ed545ae8532858c6e4f8145`.

The model loader validates the safetensors/configuration signature and strictly
loads every checkpoint tensor. See [assets.lock.json](assets.lock.json) for
download hashes and [vendor/README.md](vendor/README.md) for the unmodified
reference files, evaluator, and license notices. The pinned model and dataset
cards declare MIT; Cua and Minimal Labs source notices are retained.

## INT8 weight trial

A matched run over all **24,370 synthetic decisions** on the same M5 Pro:

| Export | Package size | Accuracy | Median | p95 |
| --- | ---: | ---: | ---: | ---: |
| Original FP16 | 1.51 MB | 99.9549% | 0.990 ms | 1.102 ms |
| INT8 weights, FP16 compute | 0.81 MB | 99.9549% | 0.990 ms | 1.104 ms |

**46.2% smaller**, with every selected option unchanged and effectively identical
latency. Numerical parity still fails: 64 rows exceed the 0.005 probability-error
limit (maximum 0.067738), versus 11 for FP16. No INT8 probability-sum violations
were observed. The original remains the default.

Weight-only, per-channel symmetric INT8 quantization of the original FP16 export,
using coremltools 9.0 and a 2,048-element threshold. Nineteen weight tensors are
compressed; activations and computation remain floating point. No calibration,
training, or test-based tuning is used. Both model variants stay loaded for one
full test pass, with three warmups each and alternating AB/BA order. Accuracy,
paired choices, strict probability gates, and timings use the existing harness.

The original 196-row demo retains 196/196 choices, but maximum probability error
0.005336 exceeds the unchanged 0.005 threshold. The full-test result above exposes larger score drift than the demo.

The public compute plan assigns **149 ANE and 32 CPU operations**, plus 19
unassigned constant-dequantization operations. `coreml-cli --fallback` counts
those 19 as CPU fallback (51 total); this is not evidence that all 19 execute
on every prediction. These are scheduler counts, not utilization or energy.
The trial does not establish INT8 activation computation or an ANE speedup.

The quantizer emits a zero-scale warning for the all-zero embedding padding row.
Tests verify that all 19 dequantized weight tensors are finite and that zero
channels remain zero. The tensor interface and minimum deployment version match
the original. The quantization manifest retains source hashes and exact settings.

Reproduce after preparing the original baseline:

```bash
uv run --frozen python quantize-int8.py
uv run --frozen python verify.py --build-dir build/int8-weights \
  --compute-units CPU_AND_NE --report build/int8-demo-reproduction.json
uv run --frozen python benchmark-synthetic.py --candidate-name int8-weights \
  --report build/int8-test-reproduction.json --trace build/int8-test-reproduction.jsonl.gz \
  --require-parity
uv run --frozen python profile-coreml.py --build-dir build/int8-weights \
  --compute-units CPU_AND_NE --report build/int8-profile-reproduction.json
uv run --frozen pytest -q tests/test_quantized_weights.py tests/test_synthetic_test.py
```

Use fresh output paths. The numerical checks return exit 1 for the recorded
failures; no tolerance is relaxed. The quantization and benchmark tests cover real
compression/interface preservation, finite dequantized weights, zero padding,
source/variant validation, full-test coverage, and metric edge cases.

Reports: [full test](reports/int8-synthetic-test.json),
[demo](reports/int8-demo-verification.json), [compute plan](reports/int8-profile.json),
and [CLI fallback](reports/int8-fallback.json).
The experimental package and complete per-row trace are under `int8-weights/` and
`reports/int8-synthetic-test-decisions.jsonl.gz` in the
[model repository](https://huggingface.co/FluidInference/cua-s1-forms-coreml).
Load the portable INT8 package locally with the existing Swift manager; the
standard model download continues to use the original FP16 artifact.

## INT4 weight trial

A matched run over all **24,370 synthetic decisions** on M5 Pro, CPU+ANE:

| Export | Package size | Accuracy | Median | p95 |
| --- | ---: | ---: | ---: | ---: |
| FP16 control, iOS 18 target | 1.51 MB | 99.9549% | 0.982 ms | 1.081 ms |
| INT4 weights, FP16 compute | 0.45 MB | 99.9302% | 0.982 ms | 1.082 ms |

**70.1% smaller**, with essentially unchanged latency. INT4 makes **17 errors
versus 11** for FP16: 14 choices change, introducing 10 errors and correcting four.
Numerical parity fails: 356 rows exceed the 0.005 probability-error limit
(maximum 0.754359); no INT4 probability-sum violations were observed.

Packed INT4 requires **iOS 18/macOS 15**. Both exports use the same decomposed
attention graph and FP16 computation. Batch-1 timing excludes encoding/loading/UI.
INT8 preserves all choices at 0.81 MB; the original FP16 remains the default.

The exact portable sizes are 1,511,080 bytes for the retargeted FP16 control and
451,595 bytes for INT4. `prepare-int4-source.py` loads the hash-verified original
FP16 MIL at specification 9 with an empty pass pipeline, preserving its weights
and decomposed attention. A direct iOS18 PyTorch conversion uses fused attention
and failed the demo even before quantization; that graph is excluded from this
comparison. No full synthetic run used that failed graph.

`quantize-int4.py` applies per-channel symmetric INT4, threshold 2,048, to 19
weight tensors. Packed INT4 blobs feed `constexpr_blockwise_shift_scale`; scales,
activations and computation remain FP16. No calibration, training, or test-based
tuning was applied. The original and retargeted FP16 controls have identical
weights and nonconstant operation counts, checked against the actual artifacts.

The 196-row demo retains all choices; its INT4 maximum probability error is
0.200770. The full run above exposes changed choices and fails the original
conversion gates. All 28 focused tests pass, covering both compressed artifacts,
finite dequantized weights and zero padding channels, packed storage and target,
source identity, interface preservation, dataset coverage and metric accounting.

The public compute plan assigns **149 ANE and 32 CPU operations**, plus 19
unassigned constant-dequantization operations. CLI fallback counts those 19 as
CPU (51 total). These are scheduler assignments, not runtime utilization or
proof of INT4 activation computation. No speed or energy gain is established.

Reproduce after preparing the original baseline; use fresh output paths:

```bash
uv run --frozen python prepare-int4-source.py
uv run --frozen python verify.py --build-dir build/int4-source-fp16 \
  --compute-units CPU_AND_NE --report build/int4-source-demo-reproduction.json
uv run --frozen python quantize-int4.py
uv run --frozen python verify.py --build-dir build/int4-weights \
  --compute-units CPU_AND_NE --report build/int4-demo-reproduction.json
uv run --frozen python benchmark-synthetic.py --baseline build/int4-source-fp16 \
  --candidate-name int4-weights --report build/int4-test-reproduction.json \
  --trace build/int4-test-reproduction.jsonl.gz --require-parity
uv run --frozen python profile-coreml.py --build-dir build/int4-weights \
  --compute-units CPU_AND_NE --report build/int4-profile-reproduction.json
uv run --frozen pytest -q tests/test_quantized_weights.py tests/test_synthetic_test.py
```

INT4 demo/full-test numerical checks return exit 1 after saving their reports.
The original target and default download remain unchanged. Experimental packages
and conversion manifests are under `int4-weights/` and `int4-source-fp16/` in the
[model repository](https://huggingface.co/FluidInference/cua-s1-forms-coreml).
Reports: [full test](reports/int4-synthetic-test.json),
[INT4 demo](reports/int4-demo-verification.json),
[FP16 control demo](reports/int4-source-demo-verification.json),
[compute plan](reports/int4-profile.json), [CLI fallback](reports/int4-fallback.json).
The complete per-row trace is `reports/int4-synthetic-test-decisions.jsonl.gz` in
the model repository. Source, package, script and trace hashes are retained in manifests.

## Swift probability fix

The Swift manager computes a stable softmax from live logits using Double arithmetic
and returns Float `probabilities`. The model's original softmax output remains
available as `rawProbabilities`, including FP16 rounding errors.

All **73,110 real Swift API calls completed** on the pinned synthetic split
(Apple M5 Pro, 24 GB, macOS 27.0; Swift 6.2.3, release build, CPU+ANE):

| Variant | Calls completed | Correct decisions | Changed choices after fix | Probability-sum failures after fix |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 24,370 | 24,359 (99.9549%) | 0 | 0 |
| INT8 | 24,370 | 24,359 (99.9549%) | 0 | 0 |
| INT4 | 24,370 | 24,353 (99.9302%) | 0 | 0 |

The recorded FP16 sum failure is fixed. Stable probabilities agree with an
independent float64 softmax within **0.000000030**. Raw conversion-parity failures
and INT4's accuracy loss remain. This validation changes no model artifacts and
makes no new latency claim. Earlier complete Swift timings predate this fix.

[Runtime report](reports/swift-runtime/report.json) · [Full runtime trace](https://huggingface.co/FluidInference/cua-s1-forms-coreml/resolve/7f632a6b137aab69b20e517ecf51cf6eaa599b6b/reports/swift-runtime/decisions.jsonl.gz)
The report pins the dataset, packages, saved reference traces, Swift sources,
and validation harness by SHA-256.

Reproduce on an Apple silicon Mac after the base setup. Supply a FluidAudio
checkout containing the fix and a complete local download of the model repository,
including its saved reports and traces:

```bash
uv run --frozen python validate-swift-runtime.py \
  --fluidaudio /path/to/FluidAudio \
  --models /path/to/cua-coreml \
  --output-dir build/swift-runtime-reproduction
```

Use a fresh output directory. The command builds the actual Swift library in
release mode, checks every row once per variant, and audits its saved output
against the original raw decisions and an independent softmax. The Python
regressions use the recorded real-model failure and run with `uv run --frozen pytest -q`.

## Live browser proof and expanded Swift benchmark

The [native Swift browser demo](https://github.com/FluidInference/FluidAudio/tree/7f9eb92b0af8594c4e048a9e57f697340aacfa67/Examples/CuaS1FormsDemo)
loads both variants into independent WKWebViews. It reads actual DOM labels,
roles and state, asks the model for a choice, applies compatible fill/check
actions, dispatches events, and independently verifies the resulting DOM.
Source values are user-entered or supplied by the original public examples.
HTML contains controls, not source values or expected choices. The
[recording](https://github.com/FluidInference/FluidAudio/blob/7f9eb92b0af8594c4e048a9e57f697340aacfa67/Examples/CuaS1FormsDemo/browser-demo.mp4)
shows patient, job and insurance forms: **100/100 original decisions** across
both models, with event-count, stale-observation and explicit-click checks.
Full actual contexts/candidates/actions are in [browser-validation.json](reports/browser-validation.json).
This is bounded local browser automation, not arbitrary desktop control or PDF extraction.

The expanded **release Swift** comparison uses all 50 initial controls, both
models resident, one warmup pass per model and ABBA with two full passes per
block (200 timed calls/model). All 400 choices match upstream labels. On the
M5 Pro / 24 GB / macOS 27.0 (26A428), original median/p95 is **0.912/0.933 ms**;
ANE gather is **0.961/0.984 ms**, about 5.4% slower by median. This timer includes
Swift encoding + Core ML + output decoding and excludes browser/rendering/animation.
See [swift-variant-comparison.json](reports/swift-variant-comparison.json) for
raw samples, exact model hashes, per-form statistics, and load/first-call costs.
The earlier compute-plan counts still apply to these unchanged artifacts;
no utilization, energy saving or held-out accuracy claim is made.

Reproduce from FluidAudio commit `7f9eb92b0af8594c4e048a9e57f697340aacfa67` (the example is
retained in history and is not part of the current library PR):

```bash
git worktree add --detach /tmp/cua-s1-browser-repro 7f9eb92b0af8594c4e048a9e57f697340aacfa67
cd /tmp/cua-s1-browser-repro
Examples/CuaS1FormsDemo/run.sh --browser
swift run --package-path Examples/CuaS1FormsDemo -c release CuaS1FormsDemo \
  --benchmark --report /absolute/path/to/variant-comparison.json \
  --hardware "Describe the measured Mac"
```

Both packages are fetched by pinned revisions and verified hashes, or supplied
with `--model /path/to/original.mlpackage --ane-model /path/to/ane-gather.mlpackage`.
The [demo README](https://github.com/FluidInference/FluidAudio/tree/7f9eb92b0af8594c4e048a9e57f697340aacfa67/Examples/CuaS1FormsDemo#matched-swift-benchmark)
also documents real-browser recording and its separate validation trace.
