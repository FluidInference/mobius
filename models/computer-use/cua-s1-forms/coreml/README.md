# CUA-S1-FORMS → Core ML

Converts the real **706,048-parameter** CUA-S1-FORMS checkpoint to a **1.51 MB
FP16 Core ML package**. The complete pinned 196-decision demo passes: PyTorch and
Core ML both get **196/196 correct**, with **196/196 matching selected options**.
The unmodified upstream Cua evaluation harness also reports no wrong actions or
wrong targets on these saved decisions.

This is a bounded local conversion verification. The demo covers three forms and
three PDFs; it does not establish generalization to other forms or successful
execution in a live application. No training or full synthetic-corpus evaluation
is needed to reproduce it.

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
SHA-256. The [FluidAudio integration checks](https://github.com/FluidInference/FluidAudio/blob/codex/cua-s1-forms/Documentation/Decision/CuaS1Forms.md)
consume this file without requiring PyTorch in the application.

The converted package and compiled bundle are proposed in
[Hugging Face model PR #1](https://huggingface.co/FluidInference/cua-s1-forms-coreml/discussions/1).
During review, use the model PR revision; automatic FluidAudio downloading uses
the model repository's `main` branch after that PR lands.

Python **3.11.11**, PyTorch **2.7.0**, and coremltools **9.0** are pinned. Python
3.11 follows the upstream package's minimum version rather than the older Mobius
Python 3.10 default. `uv.lock` pins the remaining dependencies.

`assets.py` verifies SHA-256 hashes and downloads only missing pinned assets:
the safetensors checkpoint, configuration, model/dataset cards, and `demo.jsonl`.
The original pickle checkpoint and the train/validation/synthetic-test datasets
are not downloaded. Existing files with incorrect hashes fail validation.

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

The export adapter replaces upstream Boolean mask slice assignments with
equivalent concatenation, uses a floating-point pooling clamp constant, and
makes only padded output logits finite. PyTorch's fused Transformer inference
fast path is disabled for tracing. No trained layer or real option is removed.
Upstream masked attention constants can emit a float16 cast-overflow warning
during conversion; the regression checks verify finite outputs for supported
inputs, including fully occupied option slots and truncated text.

Eighteen tests cover the real checkpoint, byte encoding, truncation across a
multibyte boundary, padding, invalid input rejection, option-count changes,
option reordering, complete reference export, and coverage when adapting decisions to the original
evaluation harness. Derived text fixtures are numerical regression cases; they
are excluded from the 196-example benchmark accuracy.

Pinned inputs:

- [Model](https://huggingface.co/cua-ai/cua-s1-forms/tree/f54adbf447f4ca6ec259f529ee3f2e3e09f8cc71):
  `f54adbf447f4ca6ec259f529ee3f2e3e09f8cc71`.
- [Dataset](https://huggingface.co/datasets/cua-ai/cua-s1-forms/tree/8273f34778b99ac2e12d9f6e7d57dad99ae20845):
  `8273f34778b99ac2e12d9f6e7d57dad99ae20845`, `demo.jsonl` only.
- [Source and evaluator](https://github.com/trycua/cua/tree/83f142c4290a0f7d9ed545ae8532858c6e4f8145/libs/cua-s1):
  `83f142c4290a0f7d9ed545ae8532858c6e4f8145`.

The model loader validates the safetensors/configuration signature and strictly
loads every checkpoint tensor. See [assets.lock.json](assets.lock.json) for
download hashes and [vendor/README.md](vendor/README.md) for the unmodified
reference files, evaluator, and license notices. The pinned model and dataset
cards declare MIT; Cua and Minimal Labs source notices are retained.
