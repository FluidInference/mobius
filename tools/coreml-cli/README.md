# coreml-cli

A command-line tool to profile CoreML models — showing per-operation compute device assignments (CPU/GPU/ANE), compilation time, and prediction latency across all `MLComputeUnits` configurations.

Replicates what Xcode's CoreML Performance Report does, but from the terminal and designed for programmatic use by coding agents.

## Example

```
$ coreml-cli test_models/160ms/

Device:    Apple M4 Pro (arm64)
RAM:       48GB
OS:        macOS 26.3.1

── decoder ────────────────────────────────────────────────────────────────────
  Parakeet EOU decoder (RNNT prediction network) (Fluid Inference)
  Mixed (Float16, Float32, Int16, Int32) | torch==2.4.0 | coremltools 8.3.0
  inputs:  targets(Int32 1×1), target_length(Int32 1), h_in(Float32 1×1×640),
           c_in(Float32 1×1×640)
  outputs: decoder(Float32 1×640×1), h_out(Float32 1×1×640),
           c_out(Float32 1×1×640)
  cold compile: 128ms

  Compute Unit                 CPU    GPU    ANE   Compile   Predict
  ────────────────────────────────────────────────────────────────
  all                       100.0%   0.0%   0.0%       7ms    0.22ms
  cpu_only                  100.0%   0.0%   0.0%       6ms    0.22ms
  cpu_and_gpu               100.0%   0.0%   0.0%       6ms    0.23ms
  cpu_and_neural_engine     100.0%   0.0%   0.0%       5ms    0.26ms

── streaming_encoder ──────────────────────────────────────────────────────────
  Mixed (Float16, Float32, Int32) | torch==2.4.0 | coremltools 8.3.0
  ...
  cold compile: 3512ms

  Compute Unit                 CPU    GPU    ANE   Compile   Predict
  ────────────────────────────────────────────────────────────────
  all                         0.0% 100.0%   0.0%      46ms    6.78ms
  cpu_only                  100.0%   0.0%   0.0%      47ms    5.45ms
  cpu_and_gpu                 0.0% 100.0%   0.0%      49ms    6.67ms
  cpu_and_neural_engine       1.2%   0.0%  98.8%      51ms    2.82ms
```

## Install

Requires macOS 14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yourusername/coreml-cli
cd coreml-cli
uv sync
```

## Usage

```bash
# Profile a single model (all compute unit configs)
uv run coreml-cli model.mlmodelc

# Profile all models in a directory
uv run coreml-cli path/to/models/

# Specific compute unit config
uv run coreml-cli model.mlmodelc --units cpu_and_neural_engine

# JSON output (for programmatic use)
uv run coreml-cli model.mlmodelc --json

# Include per-operation breakdown
uv run coreml-cli model.mlmodelc --ops

# Per-op data with private API details (backend support, estimated runtimes)
uv run coreml-cli model.mlmodelc --detailed

# ANE fallback analysis — show CPU ops grouped by rejection reason
uv run coreml-cli model.mlmodelc --fallback

# Fallback analysis as JSON (for agent consumption)
uv run coreml-cli model.mlmodelc --fallback --json

# Control benchmark iterations
uv run coreml-cli model.mlmodelc --iterations 50

# Debug logging to stderr
uv run coreml-cli model.mlmodelc --debug
```

## What it reports

### Benchmark mode (default)

For each model:

- **Cold compile time** — measured once per model by bypassing the E5 compilation cache (private API: `setExperimentalMLProgramEncryptedCacheUsage_(0)`). Reflects what users experience the first time the model runs on their device — if this is too high, the model may not be usable. For a true first-launch measurement, restart `ANECompilerService` before benchmarking: `sudo killall ANECompilerService`.

For each compute unit configuration (`all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_neural_engine`):

- **Device assignment** — % of operations on CPU, GPU, and ANE (Neural Engine)
- **Compile time** — cached load time (E5 bundle cache populated). This is the cost paid on every app launch.
- **Predict latency** — median prediction time (5 warmup + 10 timed iterations)
- **Model metadata** — precision, I/O shapes, author, description, coremltools version
- **Per-op breakdown** (`--ops`) — each operation's name, type, assigned device, and cost weight
- **Private API data** (`--detailed`) — selected backend, all supported backends, estimated runtime per backend, validation messages explaining why backends were rejected

### Fallback analysis mode (`--fallback`)

Shows only ops that are **not** on ANE, grouped by rejection reason. Designed for the ANE optimization loop: change conversion → reconvert → `--fallback` → identify blockers → fix → repeat.

For each CPU-fallback op, reports:
- **Why ANE rejected it** — e.g., "Unsupported tensor data type: int32", "Unsupported MIL operation"
- **How many ops** — grouped by rejection reason with op type counts
- **Estimated CPU cost** — how much latency the fallback adds
- **Which ops** — names for tracing back to the conversion script

Common ANE rejection reasons and fixes:
- `Unsupported tensor data type: int32` — cast to float16 before these operations
- `Unsupported MIL operation "lstm"` — decompose into supported ops (matmul, sigmoid, tanh)
- `Unsupported MIL operation "logical_and"` — replace with float multiply workaround
- `Unable to resolve operation input` — cascading from another CPU op; fix the upstream op first
- `ANE supported but scheduler chose CPU` — data transfer overhead; often not worth fixing

## How it works

Uses [PyObjC](https://pyobjc.readthedocs.io/) to call macOS CoreML framework APIs directly from Python:

1. **Public API** — `MLComputePlan` (macOS 14+) for per-operation device assignment and cost weights
2. **Private API** — `MLE5Engine.segmentationAnalyticsAndReturnError:` for richer data including backend support matrices and estimated runtimes per backend

Heavily inspired by: 

- **[maderix/ANE](https://github.com/maderix/ANE)** — reverse-engineered private `_ANEClient`/`_ANECompiler` APIs for direct Neural Engine access. Their runtime introspection approach (`objc_msgSend`, `NSClassFromString`) informed how we navigate CoreML's internal object graph.
- **[freedomtan/coreml_modelc_profling](https://github.com/freedomtan/coreml_modelc_profling)** — per-operation profiling using both public `MLComputePlan` and undocumented `MLE5Engine` APIs. Their Objective-C implementation was the direct reference for our private profiler.

## Caveats

**Note that this was a weekend project, built with Claude Code.**

- **Hardware-specific** — compute plans and compilation are tied to the local chip. Results on an M4 Pro will differ from an M1 or A17 Pro.
- **Private APIs may break** — the `MLE5Engine` path (`--detailed`) uses undocumented APIs that may change across macOS versions.
- **macOS 26 tested** — CoreML enum values changed in macOS 26 (Tahoe). The tool uses framework constants to stay portable, but has only been tested on macOS 26.

## License

MIT
