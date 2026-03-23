# coreml-cli

## Project
Python CLI tool to profile CoreML model compute device assignments (CPU/GPU/ANE) and latency.
Uses PyObjC to call CoreML framework APIs (both public and private/reverse-engineered).
Managed by `uv`. Entry point: `coreml-cli = "coreml_cli.cli:app"`.

## Commands
- `uv run coreml-cli <model>` — benchmark: latency, compile time, device % across all compute unit configs
- `uv run coreml-cli <model> --fallback` — ANE optimization: show CPU fallback ops grouped by rejection reason
- `uv run coreml-cli <model> --fallback --json` — structured fallback analysis for agent parsing
- `uv run coreml-cli <model> --json` — full benchmark as JSON
- `uv run coreml-cli <model> --ops` — include per-operation breakdown
- `uv run coreml-cli <model> --detailed` — private API data per op (implies --ops)
- `uv run coreml-cli <model> --debug` — progress logs to stderr
- `uv sync` — install/update dependencies

## Architecture
- `cli.py` — Typer CLI, wiring, merge logic
- `compute_plan.py` — Public MLComputePlan API (device assignment + cost per op)
- `private_profiler.py` — Private MLE5Engine API (backend support, estimated runtimes)
- `fallback.py` — Analyze CPU fallback ops, group by ANE rejection reason
- `latency.py` — Prediction timing via PyObjC MLModel (5 warmup + N iterations)
- `metadata.py` — Model metadata from metadata.json + MLModel description
- `model_loader.py` — Discover .mlmodelc/.mlpackage, compile via coremltools
- `output.py` — Table (default), fallback table, and JSON formatters

## Critical: macOS 26 (Tahoe) Enum Changes
CoreML enum values differ from older macOS. NEVER hardcode enum ints.
Use framework constants: `CoreML.MLComputeUnitsAll`, `CoreML.MLFeatureTypeMultiArray`, etc.
Example: MLComputeUnitsCPUOnly=0 on macOS 26 vs 1 on older versions.

## PyObjC Gotchas
- `MLComputePlan` has no sync class method — use async `loadContentsOfURL_configuration_completionHandler_` with threading.Event
- `op.outputs()` returns NSArray of MLModelStructureProgramNamedValueType, not a dict — use `outputs[0].name()`
- `MLMultiArray.dataPointer()` returns `objc.varlist`, not int — use `setObject_atIndexedSubscript_` to fill values
- `segmentationAnalyticsAndReturnError_` returns single value (not tuple) — don't unpack
- `SelectedBackend` from private API has embedded quotes like `'"bnns"'` — strip them
- `MLModelConfiguration.setExperimentalMLProgramEncryptedCacheUsage_(0)` bypasses the E5 on-disk compilation cache (used for cold compile measurement). The ANECompilerService daemon also caches in memory — for true first-launch timing, restart it: `sudo killall ANECompilerService`

## Design Principles
- Agent-first: table output default, `--json` for structured. No rich/colorama.
- stdout = data only, stderr = debug logs (gated by `--debug`)
- Suppress CoreML system logs via `OS_ACTIVITY_DT_MODE=disable` unless debugging
- Operations list excluded by default (too verbose) — opt-in via `--ops`
- Errors raise exceptions, don't print-and-continue

## Test Models
`test_models/160ms/` — downloaded from HuggingFace FluidInference/parakeet-realtime-eou-120m-coreml
Contains: decoder.mlmodelc, joint_decision.mlmodelc, streaming_encoder.mlmodelc
