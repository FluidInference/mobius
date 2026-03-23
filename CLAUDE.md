# mobius

Model conversion and optimization hub for edge AI on Apple Silicon. Converts PyTorch/ONNX models to CoreML, targeting ANE (Neural Engine).

## Structure
- `models/{class}/{model}/{target}/` — self-contained conversion pipelines (each has own `pyproject.toml`)
- `tools/coreml-cli/` — CLI tool to profile CoreML models (device assignment, latency, compile time)
- `Documentation/` — guides (ModelConversion.md is the main one)
- `knowledge/` — curated research papers and platform docs

## Key Commands
All commands run from the target directory (e.g., `models/stt/parakeet-tdt-v3-0.6b/coreml/`):
- `uv sync` — set up environment
- `uv run python convert-coreml.py --output-dir ./build/<name>` — run conversion
- `uv run python compare-models.py --audio-file <path> --coreml-dir <dir>` — validate parity

Profiling (from `tools/coreml-cli/`):
- `uv run coreml-cli path/to/model.mlmodelc` — benchmark: latency, compile time, device % across all compute unit configs
- `uv run coreml-cli model.mlmodelc --fallback` — ANE optimization: show CPU fallback ops grouped by rejection reason
- `uv run coreml-cli model.mlmodelc --fallback --json` — structured fallback analysis for agent parsing

## Constraints
- Trace with `.CpuOnly`
- Target iOS 17+ / macOS 14+
- Fixed input shapes only (no dynamic dimensions)
- Use `uv` for all dependency management
- Each conversion directory is self-contained with its own `pyproject.toml`

## Style
- 4-space indent, type hints, double-quoted strings
- Lowercase-kebab-case for directories and files
- Document trials, errors, and platform quirks — future agents rely on this context

## Model Classes
`stt`, `tts`, `vad`, `speaker-diarization`, `emb`, `segment-text`

## Workflow
See `Documentation/ModelConversion.md` for the full 11-step pipeline: directory setup → conversion → validation → profiling → documentation → HuggingFace upload → PR.
