# Repository Guidelines

## Project Structure & Module Organization
Code lives under `models/{class}/{model}/{target}`; follow the existing `vad/silero-vad/coreml` pattern when adding new work. Each target directory bundles its own `pyproject.toml`, `uv.lock`, conversion scripts, docs, and sample assets, functioning as a self-contained toolkit. Keep notes or licensing files (`README.md`, `CITATION.cff`) next to the model and push large binaries to Hugging Face.

## Build, Test, and Development Commands
- `uv sync` — create or refresh the local environment defined by the active target's `pyproject.toml` (Python 3.10.12).
- `uv run python convert-coreml.py --output-dir ./build/<name>` — run the conversion pipeline and emit CoreML bundles.
- `uv run python compare-models.py --audio-file <path> --coreml-dir <dir>` — benchmark converted models when comparison tooling is present.
- `uv run python test.py` — execute the model-specific smoke test (e.g., diarization pipeline in `pyannote-community-1`).
Run these from the target directory so relative paths resolve.

## Profiling Converted Models
After conversion, use `coreml-cli` to verify ANE compatibility and measure performance.

**Setup** (one-time): `cd tools/coreml-cli && uv sync`

**Benchmarking** — latency, compile time, and device assignment across compute units:
- `uv run coreml-cli path/to/model.mlmodelc` — profile a single model across all compute-unit configs.
- `uv run coreml-cli path/to/models/` — profile all `.mlmodelc` models in a directory.
- `uv run coreml-cli model.mlmodelc --json` — JSON output for programmatic use.
- `uv run coreml-cli model.mlmodelc --ops` — include per-operation device breakdown.
- `uv run coreml-cli model.mlmodelc --detailed` — add private API data (backend support, estimated runtimes per backend).

**ANE fallback analysis** — identify why ops fall back to CPU and what to fix:
- `uv run coreml-cli model.mlmodelc --fallback` — show CPU ops grouped by ANE rejection reason.
- `uv run coreml-cli model.mlmodelc --fallback --json` — structured JSON for agent parsing.

Use `--fallback` in the optimization loop: change conversion → reconvert → `--fallback` → fix blockers → repeat. Common fixes: cast int32 to float16, decompose unsupported ops (LSTM, logical_and), fix cascading dependencies.

## Deployment Targets & Runtime Tips
Things to keep in mind:
- Trace with `.CpuOnly`.
- Target iOS17+ (most users are on iOS17/macOS 14 right now).
- Use `uv` to manage dependencies.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints when practical, and prefer double-quoted strings. Keep filenames and directories lowercase-kebab-case, mirroring upstream model names and runtime targets (`coreml`, `onnx`, etc.). When packaging libraries, put importable code under `src/<package>` and expose CLIs via `main()` guards. Update per-model READMEs with concise variant notes and cite upstream work.

## Testing Guidelines
Ship a runnable sanity check: load bundled sample assets (e.g., `yc_first_minute.wav`) and verify end-to-end output. Document prerequisites such as `git lfs install` before cloning large checkpoints. Prefer deterministic assertions or summary prints, and capture expected metrics or speedups when adding benchmarking utilities.

## Commit & Pull Request Guidelines
Write concise, imperative commit subjects and append issue numbers when relevant (`Move parakeet to the right folder`, `... (#4)`). Pull requests should describe the model, destination runtime, conversion steps, and validation evidence (logs, plots, or HF links). Highlight deviations from existing structure, call out new dependencies, and note follow-up work so reviewers can reason about downstream impact quickly.

## Model Assets & Distribution
Store hefty weights, notebooks, and rendered plots externally (Hugging Face Hub or blog posts) and keep only lightweight helpers in-repo. Include download instructions or automation scripts when third-party assets are required, and double-check that redistribution complies with upstream licenses before publishing.
