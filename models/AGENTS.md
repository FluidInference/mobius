# Repository Guidelines

## Project Structure & Module Organization
- Code lives under `models/{class}/{model}/{target}`; mirror existing patterns like `vad/silero-vad/coreml`.
- Each target directory is self-contained: `pyproject.toml`, `uv.lock`, conversion scripts, docs, and sample assets.
- Keep `README.md`/`CITATION.cff` next to the model. Push large binaries to Hugging Face and reference them here.

## Build, Test, and Development Commands
Run these from the target directory (Python 3.10.12):
- `uv sync` — create/refresh the env defined by `pyproject.toml`.
- `uv run python convert-coreml.py --output-dir ./build/<name>` — run conversion and emit CoreML bundles.
- `uv run python compare-models.py --audio-file <path> --coreml-dir <dir>` — benchmark converted models (if present).
- `uv run python test.py` — execute the model-specific smoke test.

## Deployment Targets & Runtime Tips
- Trace with `.CpuOnly`. Target iOS 17+ and macOS 14+.
- Use `uv` for reproducible installs; avoid system Python.
- Keep bundles small; prefer float16 where supported.

## Coding Style & Naming Conventions
- 4-space indentation, type hints when practical, and double-quoted strings.
- Lowercase-kebab-case for files/dirs; mirror upstream model names and runtime targets (`coreml`, `onnx`, etc.).
- When packaging libraries, place importable code under `src/<package>` and expose CLIs via `if __name__ == "__main__": main()`.

## Testing Guidelines
- Ship a runnable sanity check using bundled assets (e.g., `yc_first_minute.wav`) and verify end-to-end output.
- Prefer deterministic assertions or concise summary prints; record expected metrics/speedups for benchmarking utilities.
- Document prerequisites such as `git lfs install` before fetching large checkpoints.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subjects; append issue numbers when relevant (e.g., `Move parakeet to the right folder (#4)`).
- Pull requests: describe the model, destination runtime, conversion steps, and validation evidence (logs, plots, or HF links). Call out deviations, new dependencies, and follow-up work.

## Model Assets & Distribution
- Store heavy weights, notebooks, and rendered plots externally (Hugging Face Hub). Include download instructions or automation scripts.
- Verify upstream license compliance before redistribution.

