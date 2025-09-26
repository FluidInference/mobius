# Agent Notes

- Preferred Python workflow uses `uv` (https://github.com/astral-sh/uv).
  - Create and manage environments with `uv venv`.
  - Install dependencies with `uv pip install` or `uv pip sync` as needed.
- When working in this repo, activate the local `.venv` and run tooling through `uv run` to keep resolutions reproducible.
- Keep CoreML conversions constrained to the fixed 15-second audio window when exporting or validating Parakeet components.
