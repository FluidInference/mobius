# Exploratory demo scope

Read the [shared project handoff](../../project-handoff.md) for cross-environment
continuation. This folder's demo-only rules do not prohibit already-authorized
training work in the designated environment; they prevent treating saved demo
evidence or a documentation task as authority to start such work.

This folder contains saved evidence from a 12-clip local sampling/demo exercise,
not an official quality baseline or a trainer. Keep the source Mac for small
demos, code preparation, and lightweight integrity/unit checks only.

- Do not respond to a request about this demo by launching model inference,
  full-suite evaluation, full-corpus preprocessing, or training.
- Prefer existing WAVs and saved evidence. Audio stays local and ignored by Git.
- Official evaluation/training requires a separately agreed environment,
  data/split manifest, protocol, and budget. Do not implicitly provision GPUs,
  upload recordings, or start remote jobs.
- The Swift/Core ML and MLX source tooling is not a ready-made CUDA stack.
- Treat ASR/naturalness/speaker diagnostics as proxies, not acoustic ground
  truth. Preserve alternate hypotheses and the distinction between frontend
  findings and unresolved acoustic/evaluator questions.
- Run `uv run python verify-demo.py` and the standard-library unit tests when
  editing the package. Add `--audio-dir` only to check existing WAV hashes;
  this must not synthesize or rescore audio.
- Keep internal plans under ignored `.mobius/`; do not commit them here.
