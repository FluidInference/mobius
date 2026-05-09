"""CoreML fp32 conversion for StyleTTS2 LibriTTS.

Per-stage `nn.Module` wrappers (`wrappers.py`), trace + convert
(`convert.py`), and per-stage parity vs `pipeline.stages`
(`parity.py`). All stages re-use the same `model[k]` instances loaded
by `run_inference.load_styletts2` — see `trials.md`.
"""
