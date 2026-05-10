# Magpie-TTS CoreML — experiments

Diagnostic and dead-end scripts kept for reproducibility; not part of
the production conversion pipeline. Production exporters live one
level up in [`../`](../) (e.g. `convert_decoder_step.py`,
`convert_text_encoder.py`, `convert_nanocodec.py`,
`convert_local_transformer.py`).

Each script here corresponds to a numbered trial in
[`../PERF.md`](../PERF.md). Run any of them in isolation to reproduce
the captured findings.

Note: the `nanocodec_experiments/` directory at `../nanocodec_experiments/`
is a separate experimental track (NanoCodec W ≤ 16,384 / variant
sweeps) with its own `results/STATUS.md` and is not consolidated here.
