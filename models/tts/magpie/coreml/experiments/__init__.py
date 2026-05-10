"""Magpie-TTS → CoreML experiments — diagnostic and dead-end scripts.

Standalone CLI drivers used to probe specific levers (AR-loop unroll,
weight quantization, MLState KV caches). None of these are part of the
production conversion pipeline; production exporters live one level up
in `coreml/` (e.g. `convert_decoder_step.py`, `convert_text_encoder.py`,
`convert_nanocodec.py`, `convert_local_transformer.py`).

Kept in tree for reproducibility — each entry below points at the trial
entry in `coreml/PERF.md` that motivated it.

CLI driver scripts:

| Module                              | Trial / verdict                                          |
|-------------------------------------|----------------------------------------------------------|
| `quantize_decoder_step_int8.py`     | Trial 2 — int8 weight quant (DEAD-END; EOS runaway)      |
| `convert_decoder_step_n2.py`        | Trial 4a — N=2 AR-loop unroll (DEAD-END; ANE rejects)    |
| `convert_decoder_step_stateful.py`  | STATEFUL MLState variant (DEAD-END; 2.2× CPU+GPU regress)|

Experimental traceable wrappers (in `traceable/`):

| Module                                       | Used by                              |
|----------------------------------------------|--------------------------------------|
| `traceable/traceable_decoder_step_n2.py`     | `convert_decoder_step_n2.py`         |
| `traceable/traceable_decoder_step_stateful.py` | `convert_decoder_step_stateful.py` |

Sub-package: `baseline_fp32/` — fp32 stage-by-stage diagnostic bench
(see `coreml/BASELINE_FP32.md`) plus the Option 1 tail-fp16 MIL-rewrite
probe (Trial 11 in `PERF.md`).

The mlpackages produced by these scripts land in `coreml/build/` and
`coreml/compiled/build/` (gitignored) — they're throwaway diagnostic
artifacts, not part of any HuggingFace upload.

Each CLI script prepends the parent (`coreml/`) directory to `sys.path`
so the production wrappers under `coreml/traceable/` and the
experimental wrappers under `coreml/experiments/traceable/` both
resolve via absolute imports.
"""
