"""fp32 baseline harness for Magpie-TTS — diagnostic-only.

Captures per-stage parity and ANE residency for fp32 builds vs the
shipping fp16 production models. Outputs land in
`coreml/build/fp32/<stage>_fp32.mlpackage` and are not uploaded to
HuggingFace.

The aggregated table lives in `coreml/BASELINE_FP32.md`. Each entry in
that table corresponds to one row produced by `bench_one.py`.

Scripts here are diagnostic — they do not modify production
`MagpieModelStore` selection or any shipping artifact.
"""
