# Kev 0.5B Core ML

This directory converts the Apache-2.0
[`jaredpalmer/kev-0.5b`](https://huggingface.co/jaredpalmer/kev-0.5b) decision model to a fixed-shape
Core ML program. Kev uses a Qwen2.5-0.5B causal backbone, a LoRA adapter and a pointer head over
typed option markers. The converter merges the adapter before export and keeps the trained pointer
head.

The first proof-of-concept bucket accepts one question per call, 128 tokens and up to 32 options.
Long states are truncated on the right so the complete question and option branch remains intact.

```bash
uv sync
uv run python convert-coreml.py --parity-only
uv run python convert-coreml.py --length 128 --max-options 32
uv run python verify.py build/kev_0_5b_fp16_L128_options32.mlpackage
```

Generated packages live under `build/` and are not repository assets. Run `coreml-cli` on the
compiled model before publishing any package or latency claim.

## Proof-of-concept result

The FP16 L128 package converts and runs. On an Apple M5 Pro it is 990 MB (944 MiB) and profiles at
8.22 ms on all compute units (GPU) or 7.24 ms on CPU+Neural Engine. The latter assigns 1,394 of
1,399 operations to the Neural Engine, although the embedding gather keeps 25.8% of estimated
runtime on CPU.

This is not a good laya replacement for the current FluidUse workload. Kev scored 57.2% micro on
the same 3,899 application questions where laya scores 71.2%. See [RESULTS.md](RESULTS.md) for the
accuracy, parity, placement and compression results.

