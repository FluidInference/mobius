---
license: apache-2.0
library_name: coremltools
pipeline_tag: text-classification
tags:
- coreml
- kev
- qwen2.5
- apple-silicon
---

# Kev 0.5B Core ML

Core ML FP16 conversion of [jaredpalmer/kev-0.5b](https://huggingface.co/jaredpalmer/kev-0.5b)
(Apache-2.0), source revision `9ce2fd39db3a397c89733f94af948e3d1fdfffcd`.
The pinned Qwen backbone is
[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` (Apache-2.0).
This package merges the Kev LoRA adapter and keeps its trained pointer head.
`assets.lock.json` pins both sources and the upstream Kev code revision.

The L128/32-option package is **989,581,157 bytes** and requires iOS 17 or
macOS 14. It handles one typed question per call. The included tokenizer and
`runtime.py` run the Core ML artifact without loading the source Qwen weights.
The serving runtime shortens the state's right edge if necessary and rejects
a question branch that cannot fit 128 tokens. Shortened requests are protocol
variants and cannot be called full native Decision Index evaluations.

```bash
uv sync
uv run python runtime.py --model-dir . --request-json request.json
```

`request.json` must contain a Kev System One `state` and one question under
`questions`; see the examples in `verify.py`. The serving request does not
need training-only `label` or `src` fields. The runtime returns the typed
answer, usage, option keys and probabilities. Reuse the `KevCoreML` class for
multiple requests so the tokenizer and compiled package load once. The special
tokens in `tokenizer/` are essential.
When a Hugging Face snapshot stores package files as blob symlinks, the runtime
stages a real-file copy before Core ML compilation; allow about 1 GB of
temporary disk space for that case. A regular local package loads in place.

## Validation

The PyTorch export wrapper's maximum logit difference from the merged native
model was 0.0000131. The Core ML package matched the native chosen option for
19 of 19 runnable fixed fixtures; four selected suite requests exceeded the
L128 capacity. Maximum probability error on `ComputeUnit.ALL` was 0.002068.
On an Apple M5 Pro, a previous `coreml-cli` profile measured 8.22 ms median
for `ComputeUnit.ALL` and 7.24 ms for CPU plus Neural Engine. A later Python
verification call measured 18.01 ms median; these are different measurement
paths, so do not substitute one for the other. See `RESULTS.md` and the report
JSON for scope and methodology.

On FluidUse's separate 3,899-question application suite, Kev scored 57.2%
versus 71.2% for the previously ported Laya. The tracker gives Kev 30.34 and
Laya 16.39 on its different frozen Decision Index protocol. This package has
not been run on the complete frozen Decision Index suite.

The converter, tokenizer and pinned dependencies are included. Fluid Inference
performed the Core ML conversion; Jared Palmer authored Kev and Qwen authored
the backbone.

The optional `kev_0_5b_e8_L128_options32.mlpackage` targets only the embedding
constant with per-channel int8; the other weights stay FP16. It reduces package
size from **989,581,157 to 853,903,798 bytes** (13.7%). On `ComputeUnit.ALL`,
the first two rows per application suite plus three fixed typed fixtures gave
**19/19** matching native choices; four selected requests exceeded L128 and
were skipped. The largest probability difference was **0.010888**, within the
0.02 gate. This is selected parity evidence, not a Decision Index score.
Pass the optional package path with `runtime.py --package` or `KevCoreML(...,
package=...)`. An earlier battery-powered comparison measured 9.90 ms W8
versus 9.60 ms FP16 request p50 in separate passes, so W8 is a size option
without a demonstrated speed gain. Forced CPU+ANE exceeded the probability
gate in prior checks; keep automatic compute-unit selection. Exact package
hashes and the selected-case protocol are in `reports/e8-selected20.json`.
