---
license: apache-2.0
library_name: coreml
pipeline_tag: image-text-to-text
base_model:
  - cua-ai/cua-s1-4b-0.2
  - Qwen/Qwen3.5-4B
tags:
  - coreml
  - computer-use
  - decision-model
  - apple-silicon
  - qwen3.5
  - fluiduse
---

# cua-s1-4b-coreml

Core ML conversion of [**Cua-S1-4B-0.2**](https://huggingface.co/cua-ai/cua-s1-4b-0.2) (Cua, Apache-2.0):
LoRA adapters on [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0) that pick one
`(element, action)` option for a GUI screen state. Both adapters are converted, each merged into its own
copy of the base: `text/` (accessibility tree) and `multimodal/` (screenshot).

The model is a one-pass chooser, not a generator: one forward pass over the prompt, then a softmax over
the answer-letter logits `A..Z` at the last position (Cua's `FourBModel` readout). The Core ML graphs keep
exactly that: prefill only, no KV cache, no vocabulary head.

Runs through [FluidUse](https://github.com/FluidInference/FluidUse) (`CuaS1FourBManager`) on macOS 14+.

```swift
let cua = try await CuaS1FourBManager.load(configuration: .init(modality: .text, variant: "gptq"))
let decision = try await cua.decide(CuaS1FourBState(
    app: "portal", taskFamily: "login_auth", goal: "Log in as the demo user",
    accessibilityTree: "- [el_0] Edit \"Username\" value=\"\"\n- [el_1] Button \"Log in\"",
    options: [
        .init(elementId: "el_0", role: "Edit", label: "Username", action: "fill", entityId: "ent_0"),
        .init(elementId: "el_0", role: "Edit", label: "Username", action: "skip"),
        .init(elementId: "el_1", role: "Button", label: "Log in", action: "click"),
    ]))
print(decision.best?.option, decision.bestPerElement())
```

## Files

| Path | What | Size |
| --- | --- | ---: |
| `tokenizer.json` | Qwen3.5 tokenizer | 12 MB |
| `embeddings.f16` | tied embedding table, row-major `[248320, 2560]` fp16, gathered on the host | 1.27 GB |
| `text/L1024/` | text decoder, fp16, 4 parts x 8 layers, prompts up to 1024 tokens | 6.8 GB |
| `text/L1024-w8/` | int8 linears | 3.4 GB |
| `text/L1024-gptq/` | GPTQ: MLP int4 (block 16) + other linears int8 | 2.6 GB |
| `multimodal/L2048/` | multimodal decoder, fp16, prompts up to 2048 tokens | 6.8 GB |
| `multimodal/L2048-w8/` | int8 linears | 3.4 GB |
| `multimodal/vision/` | vision tower + merger (up to 4096 patches), position table, config | 0.66 GB |

Each decoder part takes `hidden [1, L, 2560]`, `cos`/`sin [L, 64]` (interleaved M-RoPE, host computed)
and returns the next hidden state; the last part also takes `last_onehot [1, L]` and returns
`letter_logits [1, 26]`. Prompts are right-padded; the model is causal, so padding is exact.

## Results

Apple M5 Pro (24 GB), macOS 27, GPU (`cpuAndGPU`). Parity reference: fp32 transformers decoder layers on
the merged weights and the unpadded prompt, over 38 tasks from Cua's own `cua_bench_s1` generator.
Accuracy: GUI-360 test split (613 text tasks) rebuilt with Cua's `gui360` converter; Cua's own frozen
615-task split is published by hash only, so this is protocol-adjacent, not the same task set.

| Build | Size | Argmax vs fp32 (38) | Max \|Δp\| | GUI-360 text (613) |
| --- | ---: | ---: | ---: | ---: |
| text fp16 | 6.8 GB | 38/38 | 0.005 | 85.5% |
| text gptq | 2.6 GB | 38/38 | 0.133 | 85.5% |
| text w8 | 3.4 GB | 37/38 | 0.019 | — |
| multimodal fp16 (+ vision) | 7.4 GB | 38/38 | 0.153 | — |
| multimodal w8 (+ vision) | 4.0 GB | 38/38 | 0.163 | — |

One decision on the GPU: text ~0.7 s (1024-token bucket) and multimodal ~1.65 s (vision 0.2 s +
2048-token decoder) on an idle machine; about 1.1 s and 3 s while other GPU work was running.

On the same first 100 GUI-360 tasks, Cua's PyTorch runtime (bf16) scores 84%, Core ML fp16 88% and gptq
87%; the Core ML task outcome matches PyTorch on 96-97 of 100. Post-training int4 without calibration and 4/3/2-bit palettes are not
published: they flip 10-32 of the 38 decisions.

## Limits

- GPU only: the Neural Engine path falls back to the CPU for most of the graph (19 s / decision).
- At most 26 options per decision (one letter each). Longer prompts than the loaded bucket are rejected.
- Screenshots are smart-resized as in the reference processor, capped at 4096 patches (about 1280x800
  px); larger screenshots are downscaled further than the PyTorch reference would.
- fp16 vision features move multimodal probabilities by up to 0.15 (argmax unchanged on the fixtures).

## Provenance

Conversion code, parity and benchmark scripts: [mobius](https://github.com/FluidInference/mobius)
`models/computer-use/cua-s1-4b/coreml`. Base weights: `Qwen/Qwen3.5-4B`; adapters: `cua-ai/cua-s1-4b-0.2`
(`text/`, `multimodal/`). Both Apache-2.0; see `NOTICE`.
