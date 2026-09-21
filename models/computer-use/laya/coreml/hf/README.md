---
license: apache-2.0
library_name: coreml
pipeline_tag: text-classification
base_model: convaiinnovations/laya
tags:
  - coreml
  - laya
  - apple-silicon
  - neural-engine
  - decision-model
  - fluidaudio
---

# laya-coreml

Core ML conversion of **laya-multilingual** (Convai Innovations, Apache-2.0): a 322M-parameter
mmBERT-base encoder with a typed decision head that answers `choice`, `score`, and `noul`
questions about a text state in one forward pass, returning calibrated probabilities and no
generated tokens. Weights are unchanged from
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) `multilingual/` at
revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.

Runs through [FluidAudio](https://github.com/FluidInference/FluidAudio) (`LayaManager`) on
macOS 14+ / iOS 17+.

```swift
let laya = try await LayaManager.load()  // downloads the 128 + 512 buckets and tokenizer.json
let answer = try await laya.answer(
    state: "The T piece dropped at column 3 leaves one hole under it.",
    question: .noul("Is this a clean placement?"))
print(answer.noul!)  // P(true)
```

```bash
swift run -c release fluidaudiocli laya --state "…" --type choice \
    --instructions "What does the customer want?" --options "refund|order status|technical help"
swift run -c release fluidaudiocli laya-tetris   # headless Tetris played by laya decisions
```

## Files

| File | Tokens | Notes |
| --- | ---: | --- |
| `laya_multilingual_fp16_L128_options32.mlmodelc` | 128 | Short prompts; runs on CPU + Neural Engine |
| `laya_multilingual_fp16_L256_options32.mlmodelc` | 256 | |
| `laya_multilingual_fp16_L512_options32.mlmodelc` | 512 | Long states; GPU is faster than ANE here |
| `tokenizer.json` | | mmBERT / Gemma vocabulary (256k), byte fallback |

Each bucket is a complete FP16 model (614 MB, 393 MB of which is the embedding table) with
32 option slots. `FluidAudio` picks the smallest loaded bucket that fits a prompt and truncates
the state on the right for the largest one, exactly like laya's `max_len`.

Inputs: `input_ids` int32 `[1, L]`, `attention_mask` int32 `[1, L]`, `marker_map` float32
`[1, 32, L]` (one-hot `[MASK]` position per option), `question_type` float32 `[1, 3]`.
Outputs: `logits` `[1, 32]`, `probabilities` `[1, 32]`, `action_probabilities` `[1, 2]`.
Sequence format: `[CLS] <type> question: <instructions> [SEP] ([MASK] <option>)* [SEP] <state> [SEP]`.

## Parity and latency

Apple M5 Pro, macOS 27.0, 16 fixture questions vs. the unmodified PyTorch FP32 runtime:
16/16 argmax agreement on every bucket and compute-unit setting, max probability error 0.0021
(`ALL`) / 0.0126 (`CPU_AND_NE`). Per-question latency, warm:

| Bucket | CPU + ANE | All units |
| --- | ---: | ---: |
| L128 | **3.6 ms** | 3.9 ms |
| L256 | 9.9 ms | **5.2 ms** |
| L512 | 27.5 ms | **9.0 ms** |

Conversion pipeline, verification reports, and Swift parity fixtures:
[mobius `models/computer-use/laya/coreml`](https://github.com/FluidInference/mobius).

## License

Apache-2.0, following the upstream weights and code by Convai Innovations
([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)). Independent conversion; not an
official Convai Innovations release.
