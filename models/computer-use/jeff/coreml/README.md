---
license: apache-2.0
library_name: coreml
tags:
  - coreml
  - text-classification
  - jeff
  - gliformer
  - on-device
---

# Jeff / GLiFormer Large classification for Core ML

This is a fixed L128, batch-1 Core ML conversion of the **trained classification path** in [knowledgator/gliformer-large-v1](https://huggingface.co/knowledgator/gliformer-large-v1), as used by [Jeff](https://github.com/logan-markewich/jeff). It includes the 24-layer DeBERTa encoder, learned prompt-marker embeddings, and trained linear classification head in one FP16 ML Program. The original checkpoint contains 575,637,510 parameters; this Core ML package is about 880 MB on disk. The source checkpoint, tokenizer and decision-engine revisions are pinned in `assets.lock.json`.

The model scores one classification group with 1–8 labels and up to 128 tokenized input tokens. Choice, yes/no and ordinal labels use the same trained classification head; the probabilities are independent sigmoid scores, not a normalized softmax. Only the first `len(labels)` logits are returned. Other GLiFormer tasks such as NER, layout extraction, vision, audio and structuring are **not** exported here. This is not a Decision Index benchmark result.

The original model's classifier uses CLS pooling, a parent prompt anchor, linear parent+category fusion and dot scoring. Its word-level RNN is evaluated upstream but cannot affect these classification logits under this checkpoint's settings. `jeff_decision.py` checks those settings and carries the trained encoder and head; `trace_compat.py` makes fixed-shape DeBERTa attention convertible without changing trained weights. The trace-only finite mask substitutes -10000 for fp32-min before FP16 conversion; patched PyTorch logits match native logits on the checked cases.

## Local inference

On macOS 15 or later with Python 3.12:

```bash
uv sync --frozen
uv run python - <<'PY'
from runtime import JeffCoreML

model = JeffCoreML(".", "JeffDecision-L128-FP16.mlpackage")
print(model.score(
    "The invoice was charged twice and the customer asks for a refund.",
    ["billing: invoice or payment issue", "support: technical product issue"],
    name="Choose the correct support queue",
))
PY
```

`JeffCoreML` uses the bundled tokenizer/config and GLiFormer processor, and loads **no original PyTorch weights**. Inputs exceeding 128 tokens or eight labels fail explicitly. Build from the pinned source checkpoint with `uv run python export.py --convert --precision fp16`; the source checkpoint must be in the local Hugging Face cache. `uv run python verify.py --precision fp16` compares Core ML logits to native PyTorch.

## Validation

Four real source-checkpoint fixtures cover billing, technical support, a three-label intent choice and yes/no classification. The mathematical decision wrapper matched native logits within **3.82e-6**. Tracing-only patches matched native logits exactly on those fixtures. The exported FP16 Core ML model preserved all four chosen labels with maximum absolute logit error **0.1139**. FP32 Core ML is a diagnostic control with four-of-four agreement and maximum logit error **3.44e-5**; the FP32 package is not included because it is much larger. Full Decision Index quality, additional input lengths, other task heads and broad latency/ANE performance have not been evaluated. See `native-parity.json` and `coreml-parity-fp16.json`.

An optional `JeffDecision-L128-W8.mlpackage` compresses the FP16 package's trained embedding and linear constants using per-channel symmetric int8. Its 488,374,793-byte package (versus 922,812,014 bytes FP16) retained the chosen label on all four source-checkpoint fixtures with maximum absolute logit error **0.129469**, below the predeclared 0.25 gate. Choose it by passing that package path to `JeffCoreML`. On a small battery-powered Apple M5 Pro probe, W8 had no demonstrated full-request speed advantage: All-compute p50 was 10.07 ms for W8 and 10.12 ms for FP16, with an outlier in W8's eight-call p95. Forcing CPU plus Neural Engine was about 19 ms for both. This is a size option with four-fixture validation, not a broad quality benchmark. Exact artifact hashes are in `reports/w8-validation.json`; see `tools/decision-coreml-profile/RESULTS.md` in the repository for timing scope and placement details.

When using `huggingface_hub.snapshot_download`, pass `local_dir="./jeff-coreml"` and point `JeffCoreML` there. Core ML compilation on the tested macOS release rejected a symlinked Hub-cache weight file; a materialized local directory avoids that path issue.

This conversion is derived from the Apache-2.0 GLiFormer weights and [Transformers](https://github.com/huggingface/transformers) DeBERTa implementation. Jeff's decision adapter code is MIT licensed; the helper source here retains attribution. The model card makes no claim that this conversion is faster or more accurate than another model on a benchmark.
