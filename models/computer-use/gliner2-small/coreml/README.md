---
license: apache-2.0
library_name: coremltools
pipeline_tag: text-classification
tags:
- coreml
- gliner2
- apple-silicon
- neural-engine
---

# GLiNER2.5 small Core ML classification

This is a Core ML conversion of the **classification decision path** of
[Fastino's GLiNER2.5 small](https://huggingface.co/fastino/gliner2.5-small-v1),
source revision `7e6f537f10337497069276892a5ef435028252ce`.
The original model has 73,881,879 parameters and is licensed Apache-2.0.
This export contains its encoder and trained classification head (70,944,385
parameters); entity, relation, record, count and span extraction heads are **not**
included. Use the original checkpoint for those tasks.

The FP16 L128/K8 package occupies 151,542,752 bytes. It accepts up to eight
classification labels, with native GLiNER2 schema rendering. The tokenizer
files are included in this repository. The Core ML deployment target is iOS 17
or macOS 14. `preprocessing.py` and `runtime.py` implement the request path
without loading the original model weights at inference time.

A Python example, from the directory containing this README:

```bash
uv sync
uv run python runtime.py --model-dir . \
  --text "The rocket launched successfully." \
  --task topic --labels '["science","sports","politics"]'
```

The model returns a label, confidence and probabilities. Inputs that exceed the
bucket capacity need a larger bucket; the runtime must not truncate them. The
runtime requires macOS to execute Core ML prediction.

## Validation

On an Apple M5 Pro running macOS 27.0, the FP16 package matched the native
classifier's chosen label on all 100 eligible requests selected in source order
from a fixed application suite. The largest chosen-label confidence difference
was 0.001723. Median Python `MLModel.predict` call time was 7.72 ms, including
Python/Core ML dispatch; this is not an ANE-only measure. There were 300 rows
with more than eight options and seven over-length rows before 100 eligible rows
were collected. The smoke run does not establish a Decision Index score or
broader application accuracy. `verify-application100.json` contains the exact
counts and results.

An experimental LUT8 per-tensor package occupies 76,211,068 bytes and matched
all 100 chosen labels, but its largest confidence difference was 0.07859.
It is not the recommended parity artifact. Grouped-channel LUT8 requires an
iOS 18 or later Core ML target and has not yet been evaluated.

An experimental **token-embedding-only int8** package occupies 102,771,476
bytes, 32.2% less than FP16. It matched the native classifier's chosen label
on the same 100 eligible application requests (largest chosen-label confidence
difference 0.01685). A paired 14-request FP16 comparison kept every choice
with worst absolute probability difference 0.01758 on CPU+ANE. Full-request
CPU+ANE medians were 3.701 ms FP16 and 3.719 ms int8 on battery power; one
small AC-powered paired check measured 3.702 and 3.668 ms. The results are
effectively tied, so this is a size option, not a speed claim. The package
retains L128/K8 classification
scope and is selected explicitly:

```bash
uv run python runtime.py --model-dir . \
  --package gliner2_small_classification_embedding_w8_L128_K8.mlpackage \
  --text "The rocket launched successfully." \
  --task topic --labels '["science","sports","politics"]'
```

The package file hashes and selected-request reports are under `reports/`.

Conversion source, pinned dependencies and verification scripts are included.
The exporter freezes two static DeBERTa attention expressions and uses a finite
FP16 mask sentinel; the real-model tests and native/Core ML checks guard this
rewrite. Upstream authors receive credit for the original model; Fluid
Inference performed this Core ML conversion.
