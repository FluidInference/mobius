---
license: apache-2.0
library_name: coremltools
pipeline_tag: token-classification
tags:
- coreml
- gliner2
- apple-silicon
---

# GLiNER2.5 base for Core ML

This repository contains fixed-shape Core ML exports of the trained classification and extraction paths from [Fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1) at revision `1a8bc24e00dc7300b9017c81d63e3dcdabb26596`. The original Apache-2.0 checkpoint has 193,581,591 parameters. Fastino authored the source model; Fluid Inference converted it. The model packages include the learned encoder, classification, boundary, relation, explicit-span, and record heads. The Python runtime keeps the source GLiNER2 schema, candidate selection, and decoder semantics.

The [published Core ML repository](https://huggingface.co/FluidInference/gliner2-5-base-coreml) includes the pinned source `config.json`, `encoder_config/config.json`, and tokenizer configuration alongside the model packages. Their source hashes are recorded in `assets.lock.json`.

## Extraction

The FP32 extraction stage packages support entities, relations, entity attributes, enum choices, natural/latent/anchorless records, and schemas mixed with classification. In a small fixed manifest of real text and schema fixtures, FP32 matched the native structured output on **11/11** cases. The largest FP32 confidence difference was 0.00000114. These checks are selected parity fixtures, not a Decision Index score or a full dataset evaluation.

FP16 result: 11/11 structures matched; the largest confidence difference was 0.1653 on a latent-record fixture. The base extraction FP16 packages are also included for speed sensitive applications; use FP32 when confidence values or latent-record decisions need closer native agreement.


```bash
uv sync
uv run python - <<'PYCODE'
from gliner2 import Schema
from extraction_runtime import CoreMLBoundaryExtractor

model = CoreMLBoundaryExtractor('.', precision='fp32')
schema = Schema().entities(['person', 'organization', 'location'])
print(model.extract('Alice founded Acme in Toronto.', schema, include_spans=True))
PYCODE
```

The extraction bucket holds up to 128 combined subword tokens, 64 text words, 8 extraction queries, and 8 classification choices. Candidate, explicit span, relation pair, and record capacities are fixed in the package names. Requests beyond the bucket raise `ValueError`; they require a larger exported bucket. The runtime uses the original tokenizer files and `gliner2==2.0.0`, but loads no PyTorch model weights. Run `verify-full-extraction.py` with the pinned native checkpoint for the selected parity check.

On an M5 Pro with macOS 27.0, FP32 end-to-end median 9.93 ms with All. FP16 median was 8.98 ms with All, 10.26 ms with CPU+Neural Engine, and 21.39 ms with CPU Only for a selected three-label entity request after 20 warmups and over 200 Python calls. Those are local end-to-end measurements for this shape, not ANE-only latency or a device-wide benchmark.

The FP16 feature graph public compute plan assigned 53.13% of operations to ANE and 46.87% to CPU under CPU+Neural Engine on this machine. Per-tensor LUT8 compression reduced the FP16 feature package from 391 MB to 196 MB, but only 10/11 selected structures matched, so that compressed package is omitted.

### Extraction with embedding W8

Two optional feature packages use per-channel int8 **weight-only** compression of the trained word embedding. The remaining encoder and extraction heads keep their exported FP16 or FP32 weights and arithmetic. Select them with `feature_package=` in `CoreMLBoundaryExtractor`; the five corresponding uncompressed head packages are still required.

| Feature package | Original bytes | W8 bytes | Selected native structure parity | Largest confidence difference |
| --- | ---: | ---: | ---: | ---: |
| `gliner2_base_extraction_features_w8_embedding_fp16_L128_W64_Q8.mlpackage` | 390,998,448 | 293,070,914 | 11/11 | 0.16604 |
| `gliner2_base_extraction_features_w8_embedding_fp32_L128_W64_Q8.mlpackage` | 780,899,033 | 486,602,631 | 11/11 | 0.00334 |

The FP16 W8 confidence difference is concentrated in the latent-record case, as with the uncompressed FP16 export. Use the FP32 W8 variant where closer native confidence values matter. Under forced CPU+Neural Engine, a private profiler assigned 620/652 executable FP16 W8 feature operations to ANE. The other 32 include integer indexing around embeddings, `cumsum`, and a decompression constant; this is partial ANE placement. On the same M5 Pro, the W8 FP16 end-to-end entity-request median was 10.29 ms with CPU+Neural Engine and 9.19 ms with All (100 calls after 10 warmups). FP32 W8 with All was 12.35 ms. These are size options, not demonstrated speedups over the uncompressed packages.

```python
from gliner2 import Schema
from extraction_runtime import CoreMLBoundaryExtractor

model = CoreMLBoundaryExtractor(
    ".", precision="fp16", feature_package="gliner2_base_extraction_features_w8_embedding_fp16_L128_W64_Q8.mlpackage"
)
print(model.extract("Alice founded Acme in Toronto.", Schema().entities(["person", "organization", "location"])))
```

All six FP16 stages were profiled. The anchorless and record-assignment heads were fully assigned to ANE; the relation head was mixed, while the small scorer and explicit heads ran on CPU under this scheduler. `reports/extraction-w8-validation.json` records both the public compute-plan percentages and private executable-operation counts, which use different counting methods.

The W8 packages and their exact file hashes are listed in `extraction-w8-assets.lock.json`. `quantize-extraction-coreml.py` regenerates each package from the pinned FP16 or FP32 feature export; `verify-quantized-extraction.py` checks the selected native-output manifest. Weight-only W8 does not imply int8 activation or matrix arithmetic.

## Classification

The original L128/K8 classification packages remain available, with a separate `runtime.py` entry point. Up to eight labels fit that bucket. 100/100 selected choices matched in FP16. See the classification report JSONs for the exact selected samples and limits. The full checkpoint's task and dataset scores have not been reproduced here.

An optional **embedding-only W8** classifier keeps the encoder and heads FP16. It reduces the package from **388,981,604 to 291,053,994 bytes** (25.2%). On an M5 Pro plugged into AC, it agreed with FP16 on all 14 bucket-fitting requests in a fixed 20-request manifest (largest probability difference 0.00992), and matched the native classifier on 100/100 selected eligible application requests (largest confidence difference 0.01370). The 14-request paired full-request p50 was 3.662 ms FP16 versus 3.632 ms W8 with automatic device selection; this is a size option, not a demonstrated speedup. Forced CPU+ANE was slower (9.168 versus 9.666 ms). The W8 compute plan put 496/516 executable ops on ANE under forced CPU+ANE, with 20 CPU boundary/constant ops. Use `runtime.py --precision embedding_w8` for this classification package; extraction continues to use its separate FP32 or FP16 packages.

When downloading with `huggingface_hub.snapshot_download`, pass `local_dir="./gliner2-base-coreml"` and point the runtime there. Core ML compilation on the tested macOS release rejected the symlinked weight file in the default Hub cache snapshot; a materialized local directory passed the downloaded-package smoke test.

The Core ML deployment target is iOS 17/macOS 14. Conversion scripts, pinned dependencies, asset hashes, and selected verification reports are included. The source revision is a current pinned snapshot; identity with the historical Decision Index evaluation checkpoint has not been established.
