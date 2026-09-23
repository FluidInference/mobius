---
license: apache-2.0
library_name: coremltools
pipeline_tag: token-classification
tags:
- coreml
- gliner2
- multilingual
- apple-silicon
---

# GLiNER2.5 multilingual for Core ML

This repository contains fixed-shape Core ML exports of the trained classification and extraction paths from [Fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) at revision `a221b77a8baf4a613b8f8652661d41fa10a5641e`. The original Apache-2.0 checkpoint has 287,355,159 parameters. Fastino authored the source model; Fluid Inference converted it. The model packages include the learned encoder, classification, boundary, relation, explicit-span, and record heads. The Python runtime keeps the source GLiNER2 schema, candidate selection, and decoder semantics.

The [published Core ML repository](https://huggingface.co/FluidInference/gliner2-5-multi-coreml) includes the pinned source `config.json`, `encoder_config/config.json`, and tokenizer configuration alongside the model packages. Their source hashes are recorded in `assets.lock.json`.

## Extraction

The FP32 extraction stage packages support entities, relations, entity attributes, enum choices, natural/latent/anchorless records, and schemas mixed with classification. In a small fixed manifest of real text and schema fixtures, FP32 matched the native structured output on **15/15** cases. The largest FP32 confidence difference was 0.00000489. The multilingual manifest includes Spanish, French, Chinese, and German text. These checks are selected parity fixtures, not a Decision Index score or a full dataset evaluation.

Standalone FP16 result: 14/15 structures matched; the latent-record fixture differed (10 native records versus 9 Core ML records). The optional adaptive W8 runtime below uses FP32 for any schema containing a latent record and FP16 for the other schemas.


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

On an M5 Pro with macOS 27.0, FP32 end-to-end median 10.56 ms with All, 10.02 ms with CPU+GPU, 36.08 ms with CPU+Neural Engine for a selected three-label entity request after 20 warmups and over 200 Python calls. Those are local end-to-end measurements for this shape, not ANE-only latency or a device-wide benchmark.

Under CPU+Neural Engine, the FP32 feature graph compute plan assigned all operations to CPU on this machine; CPU+GPU is the faster measured setting for the uncompressed FP32 path. A local FP32-source LUT8 attempt stalled in k-means with numerical warnings.

### Extraction with embedding W8

The optional adaptive runtime uses two feature packages with per-channel int8 **weight-only** compression of the trained word embedding. The rest of each encoder and the corresponding five extraction heads retain their exported precision. It loads one precision at a time: FP16 with CPU+Neural Engine for schemas without latent records, then FP32 with All for schemas containing a latent record. Switching precision incurs a cold model load. The standalone FP16 path still differs on the latent-record fixture; use `CoreMLAdaptiveBoundaryExtractor` for complete selected parity.

| Feature package | Original bytes | W8 bytes | Selected native structure parity | Largest confidence difference |
| --- | ---: | ---: | ---: | ---: |
| `gliner2_multi_extraction_features_w8_embedding_linear_fp16_L128_W64_Q8.mlpackage` | 578,545,680 | 387,210,787 | 14/15 standalone | see adaptive result |
| `gliner2_multi_extraction_features_w8_embedding_fp32_L128_W64_Q8.mlpackage` | 1,155,993,314 | 580,986,690 | 15/15 standalone | 0.02455 |
| Adaptive FP16 plus FP32 | — | — | **15/15** | **0.01639** |

The W8 FP16 feature graph had 620/652 executable operations assigned to ANE with forced CPU+Neural Engine on an M5 Pro; the 32 CPU operations were mainly integer indexing around embeddings, plus `cumsum` and one decompression constant. This is a private profiler count, distinct from the public compute-plan grouping. On the same Mac, an adaptive W8 entity request took 9.61 ms median over 100 calls after 10 warmups, using the FP16 path. The uncompressed FP32 All baseline took 10.56 ms in a separate 200-call run. These are small local measurements, not a paired speedup claim or a Decision Index score.

```python
from gliner2 import Schema
from extraction_runtime import CoreMLAdaptiveBoundaryExtractor

model = CoreMLAdaptiveBoundaryExtractor(".")
print(model.extract("Alice founded Acme in Toronto.", Schema().entities(["person", "organization", "location"])))
```

All six FP16 stages were profiled. The anchorless and record-assignment heads were fully assigned to ANE; the relation head was mixed, while the small scorer and explicit heads ran on CPU under this scheduler. `reports/extraction-w8-validation.json` records both the public compute-plan percentages and private executable-operation counts, which use different counting methods.

Both W8 feature packages and the five FP16 head packages are included. `extraction-w8-assets.lock.json` records exact file hashes. `quantize-extraction-coreml.py` regenerates the W8 packages from the pinned feature exports; `verify-adaptive-extraction.py` checks the selected native-output manifest. Weight-only W8 does not imply int8 activation or matrix arithmetic.

## Classification

The original L128/K8 classification packages remain available, with a separate `runtime.py` entry point. Up to eight labels fit that bucket. 299/300 selected choices matched in FP16; 100/100 matched in an FP32 control. See the classification report JSONs for the exact selected samples and limits. The full checkpoint's task and dataset scores have not been reproduced here.

An optional **embedding-only W8 with asymmetric scaling** classifier keeps the encoder and heads FP16. It reduces the package from **576,528,829 to 385,193,894 bytes** (33.2%). On an M5 Pro plugged into AC, it agreed with FP16 on 100/100 selected eligible application requests (largest probability difference 0.01941). It matched the native classifier on 99/100 requests; the one near-tie disagreement at `jev.ag_news` row 100 is also present in the FP16 export. In a paired 100-request check, automatic-device full-request p50 was 4.283 ms FP16 versus 4.265 ms W8, so there is no established speed gain. The W8 compute plan put 496/516 executable ops on ANE under forced CPU+ANE, with 20 CPU boundary/constant ops; forced CPU+ANE was slower than automatic device choice. Symmetric embedding W8 exceeded the 0.02 probability-difference gate on the first selected manifest and is omitted. Use `runtime.py --precision embedding_w8_linear` for the compressed classification package; extraction has its own packages and runtime.

When downloading with `huggingface_hub.snapshot_download`, pass `local_dir="./gliner2-multi-coreml"` and point the runtime there. Core ML compilation on the tested macOS release rejected the symlinked weight file in the default Hub cache snapshot; use a materialized local directory.

The Core ML deployment target is iOS 17/macOS 14. Conversion scripts, pinned dependencies, asset hashes, and selected verification reports are included. The source revision is a current pinned snapshot; identity with the historical Decision Index evaluation checkpoint has not been established.
