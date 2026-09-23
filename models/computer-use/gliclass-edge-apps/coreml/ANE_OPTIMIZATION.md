# GLiClass Edge ANE placement experiments

The FP16 L128 graph places 432 of 437 operations on the Neural Engine. Four int32 preparation and
embedding operations plus one comparison remain on CPU. Although that is 98.9% by operation count,
the profiler assigns about half of estimated cost to CPU because token embedding gather is unsupported
on ANE.

Two graph boundaries were tested on the same Apple M5 Pro:

| Variant | ANE / CPU ops | CPU+ANE model median | Complete-call median | Application choice agreement |
| --- | ---: | ---: | ---: | ---: |
| Existing FP16 | 432 / 5 | 0.840 ms | **0.718–0.736 ms** | reference |
| Float pad bias | 430 / 4 | 0.816 ms | 0.722–0.738 ms | **100%** |
| Host-provided embeddings | **431 / 0** | **0.807 ms** | 0.732–0.753 ms | 99.92% |
| Host embeddings, FP16 I/O | **426 / 0** | 0.811 ms | slower in sequential test | fixture only |

The model-only profiler improves by 3.9% after moving embedding lookup outside the graph, and placement
reaches 100% ANE. The full call gets 2–3% slower at the median because transferring a 128×384 embedding
tensor costs more than the saved CPU/ANE partition work. It would also require a separate 38.7 MB FP16
or 77.4 MB FP32 embedding table. This variant is rejected.

The float-bias model removes one CPU cast without externalizing embeddings. It exactly matches all
3,899 application-suite choices and keeps the same 65.7 MB package size. In an alternating Swift
Tetris run over seeds 1–3 and 3,000 pieces per variant, it reduced aggregate wall time from 14.585 to
14.495 seconds: **0.62% faster**, with the median of per-run model-call medians moving from 4.914 to
4.893 ms. This is retained as the optional `fp16-mask` experiment; the gain is too small to replace
the default package without broader device testing.

The complete profiles and parity measurements are in
[`reports/coreml-ane-experiments.json`](reports/coreml-ane-experiments.json).

Recreate the experimental packages with:

```bash
uv run python convert-ane-experiments.py
```
