# GLiClass Edge Apps Core ML

This toolkit tunes the 32.7M-parameter Apache-2.0
[`knowledgator/gliclass-edge-v3.0`](https://huggingface.co/knowledgator/gliclass-edge-v3.0)
encoder for FluidUse's ten application decision families and converts it to fixed-shape Core ML.
The converted model scores 19.27 on the Decision Index headline panel, compared with laya's
published 16.39, and it also beats laya multilingual on FluidUse's separate 3,899-request
application suite.

The model takes one instruction, a serialized state and a dynamic list of option descriptions. The
Core ML export supports up to 25 options and uses L128, L256 and L512 buckets. Inputs longer than
L512 are truncated by the application harness. The Decision Index adapter does not truncate: it
reports any request over 512 tokens or 25 options as unsupported, which the scorer counts as wrong.

## Decision Index result

The actual FP16 Core ML packages were run over all 77,165 scoreable requests in the 19 static
headline benchmarks, including ChessBench. The panel was rebuilt with the official evaluation kit
at commit `52a698928a9ae5bdf16b75687c903871db29c6e5`, its pinned sources and deterministic sampler,
then filtered with the official exclusion file. No Decision Index evaluation data was used for
training.

| Headline measure | GLiClass Edge apps v2 | laya, published tracker result |
| --- | ---: | ---: |
| Decision Index | **19.27** | 16.39 |
| Relative index gain | **17.6%** | — |
| Mean area coverage | **54.96%** | 38.72% |
| Balanced skill | **2.10** | 1.46 |
| Breadth skill | **1.95** | 1.34 |

The Core ML run answered 49,354 requests, marked 27,811 unsupported and produced zero errors. Its
end-to-end request latency, including rendering and tokenization, was 2.0 ms median and 9.1 ms p95.
The published `multimodalart/decision-index-suite` bundle returned 404 during this run, so the 18
display-only static benchmarks were not run. They do not enter the headline formula, but the
official kit therefore marks this as incomplete for tracker submission. See
[`reports/decision-index-coreml.json`](reports/decision-index-coreml.json) for the area and
per-benchmark breakdown.

## Application-suite result

Measured September 22, 2026 on an Apple M5 Pro with 24 GB RAM and macOS 27.0:

| FluidUse application-suite measure | GLiClass Edge apps v2 | laya optimized e8 |
| --- | ---: | ---: |
| Parameters | **32.7M** | 322M |
| Full-suite macro accuracy | **72.75%** | 71.1% |
| Full-suite micro accuracy | **72.89%** | 71.2% |
| L128 fastest median | **0.843 ms** | 3.6 ms |
| L256 fastest median | **1.673 ms** | 5.8 ms |
| L512 fastest median | **2.008 ms** | 9.1 ms |
| Package size per bucket | **65.7–66.7 MB** | 448 MB with int8 embedding |

L128 and L256 use the Neural Engine when all compute units are allowed. Core ML selects the GPU for
L512 because it is faster than CPU+Neural Engine at that shape. See [RESULTS.md](RESULTS.md) for the
full per-suite, placement and parity results.

### Weight compression

The deployable compact variant is the 8-bit per-tensor k-means LUT package. At L128 it cuts the
package from 65.7 MB to 33.0 MB, preserves 97.1% of FP16 application-suite choices and has the same
application accuracy within sampling noise. It is a size optimization: CPU+ANE model latency is
0.903 ms instead of 0.843 ms for FP16. Six-bit LUT is 24.8 MB but changes 8.3% of choices; four-bit
LUT is rejected because accuracy falls from 71.94% to 56.19% on the L128-supported rows.

Verdict (`heman10x/rlcd-modernbert-151m`) was also converted and profiled at the same L128/25-option
shape. Its best viable result is 3.565 ms FP16; LUT8 halves its 303.2 MB package but slows it to
3.769 ms. More aggressive variants lose too many decisions or run slower. See [VERDICT.md](VERDICT.md).

ANE boundary experiments reached 100% placement by gathering token embeddings on the host, but the
larger input made complete calls slower. A smaller float-mask change preserves every application-suite
choice and is 0.62% faster in a paired Swift Tetris run. See [ANE_OPTIMIZATION.md](ANE_OPTIMIZATION.md).

## Reproduce

Run from this directory. Model and dataset revisions are pinned in `assets.lock.json`. Generated
checkpoints and Core ML packages stay under ignored `build/`; publish large artifacts externally.

```bash
uv sync

# Stage 1: 10,000 examples, last two encoder layers plus heads
uv run python train.py --per-suite 1000 --batch-size 16 --max-length 256 \
  --train-layers 2 --output build/checkpoint-v1

# Stage 2: continue on 30,000 examples with the last four layers trainable
uv run python train.py --base-model build/checkpoint-v1 --per-suite 3000 \
  --batch-size 16 --max-length 256 --train-layers 4 --output build/checkpoint-v2

for length in 128 256 512; do
  uv run python convert-coreml.py --length "$length"
done

uv run python verify.py

# Optional compact L128 package. Per-grouped-channel LUT requires an iOS 18 deployment target;
# per-tensor works with the current iOS 17 package target.
uv run python compress-coreml.py --bits 8 --granularity per_tensor
uv run python verify-compression.py \
  --reference build/coreml/gliclass_edge_apps_fp16_L128_options25.mlpackage \
  --candidate build/coreml/gliclass_edge_apps_lut8_kmeans_per_tensor_L128_options25.mlpackage \
  --out reports/coreml-L128-lut8-verify.json

# With an official or rebuilt headline rows file:
uv run python -m decision_index run \
  --engine decision_index_engine:GLiClassEdgeAppsCoreMLEngine \
  --rows /path/to/headline-panel.jsonl.gz \
  --out build/decision-index-coreml --compact
```

Profile a bucket with the repository profiler:

```bash
uv run --project ../../../../tools/coreml-cli coreml-cli \
  build/coreml/gliclass_edge_apps_fp16_L128_options25.mlpackage \
  --iterations 50 --json
```

## Scope

This checkpoint is tuned for the ten evaluated application families. Training uses real public
training splits, balances each family and excludes every exact serialized application benchmark
state before sampling. The Decision Index run uses five equally weighted areas, native
per-benchmark metrics, no truncation and unsupported-as-wrong scoring. Its 19.27 result clears
laya's 16.39, although it remains close to the tracker floor and has weak retrieval and tools-area
coverage. Support triage (26.0%) and RAG relevance (49.5%) also remain weak on the separate
application suite.
