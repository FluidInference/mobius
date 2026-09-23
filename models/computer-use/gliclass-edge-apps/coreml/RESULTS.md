# GLiClass Edge Apps v2 results

All measurements were recorded September 22, 2026 on an Apple M5 Pro with 24 GB RAM and macOS
27.0. The model was evaluated on the Decision Index headline panel and FluidUse's separate
3,899-request application suite. No Decision Index evaluation data was used for training, and exact
serialized application-suite benchmark states are excluded from every training source.

## Decision Index

The actual L128/L256/L512 Core ML packages score **19.27**, compared with laya's published 16.39.
This is a 2.88-point absolute and 17.6% relative gain. The run contains all 77,165 scoreable requests
from the 19 static headline benchmarks, including folded ChessBench, rebuilt from the official kit's
pinned sources and filtered with its published exclusion file.

| Area | Raw | Skill | Coverage | Benchmarks |
| --- | ---: | ---: | ---: | ---: |
| Knowledge & Reasoning | 27.13% | 1.86% | 85.22% | 6 |
| Language Understanding | 21.02% | 3.49% | 67.21% | 3 |
| Retrieval & Classification | 6.11% | 0.00% | 39.92% | 2 |
| Tools & Automation | 13.16% | 5.15% | 22.29% | 3 |
| Arts & Human Judgment | 28.93% | 0.00% | 60.16% | 5 |

The run answered 49,354 requests, rejected 27,811 beyond the deployed 512-token or 25-option limits,
and had zero errors. Unsupported requests score as wrong. End-to-end latency, including rendering
and tokenization, was 2.0 ms median, 9.1 ms p95 and 6.4 ms mean. The complete record is
[`reports/decision-index-coreml.json`](reports/decision-index-coreml.json).

The public 132,422-request bundle was unavailable from Hugging Face during the run. Its extra 18
static benchmarks are display-only and do not affect the headline index, but their absence makes
this run ineligible for direct tracker submission until the bundle is restored.

Laya's comparison numbers use its optimized int8-embedding (`e8`) Core ML packages, rather than an
unoptimized PyTorch checkpoint. GLiClass is 4.3× faster at L128, 3.5× at L256 and 4.5× at L512,
while each 65.7–66.7 MB FP16 package is about 6.7× smaller than laya's 448–450 MB e8 package.

## Accuracy

Bucketed FP16 Core ML scores 72.75% macro and 72.89% micro, compared with laya's 71.1% macro and
71.2% micro on this application suite. It wins six of ten suites. These percentages cannot be
compared with the Decision Index because that benchmark uses a different corpus, task panel,
coverage policy and aggregation formula.

| Suite | GLiClass Core ML | laya |
| --- | ---: | ---: |
| AG News | 85.75% | **93.5%** |
| Emotion | **55.5%** | 53.7% |
| MASSIVE intent | **67.0%** | 65.7% |
| Support triage | 26.0% | **54.2%** |
| Email spam | 92.75% | **99.3%** |
| Phishing | 94.25% | **99.3%** |
| Jailbreak | **87.25%** | 80.5% |
| Toxicity | **72.0%** | 53.5% |
| RAG relevance | 49.5% | **67.2%** |
| Model routing | **97.49%** | 44.1% |

## Core ML latency and size

Each result uses 50 timed iterations from `coreml-cli`. Package size is the complete `.mlpackage`.

| Bucket | Package | Fastest units | Median | CPU+ANE placement by cost |
| --- | ---: | --- | ---: | --- |
| L128 | 65.7 MB | CPU+ANE | **0.843 ms** | 49.05% ANE / 50.95% CPU |
| L256 | 65.9 MB | CPU+GPU | **1.673 ms** | 64.73% ANE / 35.27% CPU |
| L512 | 66.7 MB | All (GPU) | **2.008 ms** | 81.42% ANE / 18.58% CPU |

With CPU+ANE selected, 432 of 437 operations are assigned to the Neural Engine at every shape. The
five CPU operations are four int32 mask/embedding operations and one dependent comparison. The
cost percentages above account for their disproportionate runtime rather than merely counting ops.

The full suite uses L128 for 2,259 requests, L256 for 1,151 and L512 for 489. Python
`MLModel.predict` has a 1.02 ms aggregate median while switching among the three loaded packages.

## Weight compression

Post-training per-tensor k-means palettization was measured on the 3,461 application-suite rows that
fit L128. Latency is the same 50-iteration `coreml-cli` CPU+ANE profile used for FP16. Accuracy and
agreement compare each compressed package with the FP16 Core ML package.

| Weights | Package | FP16 choice agreement | Micro accuracy | CPU+ANE median |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 65.7 MB | 100% | 71.94% | **0.843 ms** |
| **LUT8** | **33.0 MB** | **97.10%** | **72.17%** | 0.903 ms |
| LUT6 | 24.8 MB | 91.69% | 71.38% | 0.884 ms |
| LUT4 | 16.6 MB | 63.48% | 56.19% | 0.903 ms |

LUT8 is the shipping-size option. It halves the package and preserves application accuracy, but it
does not accelerate this graph on the Neural Engine. LUT6 has a small aggregate accuracy loss and a
much larger decision-parity loss, so it is an aggressive size option. LUT4 fails the accuracy gate.
Grouped-channel LUT was not tested because Core ML requires an iOS 18 deployment target while these
packages currently target iOS 17. Full results are in
[`reports/coreml-compression.json`](reports/coreml-compression.json).

The LUT8 Tetris run reached the 5,000-piece cap on five of ten seeds and averaged 3,666.7 capped
pieces at 4.92 ms end-to-end per model call. Tetris is path-sensitive, so its 5.3% higher mean than
FP16 is not evidence that quantization improves model quality; the application suite is the stable
parity measure.

## Conversion parity

The explicit PyTorch export wrapper matches the source model exactly on the conversion fixture.
Across all 3,899 benchmark rows, FP16 Core ML changes 18 argmaxes (0.46%) relative to PyTorch. The
mean absolute logit difference is 0.0165 and the maximum is 0.210. Aggregate macro accuracy rises
from 72.67% in PyTorch to 72.75% after conversion.

## Training

Stage 1 trains 3.33M parameters in the last two encoder layers and heads for 625 CPU steps on 10,000
examples. Stage 2 continues from that checkpoint, trains 5.84M parameters in the last four layers
and heads, and runs 1,875 CPU steps on 30,000 balanced examples. The two stages took 105.5 seconds
and 413.2 seconds locally. Seed `20260922`, learning rate `2e-5`, batch size 16 and maximum length
256 are fixed by the documented commands.
