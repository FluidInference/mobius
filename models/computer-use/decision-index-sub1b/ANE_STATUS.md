# Sub-1B Core ML ANE and compression status

This records selected, real-request checks on an Apple M5 Pro with 24 GB RAM
and macOS 27.0 (2026-09-22). Initial runs recorded battery power; the newer
GLiNER2 classification and Kai/Lex W8 checks recorded AC power. The
base/multilingual extraction timing reports did not record power state. Timings are exploratory; package
bytes, native parity, full-request latency and actual compute placement matter
more than an ANE-compatible operation count. Automatic Core ML device choice is
preferred when forced CPU+ANE is slower or changes calibrated output. These
checks do not reproduce the Decision Index or compare 2048 performance.

| Model | Best validated/public route | ANE or compression finding |
| --- | --- | --- |
| GLiNER2.5 small | FP16 L128/K8 classification; optional [embedding-only int8](https://huggingface.co/FluidInference/gliner2-5-small-coreml) | Int8 151.5→102.8 MB, 100/100 selected native labels, worst confidence error 0.01685. CPU+ANE full-request p50 3.701→3.719 ms on battery and 3.702→3.668 ms in one AC pass; no speed win. FP16 496/512 and int8 496/516 executable ops on ANE, with an int32 CPU boundary. [Report](../gliner2-small/coreml/reports/ANE-OPTIMIZATION.md). |
| Verdict | FP16 L128/L512 | Embedding-only int8 303.2→264.7 MB passed 12/12 selected requests but had no meaningful speed gain; local experimental variant only. Whole-model LUT8 failed 96/100 selection/abstention agreement. Forced ANE estimated 68% of runtime on ANE. [Report](../verdict/coreml/STATUS.md). |
| GLiNER2.5 base | Classification FP16 and optional embedding W8; fixed-shape extraction packages | Classification W8 389.0→291.1 MB, 14/14 paired choices, 100/100 native choices, no speed gain; forced CPU+ANE slower despite 496/516 ANE ops. FP32 extraction matched 11/11 selected structures; FP16 confidence differed by 0.165 on one case, and extraction LUT8 changed 1/11 structures. [Classification report](../gliner2-base/coreml/reports/classification-embedding-w8.json); [extraction report](../gliner2-base/coreml/reports/extraction-validation.json). |
| GLiNER2.5 multilingual | Classification FP16 and optional asymmetric embedding W8; FP32 extraction packages | Classification W8 576.5→385.2 MB, 100/100 paired FP16 choices, 99/100 native choices with the same near-tie failure as FP16; no speed gain, and forced CPU+ANE slower despite 496/516 ANE ops. FP32 extraction matched 15/15 selected structures; extraction FP16 changed one structure and LUT8 stalled. [Classification report](../gliner2-multi/coreml/reports/classification-embedding-w8.json); [extraction report](../gliner2-multi/coreml/reports/extraction-validation.json). |
| Laya English | FP16 with automatic device choice | Embedding int8 failed the 0.02 probability gate (0.059 worst) and slowed requests. Forced CPU+ANE was slower than automatic/GPU despite 1,213/1,218 eligible operations. Shipped multilingual Laya int8 remains the comparison baseline. [Report](../laya/coreml/STATUS-ENGLISH.md). |
| Kev 0.5B | FP16 and optional embedding W8 with automatic device choice | Forced ANE was faster in six selected calls but differed from automatic by 0.021 probability, above the 0.02 gate. Embedding W8 989.6→853.9 MB passed 19/19 selected native choices under the 0.02 gate, with four overlength skips and no speed win; published as a size option. [W8 report](../kev-0.5b/coreml/reports/e8-selected20.json); [profile](../../../tools/decision-coreml-profile/RESULTS.md). |
| Kev 0.6B | Public FP16 and W8 | W8 1,194→599 MB retained four selected typed choices. CPU+ANE full-request p50 11.90→12.33 ms; no speed win. Six int32 boundary operations account for estimated CPU work despite many compile-time dequantization operations. [Report](../../../tools/decision-coreml-profile/RESULTS.md). |
| Decision 1.0 Kai | FP16 Choice/Noul/Score previews; optional embedding W8 for all three | Each typed package 643.8→448.0 MB, 2/2 pinned native decisions per path, worst probability difference 0.00887. Choice W8 placed 926/938 executable ops on ANE, with no repeatable speed gain; whole-model W8 and marker-map rewrite remain rejected. [W8 report](../decision-kai/coreml/reports/embedding-w8.json); [ANE report](../decision-kai/coreml/reports/ane-experiments.md). |
| Decision 1.0 Lex | FP16 Choice/Noul/Score previews; optional embedding W8 for Noul/Score | Noul and Score each 643.8→448.0 MB, 2/2 pinned native decisions per path. Symmetric and asymmetric Choice W8 each flipped a real decision, so Choice stays FP16. Noul W8 placed 926/938 executable ops on ANE, with no repeatable speed gain. [W8 report](../decision-lex/coreml/reports/embedding-w8.json); [ANE report](../decision-lex/coreml/reports/ane-choice-profile.md). |
| Jeff | FP16 and optional W8 classifiers with automatic device choice | Automatic/GPU request p50 10.12 ms versus forced CPU+ANE 19.71 ms. W8 922.8→488.4 MB passed 4/4 native labels under its 0.25 logit gate with no demonstrated speed win; published as a scoped size option. [W8 validation](../jeff/coreml/reports/w8-validation.json); [profile](../../../tools/decision-coreml-profile/RESULTS.md). |
| LFM2.5-350M-RLCD | FP16 L256/B8/V16 with automatic device choice | 9/9 selected upstream cases passed. Forced ANE fails numerical parity; stable scoring and FP32 log-sum-exp trials also failed. Local tied-embedding W8 saved 67 MB but exceeded the stricter 0.1 release score-error gate (0.291); symmetric W8 changed a selected value. Per-block W8 requires reconversion targeting iOS 18. [ANE report](../lfm350/coreml/reports/ane-optimization/README.md); [W8 report](../lfm350/coreml/reports/embedding-w8-linear.json). |
| NanoJev and system-one-gemma | Source-only repos | No publishable weights to optimize: trained-weight redistribution rights are unstated for NanoJev; the exact Gemma base is manually gated. |

The tests above select valid bucket-fitting requests and have small sample sizes.
The full 132,422-request Decision Index and a controlled multi-seed 2048 study
need a separately agreed evaluation environment, frozen manifests and run
budget. A compressed package is published only when its selected native parity
and artifact scope are stated; the current trials do not establish a globally
optimal ANE schedule.
