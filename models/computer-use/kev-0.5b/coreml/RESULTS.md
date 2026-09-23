# Kev 0.5B Core ML proof-of-concept results

Measured September 22, 2026 on an Apple M5 Pro with 24 GB RAM and macOS 27.0. The source revisions
are pinned in `assets.lock.json`.

## Result

Kev 0.5B converts cleanly, but it should not replace laya in FluidUse. It is slower and larger at
the short bucket and scores substantially worse on FluidUse's application suite, despite its much
stronger score on the broader Jev Reproductions Tracker.

| Measure | Kev 0.5B | laya |
| --- | ---: | ---: |
| Application suite, 3,899 questions | 57.2% | 71.2% |
| L128 package | 990 MB | 614 MB |
| L128 fastest profiled median | 7.24 ms | 3.64 ms |
| Tracker Decision Index | 30.34 | 16.39 |

Kev completed every application request without an error. Its per-suite accuracy was:

| Suite | Kev | laya |
| --- | ---: | ---: |
| AG News | 90.0% | 93.5% |
| Emotion | 47.7% | 53.7% |
| MASSIVE intent | 76.7% | 65.7% |
| Support triage | 37.0% | 54.2% |
| Email spam | 47.0% | 99.3% |
| Phishing | 38.3% | 99.3% |
| Jailbreak | 78.5% | 80.5% |
| Toxicity | 51.5% | 53.5% |
| RAG relevance | 52.7% | 67.2% |
| Model routing | 57.1% | 44.1% |

## Conversion parity

The explicit fixed-shape wrapper matches the merged upstream model with identical argmax and a
maximum logit difference of 0.0000131 before conversion. With Core ML on all compute units, the
three smoke fixtures have identical argmax and maximum probability error 0.000502.

CPU+Neural Engine preserved argmax on 19 of 19 runnable fixtures. The 16 sampled real-suite rows
had maximum probability error 0.0157. One additional handwritten phishing fixture reached 0.0210,
slightly beyond the 0.02 probability gate; use all compute units when strict probability parity is
required.

## L128 profile

| Compute units | Device assignment | Median | Mean | Stddev |
| --- | --- | ---: | ---: | ---: |
| All | 100% GPU | 8.22 ms | 8.23 ms | 0.18 ms |
| CPU only | 100% CPU | 30.99 ms | 31.19 ms | 0.52 ms |
| CPU + GPU | 100% GPU | 8.07 ms | 8.07 ms | 0.11 ms |
| CPU + Neural Engine | 74.2% ANE / 25.8% CPU runtime | **7.24 ms** | 7.25 ms | 0.06 ms |

The operation-level plan assigns 1,394 of 1,399 operations to the Neural Engine. The five CPU
operations are four int32 operations around masks and the embedding gather, plus one unresolved
comparison.

## Compression experiments

Compression was rejected for publication. Every tested int8 package kept fixture argmax, but none
met the 0.02 probability gate across the sampled checks.

| Scheme | Package | Maximum observed probability error |
| --- | ---: | ---: |
| FP16 | 990 MB | 0.0210 on CPU+ANE; 0.0005 on all units |
| Embedding int8 | 854 MB | 0.0289 |
| Embedding + half the MLP layers int8 | 697 MB | 0.0270 |
| Embedding + all MLP layers int8 | 541 MB | 0.0267 |
| All large weights int8 | 496 MB | 0.0676 |

Four- and six-bit compression were not pursued after whole-model int8 failed the parity gate.
