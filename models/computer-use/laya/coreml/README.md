# laya → Core ML

Converts the pinned **laya-multilingual** checkpoint (Convai Innovations, Apache-2.0;
mmBERT-base encoder + two-layer decision head, 321.9M parameters) into fixed-shape
**FP16 Core ML programs** that answer typed `choice` / `score` / `noul` questions in one
encoder pass. Four sequence-length buckets are exported: **128, 256, 512, 1024 tokens**, each
with 32 option slots (1024 is upstream's `max_len`). The whole model, including the 256k-row embedding table, the
encoder, the decision head, the option scorer and the action head, lives in the graph.

This is a bounded local conversion verification on 16 fixture questions (Tetris
placements, a form field, support triage, zh/ja/de, 20 options, prompt-injection, a
long meeting note). It establishes parity with the unmodified PyTorch runtime, not
task accuracy: laya's own benchmarks describe what the model can decide.

## Reproduce

Run from this directory on an Apple silicon Mac:

```bash
uv sync --frozen
uv run python assets.py                       # pinned download + SHA-256 checks
uv run python convert-coreml.py --length 128  # also 256 and 512
uv run python verify.py --length 128          # parity + latency, writes reports/
uv run python make_fixtures.py                # Swift parity fixtures
uv run python quantize.py --length 128 --precision e8 && uv run python verify.py --length 128 --precision e8
uv run pytest -q
```

Python 3.12, PyTorch 2.7.0, coremltools 9.0, transformers 5.17 and `laya` 0.3.4 are
pinned in `uv.lock`. The reference implementation is the published `laya` package; the
export adapter (`export_model.py`) wraps its unmodified `DecisionModel` and only replaces
dynamic shapes with fixed ones.

Outputs:

- `build/laya_multilingual_fp16_L{128,256,512,1024}_options32.mlpackage` — portable Core ML models.
- `build/*.conversion.json` — versions, source revision, package hashes.
- [reports/verification-multilingual-L128.json](reports/verification-multilingual-L128.json)
  (and L256, L512) — per-question parity and timing on `ALL` and `CPU_AND_NE`.
- [reports/ane-fallback-L128.json](reports/ane-fallback-L128.json) — `coreml-cli` compute placement.
- `fixtures/tokenizer-cases.json`, `fixtures/sequence-cases.json` — expected ids for the Swift port.

## Verified results

Apple **M5 Pro, 24 GB, macOS 27.0**, September 21, 2026. Reference is PyTorch FP32 on CPU.
Gates chosen before conversion: 100% argmax agreement, max probability error ≤ 0.02,
max action-probability error ≤ 0.02, finite outputs, padded option slots at −10⁴.

| Bucket | Units | Questions | Argmax | Max Δprob | p50 | p95 | Load |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L128 | ALL | 16 | 16/16 | 0.0021 | 3.87 ms | 4.15 ms | 1.9 s |
| L128 | CPU_AND_NE | 16 | 16/16 | 0.0126 | **3.64 ms** | 3.79 ms | 3.0 s |
| L256 | ALL | 16 | 16/16 | 0.0021 | **5.19 ms** | 5.62 ms | 1.8 s |
| L256 | CPU_AND_NE | 16 | 16/16 | 0.0126 | 9.88 ms | 10.14 ms | 4.4 s |
| L512 | ALL | 16 | 16/16 | 0.0021 | **9.00 ms** | 9.37 ms | 2.5 s |
| L512 | CPU_AND_NE | 16 | 16/16 | 0.0126 | 27.45 ms | 27.90 ms | 4.8 s |
| L1024 | ALL | 16 | 16/16 | 0.0023 | **17.89 ms** | 18.32 ms | 2.0 s |
| L1024 | CPU_AND_NE | 16 | 16/16 | 0.0126 | 80.07 ms | 80.43 ms | 6.9 s |

Timing is one `predict` call per question after 5 warm-up calls (20 repeats), measured
from Python, so it includes the Core ML call overhead but not tokenization. Upstream
reports 32.8 ms per question on a Tesla T4 and ~21 ms on an M1 Max GPU.

The FP32 export adapter matches the reference within **7e-6** on logits before any
conversion, which separates prompt-building mistakes from FP16 error. The ANE 0.0126
worst case is the two-option prompt-injection question; every other question is under
0.005.

The graph is 99.5% Neural Engine (973 of 978 ops); the five CPU ops are the int32
casts and the embedding gather. ANE latency still grows faster than GPU latency with
sequence length because the L×L attention matmuls dominate, so the FluidUse manager runs
the 128 bucket on CPU+ANE and longer buckets on `.all`.

## ANE profile

`coreml-cli --ops -n 20` on every bucket (`reports/ane-profile-L*.json`), Apple M5 Pro. The
percentages are the profiler's estimated runtime share per device for the CPU+ANE configuration;
op counts are 973 of 978 on the ANE for every bucket (`reports/ane-fallback-L*.json`), the five
CPU ops being the int32 casts and the embedding gather.

| Bucket | Cold compile | CPU only | CPU+GPU | CPU+ANE | All units |
| --- | ---: | ---: | ---: | ---: | ---: |
| L128 | 4.9 s | 14.1 ms | 4.6 ms | 3.9 ms (ANE 33% / CPU 67%) | 4.1 ms (GPU) |
| L256 | 5.5 s | 27.1 ms | 5.3 ms | 9.7 ms (ANE 52% / CPU 48%) | 6.4 ms (GPU) |
| L512 | 6.0 s | 58.9 ms | 8.8 ms | 28.0 ms (ANE 71% / CPU 29%) | 8.8 ms (GPU) |
| L1024 | 8.5 s | 149.6 ms | 18.1 ms | 80.4 ms (ANE 87% / CPU 13%) | 17.9 ms (GPU) |

Two things follow. `all` never picks the ANE for this graph; it runs 100% on the GPU, so
`.all` and `.cpuAndGPU` are the same thing here. And the ANE wins only at 128 tokens: its
estimated share grows with length but so does its cost per token, because the L×L attention
scores at fp16 are the dominant ops and the ANE handles them worse than the GPU. The ANE
configuration's variance is the lowest of the four (std 0.04 ms at L128 vs 0.33 ms on the GPU),
which matters for a decision loop that fires many short questions. That is why the FluidUse
manager defaults the 128 bucket to CPU+ANE and every longer bucket to all units.

## Accuracy benchmark — laya's published suites on device

`benchmark.py` rebuilds the application suites from laya's own research scripts (same datasets,
seed 13, 400 cases per task, 300 MASSIVE cases with 20 options; banking77 skipped because its 77
labels exceed the 32 option slots) and scores them with the unmodified PyTorch model on CPU at
`max_len` 1024. `FluidUseLaya benchmark` (FluidInference/FluidUse) then answers the same 3,899 questions with the Core
ML buckets (128/256/512/1024, smallest that fits) from Swift and compares row by row.

Apple M5 Pro, macOS 27.0, September 21, 2026:

| Suite | n | Upstream (T4) | PyTorch CPU here | Core ML (Swift) | Row agreement | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jev.ag_news | 400 | 0.930 | 0.935 | **0.935** | 1.000 | 3.8 ms |
| jev.emotion | 400 | 0.530 | 0.537 | **0.537** | 1.000 | 3.7 ms |
| massive_intent.en | 300 | 0.657 | 0.657 | **0.657** | 1.000 | 5.2 ms |
| app.support_triage | 400 | 0.522 | 0.540 | **0.542** | 0.998 | 5.3 ms |
| app.email_spam | 400 | 0.993 | 0.993 | **0.993** | 1.000 | 5.8 ms |
| app.phishing | 400 | 0.993 | 0.993 | **0.993** | 1.000 | 9.0 ms |
| app.guardrails_jailbreak | 400 | 0.755 | 0.805 | **0.805** | 0.995 | 3.8 ms |
| app.moderation_toxicity | 400 | 0.525 | 0.535 | **0.535** | 1.000 | 3.8 ms |
| app.rag_relevance | 400 | 0.657 | 0.672 | **0.672** | 1.000 | 5.3 ms |
| app.model_routing_domain | 399 | 0.123 | 0.441 | **0.441** | 1.000 | 5.3 ms |

Whole run: **3,899 questions in 22.9 s, p50 5.2 ms, p95 18.0 ms** from Swift, versus
61.6 ms per question for PyTorch FP32 on this Mac's CPU (4 threads) and 32.8 ms per question
upstream reports on a Tesla T4. Accuracy is identical to the reference on every suite to within
two flipped rows (support triage 0.542 vs 0.540); the largest per-row probability delta is on a
guardrails row the reference itself scores at ~0.5. `upstream` is laya's BENCHMARKS.md
laya-multilingual column; the PyTorch column reproduces it here except model routing, where the
published 0.123 looks like an upstream run artefact (0.441 here from the same script).

Reports: [benchmark-reference.json](reports/benchmark-reference.json),
[benchmark-coreml.json](reports/benchmark-coreml.json). Reproduce with
`uv run python benchmark.py` then
`swift run -c release FluidUseLaya benchmark --suites benchmark/suites.jsonl --reference benchmark/reference-rows.jsonl --model-dir build/laya-coreml`.

## Weight compression

`quantize.py` applies `coremltools.optimize.coreml` post-training compression to an exported bucket
and `verify.py --precision <tag>` re-runs the parity gates. Compression is restricted by name to the
2-D linear weight matrices and the embedding gather: the default `weight_threshold` would also
sweep the additive attention masks, the RoPE cos/sin tables and the biases into compression. Results
at L128 (16 fixture questions, argmax agreement · max Δprob vs PyTorch):

| Tag | Scheme | Package | ALL | CPU+ANE | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| `fp16` | reference | 644 MB | 16/16 · 0.002 | 16/16 · 0.013 | **pass** |
| `e8` | int8 embedding table, fp16 encoder + head | 448 MB | 16/16 · 0.015 | 16/16 · 0.014 | **pass** |
| `w8head` | int8 decision head + scorer only | 629 MB | 16/16 · 0.002 | 16/16 · 0.013 | **pass** |
| `w8enc` | int8 encoder linears (per-channel) | 534 MB | 16/16 · 0.032 | 11/16 · 0.665 | fail |
| `w8e` | int8 encoder + head linears | 519 MB | 16/16 · 0.029 | 11/16 · 0.666 | fail |
| `w8` | int8 everything | 324 MB | 16/16 · 0.042 | 11/16 · 0.668 | fail |
| `w6` | 6-bit k-means palette, encoder + head | 488 MB | 15/16 · 0.114 | 15/16 · 0.118 | fail |
| `w4` | 4-bit k-means palette, encoder + head | 456 MB | 12/16 · 0.727 | 12/16 · 0.728 | fail |
| `w6e8` | 6-bit palette + int8 embedding | 292 MB | 15/16 · 0.117 | 15/16 · 0.120 | fail |
| `w4e8` | 4-bit palette + int8 embedding | 261 MB | 11/16 · 0.734 | 11/16 · 0.729 | fail |

Only the embedding table tolerates compression. Every scheme that touches the encoder's linear
weights fails: per-channel int8 stays within 0.03 on the GPU but collapses on the Neural Engine
(11/16 argmax, Δprob 0.67, identical for `w8`, `w8e`, `w8enc`, so it is the encoder int8 path on
the ANE, not the head), and 6-/4-bit k-means palettes fail on both devices. Per-block int8 needs an
iOS 18 deployment target and was not tried. This matches the upstream Core ML port, which also
published no 6-/4-bit variants.

`e8` is published for every bucket (`laya_multilingual_e8_L*_options32.mlmodelc`):

| Bucket | Package | CPU+ANE Δprob | CPU+ANE p50 | ALL Δprob | ALL p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| L128 | 448 MB | 0.014 | 3.6 ms | 0.015 | 3.8 ms |
| L256 | 449 MB | 0.014 | 9.6 ms | 0.006 | 5.8 ms |
| L512 | 450 MB | 0.014 | 27.7 ms | 0.006 | 9.1 ms |
| L1024 | 453 MB | 0.014 | 80.1 ms | 0.006 | 17.9 ms |

On the full 3,899-question benchmark (`reports/benchmark-coreml-e8.json`) e8 keeps every suite's
accuracy within 0.5 points of fp16 at the same latency (p50 5.3 ms):

| Suite | fp16 | e8 | Row agreement | Max Δprob |
| --- | ---: | ---: | ---: | ---: |
| jev.ag_news | 0.935 | **0.935** | 1.000 | 0.055 |
| jev.emotion | 0.537 | **0.535** | 0.995 | 0.086 |
| massive_intent.en | 0.657 | **0.653** | 0.987 | 0.144 |
| app.support_triage | 0.542 | **0.537** | 0.998 | 0.052 |
| app.email_spam | 0.993 | **0.993** | 1.000 | 0.024 |
| app.phishing | 0.993 | **0.993** | 1.000 | 0.035 |
| app.guardrails_jailbreak | 0.805 | **0.810** | 0.990 | 0.801 |
| app.moderation_toxicity | 0.535 | **0.535** | 1.000 | 0.016 |
| app.rag_relevance | 0.672 | **0.675** | 0.993 | 0.031 |
| app.model_routing_domain | 0.441 | **0.454** | 0.975 | 0.078 |

## Input and output contract

One prediction scores one question. The host builds the sequence exactly like
`laya.common.build_sequence` (`preprocessing.py` calls it):
`[CLS] <type> question: <instructions> [SEP] ([MASK] <option>)* [SEP] <state> [SEP]`,
with `head_max_len = 256` tokens for instructions plus options and the rest for the state.

| Tensor | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `input_ids` | int32 | `[1, L]` | Token ids, `<pad>` (0) padded |
| `attention_mask` | int32 | `[1, L]` | 1 for real tokens |
| `marker_map` | float32 | `[1, 32, L]` | One-hot row per option pointing at its `[MASK]`; zero rows unused |
| `question_type` | float32 | `[1, 3]` | One-hot over choice / score / noul |
| `logits` | float32 | `[1, 32]` | Option scores; unused slots are −10⁴ |
| `probabilities` | float32 | `[1, 32]` | Softmax over supplied options at temperature 1 |
| `action_probabilities` | float32 | `[1, 2]` | Action head (`act_probability` is index 0) |

Per-question temperatures (`temperature`, `temperature_by_options`) and `head_max_len`
are stored in the package's creator-defined metadata; the multilingual checkpoint ships
all temperatures at 1.0. The sliding-window band mask (|i − j| ≤ 64 on 2 of every 3
layers) and RoPE tables are constants per bucket; padding is applied as an additive
−10⁴ mask in-graph.

## Limits

- Fixed shapes: a prompt longer than the largest bucket has its state truncated on the
  right, as upstream does for `max_len`; a question whose options do not fit is rejected.
- The FP16 embedding table is 393 MB of the 614 MB package; the three buckets share
  weights on disk only if the caller dedups them, so ship the buckets you need.
- Only the multilingual checkpoint is converted here. The English ModernBERT-large root
  checkpoint uses a different tokenizer and 421M parameters; `assets.lock.json` can take
  a second variant entry.
