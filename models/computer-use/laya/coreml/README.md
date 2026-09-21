# laya → Core ML

Converts the pinned **laya-multilingual** checkpoint (Convai Innovations, Apache-2.0;
mmBERT-base encoder + two-layer decision head, 321.9M parameters) into fixed-shape
**FP16 Core ML programs** that answer typed `choice` / `score` / `noul` questions in one
encoder pass. Three sequence-length buckets are exported: **128, 256, 512 tokens**, each
with 32 option slots. The whole model, including the 256k-row embedding table, the
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
uv run pytest -q
```

Python 3.12, PyTorch 2.7.0, coremltools 9.0, transformers 5.17 and `laya` 0.3.4 are
pinned in `uv.lock`. The reference implementation is the published `laya` package; the
export adapter (`export_model.py`) wraps its unmodified `DecisionModel` and only replaces
dynamic shapes with fixed ones.

Outputs:

- `build/laya_multilingual_fp16_L{128,256,512}_options32.mlpackage` — portable Core ML models.
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

Timing is one `predict` call per question after 5 warm-up calls (20 repeats), measured
from Python, so it includes the Core ML call overhead but not tokenization. Upstream
reports 32.8 ms per question on a Tesla T4 and ~21 ms on an M1 Max GPU.

The FP32 export adapter matches the reference within **7e-6** on logits before any
conversion, which separates prompt-building mistakes from FP16 error. The ANE 0.0126
worst case is the two-option prompt-injection question; every other question is under
0.005.

The graph is 99.5% Neural Engine (973 of 978 ops); the five CPU ops are the int32
casts and the embedding gather. ANE latency still grows faster than GPU latency with
sequence length because the L×L attention matmuls dominate, so the Swift manager runs
the 128 bucket on CPU+ANE and longer buckets on `.all`.

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
