# Decision-model candidates, scored on laya's own suite

Which open decision model should we convert to Core ML next? The
[Jev Reproductions Tracker](https://huggingface.co/spaces/multimodalart/jev-reproductions-tracker)
ranks 31 reproductions on a frozen 132,422-request suite and places laya 30th of 31. That ranking
is the reason for this directory: before converting anything on the strength of a leaderboard
position, score the candidates on the benchmark we already trust.

The benchmark is `../benchmark/suites.jsonl`, built by `../benchmark.py` from upstream's research
scripts (seed 13, 400 cases per task, MASSIVE 300 x 20 options). Every candidate sees the same
3,899 questions, the same option keys and descriptions in the same order, and is scored against the
same gold labels that produce laya's published per-suite accuracies.

## Application-suite experiment: tuned GLiClass Edge

The off-the-shelf checkpoints did not clear both gates, so the smallest promising encoder was tuned
on real public training splits for the ten application task families. The result is a 32.7M-parameter
checkpoint based on `knowledgator/gliclass-edge-v3.0`. Exact serialized states from all 3,899
benchmark rows are excluded before sampling training data. The benchmark itself remains untouched.

On FluidUse's application suite, the bucketed FP16 Core ML model is more accurate and materially
faster than laya. A subsequent deploy-faithful run over all 77,165 scoreable requests in the
Decision Index headline panel scores 19.27, above laya's published 16.39. The adapter uses the
actual Core ML packages, refuses inputs beyond their 512-token or 25-option capacities, never
truncates, and lets unsupported requests score as wrong. These measurements were taken on the same
Apple M5 Pro with 24 GB RAM and macOS 27.0.

| Measure | GLiClass Edge apps v2 | laya optimized e8 |
| --- | ---: | ---: |
| Parameters | **32.7M** | 322M |
| Full-suite macro accuracy | **72.75%** | 71.1% |
| Full-suite micro accuracy | **72.89%** | 71.2% |
| L128 fastest median | **0.843 ms** (CPU+ANE) | 3.6 ms (CPU+ANE) |
| L256 fastest median | **1.673 ms** (GPU) | 5.8 ms (all units) |
| L512 fastest median | **2.008 ms** (GPU) | 9.1 ms (all units) |
| Package size per bucket | **65.7–66.7 MB** | 448 MB (int8 embedding) |

The deployed Core ML check uses the smallest fitting L128/L256/L512 bucket for every row. It scores
72.75% macro and 72.89% micro after conversion, with 18 argmax changes in 3,899 requests (0.46%)
relative to PyTorch. The changes increase aggregate accuracy by 0.08 points. The bucket distribution
is 2,259 / 1,151 / 489, and the observed Python `MLModel.predict` median across the whole suite is
1.02 ms. The per-bucket profiler results above isolate model execution and use 50 timed iterations.

The Core ML compute plan places 432 of 437 operations on the Neural Engine for CPU+ANE. The five CPU
operations are four int32 mask/embedding operations and one dependent comparison. Cost weighting at
L128 is 49.05% ANE and 50.95% CPU; despite that boundary work, the measured L128 latency is 4.3x
lower than laya. At L512, `ALL` selects the GPU and is substantially faster than forcing CPU+ANE.

The Decision Index result has 54.96% mean area coverage, compared with laya's published 38.72%.
The public full-suite bundle was unavailable, so the 18 display-only benchmarks were omitted; they
do not affect the headline formula, but the official kit marks the run incomplete for tracker
submission. Its weak application suites remain support triage (26.0%) and RAG relevance (49.5%).

## Harnesses

| Script | Model family | Mapping |
| --- | --- | --- |
| `bench_kev.py` | Kev (`jaredpalmer/kev-*`) | Exact. Kev serves the same System One contract as laya: a shared state plus typed `choice` / `noul` / `score` questions, pointer head over option markers, one forward pass. Suite rows pass through unchanged. |
| `bench_gliner.py` | GLiNER 2.5 (`fastino/gliner2.5-*`) | Approximate. GLiNER has no instruction slot, so each suite gets a hand-written semantic task key, and the four binary suites get label strings paraphrasing laya's question. This is more favourable to GLiNER than the uniform treatment laya receives. |
| `bench_gliclass.py` | GLiClass | Direct single-label mapping. The instruction is the prompt, the serialized state is the text and option descriptions are the dynamic labels. |
| [`train.py`](../../../gliclass-edge-apps/coreml/train.py) | GLiClass Edge | Builds balanced real training data while excluding exact benchmark states, then tunes the selected encoder layers and classification heads. |
| [`convert-coreml.py`](../../../gliclass-edge-apps/coreml/convert-coreml.py) | GLiClass Edge | Exports fixed L128/L256/L512 Core ML programs with up to 25 dynamic options. |
| [`verify.py`](../../../gliclass-edge-apps/coreml/verify.py) | GLiClass Edge | Runs bucketed Core ML and PyTorch over all 3,899 application rows and records parity plus deployed accuracy. |
| [`decision_index_engine.py`](../../../gliclass-edge-apps/coreml/decision_index_engine.py) | GLiClass Edge | Runs the actual Core ML buckets through the official Decision Index engine interface without truncation. |

`bench_gliner.py --format` selects what the model reads: `state` is the serialized state alone,
`instructed` prepends laya's instruction text. Report the better of the two.

## Running

```bash
# GLiNER: its own venv, gliner2 needs protobuf + sentencepiece for the DeBERTa tokenizer
uv venv gliner-env && VIRTUAL_ENV=gliner-env uv pip install "gliner2[local]" protobuf sentencepiece
gliner-env/bin/python bench_gliner.py --model fastino/gliner2.5-small-v1 --format state

# Kev: clone the upstream repo, it carries the loader and the packed-sequence encoder
git clone --depth 1 https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python ../bench_kev.py --run jaredpalmer/kev-0.8b --device mps

# Accepted GLiClass result has its self-contained toolkit and full commands here:
cd ../../gliclass-edge-apps/coreml
uv sync
uv run python verify.py
```

## Gotchas

- `transformers.AutoTokenizer` cannot load the GLiNER 2.5 checkpoints directly: their
  `extra_special_tokens` metadata is a list, not a dict, and `_set_model_specific_special_tokens`
  calls `.keys()` on it. Use `gliner2.models.base.load_extractor_tokenizer`, which normalizes it.
- The task key in `classify_text(text, {key: labels})` is semantic, not a placeholder. A generic
  `"label"` key measurably underperforms a descriptive one such as `"intent"` or `"email_type"`.
- Kev checkpoints are LoRA adapters plus a `head.pt`; the Hub repo size (35-45 MB) is not the
  deployable size. The merged model is the base backbone, roughly 1 GB in fp16 for the 0.5B class.
- `kev-0.5b` and `kev-0.6b` are marked superseded prototypes by their own author. `kev-0.8b` is the
  current small checkpoint and is not on the tracker.

## Results

Apple M5 Pro, 2026-09-22. All 3,899 questions, zero errors on every run. GLiNER uses its better
input format (`state`); `instructed` costs it 11 points of macro average and is reported below only
as a caveat. Kev is scored through its serving path, so no question is refused.

| Model | Params | Macro, 10 suites | Macro, 8 suites | Suites beaten | ms p50 | Device |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| laya multilingual (shipped) | 322M | **0.711** | 0.641 | — | 61.5 | CPU x4 |
| Kev 0.8B | 0.8B | 0.653 | **0.674** | 5/10 | 171.2 | MPS |
| GLiNER 2.5 base | 194M | 0.609 | 0.607 | 3/10 | 54.0 | CPU x4 |
| Kev 0.5B | 494M | 0.577 | 0.614 | 2/10 | 50.0 | MPS |
| GLiNER 2.5 small | 74M | 0.549 | 0.542 | 2/10 | 18.0 | CPU x4 |

"Macro, 8 suites" drops `app.email_spam` and `app.phishing`. laya scores 0.993 on both while two
unrelated architectures land between 0.38 and 0.65; laya's own card says its base checkpoints are
"near chance on typed-decisions zero-shot" and that its headline numbers belong to checkpoints
"fine-tuned on that benchmark's own training split". The card does not list the RLCD training
corpora, so this is an observed asymmetry, not a confirmed contamination.

Per-suite, best configuration of each model:

| Suite | laya | Kev 0.8B | Kev 0.5B | GLiNER base | GLiNER small |
| --- | ---: | ---: | ---: | ---: | ---: |
| jev.ag_news | **0.935** | 0.892 | 0.900 | 0.785 | 0.733 |
| jev.emotion | 0.537 | **0.550** | 0.477 | 0.562 | 0.505 |
| massive_intent.en | 0.657 | **0.820** | 0.767 | 0.767 | 0.720 |
| app.support_triage | **0.542** | 0.375 | 0.370 | 0.352 | 0.302 |
| app.email_spam | **0.993** | 0.517 | 0.470 | 0.580 | 0.527 |
| app.phishing | **0.993** | 0.623 | 0.383 | 0.652 | 0.623 |
| app.guardrails_jailbreak | 0.807 | **0.843** | 0.785 | 0.790 | 0.720 |
| app.moderation_toxicity | 0.535 | 0.560 | 0.515 | **0.680** | 0.573 |
| app.rag_relevance | **0.672** | 0.537 | 0.527 | 0.645 | 0.608 |
| app.model_routing_domain | 0.441 | **0.817** | 0.571 | 0.278 | 0.175 |

### What this says about the tracker

The Jev Reproductions Tracker ranks laya 30th of 31 and puts GLiNER 2.5 small 46% above it and
Kev 0.5B 85% above it. Neither ordering reproduces here: on laya's own application suite both land
below it, and GLiNER small is last of the five. Tracker position on one frozen request suite did
not predict performance on this one, in either direction.

### Selection status

None of the off-the-shelf candidates is a viable laya replacement on this application workload. Kev
0.8B is the strongest of them after setting aside the two email suites, but its 0.8B decoder is
roughly 1.6 GB in FP16 and runs at 171 ms on MPS. The tuned GLiClass Edge model clears the
application-suite accuracy, Core ML latency, artifact-size and Decision Index headline gates. A
tracker submission still requires the unavailable display-only portion of the frozen suite.
