# Experiment: Cua-S1 recipe on a sub-1 GB base

**Status: experiment, not started on GPU.** Everything needed to run it is in this directory.

## Why

The Core ML build of Cua-S1-4B-0.2 is 2.6 GB at best (GPTQ, same GUI-360 accuracy as fp16).
Going under 1 GB by compressing the 4B model does not work: 2-bit post-training quantization
keeps 6/38 decisions (chance), and ~1.5 bits/weight would be needed. The other way under 1 GB is
a smaller model trained the same way.

Qwen3.5-0.8B (873M params, Apache-2.0) has the same architecture and **the same tokenizer** as
Qwen3.5-4B, so the whole Core ML pipeline (`../../coreml`) and the FluidUse Swift runtime apply
unchanged. Its 248k x 1024 embedding table is 254M of the 873M parameters, so: int8 ~0.85 GB, GPTQ
int4 decoder + int8 embeddings ~0.55 GB. Qwen3.5-2B is the fallback (~1.1-1.3 GB at int4).

## Hypothesis and bar

Cua's SFT recipe (`train_4b_v2.py`, per-element loss, validation-selected epoch) on 0.8B reaches
**>= 80% task accuracy** on the 613-task GUI-360 text test split (4B: 85.5%). Report zero-shot and
SFT side by side. RL (Cua's second stage, live cua-bench environments) is out of scope for the
first run.

## Data (`make_splits.py`)

Cua's SFT split `crossdataset_hard_v2` is not published; this rebuilds an open one with Cua's
converters: GUI-360 *train* episodes (text, hard distractors), split 90/10 by episode, plus
`cua_bench_s1` generator tasks. Test = the 613-task GUI-360 *test* split used for the Core ML
numbers (hash `e305de96...`); the script refuses test-episode leakage.

| Split | Tasks | Largest families |
| --- | ---: | --- |
| train | 41,836 | search_filter 28,517; multi_step_submit 6,661; form_filling 4,966 |
| val | 4,662 | search_filter 3,114; multi_step_submit 793 |
| test | 613 | multi_step_submit 507; search_filter 44; form_filling 37 |

Known skew: Cua's keyword family heuristic puts most GUI-360 *train* steps in `search_filter`
while the test split is mostly `multi_step_submit`. The loss is per element, not per family, so
this is a watch item, not a blocker; rebalancing is the first knob if SFT underperforms.

## Run

```bash
# local: rebuild splits (needs the GUI-360 train JSONL, 15 GB) and bundle them
hf download vyokky/GUI-360 --repo-type dataset --include "train/data/*/*/success/*" --local-dir ../../coreml/data/gui360
PYTHONPATH=<cua>/libs/cua-bench-s1/python/src:<cua>/libs/cua-s1/python/src python make_splits.py
./pack.sh                                     # build/cloud-bundle.tar.gz (~30 MB)

# GPU box (one CUDA GPU, 24 GB+):
tar xzf cloud-bundle.tar.gz && BASE=Qwen/Qwen3.5-0.8B sh cloud/run.sh
# optional: HF_REPO=FluidInference/<name> pushes the adapter + reports to a private repo
```

`cloud/run.sh` pins Cua's source at `a5f18829`, installs transformers 5.17 (Cua's `four-b-train`
extra pins `<5`, which cannot load `qwen3_5`), scores the zero-shot base, trains, and scores the
adapter with `cloud/eval_gui360.py` (the same readout and scoring as `coreml/bench_gui360.py`).
Cua's trainer was smoke-tested locally on transformers 5.17 with the 0.8B tokenizer (data
planning, letter tokens, per-element groups); training itself needs CUDA.

Compute estimate: ~42k examples x 4 epochs of ~0.6k-token prompts on 0.8B, roughly 1-3 GPU-hours
on one A100/H100-class GPU (LoRA r16, bf16).

## If it clears the bar

Merge the adapter and convert with `../../coreml`. The export module is config-driven (0.8B: 24
layers, hidden 1024, 8 heads / 2 KV, 16 linear-attention value heads), but the scripts hardcode
`BASE = "Qwen/Qwen3.5-4B"` and the adapter id, which need a flag. Then verify parity, run GPTQ,
and publish as a new FluidUse variant.
