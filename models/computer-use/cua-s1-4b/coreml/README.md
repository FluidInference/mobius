# Cua-S1-4B-0.2 → Core ML

Published at [FluidInference/cua-s1-4b-coreml](https://huggingface.co/FluidInference/cua-s1-4b-coreml);
Swift runtime: FluidUse `CuaS1FourBManager`.

Converts [`cua-ai/cua-s1-4b-0.2`](https://huggingface.co/cua-ai/cua-s1-4b-0.2) (Apache-2.0 LoRA
adapters, `text/` and `multimodal/`, on the frozen Apache-2.0
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B)) into FP16 Core ML programs for
closed-option GUI decisions. Each adapter is merged into its own copy of the base (LoRA deltas
added in fp32, then stored fp16).

Cua-S1-4B is not a generator. `cua_s1.four_b.FourBModel` runs **one forward pass** over a chat
prompt (screen state + lettered `(element, action)` options) and reads the logits of the answer
letters `A..Z` at the last position. The export keeps exactly that contract:

- **Prefill only**: no KV cache, no recurrent state I/O, no 248k-row vocabulary head. The last
  part gathers the final hidden state (one-hot matmul), applies the final norm and multiplies by
  the 26 tied-embedding rows of the letter tokens -> `letter_logits [1, 26]`.
- **Four decoder parts** of 8 layers each (1.7 GB fp16 per part), `hidden [1, L, 2560]` in and
  out, plus host-computed M-RoPE `cos`/`sin [L, 64]`. The embedding gather is host-side
  (`embeddings.f16`, row-major `[248320, 2560]`), so screenshot features can be spliced in.
- **Fixed buckets with right padding.** The model is causal and the gated delta rule is a
  forward scan, so tokens after the last real one cannot change it (verified in
  `tests/test_delta_rule.py`). Text: `L=1024`. Multimodal: `L=2048`.
- **Qwen3.5's hybrid stack**: 24 Gated-DeltaNet (linear attention) + 8 gated full-attention
  layers. The delta rule uses the chunked WY form (chunk 64). Its unit-lower-triangular solve
  is a recursive 2x2 block inverse; the nilpotent power series `(I+N)(I+N^2)...` is exact
  algebraically but its terms overflow on real layers (layer 24 went NaN). Within-chunk decay
  differences are masked sums of per-step decays (no cancellation of large cumulative sums),
  and those constant-mask matmuls are written const-first so the converter does not lower them
  to `linear` ops that weight compression would then touch.
- **Vision tower** (`multimodal/`): one static token-level graph for up to 4096 patches (1024
  image tokens, about 1280x800 px after `smart_resize`), with a key mask for padding. The host does the
  processor's work: `smart_resize` (factor 32), bicubic antialiased resize on uint8,
  merge-window patchify, bilinear resampling of the learned position table and the axial 2D
  rotary tables. The Conv3d patch embed over the processor's two identical frames folds into a
  Linear. Screenshots larger than the budget are downscaled further than the reference would.

## Reproduce

Apple silicon Mac, from this directory. Cua's `cua_s1` and `cua_bench_s1` are not on PyPI;
use the source at the pinned commit:

```bash
git clone https://github.com/trycua/cua /tmp/cua && git -C /tmp/cua checkout a5f18829df026d7b9ef80c339194b44b1c61f856
export PYTHONPATH=/tmp/cua/libs/cua-bench-s1/python/src:/tmp/cua/libs/cua-s1/python/src
uv sync
uv run python make_fixtures.py                              # 38 tasks from Cua's own generator
uv run python merge_reference.py --modality text            # merged weights + bf16 FourBModel reference (MPS)
uv run python merge_reference.py --modality multimodal
uv run python reference_fp32.py --modality text             # fp32 reference: HF decoder layers, one at a time
uv run python reference_fp32.py --modality multimodal
uv run python convert-coreml.py --modality text --length 1024
uv run python convert-coreml.py --modality multimodal --length 2048
uv run python convert-vision.py --max-patches 4096
uv run python verify.py --modality text --length 1024 --compute-units CPU_AND_GPU
uv run python verify.py --modality multimodal --length 2048 --compute-units CPU_AND_GPU
uv run python quantize.py --mode w8                          # also w4, p4/p3/p2 (one --part per process)
uv run python quantize.py --modality multimodal --length 2048 --mode w8
for s in 0 4 8 12 16 20 24 28; do uv run python gptq.py calibrate --device cpu --start $s --end $((s + 4)); done
uv run python gptq.py assemble                               # build/text/L1024-gptq
uv run python verify.py --variant gptq --compute-units CPU_AND_GPU
uv run python make_swift_fixtures.py
uv run pytest -q
./upload-hf.sh                                              # compiles .mlmodelc, stages, uploads
```

GUI-360 accuracy (`make_gui360.py`, `bench_gui360.py --backend coreml --variant gptq`) needs the
`vyokky/GUI-360` test JSONL (`hf download vyokky/GUI-360 --repo-type dataset --include
"test/data/*/*/success/*"`); GPTQ calibration uses a slice of the train JSONL
(`--include "train/data/*/in_app/success/*_1_1[0-9]*.jsonl"`).

## Verified results

Apple **M5 Pro, 24 GB, macOS 27.0**, September 23, 2026. Reference = fp32 transformers
`Qwen3_5DecoderLayer`s run layer by layer on the merged weights and the **unpadded** prompt
(`reference_fp32.py`). The shipped bf16 `FourBModel` runtime itself flips 3/38 argmaxes
against that fp32 reference (near-ties), so it is not used as the parity target.

| Build | Size | Fixture argmax (38) | Median / max \|Δp\| | GUI-360 text (613) | Median latency |
| --- | ---: | ---: | --- | ---: | ---: |
| text fp16 L1024 | 6.8 GB | 38/38 | 0.0013 / 0.0053 | **85.5%** | 1.05 s |
| text `gptq`: MLP int4 (block 16) + int8 rest | 2.6 GB | 38/38 | 0.022 / 0.133 | **85.5%** | 1.19 s |
| text `w8`: int8 (all linears, RTN) | 3.4 GB | 37/38 | 0.006 / 0.019 | — | — |
| text MLP int4 b16 + int8 (RTN) | 2.6 GB | 35/38 | 0.031 / 0.081 | — | — |
| text int4 b32 (RTN, all) | 1.9 GB | 27/38 | — / 0.29 | — | — |
| text 4 / 3 / 2-bit palettes | 1.7 / 1.3 / 0.9 GB | 28 / 24 / 6 | — / 0.64-0.81 | — | — |
| multimodal fp16 L2048 + vision P4096 | 6.8 + 0.6 GB | 38/38 | — / 0.153 (fp16 ViT) | not run | 1.65 s |
| multimodal `w8` + vision P4096 | 3.4 + 0.6 GB | 38/38 | — / 0.163 | not run | — |

Same first 100 GUI-360 tasks: PyTorch bf16 `FourBModel` (MPS) 84%, Core ML fp16 88% (same task
outcome on 96/100), GPTQ 87% (97/100). The GUI-360 split is rebuilt with Cua's own converter
(`make_gui360.py`, 613 tasks); Cua's frozen 615-task split is published by hash only. ANE:
37/38 but 19 s/decision (CPU fallback), so GPU only.

GPTQ (`gptq.py`) calibrates on GUI-360 *train* + generator prompts (seed disjoint from fixtures),
runs 4 layers per process on CPU (torch 2.7 MPS segfaults inside MPSGraph under GPTQ; processes
also died at teardown once), ~2.5 min/layer.

Not done: multimodal GPTQ and a GUI-360 screenshot run (needs the 6 GB test image archive); GPTQ int4
on all linears (~1.9 GB); int8 embedding table; a 2048 text bucket (2/613 GUI-360 prompts exceed 1024).

Latency: text fp16 0.68 s and multimodal 1.65 s were measured on an idle GPU; the other numbers
were taken while other GPU work was running and are upper bounds (idle vs loaded re-check on one
delta-rule layer: identical 94 ms for both matmul formulations).
