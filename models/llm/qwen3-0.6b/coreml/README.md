# Qwen3-0.6B → CoreML (LLM-on-ANE thesis test)

First `llm`-class conversion in mobius. Purpose: test whether a small LLM can run
real-time on-device and where it lands across compute units, per the research lead in
`knowledge/coreml/ane-cpu-scheduled-matmul.md` (prefill is compute-bound → ANE candidate;
decode is bandwidth-bound + mutates the KV cache in-graph → expected CPU/GPU per the
Surgical Inference §6.3 state-mutation cliff).

Model: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) — 28 layers, hidden
1024, 16 Q / 8 KV heads (GQA), head_dim 128, intermediate 3072, vocab 151,936, RoPE
θ=1e6, per-head q/k RMSNorm, tied embeddings. Arch constants are identical to the
in-repo `stt/qwen3-asr-0.6b` decoder, which this conversion is adapted from.

## Pipeline

Host owns embeddings + RoPE (outside the graph, matching the runtime split); the CoreML
model owns the 28 transformer layers + final RMSNorm + `lm_head`, with a stateful fp16 KV
cache. Output is `logits [1, 1, VOCAB]` for the last position only (prefill projects one
row). See `knowledge/coreml/ane-cpu-scheduled-matmul.md` for why prefill vs decode
placement is the interesting question.

## Run

```bash
uv sync
# Convert (downloads weights, traces, converts; ~minutes):
uv run python convert-coreml.py --output-dir ./build --max-seq-len 512

# Decode throughput + RTFx on a given engine:
uv run python benchmark.py --model-dir ./build --compute-units CPU_AND_NE
uv run python benchmark.py --model-dir ./build --compute-units CPU_AND_GPU

# Device placement / per-op fallback ablation:
cd ../../../../tools/coreml-cli
uv run coreml-cli <path>/build/qwen3_0_6b_decoder_stateful.mlpackage
uv run coreml-cli <path>/build/qwen3_0_6b_decoder_stateful.mlpackage --fallback
```

## RTFx

`benchmark.py` reports **RTFx = decode tok/s ÷ 15 tok/s**, where 15 tok/s ≈ brisk TTS
narration (a real-time downstream drain). RTFx > 1 means generation outpaces consumption.
Prefill latency and tok/s are reported separately.

## Findings (M5 Pro, 24 GB, macOS 26 / coremltools 9.0, 2026-07-16)

The two graphs split exactly as the Surgical Inference §6.3 thesis predicts — **the ANE
cliff is in-graph state mutation, not the transformer math.**

**Decode graph (stateful, in-graph KV-cache mutation) — `convert-coreml.py`:**
- `CPU_AND_NE`: **ANE REJECTED** — `ANECCompile() FAILED, error -14`. Matches §6.3: the
  in-graph cache write is the admission cliff.
- `CPU_AND_GPU`: **27 tok/s**, decode p50 29.6 ms/token, prefill 42.5 ms (8 tok).
  **RTFx = 1.80× (> 1, PASS** vs 15 tok/s real-time drain). Output is coherent English, so
  the port is numerically correct. (Python-`predict` overhead inflates per-token time; a
  Swift runtime would be faster.)

**Prefill graph (stateless, fixed seq_len=128) — `convert-prefill.py`:**
- `CPU_AND_NE`: **ANE ACCEPTED** — compiles and runs. `MLComputePlan`: **1918 ops on the
  Neural Engine, 1 on CPU** (rest are attribution-free consts) → essentially fully
  ANE-resident. p50 **12.5 ms / 128 tokens**.
- `CPU_ONLY`: 46.0 ms → the ANE is ~4× over CPU (proves it isn't silently CPU-fallback).
- `CPU_AND_GPU`: 9.3 ms → on this M5 Pro the (very strong) GPU is marginally ahead of the
  ANE via *standard* Core ML. The reviewer's "ANE beats the GPU on prefill" claim rests on
  the private CPU-scheduled matmul path (see `knowledge/coreml/ane-cpu-scheduled-matmul.md`),
  which is **not** what this standard-Core ML export uses — untested here.

**Bottom line:** an LLM *can* run on the ANE — for **prefill** (compute-bound, cache-free,
Dense-Static). Decode's KV-cache mutation is ANE-rejected and belongs on GPU/CPU, where it
is comfortably real-time (RTFx 1.80×). This is the split-phase placement the knowledge note
predicts.

Open next steps: stateless host-owned-cache decode (caches as I/O, per §6.3) to test if
*decode* can also reach the ANE; int8 weight streaming for the bandwidth wall; the private
matmul path for the ANE-beats-GPU claim.

> Not HF-uploaded — research/benchmark artifact, not a shipped model.
