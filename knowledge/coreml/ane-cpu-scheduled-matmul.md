# ANE CPU-Scheduled Matmul (Private API) — Research Lead

**Status:** unverified third-party claim. Not yet reproduced in mobius. Recorded as a
lead to investigate, not as confirmed guidance.

**Provenance:** reviewer comment on the *Surgical Inference* draft (M. Mireles), left by
the author of the `ds4-ssd` repo. The comment responds to the paper's "reverse
engineering the ANE" section and its conclusion that source-level op rewrites do not
change Core ML runtime placement because the MIL compiler owns lowering decisions.

## The claim

There is a private (undocumented) ANE path exposing a **matmul op that can be scheduled
directly from the CPU, with no Core ML recompile**. Key points as stated:

- **CPU-scheduled matmuls, no recompile.** Dispatch a matmul to the ANE without going
  through the public Core ML compile → `.mlmodelc` → `predict` cycle. This sidesteps
  both per-call Core ML dispatch overhead and the ahead-of-time static-graph
  requirement — effectively "ANE as a BLAS backend" rather than "ANE as a compiled-graph
  runtime."
- **int8 weight transfers.** When a stage is memory-bandwidth-bound, transfer weights as
  int8 to cut the bytes streamed per call (vs. FP16/FP32).
- **Measured (author's own numbers, unverified):** `ds4-ssd` reaches ~20 TOPS on M4 and
  **beats the M3 Ultra GPU on LLM prefill**.

## Why it matters for mobius

Our working model (see `CLAUDE.md`, `knowledge/coreml/neural-engine/`) is that the ANE
is reached only via Core ML compile-and-predict, so admission is governed by MIL
compiler acceptance (op set, static shapes, tensor-geometry limits, state-mutation
cliffs). This lead, if it holds, adds a second dispatch mechanism outside that path and
sharpens two positions we currently hold:

1. **"LLMs don't benefit from the ANE" is too coarse.** That is a *decode*-phase claim
   (decode is memory-bandwidth-bound; the ANE manufactures no bandwidth). **Prefill is
   compute-bound**, and the claim here is that CPU-scheduled ANE matmuls beat even a
   top-tier GPU on prefill. Relevant to any encoder-heavy or long-context STT/LLM work.

2. **Bandwidth-bound stages have an untested lever: int8 transfer.** Where per-call cost
   is dominated by weight bytes ÷ DRAM bandwidth, dropping weights to int8 is the direct
   attack, independent of compute unit.

## If proven right: on-device LLM on the ANE

The reason this lead is worth tracking beyond a footnote: it reopens **on-device LLM
inference on the ANE**, which the prevailing view (and the Surgical Inference paper
itself) writes off. mobius currently has no LLM model class — `stt`, `tts`, `vad`,
`speaker-diarization`, `emb`, `segment-text` — partly because the ANE has been assumed
useless for autoregressive transformers. If the CPU-scheduled matmul path holds, the map
changes:

- **Prefill on ANE, decode wherever.** Prefill (compute-bound, dominates long-context and
  RAG / tool-use prompts) could move to the ANE and off the GPU; decode (bandwidth-bound)
  stays on GPU or CPU. A split-phase LLM, not an all-or-nothing placement — the same
  decompose-and-place logic mobius already applies to STT/TTS pipelines.
- **The GPU stays free.** The payoff is the same architectural argument as our audio
  pipelines: keep the GPU available for the UI and other work while the ANE carries the
  compute-bound phase. For an agentic on-device assistant (LLM + STT + TTS co-resident),
  that is the difference between "runs" and "thermally impossible."
- **int8 weight streaming compounds it.** Decode's bandwidth wall is exactly where int8
  transfer (point 2 above) bites, so the two claims reinforce rather than compete.

**This does not yet change mobius scope.** It is a conditional: *if* the API surface is
real, shippable, and the prefill number reproduces, then an `llm` class (or an
LLM-prefill accelerator stage) becomes a concrete direction. Until the verification
checklist below clears, treat on-device LLM-on-ANE as a hypothesis this lead would
unlock, not a committed roadmap item.

## What to verify before relying on this

- Locate the actual API surface (the `ds4-ssd` repo is the pointer). Determine whether it
  is a private framework symbol, a Metal/ANE hybrid, or an `MLCustomLayer`-style hook —
  and whether it is App Store-shippable or research-only.
- Reproduce the M4 prefill number with our own `coreml-cli` profiling harness.
- Measure int8 transfer vs. FP16 on a bandwidth-bound stage we already ship (e.g. a
  decoder weight-stream stage) to quantify the bandwidth win and any accuracy cost.
- Confirm iOS availability and OS-version fragility (private paths break across releases).

## References

- `ds4-ssd` repo (author's implementation) — primary pointer, get the exact commit/URL
  from the reviewer.
- `knowledge/coreml/neural-engine/docs/reverse-engineering.md` — existing vendored ANE
  reverse-engineering notes (hollance/neural-engine).
- `knowledge/coreml/neural-engine/docs/ane-vs-gpu.md` — ANE-vs-GPU tradeoff context.
- `knowledge/coreml/core-ml-on-device-llama.md` — Apple's public Core ML LLM path
  (decode-focused, KV-cache + Int4), the baseline this lead claims to beat on prefill.
