# CosyVoice3 → CoreML Progress Report

Status snapshot of the CosyVoice3 (Mandarin) CoreML conversion pipeline.
The authoritative deep history — every trial, every NaN, every revert —
lives in [`TRIALS_AND_ERRORS.md`](./TRIALS_AND_ERRORS.md). This file is
the single-page "what ships today, what's in-flight" view.

## Shipping configuration (frozen)

Four CoreML models + two runtime tables + tokenizer assets, staged in
`build/upload/cosyvoice3-coreml/` for the `FluidInference/cosyvoice3-coreml`
HuggingFace repo. Minimum OS: **macOS 15 / iOS 18** (required by the
stateful decode model's `MLState` buffers).

| Model | Compute | Purpose | dtype | Size |
|---|---|---|---|---|
| `LLM-Prefill-T256-M768-fp16` | CPU + ANE | Qwen2-0.5B prefill over 256-token context, initializes 768-slot KV cache | fp16 | 695 MB |
| `LLM-Decode-M768-fp16-stateful` | CPU + GPU | Single-step AR decode against 768-slot KV cache held in `MLState` (48 per-layer buffers) | fp16 | 695 MB |
| `Flow-N250-fp16` | CPU + GPU | Speech tokens → 80-bin log-mel @ 24 kHz | fp16 | 638 MB |
| `HiFT-T500-fp16` | CPU + ANE | Mel → 24 kHz PCM via iSTFT vocoder | fp16 | 44 MB |

Runtime tables:
- `embeddings/embeddings-runtime-fp32.safetensors` (542 MB) — Qwen2
  `embed_tokens` at runtime fp32 dtype (required for bit-exact parity).
- `embeddings/speech_embedding-fp16.safetensors` (12 MB) — CosyVoice3
  `speech_embedding` table (6761 × 896 fp16).

Voices:
- `voices/cosyvoice3-default-zh.safetensors` — upstream `zero_shot_prompt.wav`
  (female, 希望你以后能够做的比我还好呦。).
- 10 AISHELL-3 bootstrapped voices (5 F + 5 M, north + south accents).

Total disk footprint (`.mlmodelc` + `.mlpackage` + runtime tables): **~6.6 GB**.

## End-to-end performance (Swift, M-series)

Measured via `fluidaudio tts --backend cosyvoice3-parity`, shipping
fixture, N_new=87, N_prompt=87 → 3.48 s audio @ 24 kHz:

| Stage   | Wall time | Compute unit       |
|---------|----------:|--------------------|
| prefill | ~0.9 s    | cpuAndNeuralEngine |
| decode  | ~2.0 s    | cpuAndGPU (stateful KV) |
| flow    | ~6.9 s    | cpuAndGPU          |
| hift    | ~0.9 s    | cpuAndNeuralEngine |
| **total** | **~10.7 s** (RTFx ~0.33×) | — |

Flow is the dominant cost at ~65% of total synth. See
"Flow ANE port" below for the attempted — and reverted — ANE variant.

## Parity

End-to-end ASR round-trip on the shipping fixture
(`build/wavs/e2e_shipping.wav`):
- Peak amplitude 0.815, mean |x| 0.052.
- CTC-ZH: 希望你以后能够做得比我还好哟。
- Qwen3: 希望你以后能够做得比我还好哟。
- Matches prompt text modulo one homophone (`的` ↔ `得`), which is
  upstream CosyVoice3 behavior.

Per-model parity vs PyTorch fp32 reference:

| Model | Metric | Value |
|---|---|---|
| LLM-Prefill fp16 | logits MAE | 0.068 (argmax OK) |
| LLM-Decode fp16 (stateful) | logits MAE | 0.018 (argmax OK) |
| Flow-N250 fp16 (cpuAndGPU) | mel MAE | 4.7e-02 |
| HiFT-T500 fp16 | audio MAE | within fp16 envelope |

## Conversion scripts

| Script | Produces | Notes |
|---|---|---|
| `convert-llm.py` | LLM-Prefill + LLM-Decode (stateful) | Selective fp32 pins (`pow/reduce_mean/rsqrt/softmax`) for RMSNorm stability |
| `convert-flow.py` | Flow-N250-fp16 | Default path is cpuAndGPU fp16; `--ane-port` path is experimental (see ANE port section) |
| `convert-coreml.py` | HiFT-T500-fp16 | Folded weight-norm, iSTFT stays on ANE |
| `convert-campplus.py` | CAMPPlus speaker embed | Python-side only (not in shipping pipeline — prompt embeddings pre-extracted per voice) |
| `convert-speech-tokenizer.py` | SpeechTokenizerV3 | Python-side only (CoreML version had 44/87 token drift from MIL argmax instability) |
| `export-embeddings.py` | Runtime embedding tables | Qwen2 + speech_embedding safetensors |

CAMPPlus and SpeechTokenizerV3 were intentionally dropped from the
shipping Swift pipeline: they run server-side once per voice, not per
synthesis. Their outputs (speaker embedding, prompt speech tokens) are
baked into the per-voice safetensors bundle.

## Flow ANE port — attempted and reverted

An ANE-port BC1S rewrite of the Flow DiT was built and benchmarked end
to end. Compiled cleanly, ran ~3× faster, and passed the Stage 0 "no
NaN" gate — but collapsed the mel dynamic range on the parity fixture,
yielding audio unintelligible to both CTC-ZH and Qwen3 ASR. Reverted
to the cpuAndGPU fp16 baseline.

| Compute unit | mel range | MAE vs fp32 ref | NaN | Audio peak |
|---|---|---:|---:|---:|
| PyTorch fp32 reference | [-12.443, 5.157] | 0.000 | 0 | — |
| Baseline Flow cpuAndGPU (**shipping**) | [-12.500, 5.172] | 4.7e-02 | 0 | 0.815 |
| ANE-port Flow CPU_AND_NE | **[-10.094, -0.825]** | **2.582** | 0 | **0.019** |

Hypothesis: precision loss in the AdaLN `(1+scale)*norm` or manual-SDPA
softmax path accumulates across 22 DiT blocks × 10 Euler steps ×
CFG batch=2, manifesting as progressive magnitude attenuation rather
than a NaN blowup. Stage 1's NaN probe was skipped because Stage 0
produced no NaN; a **range** probe (not a NaN probe) is the first
next step to pin the regressing block.

Debugging artifacts kept for follow-up:

```
src/
├── ane_attention.py       ANEAttention (manual SDPA, einsum-based)
├── ane_layernorm.py       ANEUnfusedLayerNorm (axis-1 BC1S) + patch helper
├── ane_layers.py          ANELinear (Conv2d 1x1), ANEGELU, ANEFeedForward, ANERotaryEmbedding
├── conv_pos_ane.py        Attempts (A, C, L) to break the 77-op conv_pos_embed CPU island
├── dit_ane.py             ANEDiTBlock, ANEAdaLayerNormZero(Final), ANEDiT top-level
├── flow_coreml_ane.py     FlowCoreML mirror with ANEDiT replacing flow.decoder.estimator
├── state_dict_port.py     Linear→Conv2d + LN affine reshape + rename map
└── nan_probe.py           Binary-search and shadow-fp32 block bisection helpers

build/
├── flow-ane-fp16-n250/    Original ANE-port build
└── flow-ane-n250/         Symlink shim for verify/test_coreml_e2e_fp16.py --flow-precision ane

compare-flow-ane.py        Per-block fp32 parity between host DiT and ANE port
```

Swift side, the `runHiFT` dtype branch (`fullMel.dataType` → fp16/fp32)
was kept — it's a no-op for the shipping fp32 Flow and makes the path
future-proof if a correct fp16 ANE Flow ever ships.

Full debugging trail is in [`TRIALS_AND_ERRORS.md`](./TRIALS_AND_ERRORS.md)
under **"Stage 4: Swift integration — ATTEMPTED, REVERTED"** (lines 832+).

## ANE profiling status

`tools/coreml-cli --fallback` currently fails with "unknown error" on
`MLComputePlan.loadContentsOfURL` for all four CosyVoice3 shipping
models on macOS 26.5 / M2 — likely because stateful-KV ops (LLM-Decode)
and the Flow DiT's fused ops exceed what `MLComputePlan`'s introspection
classifier handles. Sanity-check models (e.g. `silero-vad-unified-v6.0.0`)
profile fine on the same machine.

Workaround: `MLModel.compileModel(at:)` + `MLComputePlan.load(...)` in
Swift returns a structured fallback diagnostic. A one-shot Swift
harness (no existing tracker issue) would restore per-op ANE residency
visibility.

Measured Flow + HiFT + Prefill ANE residency via wall-clock A/B (model
on `.cpuAndNeuralEngine` vs `.cpuAndGPU`) is the current substitute.

## Verification

```bash
cd mobius/models/tts/cosyvoice3/coreml

# End-to-end Python parity (shipping config)
uv run python verify/test_coreml_e2e_fp16.py --flow-precision fp16

# End-to-end with the kept ANE Flow (reproduces the silence defect)
uv run python verify/test_coreml_e2e_fp16.py --flow-precision ane --compute-units CPU_AND_NE

# Per-block ANE-port parity (fp32 host vs ANE port on random input)
uv run python compare-flow-ane.py

# Swift end-to-end
cd ../../../../FluidAudio
swift build -c release
.build/release/fluidaudio tts --backend cosyvoice3-parity \
    --fixture .../shipping.safetensors \
    --models-dir mobius/.../coreml/build \
    --reference mobius/.../coreml/build/wavs/e2e_shipping.wav \
    --output /tmp/swift.wav --seed 42
```

## Ready for Swift / FluidAudio

Everything the Swift `CosyVoice3TtsManager` consumes is present:
- 4 CoreML mlpackages + precompiled mlmodelcs in `build/upload/cosyvoice3-coreml/`.
- Runtime embedding tables under `embeddings/`.
- 11 zero-shot voice bundles under `voices/`.
- Tokenizer assets + 281-entry special-tokens map under `tokenizer/`.

See `build/upload/cosyvoice3-coreml/README.md` + `manifest.json` for the
HuggingFace-facing layout.
