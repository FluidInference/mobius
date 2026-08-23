# WebGPU/WASM ports — fluidaudio-web runtime knowledge

Cross-model findings from porting FluidAudio's models to the browser in
[fluidaudio-web](https://github.com/FluidInference/fluidaudio-web) as
hand-written WebGPU/WASM forwards (no onnxruntime). Per-model port notes live
next to each model's trial README (see table); this note holds what transfers
across models. Extracted raw weights for every engine are hosted at
[`FluidInference/fluidaudio-web`](https://huggingface.co/FluidInference/fluidaudio-web)
(flat `.bin` + `manifest.json` of `name → {dims, offset, len}` in elements).

## Port index

| model | engine | in-browser result | per-model notes |
|---|---|---|---|
| Parakeet TDT v3 0.6B | `asr-parakeet` | WER 2.15% LibriSpeech (native parity), 282× RT | `models/stt/parakeet-tdt-v3-0.6b/coreml/README.md` |
| Parakeet Realtime EOU 120M | `eou-parakeet` | 297× RT, streaming == offline bit-exact | `models/stt/parakeet-realtime-eou-120m/coreml/README.md` |
| Nemotron 3.5 multilingual 0.6B | `asr-nemotron` | matches ORT streaming reference | `models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/README.md` |
| VoiceChat-11B STT (609M enc) | `asr-voicechat` | byte-identical to torch/CoreML oracle, 34.6× RT | `models/s2s/nemotron-voicechat-11b/README.md` |
| Whisper base | `asr-whisper` | matches transformers.js transcript | below (no mobius trial dir) |
| Sortformer 4-spk v2.1 | `diarization-sortformer` | fp32 parity 1.79e-7 | `models/speaker-diarization/sortformer-streaming/README.md` |
| Kokoro 82M en/zh | `tts-kokoro` | waveform corr ~0.97, ~10× RT | `models/tts/kokoro/coreml/README.md`, `models/tts/kokoro-v1.1-zh/coreml/README.md` |
| Silero VAD v5 | `vad-silero` | prob parity ~1e-6, weights bundled (1.2 MB) | `models/vad/silero-vad/coreml/README.md` |
| ACE-Step 1.5 Turbo (3.5B) | `musicgen-acestep` | ~1.9× RT, seed-deterministic | upstream port (Hamza Q.), vendored `packages/acestep` |

One shared hand-written FastConformer forward serves Parakeet / Nemotron /
EOU / Sortformer / VoiceChat — the config (d_model, layers, heads, d_ff, dw
kernel, subsampling, mel bins) is **inferred from the weight manifest**, so a
new FastConformer variant is a weight-extraction job, not a runtime job.

## Cross-cutting runtime findings

**ORT-web boundaries (why the raw runtime exists):**
- ORT's WebGPU EP has **no int8/int4 kernels**. Int-quantized encoders either
  silently fall back to WASM or (int4 MatMulNBits) run a buggy WebGPU kernel
  that returns garbage without falling back.
- Chrome caps ArrayBuffers at ~2 GB — multi-GB fp32 external-data ONNX files
  cannot load at all; self-contained fp16 exports are the fix.
- Single-threaded WASM (no cross-origin isolation on GitHub Pages) blocks the
  main thread for seconds on ~700 MB encoders — workerize, or go WebGPU.

**Quantization tolerance is a function of model size** (same RTN scheme):
0.6B encoders are int8-robust; the 120M EOU degrades under int8 and needs
fp16; the 9B VoiceChat LLM loses 25% top-1 to int4 RTN and is only lossless
at int8. Check encoder output std (healthy O(1) vs collapsed ~0.02) before
blaming anything downstream.

**Mel frontend matrix — mixups present exactly like decode bugs** (flat,
content-free encoder output → all-blank RNNT):
| model | normalization | log guard |
|---|---|---|
| Parakeet v3 / Sortformer | per-feature CMVN | 2^-24 |
| Nemotron / EOU | NA (none) | 1e-10 |
| VoiceChat | NA (none) | 2^-24 |

**Portable-WGSL performance ceiling (measured, M5 Pro, closed investigation):**
- Register-blocked GEMM plateaus at ~2.2 TFLOP/s ≈ 16% of fp32 peak; the rest
  needs simdgroup/cooperative-matrix units, which portable WGSL does not
  expose. This is also why hand-rolled WGSL only *ties* ORT-WebGPU on speed.
- f16 storage gives 1.5–2× only on large **square** GEMMs (≥2048³). Real
  speech-model GEMMs are thin (K=N≈1024, small M) and occupancy-bound —
  f16 measured 0× end-to-end on Kokoro AND Parakeet.
- **Denormals cost ~2×** on Metal-backed WebGPU: uninitialized GPU buffers
  full of garbage denormals doubled a full-forward replay (389→202 ms).
  Zero-init buffers; time submit→onSubmittedWorkDone, not the record loop.
- The one genuine raw-WGSL capability over ORT-web: **in-shader int4/int8
  dequant matmul** (ONNX MatMulNBits layout: packed nibbles [N, nblk, 16] +
  per-block scales/zero-points) — runs models ORT-WebGPU cannot run at all.
- Metal `tanh(x)` = exp/exp → NaN for large |x| (CPU saturates); clamp the
  gelu/tanh argument (±20). Large-K accumulators surface it; small tests hide it.

## Whisper base (no mobius trial dir)

Hand-written raw port (`asr-whisper`): WhisperMel 80-bin log-mel via direct
400-pt DFT in JS (1.4e-5 vs transformers.js), conv stem + 6 PRE-LN
transformer layers with **erf-GELU** (not tanh-approx), autoregressive greedy
decoder (causal self-attn + cross-attn) with the forced prefix
`<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`, suppress-token
list, and a GPT-2 byte-level BPE detokenizer. Single 30 s window; long-audio
chunking is a follow-up. Weights: `whisper/` (fp32 encoder + decoder).
