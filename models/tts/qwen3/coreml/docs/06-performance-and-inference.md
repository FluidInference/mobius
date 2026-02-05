# Performance and CoreML Inference Issues

Issues related to CoreML runtime behavior, compute unit selection, and pipeline performance.

---

## 1. GPU vs CPU Performance Inversion for Decode

**Problem:** GPU was 85x faster than CPU for prefill (125ms vs 10,672ms), but actually *slower* for the single-token decode step (~3.2 tok/s on GPU vs ~5 tok/s on CPU).

**Root Cause:** The decode step processes a single token at a time. GPU parallelism provides no benefit for such small operations, and the dispatch overhead of shipping tiny tensors to the GPU outweighs any compute savings.

**Fix:** Adopted a hybrid approach: GPU for prefill (large batch of ~139 tokens), CPU for the autoregressive decode loop (1 token at a time). Final pipeline uses `.cpuAndGPU` compute units and lets CoreML route internally. Overall RTF improved from 9.3x to 5.9x.

---

## 2. 16 CoreML Predictions Per Codec Frame

**Problem:** Each decode frame requires 16 separate CoreML prediction calls: 1 LM decode + 1 CP prefill + 14 CP decode steps. Over ~50-60 frames of actual speech, that's ~800-960 predictions total.

**Root Cause:** The code predictor's 15-step autoregressive loop cannot be traced as a single CoreML model (SDPA `is_causal` freezes under `torch.jit.trace`), so each step must be a separate model call driven from Swift.

**Impact:** The prediction call overhead dominates inference time. The final pipeline runs at ~6x RTF (6 seconds of compute per 1 second of audio), which is slower than real-time but functional for batch synthesis.

**Potential Optimization:** Fusing the CP prefill + 14 decode steps into a single CoreML model with manual attention (avoiding SDPA) could reduce per-frame predictions from 16 to 2.

---

## 3. Decode Model KV Cache Size Incompatibility Across Versions

**Problem:** Early decode models (V2 with 10-position KV cache, V4 with 20-position KV cache) couldn't accept the V9 prefill output which produces up to 139-position KV cache.

**Root Cause:** Each decode model conversion hardcoded its maximum KV cache length. V9 prefill outputs `textLen + 11` positions (up to 139), far exceeding the earlier models' limits.

**Fix:** V10 decode uses `ct.RangeDim` for variable-length KV cache support, accepting any cache size from the prefill output. The KV cache grows by 1 position per decode step.

---

## 4. 120MB Code Predictor Embedding Table

**Problem:** The code predictor requires a 120MB embedding lookup table (`cp_embeddings.npy`, shape `[15, 2048, 1024]`) loaded entirely into memory.

**Root Cause:** Each of the 15 codebook levels has its own 2048x1024 embedding table. These aren't part of any CoreML model -- they're used in Swift as a lookup table to produce the input embedding for each CP decode step.

**Impact:** Adds ~120MB to the memory footprint on top of the CoreML models. The lookup is fast (single array index per step), but the initial load and memory residency is significant.

**Mitigation:** The file is downloaded once and cached at `~/.cache/fluidaudio/Models/qwen3-tts/`. Memory-mapping could reduce RSS but isn't implemented.

---

## 5. ANE Compilation Adds ~20s Cold Start

**Problem:** First model load after a clean state takes ~20 seconds as `anecompilerservice` compiles the CoreML models for the Apple Neural Engine.

**Root Cause:** CoreML compiles `.mlmodelc` to ANE-optimized form on first load, caching at `~/Library/Caches/python/com.apple.e5rt.e5bundlecache/`. This cache can grow to 27GB+ and is invalidated by macOS updates or sleep/wake cycles.

**Mitigation:** Using `.cpuAndGPU` compute units avoids ANE compilation entirely, trading peak throughput for consistent ~1s load times. The distributed `.mlmodelc` files are already compiled from `.mlpackage`, but ANE optimization is a separate step done by the OS at runtime.

---

## 6. Total Model Payload ~5.9GB

**Problem:** The full Qwen3-TTS model set is ~5.9GB, significantly larger than other FluidAudio backends.

**Breakdown:**
| Model | Size |
|-------|------|
| LM Prefill V9 (mlmodelc) | ~1.8GB |
| LM Decode V10 (mlmodelc) | ~1.8GB |
| CP Prefill (mlmodelc) | ~150MB |
| CP Decode (mlmodelc) | ~150MB |
| Audio Decoder 10s (mlmodelc) | ~1.7GB |
| cp_embeddings.npy | ~120MB |
| speaker + special token .npy | ~12KB |

**Mitigation:** Models are downloaded once from HuggingFace and cached locally. The `.mlmodelc` format is already compiled, avoiding the additional disk space and time cost of runtime compilation from `.mlpackage`.

---

## 7. RTF Breakdown by Pipeline Stage

Measured on Apple Silicon (M-series):

| Stage | Time | Notes |
|-------|------|-------|
| LM Prefill | ~125ms (GPU) | Single call, ~139 tokens |
| LM Decode loop | ~60-80s | ~50-60 steps x ~1.2s/step |
| CP per frame | ~1.0s | 1 prefill + 14 decode calls |
| Audio Decoder | ~200ms | Single call, fixed 125 frames |
| Silence trim | ~1ms | Post-processing |
| **Total** | ~80-100s | For ~10s of audio (~8-10x RTF) |

The LM decode + CP loop dominates at ~95% of total time. The bottleneck is per-prediction CoreML overhead, not the actual neural network compute.
