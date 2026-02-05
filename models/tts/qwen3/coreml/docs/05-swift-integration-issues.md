# Swift Integration Issues

Issues encountered while porting the CoreML Python pipeline to Swift.

---

## 1. mlpackage Cannot Be Loaded at Runtime

**Problem:** The initial Swift model store tried to load `.mlpackage` files directly and failed with "Compile the model with Xcode or MLModel.compileModel()".

**Root Cause:** `.mlpackage` is the uncompiled source format. CoreML requires `.mlmodelc` (compiled) at runtime. The model store had no compilation step.

**Fix:** Initially added a `loadModel()` helper that calls `MLModel.compileModel(at:)` to compile at first launch. Later switched to distributing pre-compiled `.mlmodelc` files from HuggingFace to avoid the ~20s compilation overhead on first run.

---

## 2. `compatPrediction` Missing `options` Parameter

**Problem:** Swift compilation failed because `model.compatPrediction(input)` was called without the required `options` parameter.

**Root Cause:** FluidAudio's CoreML prediction helper requires an explicit `MLPredictionOptions()` argument, but the initial code omitted it.

**Fix:** Changed all calls to `model.compatPrediction(input, options: MLPredictionOptions())`.

---

## 3. No `extractInt32Array` Helper

**Problem:** The decode model outputs codebook IDs as Int32 MLMultiArrays, but only `extractFloatArray` existed in the synthesizer.

**Root Cause:** Previous TTS backends only needed float extraction from model outputs. The Qwen3-TTS pipeline outputs integer token IDs from the LM.

**Fix:** Added an `extractInt32Array` helper for reading codebook token IDs from MLMultiArray outputs.

---

## 4. MLMultiArray Does Not Support In-Place Slicing

**Problem:** Trimming the KV cache after prefill required creating a new MLMultiArray with a different shape and manually copying data -- there's no NumPy-style `array[:, :, :, :n, :]` slicing.

**Root Cause:** MLMultiArray is a flat memory buffer with shape metadata. It has no built-in slicing or view operations.

**Fix:** Implemented `trimKvCache()` that allocates a new MLMultiArray with the reduced sequence dimension `[56, 1, 8, actualLen, 128]` and copies valid entries with nested loops over all 5 dimensions using raw pointer access (`dataPointer.bindMemory`).

---

## 5. Fixed-Size Model Inputs Require Padding

**Problem:** CoreML models have fixed input shapes (prefill: `[1, 128]` text tokens, audio decoder: `[1, 16, 125]` codebooks). Variable-length inputs must be padded.

**Root Cause:** `torch.jit.trace` captures fixed tensor shapes at conversion time. The conversion scripts hardcoded maximum lengths (128 text tokens, 125 codec frames).

**Fix:** The Swift synthesizer pads text tokens to 128 (zero-padded), pads the codebook tensor to `[1, 16, 125]` with `codecPadTokenId` for unused frames, and passes `actual_len` alongside so the pipeline knows where valid data ends.

---

## 6. Temperature + Top-K Sampling Had No Swift Implementation

**Problem:** No sampling utility existed in the Swift codebase. Previous TTS backends used different decoding strategies.

**Root Cause:** Qwen3-TTS requires temperature+top_k sampling for both CB0 (to allow EOS generation) and CB1-15 (greedy produces silent audio). This is the first backend in FluidAudio requiring stochastic decoding.

**Fix:** Implemented two sampling functions in `Qwen3TtsSynthesizer.swift`:
- `sampleToken()` -- for CB0 with a suppression mask (only allows codec tokens 0-2047 + EOS at 2150)
- `sampleFromSlice()` -- for CB1-15 from the code predictor's `[15, 1, 2048]` logits tensor

Both apply: temperature scaling -> top-k filtering -> softmax (with numerical stability via max subtraction) -> multinomial sampling via cumulative probability walk.

---

## 7. OSLog Output Not Visible in CLI

**Problem:** Logger output from the Swift pipeline (using `OSLog`) was invisible when running the CLI tool, making it impossible to see debug info like frame count or EOS detection.

**Root Cause:** OSLog writes to the system log (viewable in Console.app), not stdout/stderr. The CLI captures stdout/stderr only.

**Workaround:** Used temporary `print()` statements for debugging during development, then removed them. Production logging stays in OSLog.

---

## 8. Codec Head Logits Shape is 3072, Not 152064

**Problem:** Initial confusion about whether the model outputs logits over the full text vocabulary (152064) or a reduced codec vocabulary.

**Root Cause:** In TTS mode, the model uses a separate `codec_head` that projects to 3072 logits (indices 0-2047 = codec tokens, 2149 = BOS, 2150 = EOS), not the full text vocabulary `lm_head`.

**Fix:** Updated the suppression mask and sampling code to work with the 3072-length logit space. Indices 2048-2148 and 2151-3071 are unused special tokens that get masked to `-inf`.

---

## 9. Code Predictor `head_dim` Was 128, Not 64

**Problem:** The code predictor's head dimension was initially assumed to be 64, which would produce incorrect KV cache shapes.

**Root Cause:** The code predictor config has `hidden_size=1024` with 8 KV heads and `head_dim=128`. This differs from the more common 64-dim heads seen in smaller transformer blocks.

**Fix:** Verified by inspecting `cp.config` attributes. Updated KV cache shape in both conversion scripts and Swift to `[10, 1, 8, kv_len, 128]` (10 = 5 layers x 2 for key+value).

---

## 10. Numpy (.npy) File Loading in Swift

**Problem:** The pipeline requires loading several `.npy` files (speaker embedding, code predictor embeddings, special token embeddings) which Swift has no native support for.

**Root Cause:** NumPy's `.npy` format is a binary format with a magic number, version byte, header length, and raw float32 data. No Swift library was available in the project.

**Fix:** Implemented a custom NPY parser in `Qwen3TtsModelStore.swift` that:
1. Validates the 6-byte magic (`\x93NUMPY`)
2. Reads version-dependent header length (2 or 4 bytes)
3. Skips the header string
4. Reads raw float32 data
5. Reconstructs 1D/3D arrays in row-major order

Handles `cp_embeddings.npy` (120MB, shape `[15, 2048, 1024]`), `speaker_embedding_official.npy` (4KB), and the three special token embeddings.
