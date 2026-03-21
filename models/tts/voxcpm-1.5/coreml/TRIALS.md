# VoxCPM 1.5 CoreML Conversion — Trial Log

Chronological record of all attempts, failures, and fixes to port VoxCPM 1.5 from PyTorch to CoreML.

---

## Phase 1: Simple Components

### Trial 1 — AudioVAE Encoder
**Approach:** Wrap `encoder.block` + `encoder.fc_mu` in a `TraceableEncoder`. Remove `weight_norm` recursively and replace `Snake1d` activations (which use `@torch.jit.script` with `shape[0]` indexing) with a simple module.
**Result:** PASS. Correlation 0.999989. Fixed input shape of 5s (220500 samples).
**Note:** The original Snake1d uses `x.reshape(shape[0], shape[1], -1)` which triggers CoreML's `__getitem__` error on shape tuples. Since input is already 3D, the reshape is a no-op.

### Trial 2 — AudioVAE Decoder
**Approach:** Same Snake activation patching. Wrap `decoder.model` in `TraceableDecoder`.
**Result:** PASS. Correlation 0.999921. Used `RangeDim` for flexible latent length (4–2000 frames).

### Trial 3 — feat_encoder (VoxCPMLocEnc)
**Approach:** 8-layer non-causal transformer with CLS token extraction. Patch GQA attention to expand KV heads manually (16 query heads, 2 KV heads). Use `is_causal=True` in `F.scaled_dot_product_attention` for the full-sequence forward.
**Result:** PASS. Correlation 0.999997. Fixed shape `[1, 1, 4, 64]`.

### Trial 4 — LocDiT Estimator
**Approach:** Wrap the flow matching velocity estimator. Batch=2 for CFG (conditioned + unconditioned). Same GQA patching.
**Result:** PASS. Correlation 0.999782. Initially used cond_len=25, later changed to cond_len=4 (single patch, matching actual VoxCPM usage).

---

## Phase 2: LM Step Conversion

### Trial 5 — Combined LM step (base_lm + residual_lm)
**Approach:** Single `TraceableLMStep` wrapping both the 24-layer base LM and 8-layer residual LM in one model, with 64 KV cache tensors (32 base + 16 residual) as I/O.
**Result:** FAIL. Validation showed lm_hidden correlation = 0.228 (terrible), res_hidden correlation = 0.992 (acceptable).

### Trial 6 — Debug: Trace parity
**Approach:** Test each sub-component's trace parity independently:
- FSQ layer trace parity: 0.0 (perfect)
- RoPE positional embeddings: correctly vary with position after tracing
- Causal mask construction: correctly varies with position after tracing
- Scatter-based cache update: correctly uses dynamic position after tracing
- Full `TraceableLMStep` trace parity: **1.0 correlation at all positions**

**Result:** Trace is perfect — the issue is in CoreML conversion, not tracing.

### Trial 7 — Debug: Layer-by-layer CoreML
**Approach:** Convert subsets of layers to CoreML and check correlation:
- 1 layer: 1.000000
- 2 layers: 1.000000
- 4 layers: 0.999999
- 8 layers: 0.999993
- 12 layers: 0.999990
- 16 layers: 0.999974
- 20 layers: 0.999953
- 24 layers: 0.999998

**Result:** All subsets pass individually. The CoreML conversion is correct.

### Trial 8 — Root cause found: Validation methodology
**Bug:** The `convert_lm_step.py` script loaded VoxCPM, traced the model, converted to CoreML, then compared CoreML outputs against PyTorch outputs computed from the **same model instance**. However, the model's `from_pretrained` + warmup changes internal state. When the validation loaded a fresh model for comparison, the random test inputs produced **different PyTorch reference outputs** because the model parameters shifted subtly during warmup.

**Fix:** Compare CoreML against the traced model's outputs directly (before any further model loads). The CoreML model was actually correct all along — the reference was wrong.

### Trial 9 — Split into base_lm_step + residual_lm_step
**Approach:** Split the monolithic LM step into two separate CoreML models:
1. `base_lm_step`: 24-layer base LM + norm + FSQ + stop head. 50 I/O tensors (embed, position, 24×k, 24×v → lm_hidden, lm_hidden_fsq, stop_logit, 24×out_k, 24×out_v)
2. `residual_lm_step`: 8-layer residual LM + norm. 18 I/O tensors.

**Result:** PASS.
- base_lm_step: lm_hidden corr=0.999999, fsq corr=0.999817, stop corr=1.000000
- residual_lm_step: res_hidden corr=0.999983

---

## Phase 3: Constants & Pipeline

### Trial 10 — Export constants
**Approach:** Extract text embedding table, projection weights, and config from the model.
**Issues:**
- `config.latent_dim` → `AttributeError`, should be `config.feat_dim`
- `model.tokenizer` → not loaded with `load_denoiser=False`, added conditional check

**Result:** PASS. Total constants: 298.9 MB (embed_tokens: 286.9 MB for [73448, 1024]).

### Trial 11 — AudioVAE encoder flexible shapes
**Approach:** Use `ct.RangeDim` for variable-length audio input.
**Result:** FAIL. The original AudioVAE has data-dependent assertions (`assert pad == 0`) that get baked into the trace. CoreML produces garbage for non-trace-time input sizes.

### Trial 12 — AudioVAE encoder EnumeratedShapes
**Approach:** Use `ct.EnumeratedShapes` with [1, 2, 3, 5, 7, 10] second inputs.
**Result:** FAIL. CoreML model compilation hangs (>2 minutes) when loading the EnumeratedShapes model. Reverted to fixed 5s input.

### Trial 13 — LocDiT cond_len fix
**Approach:** Changed cond_len from 25 (arbitrary) to 4 (patch_size, matching actual VoxCPM usage where `prefix_feat_cond = feat[:, -1, ...]`).
**Result:** PASS. Correlation 0.999922.

### Trial 14 — End-to-end generation pipeline
**Approach:** `generate_coreml.py` — zero-PyTorch pipeline using only CoreML models + numpy.

**Pipeline fixes applied:**
1. Added `AUDIO_START_TOKEN = 101` after text tokens (was missing)
2. Fixed residual LM input: `fsq(lm_hidden) * feat_mask + lm_hidden * text_mask + feat_mask * feat_embed` (per VoxCPM source)
3. Fixed duplicate stop check
4. Fixed prefix_cond shape: `[1, 64, 4]` (single patch), not a longer window
5. Factored out `run_base_lm_step()` and `run_residual_lm_step()` helpers

**Result:** Runs end-to-end, generates WAV audio, but output is unintelligible (empty transcription, confidence 0.100).

### Trial 15 — Intelligibility fix: dit_hidden + Euler direction
**Root cause analysis:** Two bugs in the generation pipeline:

1. **dit_hidden used raw lm_hidden instead of FSQ'd version.** VoxCPM applies `lm_hidden = fsq_layer(lm_hidden)` before `lm_to_dit_proj(lm_hidden)`. Our pipeline passed the pre-FSQ `lm_hidden` to the projection. The FSQ quantization is critical for the LocDiT to produce correct velocity fields.

2. **Euler solver ran forward (0→1) instead of backward (1→0).** VoxCPM's `solve_euler` uses `t_span = linspace(1, 1e-3, n_timesteps+1)` and `x = x - dt * velocity`. Our pipeline used `t = step/n_steps` (forward) with `x = x + v * dt` (additive). Both the time schedule and step direction were wrong.

**Fixes applied:**
1. Changed `linear(lm_hidden, ...)` to `linear(lm_hidden_fsq, ...)` for dit_hidden computation
2. Changed Euler solver: `t_span = linspace(1.0, 1e-3, n+1)`, `x = x - dt * v`

**Result:** PASS. Intelligible speech output.
- Prefill: ~27 tok/s (44 tokens in 1.6s)
- Generation: ~4.5 steps/s (30 steps in 6.7s)
- Transcription (FluidAudio ASR, confidence 0.957): "This is a test of the voice cloning system. And"
- Input text: "Hello, this is a test of the voice cloning system."
- "Hello" missing from output (first patch consumed as prompt context)

### Trial 16 — Transcription pipeline integration
**Approach:** Added `--transcribe` flag to `generate_coreml.py` that resamples output to 16kHz and runs FluidAudio CLI transcription.

**Result:** PASS. End-to-end TTS→ASR validation works.

### Trial 17 — Chinese text generation
**Root cause:** Standard `AutoTokenizer.encode()` produces 9 tokens for Chinese text like "你好，这是一个语音克隆系统的测试。", but VoxCPM uses `mask_multichar_chinese_tokens` (from `voxcpm.model.utils`) which splits multi-character Chinese tokens into individual characters, producing 18 tokens. Without this splitting, the model receives the wrong token sequence and generates garbage.

**Fix:** Wrapped the tokenizer with `mask_multichar_chinese_tokens` in `generate_coreml.py`.

**Results:**
- **Unprompted** (no prompt audio): Whisper transcription: "你好这是一个语音克隆系统的测试" — matches input text
- **Prompted** (with `--prompt` + `--prompt-text`): "你好这是一个语音克隆系统的测试" — matches input text
- **Prompted without prompt-text**: Unintelligible — VoxCPM requires `prompt_text` when using prompt audio for text/audio alignment in prefill

**Result:** PASS. Chinese generation works for both conditioned and unconditioned modes.

## Phase 4: Float16 Quantization

### Trial 18 — Float16 conversion
**Approach:** Added `compute_precision=ct.precision.FLOAT16` and `compute_units=ct.ComputeUnit.CPU_AND_GPU` to all `ct.convert()` calls. This stores weights as Float16 and runs inference in half precision.

**Per-component validation (Float16 vs Float32 PyTorch):**

| Model | Correlation | Max diff | Notes |
|-------|------------|----------|-------|
| audio_vae_encoder | — | 6.63e-01 | Acceptable on [-60, +61] range |
| audio_vae_decoder | 0.999999 | 9.31e-04 | |
| feat_encoder | 1.000000 | 1.15e-03 | |
| base_lm_step (lm_hidden) | 0.999998 | 3.16e-02 | |
| base_lm_step (fsq) | 0.999677 | 1.55e-02 | |
| base_lm_step (stop) | 1.000000 | 1.06e-02 | |
| residual_lm_step | 1.000000 | 5.04e-03 | |
| locdit_step | 0.999999 | 5.42e-03 | |

**End-to-end verification:**
- English: "Hello, this is a test of the voice cloning system." → ASR: "Hello, this is a test of the voice cloning system." (exact match)
- Chinese: "你好，这是一个语音克隆系统的测试。" → Whisper: "你好这是一个语音克隆系统的测试" (match)

**Result:** PASS. Float16 produces identical intelligible output to Float32. No quality degradation detected.

---

## Phase 5: INT8 Quantization & Stop Head Fix

### Trial 19 — Batch prefill models (INT8) + step models (FP16)
**Approach:** Created separate `base_lm_prefill` and `residual_lm_prefill` CoreML models that process all tokens at once (matching PyTorch's batch forward with causal mask). These were INT8-quantized via `linear_quantize_weights`. The idea was to use batch prefill for speed, then switch to the existing FP16 step models for autoregressive generation.

**Result:** FAIL. The step model's stop head fires "stop" from step 1 after batch prefill.
- Stop logit at prefill's last position: `[9.89, -9.88]` (correct: don't stop)
- Stop logit at step 1 from step model: `[-7.13, 7.13]` (STOP immediately)

**Root cause:** INT8-quantized prefill models produce KV caches that are numerically incompatible with the FP16 step model. The quantization noise in the cached keys/values propagates through 24 transformer layers, causing the step model's hidden states to diverge enough that the stop head classifier flips.

### Trial 20 — Revert to sequential prefill
**Approach:** Removed all batch prefill infrastructure. Reverted to sequential prefill (processing tokens one at a time through `base_lm_step`), matching exactly what `generate_coreml.py` does. This ensures the same model produces both the KV caches and the generation outputs — no quantization mismatch.

**Result:** Stop head still noisy. Even with sequential prefill through the same FP16 model, the stop head fires "stop" from step 1 in unconditioned mode (no prompt audio). Running with `--no-stop --max-len 30` revealed the pattern:
- Steps 1-13: mostly predicts STOP (noisy)
- Steps 14-23: transitions to "continue" (meaningful speech being generated)
- Steps 24+: oscillates between stop/continue

The FP16 stop head is inherently noisy in unconditioned mode. This wasn't visible in the Python pipeline because `generate_coreml.py` used `min_len=5` with a different token count.

### Trial 21 — Text-length-proportional minimum steps
**Approach:** Instead of a fixed `minLen`, use `minLen = max(15, textLen * 2)`. Each text token produces roughly 2 audio patches on average, so this ensures the model generates at least enough audio to cover the input text before the stop head gets a vote.

**Result:** PASS. The stop head fires at appropriate points after the minimum:

| Input | Text tokens | minLen | Steps | Audio | ASR confidence |
|-------|------------|--------|-------|-------|----------------|
| "Hello world." | 4 | 15 | 16 | 2.56s | — |
| "Hello, this is a test of the voice synthesis system." | 13 | 26 | 27 | 4.32s | 0.967 |
| "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet." | 20 | 40 | 41 | 6.56s | 0.997 |

### Trial 22 — Full INT8 quantization of step models
**Approach:** Applied `linear_quantize_weights` (INT8 linear-symmetric) to all step models: `base_lm_step`, `residual_lm_step`, `locdit_step`, `feat_encoder`. Since sequential prefill uses the same model for both prefill and generation, there's no quantization mismatch — INT8 KV caches feed back into the same INT8 model.

**Result:** PASS. Audio quality indistinguishable from FP16:

| Input | FP16 duration | INT8 duration |
|-------|--------------|---------------|
| "Hello world." | 2.56s | 2.88s |
| "Hello, this is a test..." | 4.32s | 4.32s |
| "The quick brown fox..." | 6.56s | 6.72s |

Stop head fires at similar points. Compiled `.mlmodelc` sizes unchanged (CoreML unpacks weights at compile time), but INT8 weights are stored more compactly in the `.mlpackage`.

### Trial 23 — Mixed precision: INT8 bulk + FP16 stop head
**Approach:** Attempted to keep the stop head layers in FP16 while INT8-quantizing the transformer bulk. Used `op_name_configs` to exclude the last transformer layer (ops 161-167) and head projections (ops 168-171) from quantization.

**Result:** FAIL. MiniCPM architecture shares bias parameters across transformer layers — `linear_0_bias_0_to_fp16` is reused by both layer 0 (INT8) and layer 23 (FP16). CoreML's quantizer raises `ValueError: compression config conflict detected` when two ops sharing a const have different quantization configs.

**Workaround:** Narrowed FP16 exceptions to only the 4 head projections (ops 168-171: lm_hidden, fsq, stop_proj, stop_head) which don't share params with transformer layers. This compiled successfully.

**Outcome:** Mixed precision works but unnecessary — full INT8 (Trial 22) produces equivalent quality since the stop head's noise is handled by the text-length-proportional minLen heuristic. Abandoned in favor of simpler full INT8.

---

## Summary of Key Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Snake1d `shape[0]` indexing | CoreML `__getitem__` error | Replace with simple `nn.Module` |
| SDPA `enable_gqa=True` on CPU | `IndexError: Dimension out of range` | Manual `repeat_interleave` for KV heads |
| In-place KV cache update | CoreML conversion failure | Functional `scatter` |
| LM validation against different model load | False 0.228 correlation | Validate against traced model outputs |
| Missing audio_start_token | Wrong embedding sequence | Append token 101 after text |
| LocDiT cond_len=25 | Shape mismatch in pipeline | Use cond_len=4 (patch_size) |
| AudioVAE encoder data-dependent assert | Flexible shapes produce garbage | Use fixed 5s input |
| `config.latent_dim` | AttributeError | Use `config.feat_dim` |
| dit_hidden used raw lm_hidden | Unintelligible output | Use `lm_hidden_fsq` (FSQ'd) |
| Euler solver forward (0→1) | Wrong diffusion trajectory | Backward (1→0): `x = x - dt * v` |
| Chinese tokenizer missing char splitting | Garbage Chinese output | `mask_multichar_chinese_tokens` wrapper |
| INT8 prefill + FP16 step KV cache mismatch | Stop head fires from step 1 | Use same model for prefill and generation |
| FP16 stop head noisy in unconditioned mode | Audio cut short (steps 1-13 predict STOP) | Text-proportional minLen: `max(15, textLen * 2)` |
| MiniCPM shared biases block mixed quantization | `ValueError: compression config conflict` | Either full INT8 or FP16-only head projections |
