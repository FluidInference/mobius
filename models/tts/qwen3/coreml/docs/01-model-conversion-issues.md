# Model Conversion Issues (PyTorch to CoreML)

Issues encountered while converting Qwen3-TTS PyTorch models to CoreML format.

---

## 1. Code Predictor Autoregressive Loop Breaks Under Tracing

**Problem:** Embedding the code predictor's 15-step autoregressive loop into a single CoreML model caused CB1-15 tokens to get "stuck" with repetitive values after the first frame.

**Root Cause:** HuggingFace's SDPA (Scaled Dot-Product Attention) has an `is_causal` check that evaluates to a fixed boolean during `torch.jit.trace`. This "freezes" the causal attention mask at whatever shape was used during tracing. Subsequent iterations with different sequence lengths use the frozen (wrong) mask, producing broken attention patterns.

**Fix:** Split the code predictor into two separate CoreML models (matching the LM decoder pattern):
- **CP Prefill** -- processes 2 tokens (past_hidden + cb0_embed) with a fixed 2x2 causal mask
- **CP Decode** -- processes 1 token at a time with KV cache (no causal mask needed for single-token input)

Both use **manual attention** (explicit Q/K/V/RoPE) instead of HuggingFace's SDPA, avoiding the `is_causal` freeze.

---

## 2. Code Predictor Attribute Error on Conversion

**Problem:** `convert_code_predictor_kv.py` crashed with `AttributeError: 'Qwen3TTSAttention' object has no attribute 'num_heads'`.

**Root Cause:** The Qwen3TTS attention class stores head counts on the config object (`self.config.num_attention_heads`), not directly on the module (`self.num_heads`).

**Fix:** Updated all references:
- `attn0.num_heads` -> `attn0.config.num_attention_heads`
- `attn0.num_key_value_heads` -> `attn0.config.num_key_value_heads`

---

## 3. V10 Decode Model Missing Codebook Embedding Sum

**Problem:** The first V10 decode conversion only accepted a single CB0 token. The official pipeline sums all 16 codebook embeddings (CB0 + CB1-15) for the input embedding.

**Root Cause:** The initial conversion script was simplified to only use CB0, missing the critical feedback loop where code predictor outputs (CB1-15) are summed with CB0's embedding.

**Fix:** Rewrote V10 to accept all 16 codebook IDs (`cb0` + `cb1_15`) and sum their embeddings internally, matching the official `TracableDecodeV3` behavior.

---

## 4. V2 Decode Model Missing Code Predictor Integration

**Problem:** V7 prefill + V2 decode only generated 52 tokens (~4.3s) while the official model generated 124 tokens (~9.9s). Tokens diverged after position 2.

**Root Cause:** V2 decode only used the basic codec embedding (codebook 0). The official model sums ALL 16 codebook embeddings via the code predictor, which changes the hidden states and therefore subsequent token predictions. The missing code predictor feedback caused the model to "run out of things to say" prematurely.

**Fix:** Switched to V9 prefill + V3 decode (which includes code predictor integration), then ultimately to V9 prefill + V10 decode + separate CP prefill/decode models.

---

## 5. CPU-Only Prefill Extremely Slow

**Problem:** The prefill step took 10-15 seconds on CPU, making the pipeline 8.5x slower than real-time.

**Root Cause:** The prefill model was configured with `CPU_ONLY` compute units, not leveraging GPU/ANE.

**Fix:** Switched to hybrid compute -- GPU for prefill (100x faster, from 10,672ms to 125ms), CPU for decode loop. Overall RTF improved from 9.3x to 5.9x.

---

## 6. Audio Decoder Fixed Output Size

**Problem:** The CoreML audio decoder always outputs exactly 10 seconds of audio regardless of how many valid codec frames were provided.

**Root Cause:** The `qwen3_tts_decoder_10s.mlpackage` has fixed output dimensions corresponding to 125 frames at 24kHz. The model pads internally with `codecPadTokenId` for frames beyond the actual content, but the output buffer is always the full 10 seconds.

**Fix:** Trim the decoder output to `actualFrames * samplesPerFrame` (1920 samples per frame) based on where EOS was detected in the CB0 generation loop.
