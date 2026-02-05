# Token ID and Sampling Issues

Issues related to special token constants, sampling strategy, and token sequence construction.

---

## 1. Wrong `codecEosTokenId` and `codecBosTokenId` Constants

**Problem:** The model never stopped generating -- it always produced the maximum 125 frames with trailing silence.

**Root Cause:** The Swift constants had wrong EOS/BOS token IDs:
- `codecBosTokenId` was 2048 (correct: **2149**)
- `codecEosTokenId` was 2049 (correct: **2150**)

The logits shape is `[1, 3072]`. Indices 0-2047 are codec tokens, 2149 is BOS, 2150 is EOS. The suppression mask allowed index 2049 instead of 2150, so EOS was **always masked out**.

**How Found:** Inspected `config.json` from the HuggingFace model: `codec_eos_token_id: 2150`, `codec_bos_id: 2149`.

**Fix:** Updated `Qwen3TtsConstants.swift`:
```swift
public static let codecBosTokenId: Int = 2149  // was 2048
public static let codecEosTokenId: Int = 2150  // was 2049
```

---

## 2. Greedy CB0 Decoding Never Produces EOS

**Problem:** After fixing the EOS constant, the model still ran to 125 frames because greedy argmax never selects EOS.

**Root Cause:** With greedy decoding, a codec token (indices 0-2047) always has a higher logit than EOS (index 2150). Even when the EOS logit rises to rank 2-4 after speech ends (steps 60-68), argmax always picks the top-1 codec token. The official PyTorch uses `do_sample=True` for exactly this reason.

**Diagnostic:** Logged the EOS logit rank per step:
```
Step 58: EOS rank 847
Step 60: EOS rank 12
Step 62: EOS rank 4
Step 64: EOS rank 2   <-- close but still not rank 1
Step 66: EOS rank 3
```

**Fix:** Changed CB0 from greedy argmax to temperature+top_k sampling (`temperature=0.9, top_k=50`), matching the model's `generation_config.json`. With sampling, EOS is naturally selected when it enters the top-50 candidates.

---

## 3. Greedy CB0 Causes Token Collapse

**Problem:** With greedy decoding, CB0 tokens collapsed into repetitive patterns: `1995, 215, 212, 1181, 462x3, 619x2, 1657x11, 706`. Audio was silent.

**Root Cause:** The model's generation config specifies `do_sample=True` as the default for a reason. Greedy decoding amplifies small logit biases into repetitive loops. This was verified by running the same greedy decoding in both CoreML and PyTorch -- both produced identical stuck patterns, confirming it's an algorithmic issue, not a conversion bug.

**Fix:** Same as above -- temperature+top_k sampling for CB0.

---

## 4. Greedy Code Predictor (CB1-15) Produces Silent Audio

**Problem:** Using `do_sample=False` for the code predictor produced completely silent audio.

**Root Cause:** The code predictor **requires** temperature sampling. The V3 decode wrapper has an explicit comment in the PyTorch source: *"do_sample=True is required for the subtalker to work correctly! Using do_sample=False produces silent/broken audio."* Greedy CB1-15 tokens produce degenerate codebook patterns that decode to silence. The sampled CB1-15 also affect the embedding sum fed back to the LM, changing subsequent CB0 predictions.

**Fix:** Always use `temperature=0.9, top_k=50` for CB1-15 generation in both Python and Swift.

---

## 5. Wrong `role_ids` Prefix

**Problem:** Generated audio was coherent speech but said the wrong content ("But here's another one..." instead of "Hello world..."). First CB0 token was 1724 instead of expected 1995.

**Root Cause:** Test script used `role_ids = [151644, 8948, 198]` which tokenizes to `<|im_start|>system\n`. The correct prefix is `[151644, 77091, 198]` which tokenizes to `<|im_start|>assistant\n`. The system role puts the model into a different generation mode.

**Fix:** Changed to `[151644, 77091, 198]` (assistant role prefix).

---

## 6. Wrong Speaker Embedding File

**Problem:** First CB0 token was 1221 instead of expected 1995. The model triggered early EOS at token 29 (~2.5s audio instead of ~4.3s).

**Root Cause:** Two speaker embedding files existed: `speaker_embedding.npy` (incorrect) and `speaker_embedding_official.npy` (correct, extracted from the official model). The wrong file produced the second-best token (1221) instead of the correct top token (1995).

**Fix:** Switched all scripts and the Swift pipeline to use `speaker_embedding_official.npy`.

---

## 7. KV Cache Not Trimmed After Prefill

**Problem:** CB0 tokens diverged after step 0 despite correct role_ids and speaker embedding.

**Root Cause:** The prefill model pads input to 139 positions (128 text + 11 special), but only ~26 are valid. The V10 decode model has no causal masking (single query token), so it attends to ALL 139 KV entries -- including 113 garbage/padding entries that corrupt attention.

The prefill conversion has an explicit comment: *"IMPORTANT: Only positions 0 to actual_len-1 are valid! Caller must slice to kv_cache[:, :, :, :actual_len, :]"*

**Fix:** Added KV cache trimming after prefill in both Python (`kv_cache[:, :, :, :actual_len, :]`) and Swift (`trimKvCache` helper).

---

## 8. Hardcoded Token IDs Didn't Match Tokenizer Output

**Problem:** Token IDs in `TTSCommand.swift` didn't match actual Qwen3 tokenizer output for the test sentences.

**Root Cause:** Hardcoded from an older tokenization run.

**Fix:** Re-ran tokenization in Python and updated to correct IDs:
- English (14 tokens): `[9707, 1879, 11, 419, 374, 264, 1273, 315, 279, 1467, 311, 8806, 1849, 13]`
- Chinese (10 tokens): `[108386, 99489, 3837, 105464, 87335, 46670, 105761, 105743, 81705, 1773]`

---

## 9. Non-Deterministic Output Makes Debugging Hard

**Problem:** The official model produced different results on each run, making A/B comparisons unreliable.

**Root Cause:** Both CB0 and CB1-15 use `do_sample=True` by default. Each run samples different codebook tokens, which feed back through the autoregressive loop and compound into completely different token sequences.

**Mitigation:** Used fixed seeds (`seed=42`) for reproducible comparison runs. Accepted that even between two official PyTorch runs, spectral cosine similarity is only ~0.12 (very different waveforms for the same text).

---

## 10. Token Divergence Compounds Through Autoregressive Decoding

**Problem:** Even when the first 15 codebook rows matched perfectly, by token 47 the sequences had completely diverged.

**Root Cause:** Autoregressive models amplify small numerical differences exponentially. A minor float32 rounding difference at step 15 produces a different token, which changes all subsequent hidden states.

**Mitigation:** Not a bug to fix -- this is intrinsic to stochastic autoregressive models. Both CoreML and PyTorch produce correct, intelligible speech despite completely different token sequences.
