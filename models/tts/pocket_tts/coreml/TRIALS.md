# PocketTTS CoreML Conversion — Trial Log

Chronological record of all attempts, failures, and fixes to port PocketTTS from PyTorch to pure CoreML.

---

## Phase 1: Monolithic Conversion Attempts

### Trial 1 — Full model trace (`convert_pocket_tts.py`)
**Approach:** Trace the entire PocketTTS model as one CoreML model.
**Result:** Failed. The model has dynamic control flow (autoregressive loop, EOS checking, variable-length generation) that `torch.jit.trace` cannot capture. CoreML requires static compute graphs.

### Trial 2 — ONNX intermediate (`convert_via_onnx.py`)
**Approach:** Export to ONNX first, then convert ONNX to CoreML.
**Result:** Failed. Same dynamic control flow issues. ONNX export also choked on the streaming KV cache scatter operations.

### Trial 3 — Split into submodels (`convert_pocket_tts_v2.py`, `v3`, `v4`)
**Approach:** Split the pipeline into separate traceable modules:
- Text encoder
- Flow decoder
- Mimi decoder
- EOS detector

**Result:** Partial success. Individual components converted but the orchestration between them still required PyTorch for KV cache management, text preparation, and conditioning.

---

## Phase 2: Step-Based Architecture

### Trial 4 — Traceable FlowLM backbone (`traceable_flowlm.py`)
**Approach:** Create a traceable wrapper for the full transformer backbone that takes `text_embeddings` as a fixed-size input and manages KV cache internally.
**Result:** Converted successfully to `flowlm_backbone_v2.mlpackage`, but required fixed `text_embeddings` shape `[1, 100, 1024]`. This forced zero-padding for shorter inputs, which corrupted the KV cache (see Trial 7).

### Trial 5 — Flexible text_embeddings shape
**Approach:** Use `ct.RangeDim` to allow variable-length `text_embeddings` input `[1, (1-200), 1024]`.
**Result:** Failed. CoreML's `scatter_along_axis` op threw `AssertionError` with dynamic shapes. The scatter operation in the streaming KV cache requires static dimensions.

### Trial 6 — Fixed T_text=150 (`convert_flowlm.py`)
**Approach:** Fix `text_embeddings` to `[1, 150, 1024]` — large enough for any voice+text combination.
**Result:** Converted to `flowlm_backbone_v3.mlpackage`. But still required zero-padding shorter conditioning sequences.

---

## Phase 3: Flow Decoder Fix

### Trial 7 — Flow decoder time values bug
**Bug:** The `TraceableFlowDecoder` was passing wrong time values to `SimpleMLPAdaLN`:
- `s` (start time) was hardcoded to `0` instead of `i/N`
- `t` (end time) received `lsd_step * dt` (the start value) instead of `(lsd_step + 1) * dt`

**Symptom:** CoreML generation produced gibberish audio despite transformer outputs matching PyTorch exactly. The 8-step LSD flow decoding was computing wrong velocity fields at every step.

**Root cause:** `SimpleMLPAdaLN.forward(c, s, t, x)` takes TWO time conditions and averages their embeddings. With `s=0` always, the flow trajectory was wrong.

**Fix:** Updated `traceable_flow_decoder.py` to accept explicit `s` and `t` inputs:
```python
# Before (wrong):
s = torch.zeros_like(t)  # always 0
velocity = self.flow_net(transformer_out, s, t, latent)

# After (correct):
def forward(self, transformer_out, latent, s, t):
    velocity = self.flow_net(transformer_out, s, t, latent)
```

And the generation loop:
```python
# Before (wrong):
t_np = np.array([[lsd_step * dt]])  # this is s, not t!

# After (correct):
s_np = np.array([[lsd_step * dt]])
t_np = np.array([[(lsd_step + 1) * dt]])
```

**Result:** CoreML generation now produced correct speech. Whisper transcribed output as "Hello, this is Pure CoreML Text to Speech Generation." — matching PyTorch reference.

---

## Phase 4: Eliminating PyTorch from Setup

### Trial 8 — Zero-padding conditioning corruption
**Approach:** Use `flowlm_backbone_v3.mlpackage` (T_text=150) with zero-padded conditioning. Pad the 141-token conditioning sequence (125 voice + 16 text) with 9 zeros to fill the fixed 150-slot input.

**Result:** Failed. Zero-padded tokens are NOT ignored — they pass through LayerNorm (which has bias terms) and FFN layers, producing non-zero KV cache entries. The model wrote KV entries at positions 0-149 instead of 0-140, advancing the position counter to 150 instead of 141. Generation started at the wrong position, producing garbage.

**Key insight:** You cannot zero-pad conditioning tokens. Each padded token creates a real (non-zero) KV cache entry because LayerNorm bias + FFN bias transform zeros into non-zero activations.

### Trial 9 — Conditioning step model (`traceable_cond_step.py`, v1)
**Approach:** Create a separate CoreML model that processes ONE conditioning token at a time. Feed all 141 tokens sequentially, no padding needed.

**Result:** Positions now correct (141), EOS triggered at step 21. But audio quality was wrong — Whisper transcribed as "Third is... Yes." instead of expected text.

**Root cause:** The attention implementation in `traceable_cond_step.py` differed from the verified `traceable_flowlm_step.py`:

| Aspect | Step model (correct) | Cond step v1 (wrong) |
|--------|---------------------|----------------------|
| QKV split | `.reshape(B, T, 3, H, D)` slicing | `.chunk(3, dim=-1)` then `.view()` |
| NaN handling | `torch.where(isnan, zeros, keys)` | None |
| Masking | Boolean mask + `F.scaled_dot_product_attention` | Float mask + manual softmax + `-1e9` |
| RoPE | `torch.exp(ds * (-math.log(10000) * 2/D))` | `1.0 / (10000 ** (2*indices/D))` |

While mathematically equivalent, these code paths trace to different CoreML ops, and the missing NaN→0 replacement caused NaN propagation through attention.

### Trial 10 — Conditioning step model (v2, fixed attention)
**Approach:** Copy the exact `_apply_rope_tensor` and `_streaming_attention` methods from the verified `traceable_flowlm_step.py` into `traceable_cond_step.py`.

**Result:** Reconverted `cond_step.mlpackage`. Still produced "Third is... Yes."

**Root cause discovered via verification script:** The **conditioning order** was wrong.

### Trial 11 — Conditioning order fix (voice-first)
**Bug:** `generate_coreml_v4.py` concatenated `[text_emb, voice_emb]` (text first), but the original PocketTTS model processes **voice first** then text:
1. `get_state_for_audio_prompt("alba")` → fills KV cache with 125 voice tokens (positions 0-124)
2. `_run_flow_lm_and_increment_step(text_tokens)` → adds 16 text tokens (positions 125-140)

**Verification:** Wrote a comparison script that ran both orders through the PyTorch cond_step model and compared KV caches against the original model:

| Order | Max diff | Cosine sim |
|-------|----------|-----------|
| Voice-first | 0.000007 | 1.000 |
| Text-first | 4.33–9.11 | 0.60–0.90 |

**Fix:**
```python
# Before (wrong):
combined = np.concatenate([text_emb, voice_emb], axis=1)

# After (correct):
combined = np.concatenate([voice_emb, text_emb], axis=1)
```

**Result:** Correct speech output. Whisper: "Hello, this is Pure CoreML Text to Speech Generation." Duration: 3.52s, 44 frames, EOS at step 41. Zero PyTorch dependency confirmed.

---

## Phase 5: Long-form audio bug

### Trial 12 — Mimi decoder transformer KV cache overflow (no modulo wrap)

**Bug:** `traceable_mimi_decoder._functional_attention_forward` writes streaming
KV cache via plain `cache[0].scatter(1, write_indexes, k)` with **no modulo
wrap**. The wrapper allocates the cache via
`init_states(mimi, batch_size=1, sequence_length=256)` (line 326), so
capacity is fixed at 256.

The mimi decoder transformer offset bumps by `upsample.convtr.stride[0] = 16`
per latent frame (line 409). After 16 latent frames the offset reaches 256
and `scatter` writes go out of bounds.

**Symptom:** First ~16 frames (≈ 1.28 s of audio at 12.5 Hz × 80 ms) sound
correct. From frame 17 onward the audio degrades to a robotic / metallic
buzz with no intelligible prosody. Listening tests: short verify utterances
(< 1 s of generation, all our verify_all_languages.sh outputs) sounded
clean and Whisper-passed; longer utterances (> 1.5 s of generation)
degraded after ~2 s of output.

**Why it slipped past parity tests:**
- `parity_mimi.py --num-frames 4` matched bit-identically (offset=64, well
  under 256).
- `parity_mimi.py --num-frames 40` raised the same out-of-bounds error in
  upstream PyTorch (`RuntimeError: index 256 is out of bounds for dimension
  1 with size 256`), confirming the wrapper itself was buggy. CoreML had
  the same bug baked in but didn't crash — it silently dropped the writes
  and produced garbage attention output.
- The real `model.generate_audio()` (used by `upstream_baseline.py`) uses
  the actual `StreamingMultiheadAttention` with proper sliding-window
  circular-buffer logic, which is why upstream PyTorch outputs sound clean
  at any length. The traced wrapper replaces the forward via
  `_patch_for_tracing` and loses that behavior.

**Comparison: same text, same voice, same seed:**

| Pipeline                    | Duration | Quality |
|-----------------------------|----------|---------|
| `upstream_baseline.py`      | 4.08 s   | Clean throughout |
| `generate_coreml_v4.py`     | 3.68 s   | Clean for 0–2 s, robotic 2–3.7 s |

**Fix:** Mirror the modulo-wrap pattern already in
`traceable_flowlm_step.py`:
```python
abs_idx = write_base + write_range                    # absolute logical positions
write_indexes = (abs_idx % capacity).view(B, T, 1, 1).expand(-1, -1, H, D)
new_k_cache = cache[0].scatter(1, write_indexes, k)
new_v_cache = cache[1].scatter(1, write_indexes, v)
```

Attention mask is rebuilt using absolute logical positions. Each cache
slot's logical position is reconstructed from the current `offset + T - 1`
and the slot index modulo `capacity`. Sliding-window mask uses the model's
`context = 250`.

**Verification:** re-traced + re-converted shared `mimi_decoder.mlpackage`,
re-ran the same 3.68 s text — audio now matches upstream quality
end-to-end for English long-form (user-confirmed `verify_long_fixed.wav`).

### Trial 13 — Per-language regression after the wrap fix (superseded by Trial 14)

> **Status:** the "modulo-wrap CoreML conversion drift" hypothesis below was
> a red herring. Trial 14 traced the regression to per-language Mimi weights
> (the assumption of a "shared codec" was wrong upstream). The diagnostic
> ladder and "Next steps" subsection here are kept for historical context;
> they do not represent open work.

After redistributing the patched `mimi_decoder.mlpackage` to every
`build/<lang>/`:

| Language    | Frames | Duration | Whisper output                                | Status |
|-------------|--------|----------|-----------------------------------------------|--------|
| english     | ~46    | 3.68 s   | "Hello, this is a text-to-speech system."     | PASS (user) |
| german      | 26     | 2.08 s   | "Hallo, das ist ein Sprachsynthesesystem."    | PASS |
| italian     | 31     | 2.48 s   | "Ciao, questo è un sistema di sintesi vocale." | PASS |
| portuguese  | 33     | 2.64 s   | "Olá, este é um sistema de síntese de voz."   | PASS |
| **spanish** | 31     | 2.48 s   | **"Gracias."**                                | FAIL (deterministic) |
| **french_24l** | 34  | 2.72 s   | **"Merci."** or MPSGraph crash               | FAIL (flaky) |

**Audio characterisation (24 kHz, 0.25 s windows, RMS):**

- italian: 0.13–0.31 throughout (continuous speech)
- german:  0.10–0.26 throughout (continuous speech)
- spanish: 0.03–0.16 with dips at t=1.25 s (0.0324) and t=1.75 s (0.0357)
- french_24l: 0.005–0.17 with near-silent gap at t=0.75 s (0.0050)

So Spanish/French_24l produce **intermittent** audio: the first phoneme
or two is intelligible, then dropouts make Whisper transcribe only the
opening word.

**Critical observation:** all 6 `mimi_decoder.mlpackage` files are
md5-identical (`3baf90b49ba8bfd8460d140cb4c987bf`), and Italian generates
the **same 31 frames in 2.48 s as Spanish** but produces clean speech.
So the bug is not in the shared mimi binary — it must be in the
language-specific `cond_step` / `flowlm_step` / `flow_decoder` packages
producing degraded latents that mimi-with-wrap can't compensate for the
way the OLD silent-drop mimi did.

**Parity diagnostic (mimi only):**

`parity_mimi.py --language spanish --num-frames 4` against the patched
wrapper now shows large divergence even at frame 0:

```
frame 0: u|mean=0.15448 c|mean=0.06452 abs_max=0.60588 abs_mean=0.15366
frame 1: u|mean=0.02129 c|mean=0.05319 abs_max=0.49712 abs_mean=0.05565
attn0_cache: abs_max=10.36, abs_mean=0.24
attn1_cache: abs_max=11.61, abs_mean=0.28
convtr0_partial: abs_max=59.93
```

This is suspicious: `parity_mimi` runs `TraceableMimiDecoder` (Python,
patched wrap) vs `mimi_decoder.mlpackage` (CoreML, traced from the same
patched wrap). They should match to fp16 noise. The fact that they
disagree at frame 0 with random latents suggests the modulo-wrap formula
(`x - floor(x/c)*c`) traces to MIL ops that CoreML's MPS backend
executes differently than PyTorch's CPU/MPS backend.

**Why German/Italian/Portuguese sound fine despite this divergence:**
those languages' real latents likely stay in a regime where the
fp16/MPS-vs-PyTorch numerical drift doesn't cross the threshold that
changes Whisper's transcription. Spanish/French latent distributions
apparently sit just past that threshold.

**Hypothesis:** `torch.floor(x / c) * c` may be lowered to a pattern that
uses int32 truncation in CoreML's MIL conversion, producing different
results for negative or large-magnitude `x` (the slot-logical-position
recovery passes both signs into `floor`). The
`traceable_flowlm_step.py` pattern this was meant to mirror operates
under different conditions (Q/K shapes, much smaller offsets) so the
same identity may not survive conversion identically here.

**Next steps (open):**

1. Inspect the converted MIL graph for the new wrap region — confirm
   whether `floor` lowers to a constant or runtime op, and whether the
   `to(torch.long)` cast maps to int32 or int64 in CoreML.
2. Try alternative wrap formulations in the wrapper: explicit `%`
   operator (PyTorch tensor mod) and re-convert; or pre-compute
   `write_indexes` as a Python int loop unrolled into the trace (T=16
   per call, capacity=256, only 16 distinct values).
3. If MIL conversion is the culprit, fall back to **bumping capacity**
   (Trial 12 Option 2) so wrap never happens within a single
   `decoder_transformer` invocation — increase
   `init_states(..., sequence_length=256)` to e.g. 4096.

---

### Trial 14 — Mimi weights ARE per-language (RESOLVED, 6/6 PASS)

Trial 13's "Critical observation" was wrong. The shared md5-identical
`mimi_decoder.mlpackage` was assumed safe because upstream's docstring
calls mimi a "language-agnostic codec." It is not.

**Diagnostic ladder, italian:**

| Stage          | abs_max | abs_mean | Verdict |
|----------------|---------|----------|---------|
| `parity_step`  | 0.00003 | ~0       | clean   |
| `parity_flow`  | 0.00000 | 0        | clean   |
| `parity_mimi` (frame 0) | 0.48 | 0.10 | BAD |
| `parity_mimi` attn0_cache | 10.34 | 0.24 | BAD |

So `cond_step` + `flowlm_step` + `flow_decoder` reproduce upstream
faithfully; the divergence is entirely inside `mimi_decoder`.

**Same-mlpackage parity contradiction:**

Running `parity_mimi.py` against the *byte-identical*
`mimi_decoder.mlpackage` (same md5) produced:

- english: frame 0 abs_max = 0.00000 (perfect)
- italian: frame 0 abs_max = 0.48    (broken)

PyTorch can't disagree with itself running the same mlpackage on the same
inputs — unless the Python-side reference (`TraceableMimiDecoder` loaded
via `TTSModel.load_model(language=<lang>)`) has different weights than
the traced mlpackage was converted from. It does.

**Direct weight comparison** (`decoder_transformer.layers[0].self_attn.in_proj.weight`):

```
english vs italian: abs_max=1.92, abs_mean=0.10
```

These are not numerical noise — they are different tensors. Upstream
`pocket-tts` ships per-language mimi weights inside each
`languages/<lang>/model.safetensors`, even though the codec architecture
is identical.

**Fix:** ran `convert_mimi_decoder.py --language <lang>` for the 5
non-English languages individually. Resulting md5s are all distinct:

```
english:    c92d3bce2c503c5a30a4071ddaa30749
spanish:    cee611647450a81ecbad7d20efeeff94
french_24l: 420c47fce6703531a8e2ba9c315cb748
german:     248d37732f19767331674d369dd3e0ee
italian:    2cc521825d1b8de2a18daf3a6a8b96ee
portuguese: 8bb1f992db1b1c4a4e5e6f94b1421868
```

**Verification:** `verify_all_languages.sh` end-to-end (Whisper-large-v3
transcription of each generated wav):

| Language    | Whisper output                                  | Status |
|-------------|-------------------------------------------------|--------|
| english     | "Hello, this is a text-to-speech system."       | PASS   |
| spanish     | "Hola, ¿este es un sistema de síntesis de voz?" | PASS   |
| french_24l  | "Bonjour, ceci est un système de synthèse vocale." | PASS |
| german      | "Hallo, das ist ein Sprachsynthesesystem."      | PASS   |
| italian     | "Ciao, questo è un sistema di sintesi vocale."  | PASS   |
| portuguese  | "Olá, este é um sistema de síntese de voz."     | PASS   |

6/6.

**Action items closed:**
- `convert_all_languages.sh` no longer copies a shared mimi mlpackage;
  the main loop calls `convert_mimi_decoder.py --language $lang` per
  target. Removed dead `ensure_shared_mimi` / `copy_shared_mimi` helpers
  and updated the comment block at the top of the script.
- The Trial 12 cap-256 modulo-wrap variant of `traceable_mimi_decoder.py`
  is the right one and is now committed; capacity-bump alternative
  reverted.
- Trial 13's "modulo-wrap CoreML conversion drift" hypothesis was a
  red herring. The actual conversion is fine; only the input weights
  were wrong.

---

## Summary of Bugs Found

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Flow decoder `s` hardcoded to 0 | Gibberish audio | Pass explicit `s = i/N` |
| 2 | Flow decoder `t` off by one | Gibberish audio | Pass `t = (i+1)/N` |
| 3 | Zero-padding conditioning | Wrong position (150 vs 141) | Use per-token cond_step model |
| 4 | Attention implementation mismatch | Wrong KV cache content | Copy exact code from verified step model |
| 5 | Conditioning order (text-first vs voice-first) | "Third is... Yes." | Swap to voice-first, then text |
| 6 | Mimi decoder KV cache overflow at frame 16 | Audio robotic after ~2 s | Add modulo wrap + sliding-window mask in traceable wrapper |
| 7 | Mimi weights assumed shared across languages | Spanish "Gracias.", French "Merci." | Trace `mimi_decoder.mlpackage` per-language (Trial 14) |
