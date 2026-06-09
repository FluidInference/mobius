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

---

## Phase 6: Split-Mimi (transformer on ANE + SEANet on GPU)

**Hypothesis:** the monolithic `mimi_decoder.mlmodelc` has the
upsample-plus-attention transformer block (which CoreML profiling
showed is ANE-eligible) and the SEANet conv stack (which is GPU/CPU
only) sharing one model. If we split the codec into:

- `mimi_transformer_no_trig.mlmodelc` — FP16, ~74% ANE-resident
- `mimi_seanet_fp16.mlmodelc` — FP16 GPU/CPU

and chain them per frame, the transformer block should run on the
Neural Engine and free the GPU for the boundary upsample / SEANet
conv path. End-to-end synthesis should get faster.

The split conversion ran clean: trace-time parity vs the monolithic
mimi was bit-exact in FP32 and within 0.1 dB MSE in FP16 on a
3 s utterance. CoreML compile + load succeeded; an end-to-end WAV
generated through Swift sounded indistinguishable from the
monolithic path.

### Trial 15 — A/B latency benchmark on M2 (debug build, host clock)

A per-stage timer was added to the Swift `StreamingGenerator`
(`prefill`, `flowlm`, `flowdec`, `mimi`) to isolate just the mimi
delta from the rest of the pipeline. Both modes ran from the same
process with two warmup iterations and five timed iterations,
**interleaved** (mono, split, mono, split, …) to cancel out drift
and thermal effects. Same 14-word sentence, same voice (`alba`),
same seed. Apple M2, 16 GB, macOS 26.5, AsyncIO actor.

```
mode    n  frames  prefill ms     flowlm ms/f   flowdec ms/f   mimi ms/f      total ms        RTFx
mono    5  97      762  ± 243     22.15 ± 2.10  12.73 ± 2.56   24.50 ± 1.59   6533 ±  957     1.20
split   5  98      953  ± 238     26.17 ± 8.50  17.16 ± 7.36   24.72 ± 6.83   7531 ± 1936     1.04
```

**Headline:** the mimi step itself is statistically indistinguishable
between the two pipelines — mono 24.50 ms/f vs split 24.72 ms/f
(Δ = -0.21 ms/f, -0.9%, well within ±1σ). The expected
ANE-on-transformer win does not appear at the mimi stage.

End-to-end is a regression: -13% RTFx (1.20× → 1.04×), +15.3% on
the per-chunk inner loop.

### Why the split path regressed

1. **Boundary tensor round-trip.** The transformer's last hidden
   state `[1, 512, 16]` has to leave one MLModel and re-enter
   another every frame. Two `MLModel.predict` calls (≈ two ANE/GPU
   submits) per frame replace the single submit on the monolithic
   path. The boundary copy itself is small but the dispatch overhead
   is paid twice.

2. **State plumbing.** Mimi's 26-tensor streaming state is
   partitioned 7 + 19 across the two stages. Each frame copies
   stage-1 outputs out, slices, copies stage-2 inputs in, then
   merges back into the unified `MimiState`. This is pure Swift
   overhead the monolithic path doesn't pay.

3. **Cross-stage scheduler contention.** The split path's per-stage
   variance is 4-5× higher across **all** stages — including the
   flow-LM and flow-decoder stages we did not touch (split flowlm
   σ=8.5 ms/f vs mono σ=2.1 ms/f). One bad iteration spiked
   total_ms +60% and dragged every stage with it. The signature
   matches ANE submit stalling the GPU queue while the GPU was
   mid-kernel on flow-decode / SEANet.

4. **Residency cost.** Two mlmodelc compiled + loaded vs one →
   prefill is +25% (953 vs 762 ms). Minor, but consistently in the
   wrong direction.

The transformer-on-ANE bet only pays off if `(ANE compute saved)`
exceeds `(boundary copy + state plumbing + cross-stage scheduling
overhead)`. On a 14-word M2 run, it does not — the savings are
zero or negative.

### Methodology notes (what to repeat for a re-test)

- Build was `swift build` (debug). Release was attempted but
  `AppLogger` only emits to `os_log` in release, so the per-stage
  ms/f line is unreachable from stdout/stderr. A relative
  comparison in debug is still valid because Swift overhead is
  paid by both modes equally.
- AppLogger's `os_log`-only behaviour in release is the reason
  the StageTimings line is grep-able only in debug.
- Interleaving (mono, split, mono, split…) matters. A sequential
  layout (5× mono then 5× split) gave wildly different aggregates
  in earlier runs because the second batch always paid lower
  thermal / cache-cold cost.
- Two warmup iterations were enough for ANE compile to stabilize.
  More warmup did not change the medians.
- Variance-control: drop the highest- and lowest-`total_ms` per
  mode and the regression magnitude shrinks slightly but never
  reverses sign.

### What would change the result

Not pursued in this round, but the levers exist:

- **Pipeline composite model.** Wrap transformer → SEANet inside
  one `MLModel` (CoreML pipeline / composite), letting CoreML
  keep the boundary tensor on-device and submit one combined
  graph. This eliminates the round-trip but reintroduces the
  monolithic compile-time problem — the whole point of the split
  was to make the transformer sub-graph tractable for ANE
  placement, which the composite hides again.
- **Explicit compute-unit pinning.** Force the transformer to
  `.cpuAndNeuralEngine` and SEANet to `.cpuAndGPU` via
  `MLModelConfiguration.computeUnits` instead of `.all`. This
  was not tested. The default `.all` may be over-eagerly placing
  SEANet ops on ANE-or-CPU and stalling.
- **Bigger workload.** ~95 frames is short; ANE dispatch cost
  amortizes more on longer utterances. A 50-word paragraph A/B
  may flip the sign.
- **Release build with stdout logging.** Adding a stderr fallback
  to AppLogger would make release-mode per-stage timing reachable.
  Swift overhead may be hiding a real ANE win that survives in
  release.

### Recommendation

**Do not ship the split path.** Keep `mimi_decoder.mlmodelc`
(monolithic) as the default. The split mlpackage pair has been
validated for parity and can stay in the conversion repo as a
reference / future-experiment artifact, but the runtime cost on
M2 is a net loss. Re-evaluate if either (a) we move to a composite
pipeline model, (b) we ship release-mode per-stage timing and
remeasure, or (c) we test on a chip where ANE bandwidth dominates
GPU dispatch overhead more strongly than M2.

### Status

**Trial 15: regression — split path shelved.** Building blocks
(conversion scripts, parity scripts, RoPE precomputation,
schema discovery, Swift scaffolding) remain available for future
re-test if any of the levers above are pulled.

---

## Phase 7: Dispatch reduction + per-model ANE placement (RTFx)

Motivated by an on-device profile of the int8 English pack where all four
models landed 100% CPU (~905 ms/utterance), a regression from the
`flowlm_step`+`flow_decoder`-on-ANE state in `IOS_COREML_ISSUES.md`:

| Model         | calls/utt | per-call | total   | share |
|---------------|-----------|----------|---------|-------|
| cond_step     | 18        | 6.8 ms   | 122 ms  | 13%   |
| flowlm_step   | 43        | 5.4 ms   | 231 ms  | 26%   |
| flow_decoder  | **336**   | 0.74 ms  | 249 ms  | 28%   |
| mimi_decoder  | 42        | 7.2 ms   | 302 ms  | 33%   |

Three independent levers, each landed as a standalone artifact so they can be
A/B'd in isolation.

### Trial 16 — Fuse the LSD Euler loop into the flow decoder

**Problem.** `flow_decoder.mlpackage` traces a SINGLE Euler step; the Swift
host runs the 8-step integration, dispatching `predict()` 8× per audio frame
(336 calls/utterance). The kernel is a 1056→32 MLP — far below the size that
amortizes ANE residency — so each 0.74 ms call is mostly MLModel dispatch +
the fp32↔fp16 IO cast (paid 8×). `transformer_out` is constant across all 8
steps, so feeding it once and looping internally is pure redundant traffic
removed.

**Fix.** `traceable_flow_decoder_fused.py` unrolls all N Euler steps in
`forward(transformer_out, latent_init) -> latent_final`; `s`/`t` become
trace-time constants (i/N, (i+1)/N) that fold into the AdaLN time embedding.
`convert_flow_decoder_fused.py --num-steps 8` emits `flow_decoder_fused.mlpackage`.
Math is bit-identical to the host loop (built-in parity check vs the single-step
decoder; fp32 max-abs-diff < 1e-5). 336 dispatches → 42; the fatter kernel is
now worth placing on ANE (`.all`).

Same fusion pattern as PR #66's Nemotron "B1 fusion" (decoder+joint → one
mlpackage, +15% throughput, output-identical).

**Host change.** `PocketTtsSynthesizer+Flow.swift::flowDecode` now does one
`predict({transformer_out, latent_init})` and reads `latent_final`; the 8-call
`runFlowDecoderStep` loop is deleted. `numSteps` must match `--num-steps`.

> `--num-steps 4` is available as a quality/speed knob (halves internal
> flow_net evals). PRECISION.md flags the LSD denoiser as precision-sensitive,
> so A/B with Whisper before shipping a lower step count. The FluidAudio host
> default (`PocketTtsConstants.numLsdSteps`) is currently set to 4 PENDING that
> A/B — revert to 8 + re-convert with `--num-steps 8` if 4 degrades WER.

### Trial 17 — One-shot conditioning prefill (cond_prefill)

**Problem.** `cond_step.mlpackage` is T=1; the host dispatches it once per
conditioning token (18 calls / 122 ms / 13% here). Trial 5 (RangeDim) and
Trial 8 (zero-padding) both failed to batch this — dynamic shapes assert in
`scatter`, and padded tokens corrupt the KV cache + position counter.

**Fix.** `traceable_cond_prefill.py` processes a fixed `T_max` block in one
call. No dynamic shapes (sidesteps Trial 5). A runtime `valid_len` VALUE (not
a shape) gates correctness (sidesteps Trial 8): padded tokens' cache writes
are redirected to a dump slot the attention mask always excludes, and the
returned position advances by `valid_len`, not `T_max`. Attention/RoPE copied
verbatim from the verified `TraceableCondStep` (Trial 10/11). Built-in parity:
one-shot prefill of 141 real tokens (padded to 256) matches 141 sequential
per-token calls on the valid KV prefix (max-abs-diff < 1e-3).
`convert_cond_prefill.py --t-max 256` emits `cond_prefill.mlpackage`.
Stays `.cpuAndGPU` (rank-5 KV still blocks ANE).

**Host change (DONE).** `PocketTtsModelStore` loads `cond_prefill` optionally
(absent → per-token fallback, mirroring Magpie's `decoder_prefill`); its output
schema is identical to cond_step so `.condStep` key-discovery is reused.
`prefillKVCacheVoice/Text` take a `useFastPrefill` Bool + non-optional prefill
model (Swift 6 can send `MLModel` across actors but not `Optional<MLModel>`),
build the voice / text block, pad to `T_max`, pass true count as `valid_len`,
and call `runCondPrefill` once. Threaded through `StreamingGenerator` and
`PocketTtsSession`. Builds clean + swift-format clean.

### Trial 18 — Per-model compute units

**Problem.** `PocketTtsModelStore.loadIfNeeded` loaded ALL four models
`.cpuAndGPU` — a global hammer for the Mimi beeping (issue #7). That ban only
needs to cover Mimi; applying it everywhere discards the `flowlm_step` 1.97×
ANE win and the (now-fused) flow decoder's ANE eligibility.

**Fix (DONE).** Per-model config in `loadIfNeeded`:

| Model               | Units        | Why |
|---------------------|--------------|-----|
| cond_step / prefill | `.cpuAndGPU` | rank-5 KV trips ANE partitioner |
| flowlm_step(/v2)    | `.all`       | 1.97× ANE win; int8 dequants to fp16 on ANE |
| flow_decoder_fused  | `.all`       | fused MLP+Euler, ANE-friendly |
| mimi_decoder        | `.cpuOnly`   | fp32 (no beep) + 1.74× faster than GPU |

This matches the "Final dispatch" column in `IOS_COREML_ISSUES.md` — it is the
documented intended end-state, not new territory.

### Expected impact (to be confirmed on-device)

| Model        | before | after (est.) | mechanism |
|--------------|--------|--------------|-----------|
| cond_step    | 122 ms | ~30-50 ms    | 18 calls → 1 (prefill) |
| flowlm_step  | 231 ms | ~120 ms      | ANE 1.97× |
| flow_decoder | 249 ms | ~80-120 ms   | 336 → 42 dispatches, ANE |
| mimi_decoder | 302 ms | ~302 ms      | architecturally CPU-locked |
| **total**    | 905 ms | **~530-590** | RTFx ↑ ~1.6×; mimi now the floor |

### Verification (Apple Silicon)

```
cd models/tts/pocket_tts && uv sync
uv run python coreml/convert_models/convert/convert_flow_decoder_fused.py --language english --num-steps 4
uv run python coreml/convert_models/convert/convert_cond_prefill.py --language english
# device residency / fallback reasons:
uv run coreml-cli build/english/flow_decoder_fused.mlpackage --fallback
uv run coreml-cli build/english/cond_prefill.mlpackage --fallback
# end-to-end Whisper parity after wiring the Swift host changes:
./coreml/verify_all_languages.sh
```

### Status

**Trials 16-18: implemented.** Swift host wiring is complete and compiles
(Swift 6 strict concurrency) + passes swift-format: per-model compute units,
fused-decoder single call, and the optional cond_prefill fast path with
per-token fallback. Remaining: re-convert + upload the fused/`cond_prefill`
mlpackages, then an interleaved A/B RTFx run (per Trial 15 methodology) plus a
Whisper A/B on `--num-steps 4` to confirm the estimates above.

---

## Phase 7 — MEASURED on-device (M-series, macOS 26, coremltools 9 / torch 2.12)

Phase 7's estimates above were validated on hardware. Several were wrong and are
corrected here. All numbers are `coreml-cli` medians (5 warmup) at the best
compute-unit config, composed against the 905 ms baseline (42-frame utterance).

### Baseline truth (existing fp32 HF pack)

| model | ops | device@`all` | predict | `cpu_and_ne` | size |
|-------|----:|------|--------:|------|-----:|
| cond_step | 492 | GPU | 5.14 ms | →CPU 131 ms | 254 MB (fp32) |
| flowlm_step | 540 | GPU | 4.37 ms | →CPU 24 ms | 306 MB |
| flow_decoder (1-step) | 165 | CPU 0.62 / GPU 1.39 | — | CPU | 37 MB |
| mimi_decoder | 307 | flat ~6.1 ms all engines | 6.0 ms | flat | 41 MB |

**Key correction:** the repo's "flowlm 70-90% on ANE, 1.97×" claim is FALSE on
this hardware. Nothing in the existing pack reaches the ANE — `cpu_and_ne`
falls 100% to CPU. The pack ships fp32 weights; the ANE is fp16-only.

### New artifacts (measured)

| model | ops | device@`all` | predict | size | parity |
|-------|----:|------|--------:|-----:|--------|
| **flow_decoder_fused8** | 1252 | **100% ANE** | 1.09 ms | 18 MB | 2.1e-2 vs loop (fp16) |
| flow_decoder_fused4 | 628 | **100% ANE** | 0.66 ms | 18 MB | 2.7e-2 |
| **cond_prefill** (T=256) | n/a | GPU (ANE compile fails) | 4.83 ms | 127 MB | logic 7.6e-6, fp16 6.6e-2 |
| **flowlm_step fp16** | 540 | GPU (ANE compile fails) | 3.46 ms | 145 MB | EOS Δ 0.042 < int8 0.099 |

- **Fusion makes the decoder ANE-eligible** (165 ops 0% ANE → 1252 ops 100% ANE).
  The tiny single-step kernel was always rejected; the fat fused graph is accepted.
  This is the ONLY model that reaches the ANE.
- **flowlm/cond cannot reach the ANE at any precision** — the rank-5 KV cache
  `(2,1,512,16,64)` → `ANECCompile FAILED`. fp32 was a red herring; rank-5 is
  the hard block. fp16 still helps: −21% GPU latency + half the size.
- **fp16 flowlm EOS is safe**: per-step EOS-logit drift vs fp32 is 0.042 max,
  *half* the already-shipped int8 variant's 0.099, against an ~11-unit margin to
  the −4.0 stop threshold. fp16 (fp16 acts + fp16 weights) is strictly more
  precise than the shipped int8 (fp16 acts + int8 weights).

### Measured roll-up

| stage | baseline | after | device | basis |
|-------|---------:|------:|--------|-------|
| cond | 122 ms | 4.8 ms | GPU | cond_prefill 1 call |
| flowlm_step | 231 ms | 149 ms | GPU | fp16 3.46×43 |
| flow_decoder | 249 ms | 46 ms | **ANE** | fused8 1.09×42 |
| mimi_decoder | 302 ms | 302 ms | CPU | unchanged |
| **total** | **905 ms** | **~502 ms** | | **1.80× RTFx** |

### mimi is compute-bound, NOT overhead-bound (hypothesis disproven)

A 2 MB-state toy (same shape as mimi's two attn caches) measured the marshalling
cost directly:

```
state passed in/out:  0.53 ms/call
resident MLState:     0.04 ms/call   (12× faster)
```

So per-call state marshalling is only ~0.5 ms — mimi's 6 ms is real compute
(2-layer attention over the 256-deep cache + SEANet transposed-conv upsampling).
`MLState` would save only ~0.5 ms/frame (~20 ms total) — NOT worth a stateful
rewrite of mimi. mimi is at its ~7 ms/frame compute floor and is now the
dominant cost (60% of the 502 ms).

### Remaining lever (not model-conversion)

mimi can't be cheaply sped up. The only remaining win is **cross-engine
pipelining** in the Swift runtime: the per-frame chain is flowlm (GPU 3.5) →
flow (ANE 1.1) → mimi (CPU 7.2) on three distinct engines, run serially.
Overlapping them across frames makes per-frame cost approach
max(7.2, 3.5+1.1) ≈ 7.2 ms instead of the ~12 ms sum, hiding flowlm+flow behind
mimi → projected total ~330 ms (~2.7× vs 905 ms). Conversion levers are exhausted.

### Status

Trials 16-18 IMPLEMENTED + MEASURED + verified within shipped tolerances.
Open: (1) literal end-to-end Whisper run (confirmation only — every fp16
component is bounded below the shipped int8 error); (2) `--num-steps 4` Whisper
A/B; (3) cross-engine pipelining (Swift, the next real win).

---

## Phase 8: flowlm on the ANE (rank-4 + scatter-free rewrite)

### Trial 19 — `flowlm_step_ane`: the rank-5 block is breakable

**Problem.** Phase 7 concluded flowlm "cannot reach the ANE at any
precision — the rank-5 KV cache `(2,1,512,16,64)` → `ANECCompile FAILED`"
and declared conversion levers exhausted. But rank-5 is a *formulation*
property, not a model property. The graph had three ANE-hostile
constructs: the rank-5 cache I/O, a second rank-5 tensor inside RoPE
(the `[B,T,H,32,2]` interleaved view), and the circular-buffer `scatter`
write.

**Fix.** `traceable_flowlm_step_ane.py` — the Trial 16 playbook applied
to the step model. Math identical, formulation changed:

1. Each `cache{i} [2,1,512,16,64]` split into `k_cache{i}` / `v_cache{i}`
   `[1,512,16,64]` (rank-4 I/O; same slot layout, so cond_step /
   cond_prefill caches feed in directly after splitting the K/V axis).
2. T=1 specialization (generation is always one frame/call): RoPE pairs
   via a rank-4 `[1,H,32,2]` reshape; valid+causal masks collapse to
   one comparison.
3. Scatter-free write: `onehot = (arange(L) == pos % L)`;
   `new_k = k_cache*(1-onehot) + k*onehot`. Bit-identical circular-buffer
   semantics including the modulo wrap.
4. Additive mask (`(mask-1)*1e4`) instead of `masked_fill(-inf)`.
5. Cache NaN-scrub dropped — the Swift host zero-fills caches
   (`emptyKVCacheState`), so unwritten slots are 0 by contract. The
   NaN-BOS `sequence` replacement is kept.

**Parity.** fp32 torch wrapper vs `TraceableFlowLMStep`: **0.0e+00**
(bit-identical) across 5 autoregressive steps — outputs, EOS, caches,
positions. fp16 CoreML vs fp32 torch: d_out 8.2e-3, d_EOS 5.5e-3 (vs the
0.042 fp16 EOS drift already accepted in Phase 7).

**Measured (Apple Silicon, macOS 26, MLComputePlan + same-harness A/B,
coremltools predict, median of 200 warm calls):**

| model | ops | device@`cpu_and_ne` | predict@`all` |
|-------|----:|------|--------:|
| flowlm_step (rank-5) | 567 | ANECCompile FAILED → 100% CPU | 3.32 ms (GPU) |
| **flowlm_step_ane** | 466 | **100% ANE** | **3.04 ms** |

~8% faster in the Python harness — which marshals ~25 MB of cache I/O
per call, so the compute-only delta is larger than it reads. The real
wins are placement, not the median: the GPU is now fully free per frame
(flowlm ANE → flowdec ANE → mimi CPU), which both simplifies
cross-engine pipelining (Phase 7's "remaining lever") and removes the
GPU↔ANE ping-pong; on iOS it moves 26% of utterance compute off the GPU.

**Host changes required (FluidAudio).** Per-layer cache I/O becomes
`k_cache{i}`/`v_cache{i}` (two rank-4 buffers instead of one rank-5) and
outputs gain `new_v_cache{i}`; cond_step/cond_prefill output caches must
be split once after prefill (zero-copy view is possible: the rank-5
buffer is contiguous with K at offset 0 and V at offset L*H*D).

**Follow-ups.**
- Fuse `flowlm_step_ane` + `flow_decoder_fused` into one ANE dispatch per
  frame (both are now 100% ANE; saves one MLModel boundary per frame).
- Same treatment for `cond_prefill` (T=256 block; the one-hot write
  generalizes to a T×L comparison matrix — bigger, but prefill runs once).
- On-device Swift A/B (Trial 15 methodology) before shipping; Whisper
  end-to-end as usual.

### Status

**Trial 19: CONVERTED + PARITY-VERIFIED + 100% ANE.** Swift host wiring
not yet done; artifact is `coreml/build/<lang>/flowlm_step_ane.mlpackage`
via `convert_flowlm_step_ane.py`.

### Trial 20 — `cond_prefill_ane`: same treatment, honest split verdict

**Fix.** `traceable_cond_prefill_ane.py` + `convert_cond_prefill_ane.py` —
the Trial 19 rewrite applied to the Trial 17 prefill: rank-4 k/v cache
split, the one-hot write generalized to a `[T_max, L]` assignment matrix
applied as a matmul (dump-slot redirect preserved; the dump slot
accumulates instead of last-write-wins, unobservable through the mask),
rank-4 RoPE, additive mask, no NaN scrub. Plus one new trick: the RoPE
rotations / assign matrix / coverage / mask bias are **layer-invariant**
(they depend only on position + valid_len), so they are hoisted out of
the 6-layer loop into a single prologue — without the hoist the
scalar-driven ops the ANE compiler rejects (`sin`/`cos`/`equal`/
`less_equal`) are rebuilt per layer and force six CPU<->ANE transitions
(measured 17.0 ms forced-ANE; hoisted: 9.6 ms, op count 477 -> 371, and
the partition-level `ANECCompile FAILED` disappears).

**Parity.** fp32 vs `TraceableCondPrefill` on the valid prefix:
**0.0e+00** (bit-identical), all 6 layers. fp16 CoreML: 6.99e-2 — same
band as Trial 17's shipped 6.6e-2.

**Measured (same harness as Trial 19, median of 100 warm calls):**

| config | cond_prefill (rank-5) | cond_prefill_ane (rank-4, hoisted) |
|--------|--------------------:|-------------------:|
| device @ `cpu_and_ne` | GPU-only model: falls to CPU | **92% ANE** (371 ops) |
| `all` | 4.86 ms (GPU) | 4.49 ms (scheduler picks GPU) |
| `cpu_and_ne` (forced) | — | 9.57 ms |

**Verdict.** ANE-*capable* (0% -> 92%) but on M-series the GPU stays
~2x faster for this single fat T=256 call, and `MLComputePlan` under
`.all` prefers GPU for BOTH new models (flowlm_step_ane included —
ANE residency requires loading `.cpuAndNeuralEngine`, the same
capability-vs-shipped split as Parakeet v3's encoder). Ship
`cond_prefill_ane` anyway:

1. its k/v-split rank-4 I/O matches `flowlm_step_ane`, so the host
   keeps ONE cache layout end-to-end (no split/stack between prefill
   and decode);
2. prefill runs once per utterance — 4.5 vs 4.9 ms is a wash on Mac,
   and on iOS (weak GPU, power budget) the `.cpuAndNeuralEngine`
   option now exists at all.

**Compute-unit recommendation (Mac):** cond_prefill_ane `.all` (GPU),
flowlm_step_ane `.cpuAndNeuralEngine` (3.68 ms vs 3.04 GPU — pay ~0.6
ms/frame to keep the GPU free and the decode loop ANE+CPU only) or
`.all` if raw Mac RTFx is the only goal. iPhone numbers TBD.

### Status

**Trial 20: CONVERTED + PARITY-VERIFIED + 92% ANE (hoisted).** Artifact:
`coreml/build/<lang>/cond_prefill_ane.mlpackage`. Swift host wiring for
the k/v split (shared with Trial 19) still open.
