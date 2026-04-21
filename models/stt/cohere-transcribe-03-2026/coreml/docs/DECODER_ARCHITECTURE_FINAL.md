# Decoder Architecture: Final Findings (PR #41, 2026-04-21)

This is the authoritative reference for which decoder variant should ship and
how INT8 quantization behaves. Older standalone docs
(`STATELESS_VS_STATEFUL.md`, `FP16_VS_INT8_FLEURS_COMPARISON.md`,
`FINAL_STATELESS_RESULTS.md`, `CACHE_EXTERNAL_ANALYSIS.md`, etc.) were
written with a broken host preprocessor in place and reached the wrong
conclusions; they have been removed.

## TL;DR

| decoder | EN WER | ES WER | FR WER | ZH CER | ship? |
|---|---|---|---|---|---|
| stateful (HF-shipped) | 58.2% | 52.9% | 63.5% | 46.5% | **no** |
| stateless (build-stateless/) | 58.2% | 52.9% | 63.5% | 45.6% | **no** |
| **cache-external (f16)** | **10.6%** | **4.9%** | **16.8%** | **14.1%** | **yes** |
| **cache-external (q8)** | **10.6%** | **4.9%** | **16.8%** | **14.1%** | **yes** |

Measured on the 12-sample FLEURS slice (3 per language × 4 languages)
with the fixed host pipeline: v2 mel preprocessor, masked cross-attention,
CJK byte-fallback detokenization, repetition penalty 1.1, no-repeat 3-gram.
See [../tests/bench-cache-external-hybrid.py](../tests/bench-cache-external-hybrid.py).

The cache-external decoder is the only variant that transcribes cleanly.
INT8 weight-only quantization of the cache-external decoder is
**token-bit-identical** to FP16 on every sample tested (12/12 identical
transcripts, identical token counts per sample).

## The three decoder architectures

### 1. Stateful (CoreML `MLState`)

- Uses `decoder.make_state()`, `decoder.predict({...}, state=state)`.
- KV cache lives inside the model as hidden `MLState`.
- Shipped today at
  `hf-upload/f16-download/f16/cohere_decoder_stateful.mlpackage` (305 MB)
  and `hf-upload/q8-download/q8/cohere_decoder_stateful.mlpackage` (150 MB).
- **Problem**: over-generates past end-of-utterance. First sentence is
  usually correct, then the model hallucinates plausible-looking
  continuations until it hits `MAX_TOKENS=108`. This happens in both
  FP16 and INT8 and is not fixed by any of the quantization-side tweaks
  we tried (per-channel, per-tensor on lm_head, threshold gating,
  skip-tied-embedding, EOS-biased decoding).
- Requires macOS 15+ / iOS 18+ (State API).

### 2. Stateless (re-encode full prefix)

- Feed the full token sequence every step, take `logits[0, -1, :]`.
- No KV cache at all. O(N²) decode cost.
- Exported at `build-stateless/cohere_decoder_stateless.mlpackage` (291 MB).
- **Problem**: same over-generation symptom as stateful. The
  `FINAL_STATELESS_RESULTS.md` "3.14% WER on LibriSpeech" claim was
  measured through a different host pipeline (broken mel, no masking);
  re-running with the fixed pipeline on FLEURS reproduces the
  over-generation pattern.
- Would work on macOS 14 if quality were acceptable.

### 3. Cache-external (host-threaded KV cache) — **recommended**

- KV cache is an explicit tensor I/O of the function:
  - 8 layers × (k, v) × shape `[1, 8, 108, 128]` FP16
  - Inputs: `input_id`, `position_id`, `encoder_hidden_states`,
    `cross_attention_mask`, `attention_mask`, `k_cache_0..7`, `v_cache_0..7`
  - Outputs: `logits`, `k_cache_0_out..7_out`, `v_cache_0_out..7_out`
- Host zero-inits 16 caches once, threads step-N outputs into step-N+1 inputs.
- Located at
  `hf-upload/cohere-transcribe-cache-external-coreml/cohere_decoder_cache_external.mlpackage` (291 MB).
- **Produces clean transcripts**: EN 10.6% / ES 4.9% / FR 16.8% / ZH 14.1%
  on the FLEURS slice.
- Works on macOS 14 / iOS 17 (no State API dependency).

## INT8 quantization of the cache-external decoder

Quantization pass: `coremltools.optimize.coreml.linear_quantize_weights`,
per-channel weight-only, symmetric, `weight_threshold=2048`. No special
handling of the tied embedding (`embedding_token_embedding_weight_to_fp16`)
is required — the single-consumer shape of the cache-external MIL
program lets the default config succeed.

See [../tests/quantize-cache-external.py](../tests/quantize-cache-external.py).

Op-name notes for this export (differ from the stateful decoder's):

```
lm_head op     : linear_80_cast_fp16
input embed op : var_339_cast_fp16_cast_uint16
```

Result: FP16 291 MB → INT8 146 MB, **zero measurable token-level quality loss**
across EN/ES/FR/ZH on the 12-sample FLEURS slice.

## Encoder pairing constraint (IMPORTANT)

The cache-external decoder was exported **together with its companion encoder**.
Its hidden-state statistics are matched to that encoder only.

| artifact | encoder | WER with cache-external decoder (EN) |
|---|---|---|
| companion (`hf-upload/cohere-transcribe-cache-external-coreml/cohere_encoder.mlpackage`, 7.0 GB FP32) | ✓ matches | **10.6%** |
| `hf-upload/f16-download/f16/cohere_encoder.mlpackage` (3.6 GB FP16) | different export | 58.2% |
| `hf-upload/q8-download/q8/cohere_encoder.mlpackage` (1.8 GB INT8) | different export | 78.9% |

Confirmed empirically in
[../tests/bench-cache-external-hybrid.py](../tests/bench-cache-external-hybrid.py).
The f16-download and q8-download encoders are from a **different export
pass** with a different normalization / projection calibration; they are
not drop-in replacements for the companion encoder.

**Implication**: when we ship cache-external, we must also ship the
companion encoder. The 7 GB FP32 encoder still needs to be quantized
(separate follow-up); we have not yet tested `companion_encoder_q8 +
decoder_q8` as a full pipeline.

## Known issue: ANE / GPU predict crash on q8 decoder

The q8 cache-external decoder crashes `MPSGraph` at predict time on
GPU/ANE compute units:

```
MPSGraphExecutable.mm:5070: failed assertion
`Error: MLIR pass manager failed'
```

Running with `compute_units=ComputeUnit.CPU_ONLY` works and produces the
expected lossless output. This is a `coremltools` / macOS toolchain
issue, not a model issue — the INT8 weights are valid and decode
correctly on CPU. Needs investigation (possibly unsupported op pattern
for INT8 on ANE, or compute-unit targeting bug) before ANE deployment.

## Dead ends ruled out during this investigation

- **Per-channel / per-tensor / skip-lmhead / threshold-gated q8 on the
  stateful decoder**: none restore quality. Quality loss in the stateful
  decoder is a decoder-architecture issue, not a quantization issue.
- **INT8 encoder + FP16 stateful decoder hybrid**: all four combos
  (f16/f16, q8/f16, f16/q8, q8/q8) produce essentially identical
  over-generation on the stateful decoder. See
  [../tests/bench-hybrid-configs.py](../tests/bench-hybrid-configs.py).
- **+4 EOS logit bias workaround**: hides the symptom on stateful in
  many cases but is not a real fix — it still fails on ZH and on
  samples where the "correct" EOS margin is very narrow.
- **Stateless export**: over-generates too (earlier "3.14% WER on
  LibriSpeech" claim was measured through a broken preprocessor and
  does not reproduce on the fixed pipeline).

## Host integration changes (Swift / FluidAudio)

Moving from stateful to cache-external on the FluidAudio host side
removes the `MLState` dependency and replaces it with explicit tensor
threading:

```swift
// Before (stateful):
let state = try decoder.makeState()
for step in 0..<maxTokens {
    let out = try decoder.prediction(from: inputs, state: state)
    // ...
}

// After (cache-external):
var kCaches = (0..<8).map { _ in
    MLMultiArray.zeros(shape: [1, 8, 108, 128], dataType: .float16)
}
var vCaches = (0..<8).map { _ in
    MLMultiArray.zeros(shape: [1, 8, 108, 128], dataType: .float16)
}
for step in 0..<maxTokens {
    // ... build inputs with k_cache_i / v_cache_i from previous step
    let out = try decoder.prediction(from: inputs)
    for i in 0..<8 {
        kCaches[i] = out.featureValue(for: "k_cache_\(i)_out")!.multiArrayValue!
        vCaches[i] = out.featureValue(for: "v_cache_\(i)_out")!.multiArrayValue!
    }
}
```

Lowers the minimum platform requirement from macOS 15 / iOS 18 back to
macOS 14 / iOS 17 (no State API).

## Outstanding work for PR #41

1. Quantize the 7 GB FP32 companion encoder to INT8, benchmark the
   all-q8 pipeline (companion_q8 + decoder_q8).
2. Investigate and fix the MPSGraph crash on the q8 decoder for GPU/ANE.
3. Update `hf-upload/` canonical layout: promote
   `cohere-transcribe-cache-external-coreml/` to primary, retire the
   stateful variants in `f16-download/` and `q8-download/` (or clearly
   mark them deprecated).
4. Update Swift host integration in `FluidAudio` to use the cache-external
   decode loop.

## Relevant commits on `docs/cohere-transcribe-coreml-decoder-fix`

```
2b2a624  test(cohere): cache-external decoder survives q8 quantization losslessly
53448d7  test(cohere): bench hybrid q8/f16 and stateless decoders on FLEURS
5281011  test(cohere): try re-quantization fixes for q8 decoder
d0bbbff  test(cohere): diagnose q8 over-generation + demonstrate +4 EOS bias fix
bf97ae0  test(cohere): add FLEURS benchmark for HF-shipped q8 stateful decoder
0da224a  fix(cohere): correct host-side preprocessing + CJK detokenization
```

## Relevant scripts

- `../tests/bench-fix-vs-broken.py` — baseline reference bench (cache-external f16)
- `../tests/bench-hybrid-configs.py` — stateful f16/q8 × enc/dec combos
- `../tests/bench-stateless-fleurs.py` — stateless decoder FLEURS bench
- `../tests/bench-q8-variants.py` — re-quantization A/B on stateful
- `../tests/bench-q8-eosboost.py` — +4 EOS bias workaround diagnostic
- `../tests/bench-cache-external-hybrid.py` — cache-external f16 vs q8
- `../tests/quantize-cache-external.py` — quantize cache-external to q8
- `../tests/requantize-decoder.py` — attempted per-layer q8 fixes for stateful
