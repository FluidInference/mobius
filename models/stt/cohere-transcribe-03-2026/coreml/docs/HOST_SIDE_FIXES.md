# Host-Side Fixes for Cohere Transcribe CoreML

Three host-side bugs in the reference inference code caused severely degraded
transcription quality for every language on both f16 and q8 weights. None of
these require weight re-export or re-quantization — they are all in the
Python/Swift code that calls the CoreML models.

## TL;DR

| Bug | Where | Symptom |
|-----|-------|---------|
| Mel spectrogram mismatch | preprocessing | every encoder frame drifts from the HF extractor; errors compound |
| `cross_attention_mask` all-ones | decoder call | decoder attends to zero-padded encoder frames |
| CJK detokenization token-by-token | output | SentencePiece byte-fallback triples produce gibberish for zh/ja/ko |

After fixing all three, FLEURS quality moves from near-unusable (WER / CER
200–500%) on the original `hf-upload/example.py` path to within a few points
of the PyTorch reference on f16 across the 14 tested languages.

## 1. Mel Spectrogram Parity

### The bug

The shipped `hf-upload/cohere-transcribe-cache-external-coreml/example.py`
computes mel spectrograms with `librosa.feature.melspectrogram` followed by
`librosa.power_to_db` and a manual `(mel + 80) / 80` normalization. The HF
`CohereAsrFeatureExtractor` does not use librosa — it uses its own STFT with
a different window, a different `n_fft` rounding rule, and a different
normalization. The two pipelines disagree on every frame by a small amount
that compounds through the 24-layer encoder.

### The fix

`tools/cohere_features_v2.py` — a numpy port of `CohereAsrFeatureExtractor`
with bit-close parity to the HF implementation. It matches the window,
`n_fft`, hop, mel filter bank (128 bins, fmin=0, fmax=8000), and
normalization exactly.

- Used by: `f16/quickstart.py`, `f16/example_inference.py`,
  `q8/quickstart.py`, `q8/example_inference.py`, and the Swift
  `CohereMelSpectrogram` in FluidAudio.
- Regression test: `tests/test-feature-parity.py` loads the real HF
  extractor (`trust_remote_code=True`) and compares frame-by-frame against
  the numpy port on real audio across languages.

### How to reproduce

```bash
uv run python tests/test-feature-parity.py
```

The test asserts max-abs and mean-abs diff vs HF are within tolerance and
exits non-zero on regression.

## 2. `cross_attention_mask` Zeroing Padded Encoder Frames

### The bug

The encoder ingests mel features padded to 3500 frames and outputs 438
encoder frames. The decoder's cross-attention reads encoder hidden states
via a `cross_attention_mask` input. The original example passes
`np.ones((1, 1, 1, 438))`, letting the decoder attend to every padded frame
as if it carried signal. For short utterances (most FLEURS samples), the
padded frames dominate the attention, and the decoder hallucinates.

### The fix

Compute the number of *valid* encoder frames from the actual audio length
and mask the rest with a large negative value (`-1e4` is sufficient in
fp16):

```python
enc_valid = min(
    enc_out.shape[1],
    max(1, int(np.ceil(feat_len / (MEL_FRAMES_FIXED / ENCODER_FRAMES_FIXED)))),
)
cross_mask = np.zeros((1, 1, 1, enc_out.shape[1]), dtype=np.float16)
if enc_valid < enc_out.shape[1]:
    cross_mask[:, :, :, enc_valid:] = -1.0e4
```

- Constants: `MEL_FRAMES_FIXED = 3500`, `ENCODER_FRAMES_FIXED = 438` (the
  encoder down-samples mel frames by ~8x).
- Applied in: `f16/quickstart.py`, `q8/quickstart.py`, and
  `CohereFixedPipeline` in FluidAudio.

## 3. CJK Detokenization with Byte-Fallback

### The bug

The shipped example detokenized token-by-token via
`sp.id_to_piece(token)`, then concatenated pieces. Cohere's SentencePiece
model encodes CJK characters as *byte-fallback triples* — e.g. the
character 中 becomes three tokens that render as `<0xE4>`, `<0xB8>`,
`<0xAD>`. Rendered one at a time, these are placeholder strings, not the
UTF-8 character they encode.

### The fix

Detect byte-fallback pieces (`<0xHH>` pattern), buffer the raw bytes, and
decode the buffer as UTF-8 when a non-byte-fallback piece arrives or at
end-of-stream:

```python
_BFR = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

def detok(tokens, vocab):
    out, bb = [], []
    def flush():
        if bb:
            out.append(bytes(bb).decode("utf-8", errors="replace"))
            bb.clear()
    for t in tokens:
        if t <= 4 or t == EOS: continue
        s = vocab.get(t, "")
        if s.startswith("<|"): continue
        m = _BFR.match(s)
        if m:
            bb.append(int(m.group(1), 16))
            continue
        flush()
        out.append(s)
    flush()
    return "".join(out).replace("\u2581", " ").strip()
```

- Applied in: `f16/quickstart.py`, `q8/quickstart.py`, and the Swift
  `CohereTokenizer.decode` in FluidAudio.

## Impact

Running the fixed pipeline on the same FLEURS samples that the original
example produced 200–500% WER on:

- **f16 f16** — Within a few points of the PyTorch reference on all 14
  tested languages (en, fr, es, de, it, pt, pl, nl, sv, tr, ru, zh, ja, ko).
- **q8** — Same pattern, except EOS behavior diverges on short utterances
  because of weight quantization noise on the EOS logit. See
  [Q8_EOS_BIAS.md](./Q8_EOS_BIAS.md).

See [FP16_VS_INT8_FLEURS_COMPARISON.md](./FP16_VS_INT8_FLEURS_COMPARISON.md)
for the full per-language breakdown (note: the numbers in that doc predate
these fixes — it is kept as historical context for why this investigation
happened).

## Reproduction

All three fixes are validated end-to-end by:

```bash
# Multi-dataset, multi-precision benchmark (f16 and q8):
uv run python tests/benchmark-librispeech.py --precision f16 --samples 100 --dataset fleurs --language en_us --normalize
uv run python tests/benchmark-librispeech.py --precision q8  --samples 100 --dataset fleurs --language en_us --normalize

# CJK CER benchmark (uses byte-fallback-aware detokenization):
uv run python tests/benchmark-cjk-cer.py --precision f16 --samples 100 --language ja_jp

# FLEURS q8 bench against the HF-shipped stateful decoder:
uv run python tests/bench-q8-fleurs.py --language en_us --n 3
```

Mel parity alone is validated by `tests/test-feature-parity.py`.
