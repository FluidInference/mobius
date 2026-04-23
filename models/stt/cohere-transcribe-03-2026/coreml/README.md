# Cohere Transcribe CoreML Export

CoreML export of [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) for on-device speech recognition on Apple Silicon.

## Status: Cache-External Decoder

The canonical pipeline exports the decoder with **host-managed KV cache**: the Swift loader allocates K/V cache tensors and passes them through each decoding step. This avoids the CoreML State API (macOS 15+/iOS 18+) and works everywhere FluidAudio runs.

| Component | Format | Notes |
|-----------|--------|-------|
| Encoder | `.mlpackage` (FP16) | 3500 frames (35 s), projection fused |
| Decoder (cache-external) | `.mlpackage` (FP16/INT8) | Per-step `input_id` + `k_cache_*` / `v_cache_*` I/O |
| Mel preprocessing | Pure Python / numpy | No `transformers` dependency |

### Shipped artifacts

The corresponding HF model repos consumed by FluidAudio:

- [`FluidInference/cohere-transcribe-cache-external-coreml`](https://huggingface.co/FluidInference/cohere-transcribe-cache-external-coreml) — FP16
- [`FluidInference/cohere-transcribe-q8-cache-external-coreml`](https://huggingface.co/FluidInference/cohere-transcribe-q8-cache-external-coreml) — INT8 hybrid (encoder Q8, decoder FP16)

Loader: `CohereFixedPipeline` in [FluidAudio PR #487](https://github.com/FluidInference/FluidAudio/pull/487).

## Decoder I/O contract

The Swift loader expects this exact decoder signature.

**Inputs**
- `input_id`: shape `(1, 1)` — token at the current step
- `position_id`: shape `(1, 1)` — current position
- `encoder_hidden_states`: shape `(1, enc_len, 1024)`
- `cross_attention_mask`: shape `(1, 1, 1, enc_len)`
- `attention_mask`: shape `(1, 1, 1, max_seq_len)` — masks unfilled cache slots
- `k_cache_0..7`, `v_cache_0..7`: shape `(1, num_heads, max_seq_len, head_dim)`

**Outputs**
- `logits`: shape `(1, 16384)`
- `k_cache_0_out..7_out`, `v_cache_0_out..7_out`: updated caches written back to host

Constants: `num_layers=8`, `num_heads=8`, `head_dim=128`, `hidden=1024`, `vocab=16384`, `max_seq_len=108`.

## Quick Start

```bash
# Encoder (FP16)
uv run python3 exports/export-encoder.py --output-dir build --precision float16

# Decoder (cache-external, FP16)
uv run python3 exports/export-decoder-cache-external.py --output-dir build --precision float16

# Optional: INT8 encoder
uv run python3 tools/quantize_to_int8.py
uv run python3 tools/compile_encoder_to_mlmodelc.py
```

## Decoding loop (Python reference)

```python
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
# ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
# <|en|> <|en|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
EOS_TOKEN_ID = 3
MAX_SEQ_LEN = 108

# Pre-fill caches with the prompt, then loop one token at a time:
for step in range(len(PROMPT_IDS), MAX_SEQ_LEN):
    out = decoder.predict({
        "input_id": np.array([[token]], dtype=np.int32),
        "position_id": np.array([[step - 1]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden,
        "cross_attention_mask": cross_mask,
        "attention_mask": self_mask,
        **k_caches, **v_caches,
    })
    token = int(np.argmax(out["logits"][0]))
    # write k_cache_*_out / v_cache_*_out back into k_caches / v_caches
    if token == EOS_TOKEN_ID:
        break
```

## Files

```
exports/
  export-encoder.py                 # Encoder + projection
  export-decoder-cache-external.py  # Canonical decoder export

tools/
  cohere_features_v2.py             # Numpy mel-spectrogram extractor
  compile_encoder_to_mlmodelc.py    # mlpackage → mlmodelc (encoder)
  download-fleurs-for-swift.py      # FLEURS fetcher for Swift benches
  quantize_to_int8.py               # Encoder INT8 quantization

tests/
  test-feature-parity.py            # PyTorch vs CoreML mel parity check

docs/
  CACHE_EXTERNAL_ANALYSIS.md
  CACHE_EXTERNAL_DELIVERED.md
  CACHE_INVESTIGATION_SUMMARY.md
  COHERE_ARCHITECTURE_ANALYSIS.md
  HOST_SIDE_FIXES.md
  Q8_EOS_BIAS.md
```

## Background

Earlier iterations explored a stateful (CoreML State API) decoder and a stateless (re-process all tokens per step) decoder. Both were dropped:

- **Stateful** — required macOS 15+/iOS 18+ and added cache-management complexity in the model.
- **Stateless** — O(n²), produced wrong outputs on longer sequences during validation.

`docs/CACHE_INVESTIGATION_SUMMARY.md` documents the earlier sliding-window cache bug that motivated moving cache management out of the model entirely.

## Requirements

- macOS 14+ / iOS 17+
- Python 3.10+, dependencies in `pyproject.toml` (managed with `uv`)

## License

GPL-3.0 (matches upstream CoreML conversion). Base model: Apache-2.0 ([CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)).
