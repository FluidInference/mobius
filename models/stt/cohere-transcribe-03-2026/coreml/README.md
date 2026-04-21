# Cohere Transcribe CoreML Export

CoreML export of [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) for on-device speech recognition on Apple Silicon.

## Status

The current shipping recommendation is the **cache-external decoder** paired with
its companion encoder. See [docs/DECODER_ARCHITECTURE_FINAL.md](docs/DECODER_ARCHITECTURE_FINAL.md)
for the full investigation and numbers.

| Component | Variant | Status |
|---|---|---|
| Encoder | companion FP32 (7.0 GB) | ✅ works, ships |
| Encoder | f16-download (3.6 GB) | ❌ wrong pairing — incompatible with cache-external decoder |
| Encoder | q8-download (1.8 GB) | ❌ wrong pairing |
| Decoder | cache-external FP16 (291 MB) | ✅ works: EN 10.6% / ES 4.9% / FR 16.8% / ZH 14.1% |
| Decoder | cache-external INT8 (146 MB) | ✅ token-identical to FP16 on CPU; ⚠️ MPSGraph crash on GPU/ANE |
| Decoder | stateful (HF-shipped, f16/q8) | ❌ over-generates past EOS (58–73% WER) |
| Decoder | stateless | ❌ over-generates (older "3.14% WER" claim measured with broken preprocessor) |
| Host preprocessing | `tools/cohere_features_v2.py` (v2 mel + CMVN) | ✅ required — old `hf-upload/example.py` mel is broken |

All WER/CER numbers are on a 12-sample FLEURS slice (3 per language) with
the fixed host pipeline (v2 mel, masked cross-attention, CJK byte-fallback
detokenization, repetition penalty 1.1, no-repeat 3-gram).

## Decoder architectures

Three architectures were explored:

1. **Stateful** (`decoder.make_state()`, `state=state` in `predict`) — KV cache
   inside the model as `MLState`. Requires macOS 15+ / iOS 18+. Over-generates
   past EOS on FLEURS regardless of quantization — do not ship.
2. **Stateless** — re-encode full prefix each step, take `logits[-1]`.
   No cache. Also over-generates on the fixed pipeline.
3. **Cache-external** — host-threads 16 KV cache tensors (8 layers × K/V,
   shape `[1, 8, 108, 128]`) as explicit function I/O. No `MLState`,
   runs on macOS 14+. **Only variant that produces clean transcripts.**

See [docs/DECODER_ARCHITECTURE_FINAL.md](docs/DECODER_ARCHITECTURE_FINAL.md)
for op names, benchmark results, quantization recipe, and host integration
notes.

## Cache-external decode loop

```python
k_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]
v_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]

for step in range(MAX_TOKENS):
    current = prompt_ids[step] if step < len(prompt_ids) else last
    inp = {
        "input_id": np.array([[current]], dtype=np.int32),
        "position_id": np.array([[step]], dtype=np.int32),
        "encoder_hidden_states": enc_out.astype(np.float32),
        "cross_attention_mask": cross_mask.astype(np.float32),
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float32),
    }
    for i in range(8):
        inp[f"k_cache_{i}"] = k_caches[i]
        inp[f"v_cache_{i}"] = v_caches[i]
    out = decoder.predict(inp)
    for i in range(8):
        k_caches[i] = out[f"k_cache_{i}_out"]
        v_caches[i] = out[f"v_cache_{i}_out"]
    last = int(np.argmax(out["logits"][0]))
    if step >= len(prompt_ids) - 1 and last == EOS:
        break
```

Fully working reference: [tests/bench-fix-vs-broken.py](tests/bench-fix-vs-broken.py).

## Language prompt (required)

The decoder is seeded with a 10-token prompt that encodes task + language:

```python
PROMPTS = {
    "en_us":       [13764, 7, 4, 16,  62,  62, 5, 9, 11, 13],
    "es_419":      [13764, 7, 4, 16, 169, 169, 5, 9, 11, 13],
    "fr_fr":       [13764, 7, 4, 16,  69,  69, 5, 9, 11, 13],
    "cmn_hans_cn": [13764, 7, 4, 16,  50,  50, 5, 9, 11, 13],
}
# ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
# <|lang|> <|lang|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
```

## Models (local layout)

```
hf-upload/cohere-transcribe-cache-external-coreml/     ← canonical, ships
    cohere_encoder.mlpackage                           (7.0 GB FP32, companion)
    cohere_decoder_cache_external.mlpackage            (291 MB FP16)
    tokenizer.model, example.py, README.md

build-cache-external-q8/                               ← from tests/quantize-cache-external.py
    cohere_decoder_cache_external_q8.mlpackage         (146 MB INT8)

hf-upload/f16-download/, hf-upload/q8-download/        ← stateful export, deprecated
    cohere_encoder.mlpackage                           (3.6 GB / 1.8 GB — WRONG for cache-external)
    cohere_decoder_stateful.mlpackage                  (305 MB / 150 MB — broken, over-generates)
```

## Fixed host preprocessor

The mel extractor in `hf-upload/example.py` is broken (`n_fft=400`,
no CMVN). Use `tools/cohere_features_v2.py` (`CohereMelSpectrogram`)
instead — it matches the HF `CohereAsrFeatureExtractor` reference.

Fixed in commit `0da224a` (`fix(cohere): correct host-side preprocessing + CJK detokenization`).

## Benchmarks and scripts

| script | purpose |
|---|---|
| `tests/bench-fix-vs-broken.py` | reference cache-external FP16 benchmark |
| `tests/bench-cache-external-hybrid.py` | cache-external f16 vs q8 |
| `tests/quantize-cache-external.py` | quantize cache-external decoder to INT8 |
| `tests/bench-hybrid-configs.py` | stateful f16/q8 × enc/dec combos (confirms stateful is broken) |
| `tests/bench-stateless-fleurs.py` | stateless decoder FLEURS bench (confirms stateless is broken) |
| `tests/bench-q8-variants.py` | re-quantization A/B on stateful (quantization tweaks don't fix stateful) |

## Open tasks

1. Quantize the 7 GB FP32 companion encoder to INT8, benchmark the all-q8 pipeline.
2. Fix the MPSGraph `MLIR pass manager failed` crash on the q8 cache-external decoder
   for GPU/ANE compute units (CPU runs fine).
3. Promote `cohere-transcribe-cache-external-coreml/` as the canonical HF layout;
   retire/deprecate `f16-download/` and `q8-download/` (they ship the broken
   stateful decoder).
4. Update Swift host integration in FluidAudio to use the cache-external decode
   loop (drops `MLState` dependency, allows macOS 14 / iOS 17).

## Requirements

- macOS 14+ / iOS 17+ (cache-external path; stateful path would need macOS 15+)
- Python 3.10+
- See `pyproject.toml` (coremltools, PyTorch, transformers, datasets, sentencepiece)

## Remaining documentation

- [docs/DECODER_ARCHITECTURE_FINAL.md](docs/DECODER_ARCHITECTURE_FINAL.md) — authoritative decoder comparison and q8 findings
- [docs/REVERSE_ENGINEERING.md](docs/REVERSE_ENGINEERING.md) — Cohere Transcribe architecture details
- [docs/RESEARCH_INSIGHTS.md](docs/RESEARCH_INSIGHTS.md) — general encoder-decoder ASR background
- [MLMODELC_LIMITATION.md](MLMODELC_LIMITATION.md) — why the stateful decoder cannot be `.mlmodelc`

## License

GPL-3.0 (matching upstream CoreML conversion).
Base model: Apache-2.0 ([CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)).
