# Cohere Transcribe CoreML Export

CoreML export of [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) for on-device speech recognition on Apple Silicon.

## Status: ⚠️ Decoder Quality Issues

| Component | Status | Notes |
|-----------|--------|-------|
| **Encoder** | ✅ Working | Perfect parity with reference (max diff 0.041) |
| **Decoder** | ⚠️ Poor Quality | 292.89% average WER on LibriSpeech test-clean |
| **Mel Preprocessing** | ✅ Working | Python implementation matches reference |

### Test Results (LibriSpeech test-clean, 10 samples)

```
Average WER:  292.89%
Median WER:   100.00%
Min WER:      9.09%
Max WER:      1581.82%
```

**Issue**: Decoder gets stuck in severe repetition loops on most samples, especially longer audio (>10s).

## Current Models

**FP16 Models (build/):**
- `cohere_encoder.mlpackage` (3.6 GB) - ✅ Working perfectly
- `cohere_decoder_cached.mlpackage` (291 MB) - ⚠️ Generates output but poor quality
- `cohere_cross_kv_projector.mlpackage` (32 MB)

**Important**: Quantization (INT8, INT6) either crashes or produces worse quality. Only FP16 models are functional.

## Usage

### Testing

```bash
# Test on 10 LibriSpeech test-clean samples
uv run python test-librispeech.py
```

### Exporting Models

```bash
# Export encoder (FP16)
uv run python export-encoder.py --output-dir build --precision float16

# Export decoder (FP16)
uv run python export-decoder-cached.py --output-dir build --precision float16
```

## Critical Implementation Details

### 10-Token Prompt Required

The decoder requires a 10-token configuration prompt (not just start token):

```python
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
# ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|> 
# <|en|> <|en|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
```

Without this prompt: **135% WER**  
With prompt: **41-292% WER** (varies by sample length)

### Decoder Cache Handling

The decoder uses:
- Cache shape: `(8, 8, 108, 128)` per K/V tensor
- Masking approach (not truncation) to handle variable-length cache
- Self-attention and cross-attention cache separate

## Known Issues

1. **Repetition loops**: Decoder repeats phrases ("then, then, then..." for 100+ tokens)
2. **Quality degrades with length**: Short samples (~3s) work better than long ones (>10s)
3. **Quantization fails**: INT8/INT6 either crash (MPS errors) or produce worse quality
4. **Cache handling**: Suspected issue with KV cache update logic causing repetitions

## Files

**Export Scripts:**
- `export-encoder.py` - Encoder + projection layer
- `export-decoder-cached.py` - Decoder with KV cache (current best)
- `export-decoder-cached-v2.py` - Alternative decoder export (shape mismatch errors)
- `export-decoder-with-cross-kv.py` - Optimized decoder with pre-computed cross-KV
- `export-cross-kv-projector.py` - Cross-attention KV projector

**Testing:**
- `test-librispeech.py` - WER test on 10 LibriSpeech samples

**Preprocessing:**
- `cohere_mel_spectrogram.py` - Mel spectrogram computation (Python)

**Documentation:**
- `README.md` - This file
- `REVERSE_ENGINEERING.md` - Technical details and investigation notes

## Next Steps

1. **Debug decoder cache handling** - Primary issue causing repetitions
2. **Investigate why short samples work better** - Cache position handling?
3. **Compare with BarathwajAnandan's export** - Their models also show 100% WER, suggesting fundamental export issue
4. **Consider alternative decoder export approach** - Current masking approach may have bugs

## Requirements

- macOS 14+ / iOS 17+
- Python 3.10+
- coremltools
- PyTorch
- transformers
- datasets (for testing)
- sentencepiece (for tokenization)

## License

GPL-3.0 (matching upstream CoreML conversion)

Base model: Apache-2.0 ([CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026))
