# Per-Language Decoder Test Results - FAILED

**Date**: April 8, 2026
**Status**: Complete Failure (100% WER all languages)
**Approach**: Cache-external decoders with language embeddings permanently baked in

## Executive Summary

After exhausting all other fixes (language prompts, dynamic language embeddings, multilingual encoder), we attempted exporting separate cache-external decoders for each language with language bias permanently baked into the model weights during export.

**Result**: Complete catastrophic failure. All decoders output only special control tokens (language tags) instead of actual text. 100% WER across all languages.

---

## Test Results

| Language | WER | Output Pattern |
|----------|-----|----------------|
| **English** | 100.0% | `<|emo:undefined|><|en|><|en|><|en|><|en|>...` |
| **French** | 100.0% | `<|emo:undefined|><|ar|><|ar|><|ar|>...` or `<|fr|>` loops |
| **Spanish** | 100.0% | `<|emo:undefined|><|es|><|es|><|es|><|es|>...` |
| **Chinese** | 100.0% | `<|emo:undefined|><|pl|><|pl|><|pl|>...` or `<|ar|>` loops |

**Total samples**: 10 per language (40 total)
**Success rate**: 0/40 (0%)

---

## Sample Outputs

### English
```
Reference: "however due to the slow communication channels styles in the west could lag behind by 25 to 30 year..."
Hypothesis: "<|emo:undefined|><|ar|><|ar|><|ar|>..."
WER: 100%
```

### French
```
Reference: "l'accident a eu lieu en terrain montagneux et il semblerait que cela ait été causé par un incendie m..."
Hypothesis: "<|emo:undefined|><|ar|><|ar|><|ar|>..."
WER: 100%
```

### Spanish
```
Reference: "se recomienda enfáticamente a los viajeros que se informen sobre cualquier riesgo de clima extremo e..."
Hypothesis: "<|emo:undefined|><|es|><|es|><|es|><|es|>..."
WER: 100%
```

### Chinese
```
Reference: "这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束 也 是 新 篇 章 的 开 始..."
Hypothesis: "<|emo:undefined|><|pl|><|pl|><|pl|>..."
WER: 100%
```

---

## Implementation Details

### Export Strategy

Created separate decoder models with language bias permanently baked in:

```python
class LanguageSpecificDecoder(nn.Module):
    def __init__(self, decoder_wrapper, lm_head, language_token_id: int,
                 language_strength: float = 0.5):
        super().__init__()
        # ... extract language embedding from token table

        # Store as frozen parameter
        self.language_bias = nn.Parameter(
            language_strength * lang_emb.squeeze(0),
            requires_grad=False
        )

    def forward(self, input_id, position_id, ...):
        hidden_states = self.embedding(input_id, position_id)

        # Add permanent language bias
        hidden_states = hidden_states + self.language_bias.unsqueeze(0)

        # ... rest of decoding
```

**Exported Models**:
- `cohere_decoder_english.mlpackage` (291MB)
- `cohere_decoder_french.mlpackage` (291MB)
- `cohere_decoder_spanish.mlpackage` (291MB)
- `cohere_decoder_chinese.mlpackage` (291MB)

**Total storage**: 1164 MB (4× 291MB)

### Test Configuration

- **Encoder**: PyTorch (CohereLabs/cohere-transcribe-03-2026)
- **Decoder**: Per-language CoreML cache-external
- **Dataset**: FLEURS (en_us, fr_fr, es_419, cmn_hans_cn)
- **Samples**: 10 per language
- **Language strength**: 0.5 (50% of embedding magnitude)

---

## Root Cause Analysis

### Why This Failed

1. **Language bias too strong**: Adding 0.5× language embedding to every token's hidden state overpowered the actual text generation
2. **Token generation stuck in loop**: Decoders got stuck generating language control tokens instead of actual words
3. **No conditioning signal**: Without proper prompt sequence or starting tokens, the decoder defaults to outputting special tokens
4. **Interference with attention**: The baked-in bias may be interfering with self-attention and cross-attention mechanisms

### Comparison to Previous Attempts

| Approach | English WER | French WER | Spanish WER | Chinese WER | Status |
|----------|------------|------------|-------------|-------------|--------|
| Baseline cache-external | 55% | 92% | 24% ✅ | 105% | Spanish works |
| Language prompts (10 tokens) | 142% | 129% | 18.6% ✅ | 100% | Worse |
| Decoder V2 (dynamic language_id) | 57.5% | 149% | 18.6% ✅ | 113.5% | No improvement |
| Multilingual encoder | 57.5% | 88% | 18.6% ✅ | 113.5% | No improvement |
| **Per-language decoders (baked-in)** | **100%** | **100%** | **100%** | **100%** | **Complete failure** |

**Baseline cache-external is still the best approach** (despite only working for Spanish).

---

## Lessons Learned

1. **Baking language bias into model weights breaks text generation**: The language conditioning needs to be dynamic, not static
2. **Special token loops**: Models can get stuck in degenerate states when bias is too strong
3. **Spanish-only deployment remains the recommendation**: No fix has successfully enabled multilingual support
4. **Storage cost**: 1.2GB for 4 languages is prohibitive compared to 291MB for single universal decoder

---

## Conclusion

After 4 attempted fixes:
1. ❌ Language prompts (10-token sequences)
2. ❌ Decoder V2 (dynamic language embeddings)
3. ❌ Multilingual encoder (averaged mel spectrograms)
4. ❌ Per-language decoders (baked-in language bias)

**None have successfully enabled multilingual support for cache-external decoders.**

The cache-external decoder architecture is fundamentally incompatible with multilingual ASR when exported to CoreML. The encoder loses language information during export, and all decoder-side fixes either fail to help or make results worse.

---

## Final Recommendation

**Deploy cache-external decoder for Spanish-only.**

For multilingual ASR:
- Use **Whisper CoreML** (90+ languages, proven track record)
- Use **Qwen3 ASR** (Chinese/English, already in FluidAudio)
- Wait for Cohere to release properly-exported multilingual models

**Do NOT attempt further decoder-side fixes.** The issue is architectural and cannot be solved without re-exporting from Cohere's PyTorch model with proper language conditioning preserved.

---

## Files

### Test Scripts
- `export-per-language-decoders.py` - Export script for language-specific decoders
- `test-per-language-decoders.py` - FLEURS evaluation script

### Results
- `per_language_results.json` - Full test results (100% WER all languages)

### Models (FAILED - do not use)
- `build-per-language/cohere_decoder_english.mlpackage`
- `build-per-language/cohere_decoder_french.mlpackage`
- `build-per-language/cohere_decoder_spanish.mlpackage`
- `build-per-language/cohere_decoder_chinese.mlpackage`

### Documentation
- `MULTILINGUAL_INVESTIGATION_FINAL.md` - Summary of first 3 attempts
- `PER_LANGUAGE_DECODER_FAILURE.md` - This file (4th attempt)
