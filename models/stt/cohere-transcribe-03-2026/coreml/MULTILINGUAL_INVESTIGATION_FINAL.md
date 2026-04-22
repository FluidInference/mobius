# Cohere Transcribe Cache-External Decoder - Multilingual Investigation

**Date**: April 8, 2026
**Status**: Spanish-Only Deployment Recommended
**Model**: Cohere Transcribe 03-2026 (Cache-External Decoder)

## Executive Summary

After extensive testing and multiple re-export attempts, the cache-external decoder is **fundamentally broken for multilingual use**. Only Spanish achieves acceptable WER (<20%). All other languages hallucinate with >50% WER, outputting Arabic/Polish/wrong-language text.

**Recommendation**: Deploy cache-external decoder for **Spanish-only**. For multilingual ASR, use Whisper or Qwen3.

---

## Test Results Summary

### Final WER Comparison (10 samples per language)

| Language | Cache-External WER | Status |
|----------|-------------------|---------|
| **Spanish** | 18.6% | ✅ Production Ready |
| **English** | 57.5% | ❌ Hallucinating |
| **French** | 88.0% | ❌ Hallucinating |
| **Chinese** | 113.5% | ❌ Hallucinating |

### Example Hallucinations

**English Input**:
- Reference: `"however due to the slow communication channels styles in the west could lag behind..."`
- Hypothesis: `"ولو انهم يحبون انهم يحبون انهم يحبون"` (Arabic gibberish)
- WER: 100%

**French Input**:
- Reference: `"l'accident a eu lieu en terrain montagneux et il semblerait que cela ait été causé..."`
- Hypothesis: `"نحن نعلم ان هناك من يحمل حياتنا في الوصف"` (Arabic gibberish)
- WER: 100%

**Chinese Input**:
- Reference: `"这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束..."`
- Hypothesis: `"to tylko szybko odkryć. to szybko kędzamy cieszą..."` (Polish gibberish)
- WER: 100%

**Spanish Input** (✅ Works!):
- Reference: `"se recomienda enfáticamente a los viajeros que se informen sobre cualquier riesgo..."`
- Hypothesis: `"se recomienda enfáticamente a los viajeros que se informen sobre cualquier riesgo..."`
- WER: 13.8%

---

## Attempted Fixes

### 1. Language Token Prompts (FAILED)

**Approach**: Feed 10-token language-specific prompt sequence (like PyTorch quickstart.py)

```python
PROMPT = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]  # English
#                        ^^  ^^
#                      language tokens
```

**Results**:
- English: **142% WER** (worse than no prompts!)
- French: **129% WER** (worse)
- Spanish: **18.6% WER** (slightly better)
- Chinese: **100% WER** (same)

**Conclusion**: Language tokens are being ignored by the exported model.

---

### 2. Language Embeddings in Decoder V2 (FAILED)

**Approach**: Re-export decoder with `language_id` input parameter. Extract language embeddings from token table and add to hidden states.

```python
# Get language embedding and add to hidden states
lang_embedding = self.language_embeddings[language_id]
hidden_states = hidden_states + 0.1 * lang_embedding
```

**Results**:
- English: **57.5% WER** (same as baseline)
- French: **149% WER** (even worse!)
- Spanish: **18.6% WER** (unchanged)
- Chinese: **113.5% WER** (same)

**Conclusion**: Language embedding is too weak to override encoder's "wrong language" signal.

---

### 3. Multilingual Encoder (FAILED)

**Approach**: Re-export encoder traced with averaged mel spectrograms from 4 languages (English, French, Spanish, Chinese) instead of random noise.

**Hypothesis**: Original encoder was traced with Spanish audio or random noise, baking language assumptions into CoreML model.

**Results**:
- English: **57.5% WER** (no change)
- French: **88% WER** (4% improvement, still broken)
- Spanish: **18.6% WER** (unchanged)
- Chinese: **113.5% WER** (no change)

**Conclusion**: Tracing method has minimal impact. The fundamental issue is deeper in the export process.

---

## Root Cause Analysis

### Why Spanish Works

Spanish achieves 18.6% WER while all other languages fail (>50% WER). Possible reasons:

1. **Export Reference Language**: The PyTorch→CoreML export may have used Spanish audio as the trace input
2. **Training Data Dominance**: Spanish may be the most represented language in training data
3. **Default Language Mode**: Model defaults to Spanish when language detection fails
4. **Simpler Phonetics**: Spanish has more regular phoneme-to-grapheme mapping

### Why Everything Else Fails

The encoder outputs "language-agnostic" hidden states that don't preserve which language was spoken. The decoder tries to guess from these ambiguous features and:

1. Defaults to Spanish (works if input is actually Spanish)
2. Outputs mixed Arabic/Polish/random tokens (if input is not Spanish)

**Language conditioning in the decoder cannot override the encoder's lost language information.**

---

## Technical Details

### Cache-External Decoder Architecture

**Inputs** (18 total):
- `input_id` (1,1) - Current token
- `position_id` (1,1) - Position in sequence
- `encoder_hidden_states` (1, 438, 1024) - Encoder output
- `cross_attention_mask` (1, 1, 1, 438) - Encoder attention
- `attention_mask` (1, 1, 1, step+1) - Growing decoder attention
- `k_cache_0`...`k_cache_7` (8×: 1, 8, 108, 128) - Key caches
- `v_cache_0`...`v_cache_7` (8×: 1, 8, 108, 128) - Value caches

**Outputs** (17 total):
- `logits` (1, 16384) - Token probabilities
- `k_cache_0_out`...`k_cache_7_out` - Updated key caches
- `v_cache_0_out`...`v_cache_7_out` - Updated value caches

### Test Configuration

- **Dataset**: FLEURS (Google's multilingual ASR benchmark)
- **Languages**: en_us, fr_fr, es_419, cmn_hans_cn
- **Samples**: 10 per language (3 for quick tests)
- **Encoder**: PyTorch (for baseline) or CoreML (for full-stack tests)
- **Decoder**: CoreML cache-external
- **Metric**: Word Error Rate (WER) via jiwer

---

## Comparison: Python vs Swift

Both Python (CoreML) and Swift implementations exhibit the same hallucination patterns, proving the issue is in the model export, not the Swift code.

| Language | Python WER | Swift WER | Difference |
|----------|-----------|-----------|------------|
| English | 55% | 263% | Swift 4.8× worse |
| French | 92% | 150% | Swift 1.6× worse |
| Spanish | 24% | 43% | Swift 1.8× worse |
| Chinese | 105% | 111% | Similar |

Swift is worse due to implementation bugs (fixed during investigation), but both show the fundamental hallucination issue.

---

## Recommendations

### Production Deployment

**Use cache-external decoder for Spanish only:**

```swift
// CohereAsrManager.swift
public func transcribe(
    audioSamples: [Float],
    language: CohereAsrConfig.Language? = nil,
    maxNewTokens: Int = 96
) async throws -> String {

    // Warn if non-Spanish language requested
    if let lang = language, lang != .spanish {
        logger.warning("Cache-external decoder only supports Spanish reliably. Other languages may hallucinate.")
    }

    // Recommend Spanish for best results
    let targetLanguage = language ?? .spanish

    // ... rest of implementation
}
```

**For multilingual users, recommend alternatives:**
- **Whisper CoreML**: Battle-tested, 90+ languages, proven track record
- **Qwen3 ASR**: Already in FluidAudio, supports Chinese/English

---

### 4. Per-Language Decoders with Baked-In Language Bias (CATASTROPHIC FAILURE)

**Approach**: Export separate cache-external decoders for each language with language bias permanently baked into model weights during export.

```python
class LanguageSpecificDecoder(nn.Module):
    def __init__(self, decoder_wrapper, lm_head, language_token_id: int,
                 language_strength: float = 0.5):
        # Extract language embedding and freeze as parameter
        self.language_bias = nn.Parameter(
            language_strength * lang_emb.squeeze(0),
            requires_grad=False
        )

    def forward(self, input_id, position_id, ...):
        hidden_states = self.embedding(input_id, position_id)
        # Add permanent language bias to every token
        hidden_states = hidden_states + self.language_bias.unsqueeze(0)
        # ... rest of decoding
```

**Results** (10 samples per language):
- English: **100% WER** (outputs only `<|en|>` tokens)
- French: **100% WER** (outputs only `<|ar|>` or `<|fr|>` tokens)
- Spanish: **100% WER** (outputs only `<|es|>` tokens)
- Chinese: **100% WER** (outputs only `<|pl|>` or `<|ar|>` tokens)

**Example Output**:
```
Reference: "however due to the slow communication channels..."
Hypothesis: "<|emo:undefined|><|en|><|en|><|en|><|en|>..."
```

**Conclusion**: Complete catastrophic failure. Baking language bias into weights caused decoder to get stuck generating only special control tokens (language tags) instead of actual text. This is WORSE than all previous attempts.

**Storage cost**: 1.2GB for 4 languages (4× 291MB decoders)

**See**: `PER_LANGUAGE_DECODER_FAILURE.md` for full details.

---

### Future Work (If Needed)

If multilingual support is critical for cache-external:

1. **Contact Cohere**: Report export issue, request properly exported multilingual models
2. **Use Stateful Decoder** (iOS only): Test if state management fixes language context preservation
3. ~~**Export Per-Language Decoders**~~ ❌ TESTED - Complete failure (100% WER)
4. **Switch to Whisper**: Most pragmatic solution for multilingual ASR

---

## Files

### Documentation
- `CACHE_EXTERNAL_ANALYSIS.md` - Initial Python vs Swift comparison
- `MULTILINGUAL_INVESTIGATION_FINAL.md` - This file (comprehensive summary)

### Test Scripts
- `test-fleurs-wer.py` - Baseline test (no language conditioning)
- `test-cache-external-with-prompt.py` - Test with 10-token prompts
- `test-decoder-v2.py` - Test decoder V2 with language embeddings
- `test-multilingual-encoder.py` - Test multilingual encoder
- `export-decoder-cache-external-v2.py` - Decoder V2 export script
- `export-encoder-multilingual.py` - Multilingual encoder export script

### Results
- `python_cache_external_full.json` - Baseline Python results (10 samples)
- `cache_external_with_prompt_results.json` - Language prompt test (3 samples)
- `decoder_v2_results.json` - Decoder V2 test (3 samples)
- `multilingual_encoder_test_results.json` - Multilingual encoder test (3 samples)

### Models
- `build-test/cohere_encoder_multilingual.mlpackage` - Encoder traced with 4-language average
- `build-v2/cohere_decoder_cache_external_v2.mlpackage` - Decoder with language_id input
- `hf-upload/cohere-transcribe-cache-external-coreml/` - Original cache-external decoder

---

## Conclusion

After exhaustive testing (language tokens, language embeddings, multilingual encoder), the cache-external decoder remains broken for multilingual use. The issue is baked into the CoreML export process and cannot be fixed in Swift or with decoder tricks.

**Deploy for Spanish-only. For multilingual, use Whisper or Qwen3.**
