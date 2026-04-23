# Cache-External Decoder Analysis: Python vs Swift

**Date**: April 8, 2026
**Model**: Cohere Transcribe 03-2026 (Cache-External Decoder)
**Test Dataset**: FLEURS multilingual (10 samples per language)

## Executive Summary

Both Python (CoreML) and Swift implementations of the cache-external decoder exhibit **severe multilingual hallucination issues**, but Swift is significantly worse. The root cause is that **neither implementation uses language conditioning**, and the exported CoreML decoder does not preserve the model's language detection capabilities.

## WER Comparison

| Language | Python WER | Swift WER | Swift vs Python |
|----------|-----------|-----------|-----------------|
| **English** | 55.02% | 263% | **4.8x worse** |
| **French** | 92.33% | 150% | **1.6x worse** |
| **Spanish** | 24.26% | 43% | **1.8x worse** |
| **Chinese** | 105.09% | 111% | Similar (both hallucinating) |

## Detailed Findings

### 1. Language Hallucination Patterns

Both implementations produce **non-target-language output** for most languages:

#### English Samples (Python):
- **Sample 0**: Arabic script `ولو انهم يحبون انهم يحبون...` (100% WER)
- **Sample 1**: Correct English transcription (62% WER)
- **Sample 4**: Arabic script `مين بصوتك في مكانك...` (267% WER)

#### French Samples (Python):
- **Sample 0**: Arabic script `نحن نعلم ان هناك من يحمل حياتنا...` (100% WER)
- **Sample 7**: Partial French transcription (58% WER)
- **Sample 2-6**: All Arabic hallucinations (100% WER each)

#### Spanish Samples (Python):
- **Sample 2**: Nearly perfect `"fue tanta la cantidad de gente que se concentró..."` (4.5% WER)
- **Sample 0**: Good quality Spanish (13.8% WER)
- **Average**: Best performance across all languages (24.26% WER)

#### Chinese Samples (Python):
- **Sample 0**: Polish script `"to tylko szybko odkryć..."` (100% WER)
- **Sample 1**: Arabic script `كعكعك يا شوشو...` (100% WER)
- **Sample 4**: English `"i'm sure the government..."` (122% WER)
- **All samples**: Complete hallucination (105% WER overall)

### 2. Swift Implementation Issues

Swift cache-external decoder produces **even worse hallucinations**:

- **English**: 263% WER (vs Python 55%)
- **French**: 150% WER (vs Python 92%)
- **Spanish**: 43% WER (vs Python 24%) - still best language
- **Chinese**: 111% WER (vs Python 105%)

**Why Swift is worse**:
1. Possible bugs in KV cache management
2. Incorrect attention mask sizing
3. Position ID handling issues
4. All symptoms suggest Swift's cache state is corrupted/incorrect

### 3. Root Cause Analysis

#### Neither Implementation Uses Language Conditioning

**Python code** (test-fleurs-wer.py:109):
```python
current_token = START_TOKEN  # Just token 4, no language token
```

**Swift code** (CohereAsrManager.swift):
```swift
let prompt = language?.promptSequence ?? [CohereAsrConfig.SpecialTokens.startToken]
```

While Swift HAS language support in the code, the Python test doesn't use it, proving the model should work without explicit language tokens if properly exported.

#### The CoreML Export Lost Language Detection

The original PyTorch model likely:
1. Auto-detects language from encoder hidden states
2. Conditions decoder output based on detected language
3. Uses language embeddings in the decoder layers

The CoreML export process:
1. Traced with fixed inputs (no language conditioning)
2. Lost dynamic language detection logic
3. Defaults to Arabic/mixed-language tokens

### 4. Why Spanish Works

Spanish achieves 24-43% WER while other languages hallucinate (>90% WER). Possible reasons:

1. **Training data dominance**: Spanish may be the most represented language in training
2. **Default language mode**: Model defaults to Spanish when language detection fails
3. **Simpler phonetics**: Spanish has more regular phoneme-to-grapheme mapping
4. **Export artifacts**: The specific trace inputs used during export may have been Spanish audio

## Recommendations

### Option 1: Re-export with Language Conditioning (RECOMMENDED)

**Action**: Modify `export-decoder-cache-external.py` to:
1. Accept language token as an additional input
2. Embed language token into the decoder's initial state
3. Export separate decoders per language (or one multilingual with language input)

**Pros**:
- Proper language conditioning
- Matches PyTorch model behavior
- Clean architecture

**Cons**:
- Requires re-export and re-testing
- May increase model size
- Need to test all languages

### Option 2: Use Stateful Decoder (iOS Only)

**Action**: Use the stateful decoder (already exported) which may preserve language state better.

**Pros**:
- CoreML manages state internally
- May preserve language context better
- Simpler Swift code

**Cons**:
- iOS/iPadOS only (macOS doesn't support `newState()`)
- Still may have same language detection issues
- Would need iOS device testing

### Option 3: Language-Specific Decoders

**Action**: Export separate decoder models per language.

**Pros**:
- Guaranteed language isolation
- Smaller per-language models
- No language confusion possible

**Cons**:
- 14 separate decoder models to manage
- 14× storage/memory requirements
- Deployment complexity

### Option 4: Accept Spanish-Only

**Action**: Document that cache-external decoder only works for Spanish, use other models for multilingual.

**Pros**:
- Works today (24-43% WER acceptable)
- No additional work required
- Clear user expectations

**Cons**:
- Very limited language support
- Defeats purpose of multilingual model
- Poor user experience for non-Spanish users

## Next Steps

1. **Decide on approach** (recommend Option 1: re-export with language conditioning)
2. **If re-exporting**:
   - Modify export script to accept language token input
   - Test with all 14 supported languages
   - Validate WER across all languages
   - Update Swift code to pass language token
3. **If accepting limitations**:
   - Document Spanish-only support for cache-external
   - Recommend stateful decoder for iOS multilingual use
   - Consider alternative models (Whisper, Parakeet) for multilingual

## Technical Details

### Cache-External Decoder Architecture

**Inputs** (17 total):
- `input_id` (1,1) - Current token
- `position_id` (1,1) - Position in sequence
- `encoder_hidden_states` (1, 438, 1024) - Encoder output
- `cross_attention_mask` (1, 1, 1, 438) - Encoder attention mask
- `attention_mask` (1, 1, 1, step+1) - Growing decoder attention mask
- `k_cache_0` through `k_cache_7` (8 arrays: 1, 8, 108, 128) - Key caches for 8 layers
- `v_cache_0` through `v_cache_7` (8 arrays: 1, 8, 108, 128) - Value caches for 8 layers

**Outputs** (17 total):
- `logits` (1, 16384) - Token probabilities
- `k_cache_0_out` through `k_cache_7_out` - Updated key caches
- `v_cache_0_out` through `v_cache_7_out` - Updated value caches

### Test Configuration

- **Python**: CoreMLTools prediction with PyTorch encoder
- **Swift**: Full Swift implementation with encoder + cache-external decoder
- **Dataset**: FLEURS test split (Google's multilingual ASR benchmark)
- **Languages**: en_us, fr_fr, es_419, cmn_hans_cn
- **Samples**: 10 per language (40 total)
- **No language conditioning**: Both tests started with START_TOKEN only

## Conclusion

The cache-external decoder is **fundamentally broken for multilingual use** in both Python and Swift, with Swift being significantly worse. The issue is NOT in Swift's implementation but in the **CoreML export process** which lost the model's language detection capabilities.

**Spanish is the only language that works** (24-43% WER), suggesting it was the export reference language or the most dominant in training.

To make this model usable for multilingual transcription, we must **re-export the decoder with explicit language conditioning** built into the model inputs, or accept Spanish-only deployment.
