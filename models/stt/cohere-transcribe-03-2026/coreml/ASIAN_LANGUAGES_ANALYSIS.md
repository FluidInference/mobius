# Asian Languages Analysis - Japanese, Korean, Chinese

## Executive Summary

Initial benchmark results showed **Japanese: 99.47% WER** and **Korean: 14.45% WER**, suggesting poor performance. However, investigation revealed that **the model works correctly** - the issue is that **WER (Word Error Rate) is an inappropriate metric for these languages**.

## Root Cause Analysis

### The Problem with WER for Japanese/Korean/Chinese

**WER assumes space-separated words**, which doesn't apply to:
- **Japanese**: No spaces between words (e.g., `インターネットで敵対的環境について`)
- **Korean**: Spaces used differently than Western languages
- **Chinese**: No spaces between characters/words

### Example: Japanese Transcription

**Reference (FLEURS dataset):**
```
インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう
```
*(Has artificial spaces for annotation purposes)*

**Model Output:**
```
インターネットで敵対的環境構想について検索すると、おそらく現地企業の住所が出てくるでしょう。
```
*(Natural Japanese without arbitrary spaces)*

**WER Calculation:**
- Reference splits into 4 "words" by spaces
- Hypothesis splits into 1 "word" (no spaces after normalization)
- Result: **100% WER** ❌ (Incorrect metric)

**CER Calculation:**
- Character-level comparison shows minimal differences
- Result: **7.25% CER** ✅ (Correct metric)

## Corrected Results

### Use CER as Primary Metric for Asian Languages

| Language | WER (Misleading) | **CER (Correct)** | Status |
|----------|------------------|-------------------|--------|
| 🇯🇵 Japanese | 99.47% | **7.25%** | ✅ Good |
| 🇰🇷 Korean | 14.45% | **3.48%** | ✅ Excellent |
| 🇨🇳 Chinese | TBD | TBD | Testing... |

## Model Performance Validation

### Japanese Test Case

```python
reference = "インターネットで 敵対的環境コース について検索すると..."
generated = "インターネットで敵対的環境構想について検索すると、..."

# The model correctly:
# ✓ Transcribes the audio to Japanese text
# ✓ Captures the meaning accurately
# ✓ Uses proper Japanese grammar and punctuation
# ✗ Minor word choice difference: "コース" (course) vs "構想" (concept)
```

**Actual differences:**
- One word substitution: `コース` → `構想` (semantically close)
- Punctuation placement (natural in Japanese)
- Space removal (correct Japanese formatting)

## Recommendations

### For Benchmark Reporting

1. **Primary Metric**:
   - Japanese/Korean/Chinese: **Use CER**
   - Western languages: Use WER

2. **Updated Performance Tiers**:
   - Excellent: CER < 5%
   - Good: CER 5-10%
   - Fair: CER 10-15%
   - Poor: CER > 15%

### For Future Benchmarks

1. **Use language-specific word segmentation**:
   - Japanese: MeCab or SudachiPy
   - Korean: KoNLPy
   - Chinese: jieba or pkuseg

2. **Report both WER and CER** for all languages

3. **Character-level metrics are more universal** and comparable across languages

## Conclusion

**The Cohere Transcribe 03-2026 model performs well on Asian languages:**

- ✅ Japanese: 7.25% CER (good accuracy)
- ✅ Korean: 3.48% CER (excellent accuracy)
- ⏳ Chinese: Testing in progress

The initial 99.47% WER for Japanese was a **metric mismatch**, not a model failure. The model correctly transcribes Japanese audio and should be considered production-ready for Japanese applications.

## Technical Details

### Why FLEURS Has Spaces in Japanese

The FLEURS dataset includes spaces in Japanese transcriptions for **annotation consistency** across languages, not because they reflect natural Japanese text. These spaces are arbitrary and not part of standard Japanese writing.

### Normalization Impact

Current normalization:
```python
def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(c for c in text if c.isalnum() or c.isspace())  # Removes punctuation
    text = unicodedata.normalize("NFKD", text)
    text = " ".join(text.split())  # Collapses whitespace
    return text
```

For Japanese/Korean/Chinese:
- Reference keeps FLEURS arbitrary spaces: `["インターネットで", "敵対的環境コース", "について検索すると"]`
- Hypothesis has no spaces after removing punctuation: `["インターネットで敵対的環境構想について検索するとおそらく現地企業の住所が出てくるでしょう"]`
- WER sees this as 1 word vs 4 words → 100% error rate

**Solution**: Use CER or proper word segmentation tools.
