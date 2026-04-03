#!/usr/bin/env python3
"""Analyze why Japanese has such high WER."""

def normalize_text(text: str) -> str:
    """Current normalization from benchmark."""
    import unicodedata
    text = text.lower()
    text = "".join(c for c in text if c.isalnum() or c.isspace())
    text = unicodedata.normalize("NFKD", text)
    text = " ".join(text.split())
    return text

def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

# Test with the actual Japanese example
reference = "インターネットで 敵対的環境コース について検索すると おそらく現地企業の住所が出てくるでしょう"
hypothesis = "インターネットで敵対的環境構想について検索すると、おそらく現地企業の住所が出てくるでしょう。"

print("=" * 80)
print("JAPANESE WER ANALYSIS")
print("=" * 80)

print("\nOriginal texts:")
print(f"Reference:  {reference}")
print(f"Hypothesis: {hypothesis}")

# Current normalization
ref_norm = normalize_text(reference)
hyp_norm = normalize_text(hypothesis)

print("\nAfter normalization:")
print(f"Reference:  '{ref_norm}'")
print(f"Hypothesis: '{hyp_norm}'")

print(f"\nReference words: {ref_norm.split()}")
print(f"Hypothesis words: {hyp_norm.split()}")

wer = compute_wer(ref_norm, hyp_norm)
print(f"\nWER: {wer * 100:.2f}%")

# The issue: Japanese doesn't use spaces between words!
# After removing punctuation and normalizing, we get one long character string
# When we split by spaces, we get character-level tokens instead of words

print("\n" + "=" * 80)
print("THE PROBLEM: Japanese doesn't use spaces!")
print("=" * 80)

print(f"\nReference has {len(ref_norm.split())} 'words' (actually characters)")
print(f"Hypothesis has {len(hyp_norm.split())} 'words' (actually characters)")

# For Japanese, we should use character-level or morpheme-level comparison
# Or keep the original segmentation
print("\nSolution: Use CER (Character Error Rate) for Japanese instead of WER")
print("Or use a proper Japanese word segmenter like MeCab")

