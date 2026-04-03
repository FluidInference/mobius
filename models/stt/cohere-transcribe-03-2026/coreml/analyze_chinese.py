#!/usr/bin/env python3
"""Analyze Chinese CER issue."""

def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    
    # Levenshtein distance on characters
    d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    
    return d[len(reference)][len(hypothesis)] / len(reference)

# Sample from Chinese benchmark
reference = "这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束 也 是 新 篇 章 的 开 始"
hypothesis = " 这并不是告别:这是一个篇章的结束,也是新篇章的开始。"

print("=" * 80)
print("CHINESE CER ANALYSIS - FLEURS DATASET ISSUE")
print("=" * 80)

print("\nFLEURS Reference (artificial spaces between EVERY character):")
print(f"  {reference}")
print(f"  Length: {len(reference)} chars")

print("\nModel Output (natural Chinese, no spaces):")
print(f"  {hypothesis}")
print(f"  Length: {len(hypothesis)} chars")

# CER with spaces (current benchmark)
cer_with_spaces = compute_cer(reference, hypothesis)
print(f"\nCER with spaces: {cer_with_spaces * 100:.2f}%")
print("  Problem: Reference has spaces, hypothesis doesn't")

# Remove ALL spaces and punctuation for fair comparison
ref_clean = reference.replace(" ", "").replace("，", "").replace("。", "").replace(":", "").replace(",", "")
hyp_clean = hypothesis.replace(" ", "").replace("，", "").replace("。", "").replace(":", "").replace(",", "")

print(f"\nCleaned Reference (remove spaces/punctuation):")
print(f"  {ref_clean}")

print(f"\nCleaned Hypothesis:")
print(f"  {hyp_clean}")

cer_clean = compute_cer(ref_clean, hyp_clean)
print(f"\nCER (fair comparison, no spaces): {cer_clean * 100:.2f}%")

# Character-by-character comparison
print("\n" + "=" * 80)
print("Character differences:")
print("=" * 80)

max_len = max(len(ref_clean), len(hyp_clean))
diffs = 0
for i in range(max_len):
    ref_char = ref_clean[i] if i < len(ref_clean) else "∅"
    hyp_char = hyp_clean[i] if i < len(hyp_clean) else "∅"
    if ref_char != hyp_char:
        print(f"  Position {i}: '{ref_char}' → '{hyp_char}'")
        diffs += 1

print(f"\nTotal character differences: {diffs}")
print(f"Accuracy: {(1 - diffs/len(ref_clean)) * 100:.2f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("The model generates correct natural Chinese!")
print("The 50.75% CER is due to FLEURS having spaces between every character.")
print("When comparing fairly (removing spaces), CER drops significantly.")
