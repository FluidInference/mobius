#!/usr/bin/env python3
"""Text normalization utilities for Chinese ASR CER computation.

Handles common mismatches between model output and reference transcriptions:
- Punctuation removal
- Number format normalization (digits <-> Chinese characters)
- Whitespace normalization
"""
import re


def normalize_chinese_text(text: str) -> str:
    """Normalize Chinese text for fair CER comparison.

    Args:
        text: Input Chinese text (may contain punctuation, numbers, etc.)

    Returns:
        Normalized text with:
        - Punctuation removed
        - Digits converted to Chinese characters
        - Whitespace normalized
    """
    # Remove punctuation
    text = re.sub(r'[，。！？、；：""''（）《》【】…—·]', '', text)
    text = re.sub(r'[,.!?;:()\[\]{}<>"\'-]', '', text)

    # Convert Arabic digits to Chinese characters
    digit_map = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }

    # Replace digits with Chinese characters
    for digit, chinese in digit_map.items():
        text = text.replace(digit, chinese)

    # Normalize whitespace (collapse multiple spaces, strip)
    text = ' '.join(text.split())

    return text


def normalize_chinese_numbers_advanced(text: str) -> str:
    """Advanced number normalization for year/date formats.

    Handles patterns like:
    - 2011 年 8 月 -> 二零一一年八月
    - 15 米 -> 十五米

    Args:
        text: Input text with Arabic numerals

    Returns:
        Text with numbers converted to Chinese characters
    """
    digit_map = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }

    # For years and dates, convert digit-by-digit (二零一一 not 两千零一十一)
    def replace_year(match):
        year = match.group(1)
        return ''.join(digit_map[d] for d in year)

    # Pattern: 4-digit year followed by 年
    text = re.sub(r'(\d{4})\s*年', lambda m: replace_year(m) + '年', text)

    # For other multi-digit numbers, also convert digit-by-digit
    def replace_number(match):
        num = match.group(0)
        return ''.join(digit_map.get(d, d) for d in num)

    text = re.sub(r'\d+', replace_number, text)

    return text


def compute_cer_normalized(reference: str, hypothesis: str) -> float:
    """Compute CER with text normalization.

    Args:
        reference: Ground truth transcription
        hypothesis: Model output transcription

    Returns:
        Character Error Rate (0.0 to 1.0+)
    """
    import editdistance

    # Normalize both texts
    ref_norm = normalize_chinese_text(reference)
    hyp_norm = normalize_chinese_text(hypothesis)

    # Remove spaces for character-level comparison
    ref_chars = ref_norm.replace(' ', '')
    hyp_chars = hyp_norm.replace(' ', '')

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    distance = editdistance.eval(ref_chars, hyp_chars)
    return distance / len(ref_chars)
