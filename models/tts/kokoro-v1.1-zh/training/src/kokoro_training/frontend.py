"""Versioned English/Mandarin frontend for the PyTorch model, with no dropped IDs."""

import hashlib
import json
import re
import unicodedata

from misaki import en, espeak, zh
from opencc import OpenCC

VERSION = "misaki-0.9.4-bilingual-v1"
OVERRIDES = {"API": "ˌA pˌi ˈI", "GitHub": "ɡˈɪt hˌʌb"}


class Frontend:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.simplify = OpenCC("t2s")
        self.english = en.G2P(
            trf=False, british=False, fallback=espeak.EspeakFallback(british=False)
        )
        self.english.lexicon.golds.update(OVERRIDES)
        self.mandarin = zh.ZHG2P(version="1.1", en_callable=lambda s: self.english(s)[0])

    def __call__(self, text: str) -> dict:
        normalized = self.simplify.convert(unicodedata.normalize("NFKC", text)).strip()
        if not normalized:
            raise ValueError("Empty text")
        has_zh = bool(re.search(r"[\u4e00-\u9fff]", normalized))
        has_en = bool(re.search(r"[A-Za-z]", normalized))
        language = "mixed" if has_zh and has_en else "zh" if has_zh else "en"
        phones = (self.mandarin(normalized)[0] if has_zh else self.english(normalized)[0]).strip()
        unknown = sorted(set(phones) - set(self.vocab))
        if unknown:
            raise ValueError(f"Frontend emitted unsupported symbols: {unknown}")
        ids = [0, *[self.vocab[p] for p in phones], 0]
        if not 3 <= len(ids) <= 512:
            raise ValueError(f"Input has {len(ids)} tokens; split text before inference")
        return {"text": text, "normalized_text": normalized, "language": language,
                "phonemes": phones, "input_ids": ids, "frontend_version": VERSION,
                "frontend_policy_sha256": hashlib.sha256(json.dumps(OVERRIDES, sort_keys=True).encode()).hexdigest(),
                "backend": "misaki.en + misaki.zh 1.1/pypinyin; espeak English fallback",
                "unknown_symbols": []}
