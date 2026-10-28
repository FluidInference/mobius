"""Build FluidAudio's `<lang>_lexicon_cache.json` ({"lower": {word: [phonemes]}, "caseSensitive": {Word: [phonemes]}})
from misaki's `<lang>_gold.json` + `<lang>_silver.json`: silver then gold (gold wins), heteronyms take their DEFAULT,
lower map = every key lower-cased (last one wins), case-sensitive map = keys with any upper-case letter. Built this way
from us_gold/us_silver, the output equals the us_lexicon_cache.json FluidAudio ships, which the script checks first.
usage: build_lexicon_cache.py SRC_DIR (holds us_/gb_ gold, silver and us_lexicon_cache.json) [--lang gb] [--out DIR]"""
import argparse
import json
import pathlib


def build(src, lang):
    def default(v):
        return v if isinstance(v, str) else v.get("DEFAULT")

    merged = {}
    for tier in ("silver", "gold"):
        for k, v in json.load(open(src / f"{lang}_{tier}.json", encoding="utf-8")).items():
            merged[k] = default(v)
    lower, case_sensitive = {}, {}
    for k, v in merged.items():
        if not v:
            continue
        lower[k.lower()] = list(v)
        if k != k.lower():
            case_sensitive[k] = list(v)
    return {"lower": lower, "caseSensitive": case_sensitive}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("--lang", default="gb")
    ap.add_argument("--out", type=pathlib.Path)
    A = ap.parse_args()
    shipped = json.load(open(A.src / "us_lexicon_cache.json", encoding="utf-8"))
    assert build(A.src, "us") == shipped, "builder no longer reproduces us_lexicon_cache.json"
    print("us: rebuilt cache == shipped us_lexicon_cache.json")
    cache = build(A.src, A.lang)
    out = (A.out or A.src) / f"{A.lang}_lexicon_cache.json"
    out.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    symbols = sorted({p for m in cache.values() for ps in m.values() for p in ps})
    print(f"{A.lang}: {len(cache['lower'])} lower, {len(cache['caseSensitive'])} case-sensitive -> {out} "
          f"({out.stat().st_size / 1e6:.1f} MB); symbols {''.join(symbols)}")


if __name__ == "__main__":
    main()
