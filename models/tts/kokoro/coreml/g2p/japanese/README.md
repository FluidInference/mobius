# Kokoro ANE Japanese frontend assets (FluidAudio #914)

FluidAudio's Japanese text frontend (`Sources/FluidAudio/TTS/KokoroAne/G2P/Japanese/`)
is a Swift port of Misaki's Cutlet: a MeCab-compatible Viterbi tokenizer over
`unidic-lite` plus Cutlet's kana → IPA rules. The tokenizer reads the standard
MeCab binary dictionary layout, so the assets are unidic-lite's own files,
trimmed to the three feature fields the frontend uses.

## Build the assets

```bash
uv run --with unidic-lite python convert_unidic_lite.py \
  "$(uv run --with unidic-lite python -c 'import unidic_lite, os; print(os.path.join(os.path.dirname(unidic_lite.__file__), "dicdir"))')" \
  out/
```

`convert_unidic_lite.py` keeps the double array and token table byte-for-byte,
rewrites every feature string to `pos1,pron,kana` (UniDic indices 0, 9, 17)
and applies the same trim to `unk.dic`; `matrix.bin` and `char.bin` are copied.
Sizes: `sys.dic` 188 MB → 41 MB, `matrix.bin` 71.5 MB, `char.bin` 0.3 MB,
`unk.dic` 4 KB. Add Misaki's `ja_words.txt` (`misaki/data/ja_words.txt`, 2 MB).

## Validate

`mecab_reference.py` is a pure-Python reader of the same binary layout with a
MeCab-style Viterbi; run it against fugashi on any sentences to confirm the
trimmed dictionary tokenizes identically:

```bash
uv run --with unidic-lite --with fugashi python mecab_reference.py out/ '今日は良い天気です。' '私は日本語を勉強しています。'
```

It prints `SAME`/`DIFF` per sentence (surface, pron, kana, unknown flag and
character category must all match). The Swift `JapaneseTokenizer` mirrors this
script.

## Publish

Upload `sys.dic`, `unk.dic`, `char.bin`, `matrix.bin`, `ja_words.txt` to
`FluidInference/kokoro-82m-coreml` under `ANE-ja/assets/` (the Mandarin tables
live under `ANE-zh/assets/`). FluidAudio downloads them into `<repoDir>/g2p/`
on the first plain-text Japanese synthesis.

Licenses: unidic-lite / UniDic (BSD), misaki (Apache-2.0), cutlet (MIT),
num2kana (MIT) — see `ThirdPartyLicenses/JapaneseG2P-LICENSE.md` in FluidAudio.
