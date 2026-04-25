# PocketTTS language packs

FluidAudio's PocketTTS wraps `kyutai-labs/pocket-tts`. Upstream ships a
separate weight pack per language under
`https://huggingface.co/kyutai/pocket-tts/tree/main/languages/`. The folder
name on HF is the exact string we accept as `--language` and as the
Swift `PocketTtsLanguage.rawValue`.

| Language    | ID                 | Transformer layers | Upstream source                            |
|-------------|--------------------|--------------------|--------------------------------------------|
| English     | `english`          | 6                  | `kyutai/pocket-tts/languages/english`       |
| German      | `german`           | 6                  | `kyutai/pocket-tts/languages/german`        |
| Italian     | `italian`          | 6                  | `kyutai/pocket-tts/languages/italian`       |
| Portuguese  | `portuguese`       | 6                  | `kyutai/pocket-tts/languages/portuguese`    |
| Spanish     | `spanish`          | 6                  | `kyutai/pocket-tts/languages/spanish`       |
| French (24L)| `french_24l`       | 24                 | `kyutai/pocket-tts/languages/french_24l`    |
| German (24L)    | `german_24l`       | 24              | `kyutai/pocket-tts/languages/german_24l`    |
| Italian (24L)   | `italian_24l`      | 24              | `kyutai/pocket-tts/languages/italian_24l`   |
| Portuguese (24L)| `portuguese_24l`   | 24              | `kyutai/pocket-tts/languages/portuguese_24l`|
| Spanish (24L)   | `spanish_24l`      | 24              | `kyutai/pocket-tts/languages/spanish_24l`   |

Notes:

- Upstream did NOT ship a 6-layer French pack; only `french_24l` is available.
- 24-layer variants are ~4× larger and ~3× slower than 6-layer variants at
  conversion time. Runtime latency on ANE is dominated by the flowlm step
  cache size, so 24L is ~2× slower per synthesis.
- Voice embeddings (`audio_prompt`) are per-language: the same 21 speaker
  names exist in every language pack but the underlying tensors differ.
- Tokenizer (`tokenizer.model`, SentencePiece) is per-language.
- The Mimi codec (`mimi_encoder` + `mimi_decoder`) is language-agnostic —
  re-exporting it per-language is a no-op (`mimi_decoder.mlpackage` is byte
  identical across languages). We still emit one copy per `build/<lang>/`
  for a self-contained upload tree.

## Target CoreML repo layout

Everything pushed to `FluidInference/pocket-tts-coreml` lives under
`languages/<id>/`:

```
languages/
├── english/
│   ├── cond_step.mlpackage/
│   ├── flowlm_step.mlpackage/
│   ├── flow_decoder.mlpackage/
│   ├── mimi_decoder.mlpackage/
│   └── constants_bin/
│       ├── bos_emb.bin
│       ├── text_embed_table.bin
│       ├── tokenizer.model
│       └── <voice>_audio_prompt.bin (×21)
├── french_24l/
│   └── …same structure…
└── spanish_24l/
    └── …
```

The legacy root-level English files (`cond_step.mlpackage`, …,
`constants_bin/`) remain untouched to preserve backward compatibility for
existing FluidAudio builds.

## Build + upload workflow

```bash
cd models/tts/pocket_tts/coreml

# Convert every language (overnight on Apple Silicon)
uv sync
./convert_all_languages.sh

# Or a subset
LANGUAGES="spanish italian" ./convert_all_languages.sh

# Upload (requires HF token — run by the user, not by agents)
huggingface-cli login
./upload_languages.sh
```
