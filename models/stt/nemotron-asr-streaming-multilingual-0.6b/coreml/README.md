# Nemotron-3.5-ASR-Streaming-Multilingual 0.6B — CoreML Conversion

CoreML conversion of NVIDIA's `nvidia/nemotron-asr-streaming-multilingual-0.6b` for real-time streaming ASR on Apple devices.

## Model Overview

| Property | Value |
|----------|-------|
| Source Model | `nvidia/nemotron-asr-streaming-multilingual-0.6b` (internal NVIDIA evaluation) |
| Architecture | FastConformer Cache-Aware RNNT **with Prompt** |
| Parameters | 0.6B |
| Languages | 38 (en, es, de, fr, it, ar, ja, ko, pt, ru, hi, zh-CN, zh-TW, vi, he, nl, cs, da, pl, no, sv, th, tr, bg, el, et, fi, hr, hu, lt, lv, ro, sk, uk, mt, sl, …) |
| Att Context | `[56,0]` / `[56,3]` / `[56,6]` / `[56,13]` |
| Default Chunk | 1.12 s (112 mel frames at 16 kHz, subsampled ×8 → 14 encoder frames) |
| Mel Features | 128 bins, 16 kHz |
| Vocab Size | 13,087 + 1 blank (= 13,088 joint output) |

The model differs from the English `nemotron-speech-streaming-en-0.6b` in one place: a `prompt_kernel` MLP (`Linear(1152 → 2048) → ReLU → Linear(2048 → 1024)`) is inserted at the encoder front-end. The 128-d half of the 1152 input is a one-hot language id (`num_prompts: 128`), and the 1024-d half is the pre-encode acoustic features.

## CoreML Output

Four mlpackages produced by the conversion script:

| Model | Function | Inputs (delta vs. English) |
|-------|----------|----------------------------|
| `preprocessor.mlpackage` | audio → 128-d mel | identical |
| `encoder.mlpackage` | mel + cache + **prompt_id** → encoded + new_cache | **+ `prompt_id: int32 [1]`** |
| `decoder.mlpackage` | token + LSTM state → decoder_out + new_state | identical structure (vocab is 13,088) |
| `joint.mlpackage` | encoder + decoder → logits | identical structure (output dim 13,088) |

Plus:
- `metadata.json` — config + `prompt_dictionary` + `lang_tag_token_ids`
- `tokenizer.json` — id → SentencePiece piece

## Layout

```
coreml/
├── ARCHITECTURE.md                # forward graph, prompt mechanics, auto-detection proof
├── pyproject.toml                 # macOS inference + benchmark env (coremltools, datasets, scipy)
├── nemo_reference.py              # NeMo PyTorch ground-truth runner
├── test_coreml_multilingual.py    # single-file CoreML smoke test
├── benchmark_fleurs.py            # multi-language WER/CER benchmark on FLEURS
├── fleurs_lang_map.py             # FLEURS code ↔ Nemotron code mapping
└── conversion_scripts/
    ├── pyproject.toml             # Linux conversion env; pins NeMo fork
    ├── inspect_model.py           # discovery helper
    ├── multilingual_components.py # prompt-aware encoder wrappers
    ├── convert_nemotron_multilingual.py
    └── dump_lang_tags.py          # extracts lang-tag token ids
```

`convert_nemotron_multilingual.py` reuses `PreprocessorWrapper`, `DecoderWrapper`, `JointWrapper`, `ExportSettings`, and `_coreml_convert` from the sibling English package at `../../../nemotron-speech-streaming-0.6b/coreml/conversion_scripts/individual_components.py`. Only the encoder wrapper is new.

## Prerequisites

- Linux + CUDA (NeMo + the `kingformatty/NeMo` fork run there cleanly)
- Python 3.10
- `uv`
- The `.nemo` file (≈ 2.37 GB; NVIDIA internal evaluation distribution)

## Steps

```bash
cd mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml/conversion_scripts
uv sync
```

### 1. Inspect (one-off, to confirm prompt API)

```bash
uv run python inspect_model.py \
    --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
    --target-lang en-US
```

This dumps:
- model class (must be `EncDecRNNTBPEModelWithPrompt`)
- top-level child modules (`encoder`, `decoder`, `joint`, `preprocessor`, `prompt_kernel`)
- the signature of `model.encoder.forward` — used to pick Layout A vs. Layout B
- `prompt_dictionary` lookup for the chosen language
- a 1 s silent dry-run

### 2. Convert

```bash
uv run python convert_nemotron_multilingual.py \
    --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
    --output-dir ./build_fp16 \
    --precision FLOAT16 \
    --att-context 56,0 \
    --chunk-mel-frames 112
```

For other att-context variants (right context = lookahead):

| `--att-context` | latency budget |
|-----------------|----------------|
| `56,0`          | lowest (default) |
| `56,3`          | small lookahead |
| `56,6`          | medium |
| `56,13`         | highest, best quality |

Each variant ships its own `metadata.json` with the matching `att_context_size`. Cache shapes (`cache_channel`, `cache_time`) are identical across variants because the conformer body is the same; only the right-context window changes.

### 3. (Optional) Dump lang-tag IDs separately

If you need the language-tag token list without re-running the full conversion (no torch/NeMo needed):

```bash
uv run python dump_lang_tags.py \
    --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
    --output lang_tag_token_ids.json
```

This is also embedded into `metadata.json` by the converter.

### 4. (Optional) Int8 encoder quantization

The encoder is the only large component (~2.4 GB). The sibling English package ships `quantize_encoder.py` which works unchanged on the multilingual encoder. Copy it next to the converted package and run:

```bash
uv run python ../../../nemotron-speech-streaming-0.6b/coreml/Streaming/scripts/quantize_encoder.py \
    --model-dir ./build_fp16 \
    --output-dir ./build_int8 \
    --granularity per_channel
```

Expected size: ~600 MB.

## Prompt Layouts

`multilingual_components.detect_prompt_layout(model)` returns:

- **`A`** — `model.encoder.forward` accepts a `prompt` (or `prompt_one_hot`/`prompt_emb`/`lang_prompt`) kwarg. Wrapper plumbs `prompt_id → one_hot → kwarg` through. Preferred path.
- **`B`** — Prompt applied outside the encoder. Wrapper calls `encoder.pre_encode`, concatenates the one-hot, runs `prompt_kernel`, then calls a body-only forward (`forward_internal` / `forward_after_pre_encode`). Used when the fork hasn't merged a kwarg into `ConformerEncoder.forward`.

In both layouts, the CoreML graph builds the one-hot internally so Swift only sends an int32. If the layout discovery picks B but neither body-only forward exists in your installed NeMo build, `EncoderStreamingWithExternalPrompt.forward` will raise — at which point fall back to manually patching the encoder.

## Python Inference + Benchmark (macOS)

These scripts run against the converted CoreML artifacts. They do **not**
need NeMo or torch — install the slim inference env at the top of this
directory:

```bash
cd mobius/models/stt/nemotron-asr-streaming-multilingual-0.6b/coreml
uv sync                 # uses ./pyproject.toml (coremltools + datasets + scipy)
```

### Single-file parity check

```bash
# PyTorch reference (must be run from conversion_scripts/ on a Linux+CUDA box)
uv run --project conversion_scripts python nemo_reference.py \
    --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
    --audio sample_zh.wav \
    --target-lang auto

# CoreML smoke test (macOS)
uv run python test_coreml_multilingual.py \
    --model-dir ./build_fp16 \
    --audio sample_zh.wav \
    --target-lang auto
```

Both print `detected_lang` and `text` separately. A healthy fp16
conversion produces the same lang tag and near-identical text. Re-run
with `--target-lang zh-CN` (or any other code from
`metadata.json["prompt_dictionary"]`) to exercise forced-language mode.

### FLEURS multi-language benchmark

`benchmark_fleurs.py` runs the CoreML pipeline across one or more FLEURS
test subsets and reports WER (Latin-script), CER (CJK/Thai), RTFx, and —
in `auto` mode — language-detection accuracy.

```bash
# Streamed from HuggingFace (recommended; no local download needed)
uv run python benchmark_fleurs.py \
    --model-dir ./build_fp16 \
    --use-hf \
    --languages cmn_hans_cn,en_us,es_419,ja_jp,de_de \
    --mode auto \
    --max-files-per-lang 100 \
    --output-json fleurs_auto.json

# Same suite in forced-language mode
uv run python benchmark_fleurs.py \
    --model-dir ./build_fp16 \
    --use-hf \
    --languages cmn_hans_cn,en_us,es_419,ja_jp,de_de \
    --mode forced \
    --max-files-per-lang 100 \
    --output-json fleurs_forced.json
```

Output table per language: `n`, `WER/CER`, `RTFx`, `detect%`
(auto mode only). The `detect%` column is the fraction of utterances
where the leading `<xx-XX>` token matched the FLEURS label — that's the
only sanity check on the implicit language-detection mechanism.

`fleurs_lang_map.py` defines the FLEURS↔Nemotron mapping for the 38
supported languages. Edit it if NVIDIA expands the prompt dictionary.

For a local FLEURS dump instead of HF streaming, pass
`--fleurs-root /path/to/fleurs` (expects `<lang>/test.tsv` +
`<lang>/audio/test/*.wav`).

## Swift Integration

The Swift-side counterpart lives at `FluidAudio/Sources/FluidAudio/ASR/Parakeet/Streaming/Nemotron/`. After conversion, the only Swift changes are:

| File | Change |
|------|--------|
| `NemotronStreamingConfig.swift` | add `numPrompts`, `promptDictionary`, `langTagTokenIds` |
| `StreamingNemotronAsrManager.swift` | accept `targetLang: String = "auto"` and resolve to int via dictionary |
| `StreamingNemotronAsrManager+Pipeline.swift` | feed `prompt_id` int32 input to encoder MLFeatureProvider |
| `Tokenizer.swift` | filter `langTagTokenIds` from `decode()` |

`metadata.json` carries all the data Swift needs.

## Verification Checklist

After conversion:

1. `metadata.json["num_prompts"] == 128`
2. `metadata.json["vocab_size"] == 13087`, `blank_idx == 13087`
3. `metadata.json["prompt_dictionary"]` has the 38 listed languages plus `auto: 101`
4. `metadata.json["lang_tag_token_ids"]` has **39 entries** (verified against this checkpoint) — `<bg-BG>` at id 1, `<en-US>` at 2947, `<zh-CN>` at 9847, `<vi-VN>` at 12944, etc. Every id maps to a `<xx-XX>`-style token. Note: these are scattered across the full 13,087-entry vocab, not bunched at the start
5. Encoder mlpackage exposes 6 inputs: `mel`, `mel_length`, `cache_channel`, `cache_time`, `cache_len`, `prompt_id`
6. Decoder + joint mlpackages are byte-similar in topology to the English variant (only output dim differs)

## License & Distribution

The source model is under the **NVIDIA Software and Model Evaluation License** and is currently marked "internal NVIDIA evaluation". Per the upstream README, contact `jaydar@nvidia.com` for access. Do **not** upload converted artifacts publicly without confirming redistribution terms. Local on-device use within VoiceLink is fine per project policy (`/Users/kikow/.claude/CLAUDE.md`, project `CLAUDE.md`).
