# kokoro-v1.1-zh-coreml

7-stage CoreML conversion of [Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)
(Mandarin fine-tune of Kokoro-82M, n_token=178, Bopomofo + tone-digit vocab),
optimized for the Apple Neural Engine (~80 MB total).

Adapted from [`models/tts/kokoro/laishere-coreml/`](../../kokoro/laishere-coreml/)
(MIT, vendored from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml))
by switching the upstream checkpoint and G2P language code. Architecture
(ALBERT, predictor, text_encoder, decoder/generator, iSTFT) is identical to
the v1.0 English chain — only the embedding `vocab_size` (177 → 178) and
weights change, so the 7-stage trace + RangeDim shape bounds stay byte-for-byte
compatible.

## What you get

7 `.mlpackage` files plus auxiliary assets, all written into the directory
passed via `--output-dir`:

| Stage | Output                          | Precision             | Compute Units |
|-------|---------------------------------|-----------------------|---------------|
| 1     | `KokoroAlbert.mlpackage`        | fp16 + int8 palettize | `CPU_AND_NE`  |
| 2     | `KokoroPostAlbert.mlpackage`    | fp16 + int8 palettize | `CPU_AND_NE`  |
| 3     | `KokoroAlignment.mlpackage`     | fp16 + int8 palettize | `CPU_AND_NE`  |
| 4     | `KokoroProsody.mlpackage`       | fp16 + int8 palettize | `ALL`         |
| 5     | `KokoroNoise.mlpackage`         | fp32 + int8 palettize | `ALL`         |
| 6     | `KokoroVocoder.mlpackage`       | fp16 + int8 palettize | `CPU_AND_NE`  |
| 7     | `KokoroTail.mlpackage`          | fp32                  | `ALL`         |
|       | `vocab.json`                    | 171-entry Bopomofo+IPA+digit→token id map | — |
|       | `zf_001.bin` / `zm_009.bin`     | voice packs `[510, 256]` flat fp32    | — |
|       | `benchmark_data.json`           | precomputed phoneme cases for benchmark.py | — |
|       | `ref.wav` / `test.wav`          | parity validation pair (24 kHz mono)  | — |

The 7-stage shape bounds, op-translation patches, and per-stage compute-unit
choices are inherited verbatim from the v1.0 chain. See
[`docs/architecture.md`](docs/architecture.md) and
[`docs/shape-bounds.md`](docs/shape-bounds.md) for the underlying details.

## Differences from v1.0 English chain

| Item                | v1.0 English             | v1.1-zh Mandarin            |
|---------------------|--------------------------|-----------------------------|
| Upstream checkpoint | `hexgrad/Kokoro-82M`     | `hexgrad/Kokoro-82M-v1.1-zh` |
| Vocab size          | 177 (IPA + arrow tones)  | 171 (IPA + Bopomofo `ㄅㄆㄇ` + tone digits `1-5`) |
| `lang_code`         | `'a'` (English)          | `'z'` (Mandarin)            |
| G2P backend         | `misaki.en` (espeak fallback) | `misaki.zh` (jieba + pypinyin → Bopomofo+digit) |
| Default voice       | `af_heart`               | `zf_001` (female) / `zm_009` (male) |
| Voice pack count    | 1 packaged (af_heart.bin) | 2 packaged (zf_001.bin, zm_009.bin) — full 96-voice set on upstream HF |

Architecture, layer counts, hidden sizes, and 7-stage trace boundaries are
unchanged.

## Setup

```bash
cd mobius/models/tts/kokoro-v1.1-zh/coreml
uv sync
# `uv sync` may resolve coremltools to the pure-python sdist (Tag: py3-none-any)
# which is missing the native libs and breaks ct.convert(...) with
# `RuntimeError: BlobWriter not loaded`. Force the platform wheel:
uv pip install --reinstall coremltools==9.0
```

Verify the wheel is the platform one (`cp311-none-macosx_11_0_arm64`), not pure
python (`py3-none-any`):

```bash
uv run python -c "import coremltools, os; print(os.path.dirname(coremltools.__file__))"
ls .venv/lib/python3.11/site-packages/coremltools/libcoremlpython.so   # must exist
```

## Convert

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python convert-coreml.py --output-dir build/kokoro-v1.1-zh
```

The script downloads `hexgrad/Kokoro-82M-v1.1-zh` on first run, runs
misaki[zh] G2P on a Mandarin trace sentence (`你好世界，今天天气很好。`),
and emits the 7 `.mlpackage` files with the same RangeDim bounds as v1.0
(`T_enc≤512`, `T_a≤2000`, `max_T2≤4000`).

Re-running with `--stages tail prosody` only converts the listed stages and
reuses existing mlpackages on disk for the E2E chain check.

## Validate parity

```bash
uv run python compare-models.py --models-dir build/kokoro-v1.1-zh \
    --text "今天天气真好，阳光明媚。" --voice zf_001 \
    --save-ref /tmp/ref.wav --save-coreml /tmp/cm.wav
```

Pass criteria: waveform corr ≥ 0.80, mel-spectrogram corr ≥ 0.99 (mobius
standard, matches the v1.0 English chain).

## End-to-end synthesis

```bash
# Mandarin text → WAV (uses misaki[zh] G2P)
uv run python inference.py --models-dir build/kokoro-v1.1-zh \
    --text "你好世界" --voice zf_001 --output /tmp/zf_001.wav

uv run python inference.py --models-dir build/kokoro-v1.1-zh \
    --text "你好世界" --voice zm_009 --output /tmp/zm_009.wav

# Pre-computed Bopomofo+digit phonemes → WAV (skips G2P, matches iOS app flow)
uv run python inference.py --models-dir build/kokoro-v1.1-zh \
    --phonemes "ㄋㄧˇㄏㄠˇ" --voice zf_001 --output /tmp/hello.wav
```

Per-stage timings are printed to stderr.

## Voice packs

The conversion + dump scripts ship with two voices to keep the artifact
bundle small:

| Voice id  | Gender | Source                                                                                  |
|-----------|--------|-----------------------------------------------------------------------------------------|
| `zf_001`  | Female | [`hexgrad/Kokoro-82M-v1.1-zh/voices/zf_001.pt`](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/blob/main/voices/zf_001.pt) |
| `zm_009`  | Male   | [`hexgrad/Kokoro-82M-v1.1-zh/voices/zm_009.pt`](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/blob/main/voices/zm_009.pt) |

Upstream provides the full 96-voice set (49 `zf_*` + 47 `zm_*` + 3 EN). To
regenerate the bin files for additional voices, edit `DEFAULT_VOICES` in
`dump-benchmark-data.py` (or pass `--voices zf_001 zm_009 zf_002 …`) and
re-run.

```bash
uv run python dump-benchmark-data.py --output-dir build/kokoro-v1.1-zh \
    --voices zf_001 zm_009
```

## Benchmark

```bash
# Seed precomputed phonemes + voice packs first (one time, after editing TEXTS)
uv run python dump-benchmark-data.py --output-dir build/kokoro-v1.1-zh

# Latency: per-stage median + chain time + speedup, over a 6-sentence Mandarin corpus
uv run python benchmark.py --models-dir build/kokoro-v1.1-zh --voice zf_001
```

## Shape bounds

Inherited verbatim from the v1.0 chain — see [`docs/shape-bounds.md`](docs/shape-bounds.md).
Mandarin phoneme strings tend to be denser per character than English (each
Hanzi typically expands to ~2 Bopomofo + 1 tone digit), so a single sentence
fills `T_enc` faster — keep an eye on the `T_a` probe in benchmark.py for
the long-passage case.

| Symbol           | Meaning                          | Default upper bound |
|------------------|----------------------------------|---------------------|
| `T_enc`          | phoneme sequence length          | 512                 |
| `T_a`            | aligned acoustic frames          | 2000                |
| `T_ns0`, `T_ns1` | noise source bank sizes          | derived             |
| `T_pre`          | pre-iSTFT frames into Tail       | derived             |

## HuggingFace upload (manual)

This directory does **not** push to HF. After conversion, compile the
`.mlpackage`s into `.mlmodelc`s and stage them locally:

```bash
mkdir -p build/kokoro-v1.1-zh-hf
for mlp in build/kokoro-v1.1-zh/Kokoro*.mlpackage; do
    xcrun coremlcompiler compile "$mlp" build/kokoro-v1.1-zh-hf/
done
cp -R build/kokoro-v1.1-zh/Kokoro*.mlpackage build/kokoro-v1.1-zh-hf/
cp build/kokoro-v1.1-zh/{vocab.json,benchmark_data.json} build/kokoro-v1.1-zh-hf/
cp build/kokoro-v1.1-zh/{zf_001.bin,zm_009.bin} build/kokoro-v1.1-zh-hf/

# Suggested HF repo (user uploads, never the assistant):
#   huggingface-cli upload FluidInference/kokoro-82m-v1.1-zh-coreml \
#       ./build/kokoro-v1.1-zh-hf/ .
```

## License

MIT — see `LICENSE`. Original conversion © laishere; v1.0 mobius adaptation +
v1.1-zh switch © FluidInference. Underlying Kokoro-82M-v1.1-zh model
© hexgrad (Apache-2.0).
