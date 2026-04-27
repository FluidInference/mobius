# kokoro-laishere-coreml

7-stage CoreML conversion of [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)
optimized for the Apple Neural Engine (~80 MB total, ~20× real-time on M-series).

Vendored from [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml)
(MIT) and adapted to mobius CLI conventions (`--output-dir` / `--models-dir`).

## What you get

7 `.mlpackage` files plus auxiliary assets, all written into the directory
passed via `--output-dir`:

| Stage | Output | Precision | Compute Units |
|---|---|---|---|
| 1 | `KokoroAlbert.mlpackage`     | fp16 + int8 palettize | `CPU_AND_NE` |
| 2 | `KokoroPostAlbert.mlpackage` | fp16 + int8 palettize | `CPU_AND_NE` |
| 3 | `KokoroAlignment.mlpackage`  | fp16 + int8 palettize | `CPU_AND_NE` |
| 4 | `KokoroProsody.mlpackage`    | fp16 + int8 palettize | `ALL` |
| 5 | `KokoroNoise.mlpackage`      | fp32 + int8 palettize | `ALL` |
| 6 | `KokoroVocoder.mlpackage`    | fp16 + int8 palettize | `CPU_AND_NE` |
| 7 | `KokoroTail.mlpackage`       | fp32                  | `ALL` |
|   | `vocab.json`                 | 177-entry IPA→token id map  | — |
|   | `af_heart.bin`               | voice pack `[510, 256]` flat fp32 | — |
|   | `benchmark_data.json`        | precomputed phoneme cases for benchmark.py | — |
|   | `ref.wav` / `test.wav`       | parity validation pair (24 kHz mono) | — |

## Setup

```bash
cd mobius/models/tts/kokoro/laishere-coreml
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
uv run python convert-coreml.py --output-dir build/laishere-kokoro
```

Re-running with `--stages tail prosody` only converts the listed stages and
reuses existing mlpackages on disk for the E2E chain check.

Reported numbers on M2 Max (verified during initial conversion):

```
[E2E] corr=0.805583, mel_corr=0.993527, chain=460.6 ms
```

## Validate parity

```bash
uv run python compare-models.py --models-dir build/laishere-kokoro \
    --save-ref /tmp/ref.wav --save-coreml /tmp/cm.wav
```

Pass criteria: waveform corr ≥ 0.80, mel-spectrogram corr ≥ 0.99.

## End-to-end synthesis

```bash
# Text → WAV (uses Kokoro G2P)
uv run python inference.py --models-dir build/laishere-kokoro \
    --text "Hello world" --output /tmp/hello.wav

# Pre-computed IPA → WAV (skips G2P, matches the iOS app flow)
uv run python inference.py --models-dir build/laishere-kokoro \
    --phonemes "həlˈoʊ wˈɜːld" --output /tmp/hello.wav
```

Per-stage timings are printed to stderr.

## Benchmark

```bash
# Seed precomputed phonemes + voice pack first (one time, after editing TEXTS)
uv run python dump-benchmark-data.py --output-dir build/laishere-kokoro

# Latency: per-stage median + chain time + speedup, over a 6-sentence corpus
uv run python benchmark.py --models-dir build/laishere-kokoro
```

Target: ≥ 20× real-time chain throughput on M-series.

## Shape bounds

Derived from `--max-frames` (default 2000). See `docs/shape-bounds.md` for the
full T_enc/T_a/T_ns/T_pre relationship table.

| Symbol | Meaning | Default upper bound |
|---|---|---|
| `T_enc` | phoneme sequence length (incl. BOS/EOS) | 512 |
| `T_a`   | aligned acoustic frames                  | 2000 |
| `T_ns0`, `T_ns1` | noise source bank sizes         | derived |
| `T_pre` | pre-iSTFT frames into Tail               | derived |

## HuggingFace upload (manual)

This directory does **not** push to HF. After conversion, compile the
`.mlpackage`s into `.mlmodelc`s and stage them locally:

```bash
mkdir -p build/laishere-kokoro-hf/ANE
for mlp in build/laishere-kokoro/Kokoro*.mlpackage; do
    xcrun coremlcompiler compile "$mlp" build/laishere-kokoro-hf/ANE/
done
cp -R build/laishere-kokoro/Kokoro*.mlpackage build/laishere-kokoro-hf/ANE/
cp build/laishere-kokoro/{vocab.json,af_heart.bin} build/laishere-kokoro-hf/ANE/

# This variant lives under the existing kokoro repo as the `ANE/` subdirectory
# (sibling to the single-graph export). Upload:
#   huggingface-cli upload FluidInference/kokoro-82m-coreml \
#       ./build/laishere-kokoro-hf/ANE/ ANE/
```

## License

MIT — see `LICENSE`. Original work © laishere; adaptations © FluidInference.
Underlying Kokoro model © hexgrad (Apache-2.0).
