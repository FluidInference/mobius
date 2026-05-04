# g2pW → CoreML

CoreML conversion for [g2pW](https://github.com/GitYCC/g2pW) — a BERT-based
Mandarin polyphone disambiguator. Picks the right reading for ambiguous
Hanzi (e.g. 行 → `xíng` vs `háng`) using sentence context.

Source paper: [G2PW: A Conditional Weighted Softmax BERT for Polyphone
Disambiguation in Mandarin (Chen et al., Interspeech 2022)](https://arxiv.org/abs/2203.10430).

## Pipeline

```
upstream ONNX  ──onnx2torch──▶  torch.nn.Module
                                       │
                                       ▼
                                 torch.jit.trace (B=1, L=512)
                                       │
                                       ▼
                          coremltools.convert (mlprogram, fp16)
                                       │
                                       ▼
                              build/g2pw/g2pw.mlpackage
```

Trace target is `.CpuOnly` per [mobius/CLAUDE.md](../../../CLAUDE.md);
runtime can override via `MLModelConfiguration.computeUnits`.

## Inputs

| name             | shape          | dtype   | meaning                                       |
|------------------|----------------|---------|-----------------------------------------------|
| `input_ids`      | `(1, 512)`     | int32   | BERT WordPiece IDs (Chinese vocab)            |
| `token_type_ids` | `(1, 512)`     | int32   | Segment IDs (always zero for single-sentence) |
| `attention_mask` | `(1, 512)`     | int32   | 1 for real tokens, 0 for padding              |
| `char_ids`       | `(1,)`         | int32   | Polyphonic-char ID for the conditional bias   |
| `position_ids`   | `(1,)`         | int32   | Index of the target char in the BERT sequence |
| `phoneme_mask`   | `(1, N_LABEL)` | float32 | 1 for valid pinyin labels for this char       |

`N_LABEL` is read from the upstream ONNX (~600 labels in the v2 model).

## Outputs

| name      | shape          | dtype   | meaning                                |
|-----------|----------------|---------|----------------------------------------|
| `probs`   | `(1, N_LABEL)` | float32 | Weighted-softmax probability per label |

Pick the reading via `argmax(probs)`, look up the bopomofo or pinyin
string in `POLYPHONIC_CHARS.txt`.

## Build

```bash
uv sync
uv run python convert-coreml.py --output-dir ./build/g2pw
```

This:
1. Downloads `https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v2-onnx.zip`
   to `~/.cache/g2pw-coreml/` (skipped if cached).
2. Extracts the upstream `G2PWModel/` directory.
3. Round-trips the ONNX graph through `onnx2torch` for tracing.
4. Traces with fixed-shape dummy inputs (`B=1, L=512`).
5. Emits `build/g2pw/g2pw.mlpackage` (fp16 mlprogram).
6. Copies side files (`POLYPHONIC_CHARS.txt`, `MONOPHONIC_CHARS.txt`,
   `config.py`, `version`) into `build/g2pw/`. The bopomofo→pinyin dict
   files are NOT shipped in the v2 ONNX archive — downstream Swift
   code maintains its own bopomofo lookup tables.

## Validate

Numerical parity vs upstream ONNX (CPU EP):

```bash
uv run python compare-models.py \
    --cache-dir ~/.cache/g2pw-coreml \
    --coreml-dir ./build/g2pw
```

Asserts:
- Top-1 argmax matches between ONNX and CoreML (the hard contract —
  picking the wrong pinyin label would change the synthesised audio).
- Per-label probability vectors stay within `atol=2e-2` (configurable).
  Empirically the fp16 round-trip lands at max diff ~1.2e-2, mean ~2e-5,
  L2 ~1.4e-2 on the seeded dummy batch.

Standalone CoreML smoke test (no ONNX needed):

```bash
uv run python test.py --coreml-dir ./build/g2pw
```

Asserts the output prob vector sums to ~1.

Real-world polyphone disambiguation (downloads `bert-base-chinese`
tokenizer on first run):

```bash
uv run python test_real.py --coreml-dir ./build/g2pw
```

Feeds 15 traditional-Chinese sentences exercising 行 / 長 / 重 / 都 /
覺 in disambiguating contexts and asserts top-1 picks the correct
bopomofo label per case. All 15 should land at confidence ≈ 1.000.

## Quantize

The baseline is fp16 (`compute_precision=FLOAT16` in convert-coreml.py).
Use `quantize.py` to emit smaller variants from the fp16 baseline:

```bash
# int8 linear per-channel (recommended): 2.0x shrink, near-lossless
uv run python quantize.py --mode linear-per-channel --out-dir ./build/g2pw-int8

# 6-bit kmeans palette: 2.7x shrink, slight drift
uv run python quantize.py --mode palettize-6bit --out-dir ./build/g2pw-pal6

# 4-bit kmeans palette: 4.0x shrink, larger drift
uv run python quantize.py --mode palettize-4bit --out-dir ./build/g2pw-pal4
```

Measured on this checkpoint:

| variant                 | size  | shrink | synthetic argmax vs ONNX | max diff | real polyphone* |
|-------------------------|-------|--------|--------------------------|----------|-----------------|
| fp16 (baseline)         | 303 MB| 1.0x   | match                    | 1.2e-2   | 15/15           |
| int8 linear per-channel | 152 MB| 2.0x   | match                    | 2.0e-2   | 15/15           |
| 6-bit kmeans palette    | 114 MB| 2.7x   | **flip**                 | 7.0e-2   | 15/15           |
| 4-bit kmeans palette    |  76 MB| 4.0x   | match                    | 1.7e-1   | 15/15           |

\* `test_real.py` 15-case suite. Real polyphone decisions hit confidence
≈ 1.000 against alternatives at ≈ 0.000, so even the 4-bit drift never
flips the chosen label. The 6-bit synthetic argmax flip is on a
random-`phoneme_mask` batch where competing labels sit very close — a
worst case that doesn't occur in production text.

**Recommendation**: int8 linear per-channel (`g2pw-int8`) is the
default ship target — half the size, no observed regressions on either
the synthetic or real-world batches.

## Profile

From `mobius/tools/coreml-cli/`:

```bash
uv run coreml-cli ../../models/segment-text/g2pw/coreml/build/g2pw/g2pw.mlpackage --fallback
```

`--fallback` surfaces any ANE-incompatible ops grouped by rejection
reason — useful for guiding follow-up optimization passes if BERT layers
land on CPU more than expected.

## Upload

After validation passes, upload `build/g2pw-int8/` to HuggingFace as
`FluidInference/g2pw-coreml` (the recommended ship variant — 152 MB,
no accuracy regression). The Swift integration in
`Sources/FluidAudio/TTS/KokoroAne/G2P/Mandarin/` will fetch it via
`KokoroAneResourceDownloader.ensureG2pwModel(...)` (separate PR — see
issue #572 item 1).

For folks who want the fp16 baseline (e.g. for re-quantization
experiments), upload `build/g2pw/` alongside under a `fp16/` subdir.

## License & attribution

g2pW is released under Apache 2.0 (upstream `LICENSE`). The converted
mlpackage inherits the same license. Cite the original paper when
shipping downstream.

## Known caveats

- The released ONNX bakes POS conditioning into the weights — there is
  no `pos_ids` input. If a future release exposes runtime POS as an
  input, the trace shape contract above grows by one tensor.
- Fixed `seq_len=512`. Sentences that exceed 510 BERT WordPieces (after
  CLS / SEP) must be windowed in the caller (matches upstream behaviour).
- `compute_precision=FLOAT16` — drop to `FLOAT32` if you observe
  argmax flips on long-tail labels. fp16 round-trip empirically lands at
  max diff ~1.2e-2 on the seeded parity batch, well under the
  ~0.5 confidence margin typical of polyphone decisions.
- **Trained on traditional Chinese**: `POLYPHONIC_CHARS.txt` contains
  3582 traditional-Hanzi entries with 1305 bopomofo labels. Simplified
  characters that don't share their glyph with traditional (e.g. 长 vs
  長, 觉 vs 覺) won't be in `chars` and must be converted (e.g. via
  OpenCC) before invoking the model.
