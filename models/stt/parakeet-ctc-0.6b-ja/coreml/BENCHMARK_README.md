# Benchmark Instructions - Parakeet CTC Japanese

## Benchmark Script

**Script**: `benchmark-fleurs-ja.py`

Benchmark the Japanese Parakeet CTC CoreML model on FLEURS Japanese (650 samples).

## Requirements

The benchmark script is ready to run. All dependencies are already in `pyproject.toml`.

Uses **streaming mode** to avoid downloading all 21k+ FLEURS files - only Japanese samples are loaded.

## Usage

```bash
# Quick test with 10 samples
uv run python benchmark-fleurs-ja.py --num-samples 10

# Full Japanese test set (650 samples)
uv run python benchmark-fleurs-ja.py

# Save detailed results to JSON
uv run python benchmark-fleurs-ja.py --output-file results.json

# Custom build directory
uv run python benchmark-fleurs-ja.py --build-dir ./build --num-samples 50
```

## Dataset

- **Source**: `FluidInference/fleurs-full`
- **Japanese Samples**: 650 samples (ja_jp)
- **License**: CC-BY 4.0 (free to use)
- **Audio**: 16kHz mono WAV
- **Language**: Japanese
- **Loading**: Streaming mode (efficient, no full download required)

## Expected Performance

Based on the NeMo paper benchmarks, the CTC decoder should achieve:

| Dataset | Expected CER |
|---------|--------------|
| FLEURS Japanese | ~10-13% |

**Note**: The paper reported 6.5% CER on JSUT basic5000 and 13.3% on Mozilla Common Voice 16.1 test for the CTC decoder.

## Output Metrics

The benchmark reports:

- **Average CER** (Character Error Rate)
- **Total Audio Duration**
- **Total Latency**
- **Average Latency/Sample**
- **Average RTFx** (Real-Time Factor - higher is better)
- **Total RTFx**

Example output:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric                 ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Samples                │ 650       │
│ Average CER            │ 12.34%    │
│ Total Audio Duration   │ 2345.67s  │
│ Total Latency          │ 234.56s   │
│ Average Latency/Sample │ 360.86ms  │
│ Average RTFx           │ 3.45x     │
│ Total RTFx             │ 10.00x    │
└────────────────────────┴───────────┘
```

## Important Implementation Details

### Raw Logits + log_softmax

The Japanese CTC model outputs **raw logits** (not log-probabilities). The benchmark script applies `log_softmax` before CTC decoding:

```python
# Apply log_softmax (CRITICAL for Japanese model!)
logits_max = np.max(raw_logits, axis=-1, keepdims=True)
logits_shifted = raw_logits - logits_max  # Numerical stability
exp_logits = np.exp(logits_shifted)
sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
log_probs = logits_shifted - np.log(sum_exp)
```

This is because the CoreML conversion had issues with `log_softmax` (see `CONVERSION_NOTES.md` for details).

### Greedy CTC Decoding

The benchmark uses greedy CTC decoding:
1. Take argmax of log-probabilities at each timestep
2. Collapse consecutive repeats
3. Remove blank tokens
4. Convert token IDs to text using SentencePiece vocabulary

## Character Error Rate (CER)

For Japanese, we use **Character Error Rate** instead of Word Error Rate because:
- Japanese doesn't have clear word boundaries
- Characters (including kanji, hiragana, katakana) are the natural unit
- All spaces are removed before computing edit distance

## Troubleshooting

### Streaming Mode

The script uses **streaming mode** which:
- ✅ Only downloads Japanese samples as needed
- ✅ Avoids downloading all 21k+ FLEURS files
- ✅ No rate limit issues (only ~650 files downloaded)

### Dataset Download Issues

If download fails:
1. Check internet connection
2. Try with fewer samples first: `--num-samples 10`
3. The script streams samples incrementally, so partial progress is saved

### Model Not Found

Ensure you've run the conversion first:
```bash
uv run python convert-parakeet-ja.py --output-dir ./build
```

The benchmark expects these files:
- `build/Preprocessor.mlpackage`
- `build/Encoder.mlpackage`
- `build/CtcDecoder.mlpackage`
- `build/vocab.json`

## Comparison with Paper Results

The original NeMo paper reports these CER scores for the CTC decoder:

| Dataset | CER |
|---------|-----|
| JSUT basic5000 | 6.5% |
| Mozilla Common Voice 8.0 test | 7.2% |
| Mozilla Common Voice 16.1 dev | 10.2% |
| Mozilla Common Voice 16.1 test | 13.3% |
| TEDxJP-10k | 9.1% |

FLEURS Japanese test should fall in the 10-13% range based on these benchmarks.

## Next Steps

After running the benchmark:

1. **Compare with NeMo results**: Run the NeMo model on the same FLEURS test set to verify CoreML accuracy
2. **Beam search decoding**: Implement beam search for potentially better accuracy (greedy is baseline)
3. **Language model integration**: Add n-gram or neural LM for improved results
4. **Other datasets**: Test on Mozilla Common Voice, JSUT, or ReazonSpeech for broader evaluation

## References

- **Model**: https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
- **Dataset**: https://huggingface.co/datasets/FluidInference/fleurs-full
- **Original FLEURS**: https://huggingface.co/datasets/google/fleurs
- **FLEURS Paper**: https://arxiv.org/abs/2205.12446
