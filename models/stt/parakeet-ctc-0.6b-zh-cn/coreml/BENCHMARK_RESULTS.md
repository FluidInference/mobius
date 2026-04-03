# Parakeet CTC 0.6B zh-CN Benchmark Results

## THCHS-30 Test Set Benchmarks

### Configuration
- **Dataset:** THCHS-30 test set (Mandarin Chinese)
- **Samples:** 100 samples (for validation)
- **Model:** parakeet-ctc-0.6b-zh-cn
- **Platform:** Apple Silicon (M-series)

### Results Summary

#### INT8 Quantized Encoder (571 MB)
- **Mean CER:** 8.37%
- **Median CER:** 6.67%
- **Mean Latency:** 1,284.6 ms
- **CER Distribution:**
  - <5%: 32 samples (32.0%)
  - <10%: 69 samples (69.0%)
  - <20%: 92 samples (92.0%)

#### FP16 Encoder (1.1 GB)
- **Mean CER:** ~8.3% (similar performance to INT8)
- **Mean Latency:** ~1,250 ms

### Comparison with FLEURS

| Dataset | Samples | Mean CER | Notes |
|---------|---------|----------|-------|
| **THCHS-30** | 100 | **8.37%** | Clean Chinese corpus |
| FLEURS zh_cn | 100 | 10.22% | Some English contamination |

THCHS-30 shows **18% better CER** compared to FLEURS, likely due to cleaner data without transcription quality issues.

### Dataset Information

- **THCHS-30:** http://www.openslr.org/18/
- **Paper:** https://arxiv.org/abs/1512.01882
- **HuggingFace Dataset:** https://huggingface.co/datasets/FluidInference/THCHS-30-tests
- **License:** Apache 2.0
- **Total Test Samples:** 2,495 utterances (10 speakers)

### Text Normalization

Character Error Rate (CER) calculated using:
- Chinese punctuation removal
- English punctuation removal
- Arabic digit to Chinese character conversion (0→零, 1→一, etc.)
- Whitespace normalization
- Levenshtein distance for edit distance calculation

### Running the Benchmark

```bash
# FluidAudio CLI (Swift)
swift run -c release fluidaudiocli ctc-zh-cn-benchmark \
  --auto-download \
  --samples 100 \
  --output results.json

# Python benchmark script
python Scripts/benchmark_ctc_zh_cn.py --max-samples 100
```

### CI Integration

GitHub Actions workflow automatically runs 100-sample benchmark on every PR:
- Auto-downloads THCHS-30 from HuggingFace
- Validates CER < 10% threshold
- Posts results as PR comment

See: `.github/workflows/ctc-zh-cn-benchmark.yml`
