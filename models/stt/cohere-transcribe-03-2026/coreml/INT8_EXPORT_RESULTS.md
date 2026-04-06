# INT8 Model Export Results (35-Second Window)

## Issue Fixed

**Critical bug:** The decoder was hardcoded to accept 376 encoder outputs (from 3001 frames), but the official encoder uses 3500 frames (35 seconds) which produces 438 encoder outputs.

**Fix:** Updated `export-decoder-stateful.py` to accept 438 encoder outputs to match the 3500 frame encoder.

## Model Specifications

### Architecture
- **Encoder input:** 3500 frames (35 seconds @ 10ms/frame)
- **Encoder output:** (1, 438, 1024) - 438 sequence length
- **Decoder cache:** 108 tokens max
- **Vocabulary:** 16,384 tokens

### Model Sizes
| Component | FP16 | INT8 | Compression |
|-----------|------|------|-------------|
| Encoder | 3.58 GB | 1.82 GB | 1.97x |
| Decoder | 290.5 MB | 145.8 MB | 1.99x |
| **Total** | **3.87 GB** | **1.97 GB** | **1.97x** |

### Quantization Settings
```python
OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int8",
    granularity="per_channel",  # Per-channel quantization
)
```

## Quality Test Results

**Test dataset:** LibriSpeech test-clean, 10 samples (3.3-23.3 seconds)

### Overall Metrics
- **Average WER:** 25.2%
- **Average duration:** 9.2s
- **Perfect matches:** 0 / 10 (0%)
- **Verdict:** ⚠️ NEED REVIEW

### Per-Sample Results

| Sample | Duration | WER | Quality |
|--------|----------|-----|---------|
| 1 | 3.50s | 12.5% | ✅ Excellent |
| 2 | 14.22s | 9.3% | ✅ Excellent |
| 3 | 5.03s | 9.1% | ✅ Excellent |
| 4 | 23.32s | **110.9%** | ❌ Complete failure |
| 5 | 11.06s | 16.1% | ⚠️ Moderate |
| 6 | 13.16s | 15.2% | ⚠️ Moderate |
| 7 | 5.85s | 17.6% | ⚠️ Moderate |
| 8 | 3.31s | 22.2% | ⚠️ Moderate |
| 9 | 4.79s | 18.2% | ⚠️ Moderate |
| 10 | 7.28s | 20.8% | ⚠️ Moderate |

### Critical Failure Case

**Sample 4 (23.32s):**
```
Reference:  "from the respect paid her on all sides she seemed like a queen..."
Hypothesis: "the world is a very important part of the world. and the world is a ve..."
WER: 110.9%
```

This sample produced complete gibberish, suggesting INT8 quantization degrades quality on longer audio.

## Analysis

### Meets Requirements
- ✅ Average WER < 30% (25.2%)
- ✅ 2x model size reduction
- ✅ Working 35-second window

### Does NOT Meet Requirements
- ❌ Perfect matches < 60% (0%)
- ❌ Unstable on long audio (23s sample failed)
- ❌ No perfect transcriptions even on short samples

### Quality Degradation Patterns
1. **Short audio (3-5s):** Good quality (9-22% WER)
2. **Medium audio (11-14s):** Moderate quality (9-16% WER)
3. **Long audio (23s):** Complete failure (110% WER)

## Recommendations

### Option 1: Upload FP16 Models (Recommended)
**Pros:**
- Known quality: 23.76% WER, 64% perfect matches
- Stable on long audio
- Matches official model precision

**Cons:**
- Larger size: 3.87 GB vs 1.97 GB
- Still has encoder bias issues (quiet/high-pitched voices)

### Option 2: Accept INT8 Quality
**Pros:**
- 2x size reduction (1.97 GB)
- Acceptable WER for short-medium audio

**Cons:**
- No perfect matches
- Unstable on long audio
- Quality degradation from FP16

### Option 3: Investigate Quantization
**Possible improvements:**
- Try `per_tensor` granularity (more stable but lower quality)
- Try `linear_symmetric` with different calibration
- Quantize encoder only, keep decoder FP16

## Files Exported

### FP16 Models (35-second window)
```
build-35s/cohere_encoder.mlpackage           # 3.58 GB
build-35s/cohere_decoder_stateful.mlpackage  # 290.5 MB
```

### INT8 Models (35-second window)
```
build-35s-int8/cohere_encoder_int8.mlpackage           # 1.82 GB
build-35s-int8/cohere_decoder_stateful_int8.mlpackage  # 145.8 MB
```

## Next Steps

**Decision needed:** Which models to upload to HuggingFace?

1. **FP16 only** - Best quality, known performance
2. **INT8 only** - Smallest size, acceptable for short audio
3. **Both** - Let users choose based on their needs

## Known Issues

From previous investigation (INVESTIGATION_SUMMARY.md):
- Encoder struggles with quiet speakers (RMS < 0.03)
- Encoder struggles with high-pitched voices (>1000 Hz)
- 36% of samples fail regardless of precision
- This is a model training data bias issue, not a conversion issue
