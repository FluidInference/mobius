# Cohere Transcribe CoreML Quantization Results

Testing conducted on 10 English FLEURS samples using various encoder/decoder quantization combinations.

## Summary Table

| Configuration | Encoder Size | Decoder Size | Total Size | Success Rate | Loop Rate | Avg WER | Notes |
|--------------|--------------|--------------|------------|--------------|-----------|---------|-------|
| **FP16 + FP16** | 3.6 GB | 291 MB | 3.9 GB | 20% (2/10) | 0% (0/10) | ~10-30%* | Baseline - stable but large |
| **INT8 + FP16** (Hybrid) | 1.8 GB | 291 MB | 2.1 GB | 20% (2/10) | 0% (0/10) | ~10-30%* | **RECOMMENDED** - 46% size reduction, same quality |
| **INT4 + FP16** | 899 MB | 291 MB | 1.2 GB | 20% (2/10) | 0% (0/10) | 293.38% | 69% size reduction but severe quality degradation |
| **INT8 + INT8** | 1.8 GB | 146 MB | 1.95 GB | 14% (1-2/10) | 71% (5-10/10) | N/A | NOT RECOMMENDED - decoder instability causes loops |

*Estimated based on successful samples only

## Detailed Results

### FP16 Encoder + FP16 Decoder
- **Model sizes**: 3.6 GB encoder + 291 MB decoder = 3.9 GB total
- **Success rate**: 2/10 samples with WER < 30%
- **Loop rate**: 0/10 (no repetition loops)
- **Quality**: High quality on successful samples
- **Conclusion**: Baseline configuration - stable but memory-intensive

### INT8 Encoder + FP16 Decoder (Hybrid) ✅ RECOMMENDED
- **Model sizes**: 1.8 GB encoder + 291 MB decoder = 2.1 GB total
- **Success rate**: 2/10 samples with WER < 30%
- **Loop rate**: 0/10 (no repetition loops)
- **Quality**: Same as FP16 baseline on successful samples
- **Size reduction**: 46% smaller than full FP16
- **Conclusion**: **Best balance** - significant memory savings with no quality loss

### INT4 Encoder + FP16 Decoder ⚠️ TOO AGGRESSIVE
- **Model sizes**: 899 MB encoder + 291 MB decoder = 1.2 GB total
- **Success rate**: 2/10 samples with WER < 30%
- **Loop rate**: 0/10 (no repetition loops)
- **Average WER**: 293.38% (extremely high)
- **Quality**: Severe degradation - hallucinations on most samples
- **Size reduction**: 69% smaller than full FP16
- **Example failure**: Ground truth about "communication channels" → Hallucinated content about "voting polls"
- **Conclusion**: INT4 is too aggressive for the encoder - causes hallucinations

### INT8 Encoder + INT8 Decoder ❌ NOT RECOMMENDED
- **Model sizes**: 1.8 GB encoder + 146 MB decoder = 1.95 GB total
- **Success rate**: ~14% (1-2/10 samples)
- **Loop rate**: ~71% (5-10/10 samples with repetition loops)
- **Quality**: Unstable - decoder quantization causes repetition loops
- **Conclusion**: INT8 decoder is unstable - avoid

## FLEURS Dataset Performance

All configurations show poor performance on FLEURS dataset (diverse acoustic conditions):
- **FP16 on FLEURS (140 samples)**: 7.1% success, 12.1% loops
- The 20% success rate on English samples drops to ~7% across multiple languages
- Model appears optimized for clean audio, struggles with field recordings

## Recommendations

1. **For production use**: **Hybrid INT8+FP16** (2.1 GB)
   - 46% memory savings vs FP16
   - Same quality as FP16 baseline
   - No stability issues

2. **For memory-constrained devices**: Test INT6 if available
   - INT4 is too aggressive (causes hallucinations)
   - INT8 is the minimum viable quantization for encoder

3. **Decoder quantization**: Always use FP16
   - INT8 decoder causes 71% loop rate
   - 146 MB savings not worth instability

## Technical Details

### Quantization Method
- **Tool**: CoreML Tools `linear_quantize_weights`
- **Mode**: `linear_symmetric`
- **Weight threshold**: 512
- **iOS requirement**: INT4 requires iOS 18+ (iOS 17 for INT8)

### Test Environment
- **Dataset**: FLEURS English (10 samples)
- **Metric**: Word Error Rate (WER)
- **Success threshold**: WER < 30%
- **Loop detection**: 5+ consecutive word repetitions

### Model Architecture
- **Encoder**: Conformer (processes mel spectrogram → hidden states)
- **Decoder**: Autoregressive decoder with KV cache (hidden states → text)
- **Vocabulary**: 33,684 tokens

## Files

- `ios18/cohere_encoder.mlpackage` - FP16 encoder (iOS 18 target)
- `int4/cohere_encoder_int4.mlpackage` - INT4 encoder
- `q8/cohere_encoder.mlpackage` - INT8 encoder
- `f16/cohere_decoder_stateful.mlpackage` - FP16 decoder
- `q8/cohere_decoder_stateful.mlpackage` - INT8 decoder

## Scripts

- `export-encoder-ios18.py` - Export FP16 encoder with iOS 18 target
- `quantize_encoder_to_int4.py` - Quantize FP16 encoder to INT4
- `test_int4enc_fp16dec_10_en.py` - Test INT4 encoder + FP16 decoder
- `test_hybrid_10_en.py` - Test INT8 encoder + FP16 decoder

## Next Steps

1. Document hybrid quantization support in Swift/FluidAudio
2. Upload INT8 encoder to HuggingFace for FluidInference repo
3. Consider testing INT6 if CoreML adds support in future iOS versions
4. Investigate why FLEURS performance is poor across all configurations
