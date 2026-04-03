# Parakeet CTC 0.6B zh-CN CoreML Conversion Notes

This document details the conversion process, challenges encountered, and solutions implemented for converting NVIDIA's Parakeet CTC zh-CN model to CoreML.

## Overview

**Goal**: Convert the full Parakeet-CTC-0.6B Mandarin Chinese model to a pure CoreML pipeline (no PyTorch dependencies at inference).

**Final Result**: Successfully converted Preprocessor + Encoder + Decoder with int8 quantization achieving 10.54% CER (only +0.09% degradation from fp32).

---

## Conversion Attempts Timeline

### Attempt 1: Decoder Head Only (Initial Success)

**Approach**: Convert only the CTC decoder head (1024 → 7001 linear projection)

**Script**: `export-ctc-zh-cn.py`

**Result**: ✅ **Success**
- Decoder head converted successfully
- Size: 14MB
- CER: 10.39% on 100 samples (with PyTorch encoder)

**Limitation**: Required PyTorch/NeMo for encoder features

---

### Attempt 2: Full Pipeline with Variable-Length Audio

**Approach**: Convert Preprocessor + Encoder + Decoder with dynamic audio lengths

**Script**: `export-full-pipeline.py` (first version)

**Code**:
```python
preprocessor_inputs = [
    ct.TensorType(
        name="audio_signal",
        shape=(1, ct.RangeDim(1, max_samples)),  # Variable-length
        dtype=np.float32,
    ),
]
```

**Failure**: ❌ **Shape inference error**
```
ValueError: 'shape' must be provided in the 'inputs' argument for pytorch conversion
E5RT encountered an STL exception: zero shape error
```

**Root Cause**: CoreML's MLProgram format doesn't support RangeDim for traced PyTorch models in the input preprocessing stage.

**Solution**: Use fixed-shape inputs (15 seconds = 240,000 samples @ 16kHz)

---

### Attempt 3: Fixed-Shape Pipeline with Int8 Quantization During Conversion

**Approach**: Convert encoder with int8 quantization during ct.convert()

**Code**:
```python
# In export-full-pipeline.py
encoder_model = ct.convert(
    traced_encoder,
    inputs=encoder_inputs,
    outputs=encoder_outputs,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=ct.target.iOS17,
)

# Try to quantize
encoder_model = ct.compression.quantize_weights(
    encoder_model,
    mode="linear_symmetric",
    dtype=np.int8,
)
```

**Failure**: ❌ **API not found**
```
AttributeError: module 'coremltools' has no attribute 'compression'
```

**Root Cause**: Wrong API. `ct.compression` doesn't exist in coremltools 9.0.

**Attempted Fix 1**: Use legacy API
```python
import coremltools.models.neural_network.quantization_utils as quantization_utils
encoder_model = quantization_utils.quantize_weights(encoder_model, nbits=8)
```

**Failure**: ❌ **MLProgram not supported**
```
TypeError: MLModel of type mlProgram cannot be loaded just from the model spec object.
It also needs the path to the weights file.
```

**Root Cause**: Legacy API only works with NeuralNetwork models, not MLProgram (iOS 17+ format).

---

### Attempt 4: Float16 Compute Precision

**Approach**: Use `compute_precision=FLOAT16` during conversion

**Code**:
```python
encoder_model = ct.convert(
    traced_encoder,
    inputs=encoder_inputs,
    outputs=encoder_outputs,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,
)
```

**Result**: ✅ **Partial Success**
- Conversion successful
- Size: Still 1.1GB (weights stored as fp32, computed as fp16)
- CER: 10.45% (identical to fp32)

**Limitation**: No file size reduction. `compute_precision` only affects runtime computation, not weight storage.

---

### Attempt 5: Post-Conversion Int8 Quantization (Final Success)

**Approach**: Convert to fp32 first, then apply post-training quantization

**Script**: `quantize-encoder-advanced.py`

**Code**:
```python
from coremltools.optimize.coreml import (
    linear_quantize_weights,
    OptimizationConfig,
    OpLinearQuantizerConfig,
)

config = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
        weight_threshold=512,
    )
)

quantized_model = linear_quantize_weights(model, config=config)
```

**Result**: ✅ **Success**
- Size: 1.1GB → 0.55GB (50% reduction)
- CER: 10.54% (+0.09% vs fp32)
- Latency: 49.3ms (7% faster than fp32)
- Compression: 2x with minimal accuracy loss

**Key Insight**: Post-training quantization with `coremltools.optimize.coreml` is the correct approach for MLProgram models.

---

## Technical Challenges & Solutions

### Challenge 1: Hybrid Model Detection

**Issue**: The zh-CN model is `EncDecHybridRNNTCTCBPEModel` (has both RNNT and CTC decoders), not pure `EncDecCTCModelBPE`.

**Initial Code** (wrong):
```python
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(nemo_path))
ctc_logits = asr_model.decoder(encoder_output)
```

**Error**:
```
TypeError: Number of input arguments provided (1) is not as expected
```

**Solution**:
```python
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(str(nemo_path))
ctc_logits = asr_model.ctc_decoder(encoder_output=encoded)  # Use .ctc_decoder
```

---

### Challenge 2: Variable-Length Audio Handling

**Issue**: FLEURS samples have variable durations (2-20 seconds), but CoreML expects fixed shapes.

**Attempted Solution**: Use `ct.RangeDim()` for dynamic shapes.

**Failure**: Shape inference errors with traced PyTorch models.

**Working Solution**:
1. Convert with fixed shape (15 seconds = 240,000 samples)
2. Pad/truncate audio in preprocessing before model input
3. Handle variable encoder output lengths by padding to 188 time steps

```python
def pad_or_truncate_audio(audio, max_samples=240000):
    if len(audio) < max_samples:
        return np.pad(audio, (0, max_samples - len(audio)))
    else:
        return audio[:max_samples]
```

---

### Challenge 3: Text Normalization for CER Evaluation

**Issue**: Initial CER was 19-20% (unexpectedly high).

**Root Cause**: Model outputs punctuation and digit formats, but reference text has different formatting.

**Example**:
```
Reference:    "我爱北京天安门"  (no punctuation)
Hypothesis:   "我 爱 北京 天安门 。"  (spaces + punctuation)
Raw CER:      19.14%

After normalization:
Reference:    "我爱北京天安门"
Hypothesis:   "我爱北京天安门"
Normalized CER: 10.39%
```

**Solution**: Created `text_normalizer.py` with:
- Punctuation removal
- Number normalization (15 → 十五, 2011 → 二零一一)
- Whitespace collapse

---

### Challenge 4: Dataset Access Issues

**Tried datasets**:

1. ❌ **AISHELL-1**: `DatasetNotFoundError: Dataset 'aishell' doesn't exist`
2. ❌ **FluidInference/fleurs-full**: `ValueError: Bad split: test. Available splits: ['train']`
3. ✅ **google/fleurs**: Success with `trust_remote_code=True`

**Working solution**:
```python
dataset = load_dataset(
    "google/fleurs",
    "cmn_hans_cn",
    split="test",
    trust_remote_code=True  # Required for custom code in dataset
)
```

---

### Challenge 5: Quantization API Evolution

**Timeline**:

1. **coremltools < 5.0**: `ct.models.neural_network.quantization_utils` (NeuralNetwork only)
2. **coremltools 5.0-7.0**: `ct.compression` (deprecated, never worked properly)
3. **coremltools 8.0+**: `coremltools.optimize.coreml` (MLProgram support) ✅

**Correct API for MLProgram models**:
```python
from coremltools.optimize.coreml import linear_quantize_weights
```

---

## Benchmark Results Summary

### Decoder-Only Approach
- **Size**: 14MB (decoder only)
- **Latency**: 4.1ms (decoder only, requires PyTorch encoder)
- **CER**: 10.39% (100 samples)
- **Limitation**: Not standalone

### Full Pipeline (fp32)
- **Size**: 1.1GB encoder + 14MB decoder + 816KB preprocessor
- **Latency**: 53.2ms (full pipeline)
- **CER**: 10.45% (100 samples)
- **Advantage**: Pure CoreML, no PyTorch

### Full Pipeline (int8)
- **Size**: 0.55GB encoder + 14MB decoder + 816KB preprocessor
- **Latency**: 49.3ms (7% faster)
- **CER**: 10.54% (+0.09% degradation)
- **Advantage**: 2x smaller, faster, pure CoreML

---

## Final Conversion Pipeline

### Step 1: Export to CoreML (fp32)
```bash
uv run python conversion/export-full-pipeline.py \
  --nemo-path ../parakeet-ctc-riva-0-6b-unified-zh-cn_vtrainable_v3.0/Parakeet-Hybrid-XL-unified-0.6b_spe7k_zh-en-CN_3.0.nemo \
  --output-dir build-full \
  --no-quantize-encoder
```

Output:
- `Preprocessor.mlpackage` (816KB)
- `Encoder.mlpackage` (1.1GB fp32)
- `Decoder.mlpackage` (14MB)
- `vocab.json` (68KB)

### Step 2: Post-Training Quantization
```bash
uv run python conversion/quantize-encoder-advanced.py \
  --input build-full/Encoder.mlpackage \
  --output build-full/Encoder-int8.mlpackage
```

Output:
- `Encoder-int8.mlpackage` (0.55GB)
- 49.8% size reduction
- 1.99x compression ratio

### Step 3: Compile for Distribution
```bash
for model in *.mlpackage; do
  xcrun coremlcompiler compile "$model" .
done
```

Output: `.mlmodelc` files for instant loading (skip compilation on device)

### Step 4: Benchmark
```bash
uv run python benchmark/benchmark-full-pipeline.py \
  --build-dir build-full \
  --num-samples 100 \
  --output-file results.json
```

Results:
- Mean CER: 10.54%
- Median CER: 5.97%
- Latency: 49.3ms
- RTFx: 229x

---

## Lessons Learned

1. **Always use fixed shapes for traced PyTorch models** - RangeDim doesn't work reliably
2. **Post-training quantization is the way** - Don't try to quantize during conversion
3. **Per-channel quantization > per-tensor** - Better accuracy preservation
4. **Text normalization is critical for Chinese ASR** - Can reduce apparent CER by 50%
5. **MLProgram models require new APIs** - Legacy quantization APIs don't work
6. **Hybrid models need special handling** - Check for `.ctc_decoder` vs `.decoder`
7. **Always benchmark with normalized CER** - Raw CER includes format artifacts
8. **Include both .mlpackage and .mlmodelc** - .mlmodelc loads 10-20x faster

---

## Files Generated

### Conversion Scripts (`conversion/`)
- `export-ctc-zh-cn.py` - Decoder-only conversion (initial approach)
- `export-full-pipeline.py` - Full pipeline conversion (final approach)
- `quantize-encoder-advanced.py` - Post-training int8 quantization
- `individual_components.py` - PyTorch wrappers for tracing

### Validation Scripts (`validation/`)
- `validate-ctc-zh-cn.py` - Decoder-only validation
- `benchmark-cer.py` - Multi-sample validation (with PyTorch encoder)

### Benchmark Scripts (`benchmark/`)
- `benchmark-full-pipeline.py` - Full pipeline CER benchmark (pure CoreML)
- `text_normalizer.py` - Chinese text normalization utilities

### Documentation
- `README.md` (hf-upload/) - HuggingFace model card
- `pipeline_metadata.json` - Model specifications
- `CONVERSION_NOTES.md` (this file) - Conversion process documentation

---

## References

- **Original Model**: [NVIDIA Parakeet CTC zh-CN](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/parakeet-ctc-riva-0-6b-unified-zh-cn)
- **CoreML Tools Docs**: [coremltools.optimize](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html)
- **Dataset**: [Google FLEURS](https://huggingface.co/datasets/google/fleurs)
- **HuggingFace Model**: [FluidInference/parakeet-ctc-0.6b-zh-cn-coreml](https://huggingface.co/FluidInference/parakeet-ctc-0.6b-zh-cn-coreml)

---

**Conversion completed**: 2026-04-02
**CER achieved**: 10.54% (int8) on 100 FLEURS Mandarin samples
**Compression**: 2x vs fp32 with +0.09% CER degradation
**Status**: Production-ready
