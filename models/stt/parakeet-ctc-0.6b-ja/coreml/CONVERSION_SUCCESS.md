# Parakeet CTC 0.6B Japanese - CoreML Conversion SUCCESS ✅

**Date**: 2026-04-03
**Model**: `nvidia/parakeet-tdt_ctc-0.6b-ja`
**Status**: ✅ Successfully converted and validated

## Summary

The Japanese Parakeet CTC model has been successfully converted to CoreML with excellent numerical accuracy. All components achieve 100% Apple Neural Engine utilization with no CPU fallbacks.

## Conversion Results

| Component | Status | Max Diff | ANE % | Notes |
|-----------|--------|----------|-------|-------|
| **Preprocessor** | ✅ | 0.148 | 100% | Audio → Mel spectrogram |
| **Encoder** | ✅ | 0.109 | 100% | Mel → Features (FastConformer) |
| **CTC Decoder** | ✅ | 0.011 | 100% | Features → Raw logits |
| **MelEncoder** | ✅ | - | 100% | Fused: Audio → Features |
| **FullPipeline** | ✅ | 0.482 | 100% | Fused: Audio → Raw logits |

### Numerical Accuracy

- **Individual CTC Decoder**: 0.011 max diff (excellent!)
- **Full Pipeline**: 0.482 max diff (1.44% relative error)
- **Preprocessor contribution**: 0.148
- **Encoder contribution**: 0.109

The 0.482 accumulated error in the full pipeline is well within acceptable bounds for CTC decoding.

## Critical Bug Fixed

### The Problem

Initial conversion attempts produced catastrophically wrong outputs:

```python
# Expected output range (raw logits):
[-18.10, 15.33]

# Broken CoreML output:
[-45440, 0]  # Completely wrong!
```

**Max difference**: 45,422 (unusable for transcription)

### Root Cause

The NeMo CTC decoder's `forward()` method applies `log_softmax`:

```python
# NeMo's ConvASRDecoder.forward()
def forward(self, encoder_output):
    return torch.nn.functional.log_softmax(
        self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
    )
```

**CoreML failed to convert `log_softmax` correctly** for this model, producing the extreme `-45440` values.

### The Solution

Bypass the `log_softmax` by directly accessing `decoder_layers` (Conv1d):

```python
# Before (broken):
class CTCDecoderWrapper(torch.nn.Module):
    def forward(self, encoder_output):
        logits = self.module(encoder_output=encoder_output)  # Calls forward() with log_softmax
        return logits

# After (fixed):
class CTCDecoderWrapper(torch.nn.Module):
    def forward(self, encoder_output):
        # Bypass forward(), use only Conv1d + transpose
        conv_output = self.module.decoder_layers(encoder_output)  # [B, V, T]
        logits = conv_output.transpose(1, 2)  # [B, T, V]
        return logits  # Raw logits
```

### Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| **Max Difference** | 45,422 ❌ | 0.011 ✅ |
| **Output Range** | [-45440, 0] | [-18.10, 15.33] |
| **Usable** | No | Yes |
| **ANE Utilization** | 100% | 100% |

## Usage

### Important: Apply log_softmax in Post-Processing

The converted models output **raw logits**, not log-probabilities. You must apply `log_softmax` before CTC decoding:

```python
import torch
import coremltools as ct

# Load CoreML model
model = ct.models.MLModel('build/FullPipeline.mlpackage')

# Get raw logits
output = model.predict({
    'audio_signal': audio_samples,
    'audio_length': np.array([len(audio_samples[0])], dtype=np.int32)
})
raw_logits = output['ctc_logits']

# Apply log_softmax for CTC decoding
logits_tensor = torch.from_numpy(raw_logits)
log_probs = torch.nn.functional.log_softmax(logits_tensor, dim=-1)

# Now use log_probs for CTC beam search decoding
```

### Why This Works

```python
# Original NeMo decoder output:
log_probs_nemo = ctc_decoder(encoder_output)  # Includes log_softmax

# Our approach:
raw_logits = ctc_decoder_wrapper(encoder_output)  # No log_softmax
log_probs_ours = torch.nn.functional.log_softmax(raw_logits, dim=-1)

# Verification:
assert torch.allclose(log_probs_nemo, log_probs_ours, atol=1e-6)  # ✅ Identical!
```

## Model Details

- **Sample Rate**: 16 kHz
- **Max Audio Duration**: 15 seconds (240,000 samples)
- **Vocabulary**: 3,072 Japanese SentencePiece BPE tokens + 1 blank
- **Blank ID**: 3072
- **Architecture**: Hybrid FastConformer-TDT-CTC (0.6B parameters)

### Component Shapes

```
Audio [1, 240000]
  ↓ Preprocessor
Mel [1, 80, 1501]
  ↓ Encoder (FastConformer, 8x downsampling)
Features [1, 1024, 188]
  ↓ CTC Decoder (Conv1d 1024→3073, kernel_size=1)
Raw Logits [1, 188, 3073]
  ↓ log_softmax (post-processing)
Log Probs [1, 188, 3073]
```

## Files Generated

```
build/
├── Preprocessor.mlpackage      # Audio → Mel spectrogram
├── Encoder.mlpackage            # Mel → Encoder features (FastConformer)
├── CtcDecoder.mlpackage         # Features → Raw CTC logits
├── MelEncoder.mlpackage         # Fused: Audio → Encoder features
├── FullPipeline.mlpackage       # Fused: Audio → Raw CTC logits
├── vocab.json                   # 3,072 Japanese BPE tokens
└── metadata.json                # Model metadata
```

## Compilation

```bash
cd build
xcrun coremlcompiler compile Preprocessor.mlpackage .
xcrun coremlcompiler compile Encoder.mlpackage .
xcrun coremlcompiler compile CtcDecoder.mlpackage .
xcrun coremlcompiler compile MelEncoder.mlpackage .
xcrun coremlcompiler compile FullPipeline.mlpackage .
```

## Performance

- **100% Apple Neural Engine utilization** across all components
- **0 CPU fallback operations**
- **Excellent numerical accuracy** (1.44% relative error)
- Suitable for production use

## Validation Scripts

Test scripts are included for verification:

```bash
# Validate CTC decoder fix
uv run python validate-fix.py

# Test full pipeline
uv run python test-full-pipeline.py

# Analyze error sources
uv run python analyze-pipeline-error.py

# Test isolated Conv1d layer
uv run python test-linear-layer.py
```

## Comparison with Other Models

| Model | Language | Status | Output Type | Notes |
|-------|----------|--------|-------------|-------|
| parakeet-ctc-0.6b-zh-cn | Chinese | ✅ Working | log-softmax | No issues |
| parakeet-tdt-v2-0.6b | English | ✅ Working | log-softmax | No issues |
| **parakeet-tdt_ctc-0.6b-ja** | **Japanese** | ✅ **Fixed** | **Raw logits** | log_softmax bug workaround |

## Key Takeaways

1. **CoreML `log_softmax` conversion can fail** silently, producing extreme values instead of errors
2. **Raw logits + post-processing** is a reliable workaround
3. **Isolation testing** (testing each layer separately) was critical to finding the bug
4. The conversion infrastructure is robust - the same process works for other models
5. **100% ANE utilization** is achievable even with the workaround

## References

- **HuggingFace Model**: https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
- **Working Comparison**: `../parakeet-ctc-0.6b-zh-cn/coreml/`
- **Conversion Script**: `convert-parakeet-ja.py`
- **Component Wrappers**: `individual_components.py`
- **Detailed Investigation**: `CONVERSION_NOTES.md`
