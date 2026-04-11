# CosyVoice3 CoreML Conversion - Status Report

**Date:** 2026-04-10  
**Model:** Fun-CosyVoice3-0.5B-2512 (HiFT Vocoder)  
**Status:** ❌ **BLOCKED - Outputs Corrupted**

---

## Summary

Successfully converted the CosyVoice3 vocoder to CoreML format (both FP16 and FP32), but the converted models produce **completely incorrect outputs** with max error of 1.98 (198%) and correlation of only 0.08 with reference outputs.

---

## What Works ✓

### 1. TorchScript Tracing
- **Status:** ✓ Perfect
- **Max difference:** 0.000000
- Traced model matches PyTorch exactly

### 2. Custom ISTFT Implementation
- **Status:** ✓ Perfect
- **Max difference:** 0.001245 (0.12%)
- CoreML-compatible inverse STFT using `torch.fft.irfft` + overlap-add
- Works correctly in isolation

### 3. CoreML Conversion Process
- **Status:** ✓ Completes successfully
- Both FP16 and FP32 conversions complete without fatal errors
- Models compile and run on device
- No crashes or runtime failures

---

## What's Broken ✗

### CoreML Model Outputs

| Metric | FP16 | FP32 | Expected |
|--------|------|------|----------|
| Max diff | 1.980000 | 1.980000 | < 0.01 |
| Mean diff | 0.875 | 0.959 | < 0.001 |
| Correlation | 0.079 | 0.079 | > 0.999 |
| Model size | 77.9 MB | 115.9 MB | ~340 MB |
| **Status** | ✗ FAIL | ✗ FAIL | ✓ PASS |

### Output Characteristics

**PyTorch (correct):**
```
[-0.125, 0.990, -0.990, 0.651, 0.990, -0.990, ...]
Natural variation, proper audio values
```

**CoreML (corrupted):**
```
[-0.462, 0.990, -0.990, -0.990, -0.990, 0.990, ...]
Heavily clipped to ±0.99, mostly extreme values
```

**Root cause:** The main generator network (convolutions/upsampling) is computing garbage values that get clipped to [-0.99, 0.99] by the `audio_limit` clamping.

---

## Technical Findings

### 1. Conversion Warnings

**Critical overflow warning (pass 58):**
```
elementwise_unary.py:889: RuntimeWarning: overflow encountered in cast
  return input_var.val.astype(dtype=string_to_nptype(dtype_val))
```

This indicates integer overflow during int64→int32 casting, likely corrupting some operation.

### 2. Model Statistics

**FP32 CoreML model:**
- Total operations: 253,810 (extremely bloated)
- Const operations: 169,114 (66% of all ops)
- The 100-frame ISTFT overlap-add loop was unrolled into:
  - 24,007 slice operations
  - 12,002 scatter operations  
  - 12,120 add operations

**Comparison:**
- Original checkpoint: 340 MB
- FP32 CoreML: 115.9 MB (missing weights?)
- FP16 CoreML: 77.9 MB

### 3. Precision Analysis

Both FP16 and FP32 produce **identical max errors (1.98)**, indicating the issue is NOT floating-point precision but rather **incorrect computation** in the network.

---

## Conversion Timeline

| Version | Time | Bottleneck Passes | Status |
|---------|------|-------------------|--------|
| FP16 | 67 min | Pass 57 (23 min), Pass 93 (24 min) | ✗ Wrong outputs |
| FP32 | 73 min | Pass 57 (27 min), Pass 90 (27 min) | ✗ Wrong outputs |

---

## Investigation Attempts

### 1. ✓ Verified Components

- [x] TorchScript tracing: Perfect (diff = 0.000000)
- [x] Custom ISTFT alone: Perfect (diff = 0.001245)
- [x] Traced model validation: Perfect

### 2. ✗ Identified Issues

- [x] CoreML full model: Broken (correlation = 0.08)
- [x] Overflow warning during conversion (pass 58)
- [x] Outputs heavily clipped (mostly ±0.99)
- [x] Both FP16 and FP32 fail identically

### 3. ⏳ In Progress

- [ ] Conversion without `skip_model_load` (running, ~60 min remaining)
- [ ] This will show if validation errors are being masked

---

## Key Code Changes

### 1. Custom ISTFT (istft_coreml.py)

Replaced unsupported `torch.istft` with:
```python
class CoreMLISTFT(nn.Module):
    def forward(self, magnitude, phase):
        # Reconstruct complex spectrum
        real = magnitude * torch.sqrt(1.0 - phase**2)
        imag = magnitude * phase
        complex_spec = torch.complex(real, imag)
        
        # Apply inverse FFT
        frames = torch.fft.irfft(complex_spec, n=self.n_fft)
        
        # Overlap-add synthesis
        output = torch.zeros(batch_size, output_length)
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            output[:, start:end] += windowed_frames[:, i, :]
        
        return output
```

### 2. Patched SineGen2 (generator_patched.py)

Replaced unsupported `torch.multiply`:
```python
# BEFORE
fn = torch.multiply(f0, harmonics)

# AFTER  
fn = f0 * harmonics
```

### 3. Attribute Rename (generator_coreml.py)

Fixed naming conflict:
```python
# BEFORE (triggered torch.istft converter)
self.istft = CoreMLISTFT(...)

# AFTER (uses custom implementation)
self.custom_istft = CoreMLISTFT(...)
```

---

## Hypotheses for Corruption

### 1. Weight Corruption
- Model size is smaller than expected (116MB vs 340MB)
- Some weights may have been quantized incorrectly or dropped

### 2. Operator Conversion Issues
- The overflow warning suggests some operation is breaking
- Could be in convolutions, weight_norm layers, or upsampling

### 3. Graph Optimization Corruption
- Passes 57, 90, 93 took 20-27 minutes each (normally 7-10s)
- These long passes may have incorrectly optimized the graph

### 4. Loop Unrolling Issues
- The ISTFT loop was massively unrolled (24k+ ops)
- This unrolling might have introduced errors

---

## Next Steps

### Immediate (waiting for results)

1. **Conversion without skip_model_load** (in progress)
   - Will reveal if validation errors are being masked
   - Running in background, ~60 min remaining

### If Validation Reveals Errors

2. **Try Alternative Conversion Strategies:**
   - Convert without optimization passes
   - Use ONNX as intermediate format
   - Manually specify which passes to run/skip
   - Try older coremltools version

3. **Component-by-Component Debugging:**
   - Convert each network layer individually
   - Find which specific layer is breaking
   - Isolate the problematic operation

### If No Validation Errors

4. **Deep Debugging:**
   - Export intermediate layer outputs from both models
   - Find where outputs first diverge
   - Check weight values in CoreML vs PyTorch

---

## Blockers

1. **Primary:** CoreML conversion corrupts network computation
2. **Secondary:** No clear error message indicating what's wrong
3. **Tertiary:** Long conversion times (60-70 min) slow iteration

---

## Files Created

### Conversion Scripts
- `convert_coreml_simple.py` - FP16 conversion (with skip_model_load)
- `convert_coreml_fp32.py` - FP32 conversion (with skip_model_load)
- `convert_without_skip.py` - FP32 with validation (running)

### Custom Implementations  
- `istft_coreml.py` - CoreML-compatible ISTFT
- `generator_coreml.py` - Modified generator (renamed istft attribute)
- `generator_patched.py` - Patched SineGen2 (fixed torch.multiply)

### Validation/Testing
- `validate_coreml.py` - Validates FP16 model
- `validate_fp32.py` - Validates FP32 model
- `debug_outputs.py` - Analyzes output patterns
- `test_istft.py` - Tests ISTFT in isolation
- `test_traced.py` - Tests TorchScript tracing
- `inspect_model.py` - Inspects CoreML model structure

### Generated Models
- `converted/hift_vocoder.mlpackage` - FP16 (77.9 MB) ✗
- `converted/hift_vocoder_fp32.mlpackage` - FP32 (115.9 MB) ✗
- `test_istft.mlpackage` - ISTFT only ✓
- `converted/hift_vocoder_validated.mlpackage` - With validation (pending)

### Logs
- `/tmp/vocoder_coreml_conversion.log` - FP16 conversion
- `/tmp/vocoder_fp32_conversion.log` - FP32 conversion
- `/tmp/convert_no_skip.log` - Validated conversion (running)

---

## Conclusion

The CosyVoice3 vocoder successfully converts to CoreML format but produces **completely incorrect outputs**. The issue is NOT related to:
- TorchScript tracing (perfect)
- Custom ISTFT implementation (perfect)
- Floating-point precision (FP16/FP32 both fail identically)

The issue IS related to:
- Corruption in the main generator network during CoreML conversion
- Likely caused by incorrect operator conversion or graph optimization
- Masked by `skip_model_load=True` flag

**Current status:** Waiting for validation-enabled conversion to complete, which may reveal the actual error being masked.
