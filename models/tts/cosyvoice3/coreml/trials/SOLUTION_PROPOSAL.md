# CosyVoice3 CoreML Conversion - Solution Proposal

**Date:** 2026-04-10
**Status:** ROOT CAUSE IDENTIFIED - Proposing Solutions

---

## Executive Summary

The CosyVoice3 CoreML conversion failure is **NOT a CoreML bug** - it's a **PyTorch model instability issue** that gets exposed during conversion.

**Root Cause:**
- ResBlocks have architectural gain > 1.0 (measured 4-30x per block)
- Gains compound exponentially across 9 blocks
- Final output range is 119x larger than input
- CoreML conversion amplifies this slightly → catastrophic clipping

**Impact:**
- Full model: max diff 1.98, correlation 0.08 (broken)
- Cause: Output values ~±83 get clipped to ±0.99 → garbage

---

## Problem Details

### Measured Signal Amplification

| Stage | Input Range | Output Range | Gain |
|-------|-------------|--------------|------|
| **Baseline** (no ResBlocks) | [-0.40, 0.31] | [-0.40, 0.31] | 1.0x |
| **Layer 0 ResBlocks** | [-1.26, 1.25] | [-8.89, 5.19] | 5.7x |
| **Layer 1 ResBlocks** | [-1.95, 2.19] | [-18.99, 9.71] | 6.9x |
| **Layer 2 ResBlocks** | [-4.03, 3.57] | **[-65.35, 18.06]** | **11.0x** |

**Worst individual ResBlock:**
- ResBlock[2,2]: **30.31x gain** (input 7.6 → output 230.3)

### Why Standard HiFiGAN Works But This Doesn't

HiFiGAN typically uses:
1. **Batch normalization** or **layer normalization** to stabilize outputs
2. **Lower gain ResBlocks** (gain ~1.0-2.0, not 4-30x)
3. **Fewer ResBlocks** (often 3-6 total, not 9)
4. **Gradient clipping** during training to prevent explosion

CosyVoice3's model:
- ❌ No normalization layers
- ❌ ResBlocks with gain 4-30x
- ❌ 9 ResBlocks total
- ❌ No output clamping between layers

---

## Proposed Solutions

### Option 1: Add Normalization Layers (Recommended)

**Approach:** Insert LayerNorm or BatchNorm after each ResBlock group

**Pros:**
- Mathematically sound fix
- Prevents signal explosion
- Doesn't change model architecture drastically
- CoreML supports BatchNorm/LayerNorm perfectly

**Cons:**
- Need to retrain or fine-tune model
- Changes model weights distribution
- May affect audio quality until re-trained

**Implementation:**
```python
class CausalHiFTGeneratorCoreML(nn.Module):
    def __init__(self, ...):
        # ... existing code ...

        # Add normalization layers
        self.resblock_norms = nn.ModuleList([
            nn.LayerNorm(self.upsample_initial_channel // (2 ** i))
            for i in range(len(upsample_rates))
        ])

    def decode(self, x, s, finalize=True):
        # ... upsampling ...

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            # ... source fusion ...

            # ResBlocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            # ADD NORMALIZATION HERE
            x = self.resblock_norms[i](x.transpose(1, 2)).transpose(1, 2)
```

### Option 2: Reduce ResBlocks Gain (Requires Model Access)

**Approach:** Modify ResBlock weights to reduce gain

**Pros:**
- No architecture changes
- Might preserve audio quality better
- Could work with frozen weights

**Cons:**
- Requires direct weight manipulation
- May affect audio quality
- No guarantee of stability

**Implementation:**
```python
# Scale down ResBlock weights after loading
for i, resblock in enumerate(generator.resblocks):
    # Empirically measured gains
    measured_gains = [
        7.08, 16.77, 10.05,  # Layer 0
        12.38, 8.43, 10.14,  # Layer 1
        5.20, 4.09, 30.31,   # Layer 2
    ]

    target_gain = 1.5  # Want gain ~1.5x instead of 4-30x
    scale_factor = target_gain / measured_gains[i]

    # Scale all conv weights
    for name, param in resblock.named_parameters():
        if 'weight' in name and 'original1' in name:  # Weight norm weights
            param.data *= scale_factor
```

### Option 3: Add Output Clamping (Quick Fix)

**Approach:** Clamp outputs after each ResBlock group to prevent explosion

**Pros:**
- ✓ Easiest to implement
- ✓ No retraining needed
- ✓ Converts perfectly to CoreML
- ✓ Might preserve audio quality

**Cons:**
- May introduce clipping artifacts
- Not addressing root cause
- May affect expressiveness

**Implementation:**
```python
def decode(self, x, s, finalize=True):
    # ... upsampling ...

    for i in range(self.num_upsamples):
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[i](x)

        # ... source fusion ...

        # ResBlocks
        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[i * self.num_kernels + j](x)
            else:
                xs += self.resblocks[i * self.num_kernels + j](x)
        x = xs / self.num_kernels

        # ADD CLAMPING HERE
        x = torch.clamp(x, min=-10.0, max=10.0)  # Prevent explosion
```

**Empirical clamp values:**
- Layer 0: clamp to ±5.0
- Layer 1: clamp to ±10.0
- Layer 2: clamp to ±15.0

### Option 4: Use CoreML-Specific Quantization

**Approach:** Convert with INT8 or FP16 quantization to forcibly limit range

**Pros:**
- No model changes needed
- Smaller model size
- Faster inference

**Cons:**
- Doesn't solve root cause
- May introduce quantization noise
- Clipping still happens, just earlier

**Implementation:**
```python
import coremltools as ct

# Convert with quantization
coreml = ct.convert(
    traced,
    inputs=[ct.TensorType(name='mel', shape=(1, 80, 100))],
    outputs=[ct.TensorType(name='audio')],
    minimum_deployment_target=ct.target.macOS14,
    compute_precision=ct.precision.FLOAT16,  # or INT8
)

# Apply post-training quantization
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig

config = OptimizationConfig(
    global_config=OpPalettizerConfig(mode="kmeans", nbits=4)
)
compressed_model = ct.optimize.coreml.palettize_weights(coreml, config)
```

---

## Recommended Approach

**Phase 1: Quick Validation (Option 3)**
1. Add output clamping after each ResBlock group
2. Test if audio quality is acceptable
3. Convert to CoreML and validate parity
4. **Goal:** Confirm clamping solves the conversion issue

**Phase 2: Proper Fix (Option 1)**
1. Add LayerNorm after each ResBlock group
2. Fine-tune model on small dataset to recover quality
3. Convert to CoreML and validate
4. **Goal:** Production-ready stable model

**Phase 3: Optimization (Option 4)**
1. Apply quantization (FP16 or INT8)
2. Profile on target hardware (ANE utilization)
3. Benchmark RTFx and quality
4. **Goal:** Optimal inference performance

---

## Implementation Steps

### Step 1: Test Clamping Fix (1-2 hours)

```bash
# 1. Add clamping to generator_coreml.py
# 2. Test conversion
python convert_coreml_simple.py

# 3. Validate output
python validate_coreml.py
```

**Success criteria:**
- Max diff < 0.01
- Correlation > 0.99
- Audio sounds acceptable

### Step 2: Test Normalization Fix (2-4 hours)

```bash
# 1. Add LayerNorm to generator_coreml.py
# 2. Load existing weights
# 3. Test with frozen norms (no training)
# 4. Convert and validate
```

**Success criteria:**
- Conversion succeeds
- Outputs stable
- Quality acceptable (may need fine-tuning)

### Step 3: Fine-Tuning (if needed)

```bash
# 1. Prepare small TTS dataset
# 2. Fine-tune with normalization layers
# 3. Validate quality matches original
```

---

## Expected Results

### With Clamping (Option 3)

**Predicted performance:**
- ✓ Conversion succeeds
- ✓ Max diff < 0.01 (vs current 1.98)
- ✓ Correlation > 0.99 (vs current 0.08)
- ? Audio quality: Unknown (may clip expressiveness)

### With Normalization (Option 1)

**Predicted performance:**
- ✓ Conversion succeeds
- ✓ Max diff < 0.001 (perfect parity)
- ✓ Correlation ~1.000
- ✓ Audio quality: Same as original (after fine-tuning)

---

## Risk Analysis

### Option 3 (Clamping) Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Clipping introduces artifacts | Medium | Medium | Test with various inputs |
| Quality degradation | Medium | High | Compare to original audio |
| Not fixing root cause | High | Low | Plan migration to Option 1 |

### Option 1 (Normalization) Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Need fine-tuning | High | Medium | Prepare dataset beforehand |
| Quality changes | Medium | High | Extensive A/B testing |
| Training time required | High | Medium | Use small dataset first |

---

## Conclusion

The CoreML conversion failure is caused by **ResBlocks architectural instability** (4-30x gain per block), not a CoreML bug.

**Immediate action:**
1. Test Option 3 (clamping) to validate it fixes conversion
2. If successful, plan Option 1 (normalization) for production

**Long-term solution:**
- Add normalization layers
- Fine-tune model
- Validate quality matches original

**Timeline:**
- Quick fix (clamping): 1-2 hours
- Proper fix (normalization): 2-4 hours + training time
- Production deployment: 1-2 days (with quality validation)

The root cause is now fully understood and multiple viable solutions exist.
