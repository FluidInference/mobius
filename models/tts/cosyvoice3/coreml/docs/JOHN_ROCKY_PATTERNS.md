# CoreML Conversion Patterns from john-rocky/CoreML-Models

**Source:** https://github.com/john-rocky/CoreML-Models

Comprehensive analysis of conversion patterns applicable to CosyVoice3 TTS.

---

## Table of Contents

1. [Model Splitting Strategy](#1-model-splitting-strategy)
2. [Flexible Input Shapes (RangeDim)](#2-flexible-input-shapes-rangedim)
3. [Bucketed Decoder Approach](#3-bucketed-decoder-approach)
4. [Audio Quality: FP32 vs FP16](#4-audio-quality-fp32-vs-fp16)
5. [Weight Normalization Removal](#5-weight-normalization-removal)
6. [ONNX Intermediate Format](#6-onnx-intermediate-format)
7. [LSTM Gate Reordering](#7-lstm-gate-reordering)
8. [Runtime Integration Patterns](#8-runtime-integration-patterns)
9. [Operation Patching](#9-operation-patching)
10. [Applicability to CosyVoice3](#10-applicability-to-cosyvoice3)

---

## 1. Model Splitting Strategy

### Pattern: Split Dynamic-Length Models into Fixed-Shape Components

**Used in:**
- **Kokoro TTS** (Predictor + Decoder buckets)
- **OpenVoice** (SpeakerEncoder + VoiceConverter)

### Kokoro Example:

```python
# Model 1: Predictor (flexible input, predicts duration)
class PredictorWrapper(nn.Module):
    def forward(self, input_ids, ref_s_style):
        # input_ids: [1, T] where T = 1..256 (flexible via RangeDim)
        # Output: duration [1, T], d_for_align [1, 640, T], t_en [1, 512, T]
        ...
        duration = torch.sigmoid(self.predictor.duration_proj(x)).sum(axis=-1)
        return duration, d_for_align, t_en

# Model 2: Decoder (fixed input, multiple buckets)
class DecoderWrapper(nn.Module):
    def forward(self, en_aligned, asr_aligned, ref_s):
        # en_aligned: [1, 640, frames] - frames is FIXED per bucket (128/256/512)
        # Output: audio [batch_size, samples]
        ...
        audio = self.decoder(asr_aligned, F0_pred, N_pred, s_decoder).squeeze(1)
        return audio
```

### OpenVoice Example:

```python
# Model 1: Speaker Encoder (flexible input)
class SpeakerEncoderWrapper(nn.Module):
    def forward(self, spec_t):
        # spec_t: [1, T, 513] where T is flexible (10-1000 via RangeDim)
        # Output: [1, 256, 1] speaker embedding
        se = self.ref_enc(spec_t)
        return se.unsqueeze(-1)

# Model 2: Voice Converter (flexible input)
class VoiceConverterWrapper(nn.Module):
    def forward(self, spec, spec_lengths, src_se, tgt_se):
        # spec: [1, 513, T] where T is flexible
        # Output: audio waveform
        ...
```

### Why Split?

1. **Dynamic lengths** (like duration-based frame counts) cannot be represented in CoreML's static graph
2. **Predictor** handles variable-length inputs using RangeDim
3. **Decoder** uses fixed shapes per bucket, chosen at runtime

---

## 2. Flexible Input Shapes (RangeDim)

### Pattern: Use `ct.RangeDim` for Variable-Length Inputs

**Used in:**
- Kokoro Predictor (1-256 phonemes)
- OpenVoice SpeakerEncoder (10-1000 spectrogram frames)
- OpenVoice VoiceConverter (10-1000 frames)

### Kokoro Example:

```python
flex_len = ct.RangeDim(lower_bound=1, upper_bound=MAX_PHONEMES, default=MAX_PHONEMES)
pred_ml = ct.convert(
    traced_pred,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, flex_len), dtype=np.int32),
        ct.TensorType(name="ref_s_style", shape=(1, 128), dtype=np.float32),
    ],
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT32,  # FP32 for audio quality!
)
```

### OpenVoice Example:

```python
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(
        name="spectrogram",
        shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=10, upper_bound=1000, default=100), 513))
    )],
    minimum_deployment_target=ct.target.iOS16,
)
```

### Benefits vs EnumeratedShapes:

| Approach | Flexibility | Padding Required | Use Case |
|----------|-------------|------------------|----------|
| **RangeDim** | Any size in range | ❌ No | Predictor, encoder (dynamic input) |
| **EnumeratedShapes** | Only specific sizes | ✅ Yes | Decoder (fixed buckets) |

### When to Use RangeDim:

- Input length varies continuously (e.g., text → phonemes, variable audio chunks)
- Want to avoid padding artifacts
- Model can handle variable-length inputs naturally (e.g., LSTM, attention)

---

## 3. Bucketed Decoder Approach

### Pattern: Multiple Fixed-Shape Decoders for Different Output Lengths

**Used in:**
- Kokoro Decoder (128, 256, 512 frames)

### Kokoro Buckets:

```python
DECODER_BUCKETS = [128, 256, 512]

for bucket in DECODER_BUCKETS:
    en_aligned = torch.randn(1, hidden_d, bucket)
    asr_aligned = torch.randn(1, hidden_t, bucket)

    traced_dec = torch.jit.trace(dec_wrapper, (en_aligned, asr_aligned, ref_s))

    dec_ml = ct.convert(
        traced_dec,
        inputs=[
            ct.TensorType(name="en_aligned", shape=(1, hidden_d, bucket)),
            ct.TensorType(name="asr_aligned", shape=(1, hidden_t, bucket)),
            ct.TensorType(name="ref_s", shape=(1, 256)),
        ],
        compute_precision=ct.precision.FLOAT32,  # FP32 for audio quality!
    )
    dec_ml.save(f"Kokoro_Decoder_{bucket}.mlpackage")
```

### Runtime Bucket Selection (Swift):

```swift
// Pick smallest bucket that fits
let totalFrames = predictedDurations.sum()
let bucket = Self.buckets.first { $0 >= totalFrames } ?? Self.buckets.last!

// Pad features to bucket size
var outIdx = 0
for i in 0..<T {
    let rep = predDur[i]
    for _ in 0..<rep {
        if outIdx >= bucket { break }
        // Copy features...
        outIdx += 1
    }
}

// Run decoder
let decOut = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [...]))

// Trim audio to actual length
let actualSamples = totalFrames * Self.samplesPerFrame
let audio = Array(audioPtr[0..<actualSamples])
```

### Our MB-MelGAN Buckets:

```python
# We're already using this pattern!
ct.EnumeratedShapes(shapes=[(1, 80, 125), (1, 80, 250), (1, 80, 500)])
```

---

## 4. Audio Quality: FP32 vs FP16

### Pattern: Use FP32 for Audio Models to Preserve Quality

**Used in:**
- **Kokoro**: Explicitly uses FP32, comments say "FP16 corrupts audio quality"
- **HTDemucs**: Uses FP32 "to prevent overflow in the frequency branch"

### Kokoro Example:

```python
dec_ml = ct.convert(
    traced_dec,
    # ...
    compute_precision=ct.precision.FLOAT32,  # FP16 corrupts audio!
)
```

### HTDemucs Example:

```python
mlmodel = ct.convert(
    onnx_path,
    # ...
    compute_precision=ct.precision.FLOAT32,  # Prevent overflow in frequency branch
)
```

### Our MB-MelGAN:

```python
# Currently using FP16 - should we switch to FP32?
mlmodel = ct.convert(
    traced_model,
    # ...
    compute_precision=ct.precision.FLOAT16,  # ⚠️ Consider FP32 for quality
)
```

### Trade-offs:

| Precision | Model Size | Quality | ANE Support | Recommendation |
|-----------|-----------|---------|-------------|----------------|
| **FP16** | 2× smaller | May degrade | ✅ Full | Embedding models, simple ops |
| **FP32** | 1× baseline | ✅ Best | ⚠️ Limited | Audio generation, frequency ops |

**Action:** Test MB-MelGAN with FP32 and compare quality!

---

## 5. Weight Normalization Removal

### Pattern: Remove `weight_norm` Before Exporting to CoreML

**Used in:** OpenVoice

### Example:

```python
# Remove weight_norm for export (prevents CoreML conversion issues)
model.dec.remove_weight_norm()
for layer in model.flow.flows:
    if hasattr(layer, 'remove_weight_norm'):
        layer.remove_weight_norm()
for layer in model.enc_q.enc.in_layers:
    torch.nn.utils.remove_weight_norm(layer)
```

### Why?

- `weight_norm` wraps parameters with `_g` and `_v` tensors
- CoreML may not handle this decomposition correctly
- Removing it "bakes" the normalization into the weight tensors

### Check Our Models:

```python
# For MB-MelGAN, check if any layers use weight_norm
for name, module in model.named_modules():
    if hasattr(module, 'weight_g') or hasattr(module, 'weight_v'):
        print(f"⚠️ {name} uses weight_norm - should remove before export!")
```

---

## 6. ONNX Intermediate Format

### Pattern: Export to ONNX First to Avoid CoreML Bugs

**Used in:** HTDemucs

### Example:

```python
# Export via ONNX to avoid coremltools int op conversion bug
torch.onnx.export(
    wrapper, dummy, "model.onnx",
    input_names=["mix"], output_names=["sources"],
    opset_version=17, do_constant_folding=True,
)

# Convert ONNX to CoreML
mlmodel = ct.convert(
    "model.onnx",
    inputs=[ct.TensorType(name="mix", shape=(1, 2, 343980))],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
)
```

### When to Use:

- Hit CoreML conversion bugs (int ops, shape ops, etc.)
- ONNX has better op coverage for some PyTorch operations
- Can validate ONNX separately with `onnxruntime` before CoreML conversion

### Trade-offs:

| Approach | Pros | Cons |
|----------|------|------|
| **Direct PyTorch → CoreML** | Simpler, one step | May hit conversion bugs |
| **PyTorch → ONNX → CoreML** | More robust, can debug ONNX | Extra step, ONNX export may also fail |

---

## 7. LSTM Gate Reordering

### Pattern: Reorder LSTM Gates When Loading ONNX Weights

**Used in:** convert_diarization.py

### Example:

```python
def reorder_lstm_gates(w):
    """ONNX gate order [i,o,f,c] → PyTorch [i,f,g,o]"""
    chunks = torch.chunk(torch.from_numpy(w), 4, dim=0)
    return torch.cat([chunks[0], chunks[2], chunks[3], chunks[1]], dim=0)

# Apply when loading LSTM weights from ONNX
for layer_idx, (w_key, r_key, b_key) in enumerate(lstm_weights):
    for d in range(2):  # bidirectional
        getattr(model.lstm, f'weight_ih_l{layer_idx}').copy_(
            reorder_lstm_gates(W[w_key][d])
        )
```

### Why?

- ONNX and PyTorch use different gate orderings for LSTM cells
- ONNX: input, output, forget, cell
- PyTorch: input, forget, gate, output

### Relevance:

- Only needed if loading ONNX weights into PyTorch
- Not needed for direct PyTorch model export

---

## 8. Runtime Integration Patterns

### Kokoro Swift Pipeline:

```swift
class KokoroTTS {
    private var predictor: MLModel?
    private var decoders: [Int: MLModel] = [:]  // Bucket → model

    func synthesize(inputIDs: [Int32], voice: String) throws -> [Float] {
        // 1. Run predictor
        let predOut = try predictor.prediction(from: ...)
        let duration = predOut.featureValue(for: "duration")?.multiArrayValue

        // 2. Convert duration to integer frames
        var totalFrames = 0
        for i in 0..<T {
            let v = max(1, Int(durPtr[i].rounded()))
            predDur[i] = v
            totalFrames += v
        }

        // 3. Pick bucket and pad
        let bucket = Self.buckets.first { $0 >= totalFrames } ?? Self.buckets.last!
        let enArr = try MLMultiArray(shape: [1, 640, bucket], dataType: .float32)
        memset(enArr.dataPointer, 0, enArr.count * MemoryLayout<Float>.size)

        // 4. Repeat-interleave features
        var outIdx = 0
        for i in 0..<T {
            for _ in 0..<predDur[i] {
                if outIdx >= bucket { break }
                enPtr[c * bucket + outIdx] = dPtr[c * T + i]
                outIdx += 1
            }
        }

        // 5. Run decoder
        let decOut = try decoder.prediction(from: ...)

        // 6. Trim audio to actual length
        let actualSamples = totalFrames * Self.samplesPerFrame
        return Array(audioPtr[0..<actualSamples])
    }
}
```

### Key Steps:

1. **Predict duration** → total frames needed
2. **Choose bucket** → smallest that fits
3. **Pad features** → zero-pad to bucket size
4. **Decode** → run fixed-shape decoder
5. **Trim** → extract actual audio length

---

## 9. Operation Patching

### Pattern: Patch CoreML Ops to Work Around Conversion Issues

**Used in:** Kokoro

### Example:

```python
from coremltools.converters.mil.frontend.torch import ops as _ct_ops
from coremltools.converters.mil import Builder as mb

def _patched_int(context, node):
    """Custom int op for shape computations."""
    inputs = _ct_ops._get_inputs(context, node)
    x = inputs[0]
    if x.val is not None:
        val = x.val
        if isinstance(val, np.ndarray):
            val = int(val.item()) if val.ndim == 0 else int(val.flat[0])
        else:
            val = int(val)
        res = mb.const(val=np.int32(val), name=node.name)
    else:
        res = mb.cast(x=x, dtype="int32", name=node.name)
    context.add(res)

# Register patched op
_ct_ops._TORCH_OPS_REGISTRY.register_func(
    _patched_int, torch_alias=["int"], override=True
)
```

### When Needed:

- Hit specific op conversion failures (int, shape, gather, etc.)
- CoreML converter doesn't handle edge cases
- Can customize op behavior for CoreML's MIL (Model Intermediate Language)

### Relevance:

- We haven't hit this yet with MB-MelGAN
- Keep in mind for future CosyVoice3 LLM/Flow conversion

---

## 10. Applicability to CosyVoice3

### Current State:

| Component | Status | Conversion Approach |
|-----------|--------|---------------------|
| **Vocoder (MB-MelGAN)** | ✅ Converted | EnumeratedShapes buckets (125, 250, 500 frames) |
| **Flow Model** | ⏸️ Not started | Would need model splitting |
| **LLM** | ⏸️ Not started | Would need model splitting |

### Immediate Actions (MB-MelGAN):

1. **Test FP32 precision** (currently using FP16)
   ```python
   compute_precision=ct.precision.FLOAT32  # Like Kokoro & HTDemucs
   ```

2. **Consider RangeDim** instead of EnumeratedShapes
   ```python
   # Current: 3 fixed sizes
   ct.EnumeratedShapes(shapes=[(1, 80, 125), (1, 80, 250), (1, 80, 500)])

   # Alternative: continuous range (like Kokoro)
   ct.RangeDim(lower_bound=50, upper_bound=500, default=125)
   ```

3. **Check for weight_norm** in MB-MelGAN
   ```python
   for name, module in model.named_modules():
       if hasattr(module, 'weight_g'):
           print(f"Remove weight_norm from {name}")
   ```

### Future Work (Flow + LLM):

When converting Flow and LLM models, apply Kokoro's pattern:

1. **Predictor Model (LLM)**:
   - Input: text tokens (flexible length via RangeDim)
   - Output: predicted token count + hidden states

2. **Decoder Models (Flow)**:
   - Input: hidden states (fixed buckets)
   - Output: mel spectrograms (fixed buckets)
   - Runtime: choose smallest bucket ≥ predicted count

### Comparison:

| Model | Kokoro | CosyVoice3 (proposed) |
|-------|--------|----------------------|
| **Predictor** | BERT + LSTM duration | LLM text → token count |
| **Decoder buckets** | 128, 256, 512 frames | TBD (based on mel length stats) |
| **Vocoder** | iSTFTNet (~3k ops) | MB-MelGAN (202 ops) ✅ |
| **Precision** | FP32 | FP16 → test FP32 |

---

## Summary of Key Patterns

1. ✅ **Model Splitting** - Separate predictor (RangeDim) from decoder (fixed buckets)
2. ✅ **RangeDim** - For flexible input lengths (proven in Kokoro, OpenVoice)
3. ✅ **Bucketed Decoders** - Multiple fixed-shape models for different lengths
4. ⚠️ **FP32 for Audio** - Kokoro & HTDemucs both emphasize this
5. ✅ **Weight Norm Removal** - Remove before export (if present)
6. ✅ **ONNX Intermediate** - Fallback if direct conversion fails
7. ℹ️ **LSTM Gate Reorder** - Only if loading ONNX weights
8. ✅ **Runtime Pattern** - predict → choose bucket → pad → decode → trim
9. ℹ️ **Op Patching** - Last resort for conversion bugs

**Most Applicable to Our Work:**
- #1, #2, #3, #4, #8 for MB-MelGAN and future Flow/LLM conversion

---

## References

- **Kokoro Conversion:** `/tmp/CoreML-Models/conversion_scripts/convert_kokoro.py`
- **Kokoro Swift App:** `/tmp/CoreML-Models/sample_apps/KokoroDemo/KokoroDemo/KokoroTTS.swift`
- **OpenVoice:** `/tmp/CoreML-Models/conversion_scripts/convert_openvoice.py`
- **HTDemucs:** `/tmp/CoreML-Models/conversion_scripts/convert_htdemucs.py`
- **Diarization:** `/tmp/CoreML-Models/conversion_scripts/convert_diarization.py`

**Repository:** https://github.com/john-rocky/CoreML-Models

This repository proves that **complex TTS models CAN be fully converted to CoreML**! 🎉
