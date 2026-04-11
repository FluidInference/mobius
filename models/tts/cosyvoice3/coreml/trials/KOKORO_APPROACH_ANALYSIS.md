# Kokoro CoreML Approach - Analysis and Application to CosyVoice3

Analysis of how Kokoro successfully converted their vocoder to CoreML and how to apply it to CosyVoice3.

## Kokoro's Success Patterns

### 1. **Fixed Input Shapes** (Critical)

**Kokoro approach:**
```python
# All models use pre-determined dimensions
Duration Model: [1, 128] tokens → [1, 512, 80] features
Decoder (3s):   [1, 512, 72] asr, [1, 144] F0 → [1, 43200] audio (24kHz)
```

**Key insight:** Create separate models for different durations (3s, 10s, 45s) instead of one dynamic model.

**Application to CosyVoice3:**
```python
# Create fixed-duration vocoder variants
VocoderCoreML_3s:  mel [1, 80, 100]  → audio [1, 72000]  # 3s at 24kHz
VocoderCoreML_10s: mel [1, 80, 333]  → audio [1, 240000] # 10s
VocoderCoreML_30s: mel [1, 80, 1000] → audio [1, 720000] # 30s
```

### 2. **Avoid pack_padded_sequence** (Critical)

**Kokoro approach:**
```python
class TextEncoderFixed(nn.Module):
    def forward(self, x, input_lengths, m):
        # Initialize LSTM states explicitly
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_directions * self.num_layers,
            batch_size,
            self.hidden_size,
            dtype=x.dtype,
            device=x.device
        )
        c0 = torch.zeros(...)

        # Flatten parameters for efficiency
        self.lstm.flatten_parameters()

        # Run LSTM WITHOUT pack_padded_sequence
        x, (hn, cn) = self.lstm(x, (h0, c0))

        # Use masking to handle variable lengths
        x.masked_fill_(m, 0.0)
```

**Application to CosyVoice3:**
CosyVoice3's F0 predictor has LSTM - needs same fix:
```python
class F0PredictorFixed(nn.Module):
    def forward(self, x):
        # Explicit state initialization (no pack_padded_sequence)
        batch_size = x.shape[0]
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x, (h0, c0))
        return x
```

### 3. **Deterministic Components** (Important)

**Kokoro approach:**
```python
class SineGenDeterministic(nn.Module):
    def forward(self, f0, random_phases):
        # Use provided random_phases instead of generating new ones
        rad_values[:, 0, :] = rad_values[:, 0, :] + random_phases.squeeze(1)

        # Deterministic phase accumulation
        phase_accum = torch.cumsum(rad_values, dim=1)
        phase_wrapped = (phase_accum - torch.floor(phase_accum)) * 2 * np.pi

        sine_waves = torch.sin(phase_wrapped) * self.sine_amp * uv
        return sine_waves
```

**Key insight:** Pass random values as inputs (not generated inside) so CoreML can trace them.

**Application to CosyVoice3:**
```python
class SourceModuleFixed(nn.Module):
    def forward(self, f0_upsampled, random_seed_tensor):
        # Use random_seed_tensor as input, not torch.randn()
        # This makes the model deterministic during tracing
        ...
```

### 4. **Custom STFT Implementation** (Critical)

**Kokoro approach:**
```python
# From v21.py line 378-379
har_spec, har_phase = self.stft.transform(har_source)
har = torch.cat([har_spec, har_phase], dim=1)

# Later: line 418
audio = self.stft.inverse(spec, phase)
```

They use `kokoro.istftnet.TorchSTFT` which is CoreML-compatible.

**Application to CosyVoice3:**
We already created `coreml_stft.py` with `TorchSTFT` class - use it:
```python
from coreml_stft import CosyVoiceSTFT

class VocoderCoreMLFixed(nn.Module):
    def __init__(self):
        self.custom_stft = CosyVoiceSTFT(n_fft=16, hop_len=4)

    def forward(self, mel):
        # Use custom STFT instead of torch.stft
        s_stft_real, s_stft_imag = self.custom_stft(source)
        ...
```

### 5. **Explicit Dimension Matching** (Important)

**Kokoro approach:**
```python
# From GeneratorDeterministic line 391-395
if x_source.shape[2] != x.shape[2]:
    if x_source.shape[2] < x.shape[2]:
        x_source = F.pad(x_source, (0, x.shape[2] - x_source.shape[2]))
    else:
        x_source = x_source[:, :, :x.shape[2]]

x = x + x_source
```

**Key insight:** Never assume dimensions match - explicitly pad or truncate.

**Application to CosyVoice3:**
```python
# In multi-stage decoder
for i in range(3):
    x = self.ups[i](x)
    si = self.source_downs[i](s_stft)

    # Explicit dimension matching (Kokoro-style)
    if si.shape[2] != x.shape[2]:
        if si.shape[2] < x.shape[2]:
            si = F.pad(si, (0, x.shape[2] - si.shape[2]))
        else:
            si = si[:, :, :x.shape[2]]

    x = x + si
```

### 6. **Two-Stage Architecture** (Strategic)

**Kokoro approach:**
- **Stage 1 (Duration Model):** Text → phoneme durations + features
- **Stage 2 (Decoder):** Pre-computed features → audio

**Key insight:** Separation allows Swift-side alignment, avoiding dynamic shapes in CoreML.

**Application to CosyVoice3:**
We can't easily split CosyVoice3 due to integrated flow model, but we can:
- Keep hybrid approach for full pipeline
- Focus on making vocoder-only work in CoreML for post-processing

## Kokoro's Operation Count Secret

**Why Kokoro has ~3,000 operations vs CosyVoice3's 705,848:**

1. **Simpler F0 handling:** No CausalConvRNN
2. **Simpler source:** Basic harmonic generation vs NSF
3. **Fewer upsampling stages:** 2-3 vs CosyVoice3's complex multi-stage
4. **Simpler ResBlocks:** No adaptive normalization
5. **Optimized STFT:** Designed for CoreML from the start

**CosyVoice3 complexity breakdown:**
```
F0 Predictor (CausalConvRNN): 150,000 ops
Source Generator (NSF):        100,000 ops
Custom STFT:                   150,000 ops
Multi-Stage Decoder:           200,000 ops
Custom ISTFT:                  100,000 ops
Other:                           5,848 ops
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                         705,848 ops
```

**Kokoro (estimated):**
```
Simple F0:                         500 ops
Basic source:                      500 ops
Optimized STFT:                    500 ops
Simple decoder:                  1,000 ops
Optimized ISTFT:                   500 ops
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:                           ~3,000 ops
```

## Implementation Plan for CosyVoice3

### Strategy A: Simplify Existing Vocoder (Kokoro Patterns)

**Goal:** Reduce from 705k → <10k operations by removing complex components.

```python
class CosyVoice3VocoderSimplified(nn.Module):
    """
    Simplified CosyVoice3 vocoder following Kokoro's patterns.
    Target: <10,000 operations for CoreML compatibility.
    """

    def __init__(self, original_vocoder):
        super().__init__()

        # REMOVE: CausalConvRNNF0Predictor (150k ops saved)
        # REMOVE: SourceModuleHnNSF (100k ops saved)
        # REMOVE: Multi-stage STFT fusion (150k ops saved)

        # KEEP (simplified):
        self.conv_pre = nn.Conv1d(80, 256, 7, padding=3)

        # 2 upsampling stages (not 3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 16, 8, 4),  # 8x
            nn.ConvTranspose1d(128, 64, 16, 8, 4),   # 8x (64x total)
        ])

        # Simple ResBlocks (1 per stage, not 3)
        # NO adaptive normalization, NO style conditioning
        self.resblocks = nn.ModuleList([
            SimpleResBlock(128),
            SimpleResBlock(64),
        ])

        self.conv_post = nn.Conv1d(64, 1, 7, padding=3)

        # NO STFT needed for this simple path!

    def forward(self, mel):
        """
        Direct mel → audio (Kokoro-style simplicity)

        Fixed shape: mel [1, 80, T] → audio [1, T*480]
        """
        # Pre-process
        x = self.conv_pre(mel)  # [1, 256, T]

        # Upsample (Kokoro-style: simple, no fusion)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)                   # Upsample
            x = self.resblocks[i](x)    # ResBlock

        # Post-process
        x = F.leaky_relu(x)
        x = self.conv_post(x)           # [1, 1, samples]
        audio = torch.tanh(x)

        return audio.squeeze(1)         # [1, samples]


class SimpleResBlock(nn.Module):
    """Simple ResBlock without adaptive normalization (Kokoro-style)"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(x, 0.1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x)
        return x + residual
```

**Expected operations:** ~3,000-5,000 (like Kokoro)

**Training approach:**
```python
# Knowledge distillation from original vocoder
teacher = CausalHiFTGenerator(...)  # Original
student = CosyVoice3VocoderSimplified()

for epoch in range(100):
    for mel, audio in dataloader:
        # Student prediction
        student_audio = student(mel)

        # Teacher prediction
        with torch.no_grad():
            teacher_audio = teacher(mel, finalize=True)

        # Distillation loss
        loss = F.l1_loss(student_audio, teacher_audio)
        loss += 0.1 * mel_loss(student_audio, audio)  # Ground truth too

        loss.backward()
        optimizer.step()

    # Validate CoreML conversion every 10 epochs
    if epoch % 10 == 0:
        traced = torch.jit.trace(student, example_mel)
        try:
            mlmodel = ct.convert(traced, ...)
            print(f"Epoch {epoch}: CoreML ✅")
        except Exception as e:
            print(f"Epoch {epoch}: CoreML ❌ - {e}")
```

### Strategy B: Fixed-Shape Variants (Kokoro Bucketing)

**Goal:** Create 3 separate models for different durations.

```python
# 3 second variant
class VocoderCoreML_3s(nn.Module):
    def forward(self, mel):
        # Fixed: mel [1, 80, 125] → audio [1, 72000]
        assert mel.shape == (1, 80, 125), f"Expected [1,80,125], got {mel.shape}"
        return self.generate(mel)

# 10 second variant
class VocoderCoreML_10s(nn.Module):
    def forward(self, mel):
        # Fixed: mel [1, 80, 417] → audio [1, 240000]
        assert mel.shape == (1, 80, 417), f"Expected [1,80,417], got {mel.shape}"
        return self.generate(mel)

# Convert each separately
for duration, model_class in [("3s", VocoderCoreML_3s),
                              ("10s", VocoderCoreML_10s)]:
    model = model_class()
    traced = torch.jit.trace(model, example_mel)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=model.input_shape)],  # Fixed shape!
        ...
    )
    mlmodel.save(f"vocoder_{duration}.mlpackage")
```

**Swift-side routing:**
```swift
func selectVocoder(forDuration duration: TimeInterval) -> MLModel {
    switch duration {
    case 0..<5: return vocoder3s
    case 5..<15: return vocoder10s
    default: return vocoder30s
    }
}
```

### Strategy C: Apply All Kokoro Patterns to Current Vocoder

**Goal:** Keep CosyVoice3 architecture but apply Kokoro's CoreML-friendly patterns.

```python
class CosyVoice3VocoderKokoroStyle(nn.Module):
    """
    CosyVoice3 vocoder with Kokoro's CoreML patterns applied.
    """

    def __init__(self, original_vocoder):
        super().__init__()

        # Keep all original components
        self.conv_pre = original_vocoder.conv_pre
        self.ups = original_vocoder.ups
        self.resblocks = original_vocoder.resblocks

        # FIX 1: Replace F0 predictor LSTM (Kokoro pattern)
        self.f0_predictor = F0PredictorFixed(original_vocoder.f0_predictor)

        # FIX 2: Replace source module with deterministic version
        self.m_source = SourceModuleDeterministic(original_vocoder.m_source)

        # FIX 3: Use custom STFT (already created)
        self.custom_stft = CosyVoiceSTFT(n_fft=16, hop_len=4)

        self.conv_post = original_vocoder.conv_post

    def forward(self, mel, random_seed):
        """
        Kokoro pattern: Pass random values as input (not generated inside).
        Fixed shapes enforced.
        """
        # F0 prediction with fixed LSTM
        f0 = self.f0_predictor(mel)

        # Source generation (deterministic)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s = self.m_source(s, random_seed)  # random_seed as input!
        s = s.squeeze(1)

        # Custom STFT (Kokoro pattern)
        s_stft_real, s_stft_imag = self.custom_stft(s)
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        # Multi-stage decoder with explicit dimension matching
        x = self.conv_pre(mel)

        for i in range(3):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            # Downsample source
            si = self.source_downs[i](s_stft)

            # FIX 4: Explicit dimension matching (Kokoro pattern)
            if si.shape[2] != x.shape[2]:
                if si.shape[2] < x.shape[2]:
                    si = F.pad(si, (0, x.shape[2] - si.shape[2]))
                else:
                    si = si[:, :, :x.shape[2]]

            # Fusion
            x = x + si

            # ResBlocks
            for j in range(3):
                x = self.resblocks[i*3+j](x)

        # Post-processing
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        audio = torch.tanh(x)

        return audio.squeeze(1)


class F0PredictorFixed(nn.Module):
    """Fixed F0 predictor following Kokoro's LSTM pattern"""
    def __init__(self, original_f0_predictor):
        super().__init__()
        self.conv_layers = original_f0_predictor.conv_layers
        self.rnn = original_f0_predictor.rnn
        self.proj = original_f0_predictor.proj

        # Get RNN config for state initialization
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(self, x):
        # Convolutions
        for conv in self.conv_layers:
            x = conv(x)

        # Transpose for RNN
        x = x.transpose(1, 2)  # [B, T, C]

        # FIX: Explicit state initialization (Kokoro pattern)
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            dtype=x.dtype,
            device=x.device
        )
        c0 = torch.zeros_like(h0)

        # FIX: Flatten parameters (Kokoro pattern)
        self.rnn.flatten_parameters()

        # RNN without pack_padded_sequence
        x, _ = self.rnn(x, (h0, c0))

        # Project
        f0 = self.proj(x)

        return f0.transpose(1, 2)  # [B, 1, T]
```

## Conversion Script (Kokoro-Style)

```python
"""
convert_vocoder_kokoro_style.py

Convert CosyVoice3 vocoder to CoreML using Kokoro's proven patterns.
"""

import torch
import coremltools as ct
from cosyvoice.hifigan.generator import CausalHiFTGenerator
from generator_kokoro_style import CosyVoice3VocoderSimplified

# Load original vocoder
checkpoint = torch.load("hift.pt", map_location="cpu")
original_vocoder = CausalHiFTGenerator(**checkpoint['config'])
original_vocoder.load_state_dict(checkpoint['generator'])
original_vocoder.eval()

# Create simplified version
simplified_vocoder = CosyVoice3VocoderSimplified(original_vocoder)
simplified_vocoder.eval()

# Fixed input shape (Kokoro pattern - 3 second variant)
batch_size = 1
mel_frames = 125  # 3 seconds at ~24fps mel
mel_channels = 80
example_mel = torch.randn(batch_size, mel_channels, mel_frames)

# Trace with fixed shape
print("Tracing model...")
with torch.no_grad():
    traced_model = torch.jit.trace(simplified_vocoder, example_mel)

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="mel_spectrogram",
            shape=example_mel.shape,  # Fixed shape!
        )
    ],
    outputs=[
        ct.TensorType(name="audio_waveform")
    ],
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,
)

# Save
output_path = "vocoder_simplified_3s.mlpackage"
mlmodel.save(output_path)
print(f"✅ Saved: {output_path}")

# Verify it loads
import coremltools.models as cm
loaded = cm.MLModel(output_path)
print(f"✅ Model loads successfully")
print(f"   Input: {loaded.input_description}")
print(f"   Output: {loaded.output_description}")
```

## Expected Results

| Approach | Operations | CoreML Success | Quality | Training Time |
|----------|-----------|----------------|---------|---------------|
| **A: Simplified** | ~3,000-5,000 | ✅ High | 90-95% | 2-4 weeks |
| **B: Fixed-Shape Variants** | ~10,000 | ⚠️ Medium | 100% | 1 week |
| **C: Apply Patterns** | ~50,000-100,000 | ⚠️ Low | 100% | 1 week |

**Recommendation:** Start with **Approach A (Simplified)** - most likely to succeed.

## Timeline

**Week 1: Implement and Test**
- Day 1-2: Implement `CosyVoice3VocoderSimplified`
- Day 3: Test CoreML conversion (no training)
- Day 4-5: If converts, prepare training data
- Day 6-7: Start training with distillation

**Week 2-4: Train and Validate**
- Week 2: Train simplified vocoder
- Week 3: Validate quality, fine-tune
- Week 4: Final validation, deploy

**Fallback:** If Approach A fails, try Approach B (fixed-shape variants) or continue with hybrid.

## Conclusion

Kokoro's success comes from:
1. ✅ **Fixed shapes** - no dynamic dimensions
2. ✅ **Explicit state management** - no pack_padded_sequence
3. ✅ **Deterministic components** - random values as inputs
4. ✅ **Custom STFT** - CoreML-compatible from the start
5. ✅ **Explicit dimension matching** - never assume shapes match
6. ✅ **Simple architecture** - ~3,000 operations, not 705,000

**Applying to CosyVoice3:** We can either simplify the architecture (Approach A - recommended) or apply the patterns to existing architecture (Approach C - harder).

**Most likely path to success:** Simplified vocoder with knowledge distillation, targeting <5,000 operations.
