# v21.py Model Architecture Documentation

## Overview

This document outlines the model class structures in `v21.py`, focusing on the neural network architectures and their CoreML-compatible modifications. Each class addresses specific CoreML conversion challenges while maintaining the original Kokoro TTS functionality.

---

## Model Class Hierarchy

```
KokoroCompleteCoreML (Main Wrapper)
├── BERT Encoder (from Kokoro)
├── TextEncoderFixed
│   ├── Embedding Layer
│   ├── CNN Layers
│   └── Bidirectional LSTM
├── TextEncoderPredictorFixed
│   ├── LSTM Blocks
│   └── AdaLayerNorm Blocks
├── Duration/Pitch Predictors (from Kokoro)
├── Decoder Blocks (4x, from Kokoro)
└── GeneratorDeterministic
    ├── SourceModuleHnNSFDeterministic
    │   └── SineGenDeterministic
    ├── Upsampling Blocks
    ├── Residual Blocks
    └── STFT Modules
```

---

## 1. TextEncoderFixed

**Purpose**: Encodes phoneme token sequences into linguistic feature representations

### Architecture

```
Input: [B, L] int32 token IDs
  ↓
Embedding: [B, L, embedding_dim]
  ↓
Transpose: [B, embedding_dim, L]
  ↓
CNN Layers: (conv1d blocks)
  ↓
Transpose: [B, L, channels]
  ↓
Bidirectional LSTM: [B, L, hidden_size*2]
  ↓
Transpose: [B, hidden_size*2, L]
  ↓
Output: [B, channels, L]
```

### Key Components

#### Initialization

```python
def __init__(self, original_text_encoder):
    self.embedding = original_text_encoder.embedding
    self.cnn = original_text_encoder.cnn
    self.lstm = original_text_encoder.lstm

    # Store LSTM configuration
    self.hidden_size = self.lstm.hidden_size
    self.num_layers = self.lstm.num_layers
    self.bidirectional = True
    self.num_directions = 2
```

### Forward Pass Details

#### Step 1: Embedding

```python
x = self.embedding(x)  # [B, L] → [B, L, emb_dim]
x = x.transpose(1, 2)   # [B, L, emb_dim] → [B, emb_dim, L]
```

#### Step 2: Masking and CNN

```python
m = m.unsqueeze(1)     # [B, L] → [B, 1, L]
x.masked_fill_(m, 0.0) # Zero out padding positions

for c in self.cnn:
    x = c(x)            # Conv1d layers
    x.masked_fill_(m, 0.0)  # Re-mask after each layer
```

#### Step 3: LSTM (CoreML-Compatible)

```python
x = x.transpose(1, 2)  # [B, emb_dim, L] → [B, L, features]

# Explicit state initialization (required for CoreML)
batch_size = x.shape[0]
h0 = torch.zeros(
    self.num_directions * self.num_layers,
    batch_size,
    self.hidden_size,
    dtype=x.dtype,
    device=x.device
)
c0 = torch.zeros(...)  # Same shape as h0

self.lstm.flatten_parameters()
x, (hn, cn) = self.lstm(x, (h0, c0))  # No pack_padded_sequence
```

**Key Difference**: Original uses `pack_padded_sequence` for efficiency; this version uses explicit masking instead.

#### Step 4: Output Processing

```python
x = x.transpose(-1, -2)  # [B, L, features] → [B, features, L]

# Padding if needed
if x.shape[-1] < m.shape[-1]:
    x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], ...)
    x_pad[:, :, :x.shape[-1]] = x
    x = x_pad

x.masked_fill_(m, 0.0)  # Final masking
return x
```

### Inputs

- `x`: Token IDs `[B, L]` int64
- `input_lengths`: Actual sequence lengths `[B]` int64
- `m`: Boolean mask `[B, L]` (True = padding)

### Output

- `x`: Encoded features `[B, channels, L]` float32

---

## 2. TextEncoderPredictorFixed

**Location**: Lines 115-194

**Purpose**: Encodes features for duration and pitch prediction with style conditioning

**Original Problem**: Duration predictor's text encoder uses `pack_padded_sequence`

### Architecture

```
Input: [B, d_model, L]
  ↓
Concatenate with Style: [B, d_model + style_dim, L]
  ↓
LSTM Block 1
  ↓
AdaLayerNorm 1 (style-conditioned)
  ↓
Concatenate Style
  ↓
LSTM Block 2
  ↓
AdaLayerNorm 2
  ↓
... (repeat pattern)
  ↓
Output: [B, L, d_model]
```

### Key Components

#### Initialization

```python
def __init__(self, text_encoder):
    self.lstms = text_encoder.lstms  # ModuleList[LSTM + AdaLayerNorm]
    self.d_model = text_encoder.d_model
    self.sty_dim = text_encoder.sty_dim
```

**Structure**: `self.lstms` is a ModuleList alternating between:
- Bidirectional LSTM layers
- AdaLayerNorm layers (style-adaptive normalization)

### Forward Pass Details

#### Step 1: Permute and Expand Style

```python
# Input x: [B, d_model, L]
x = x.permute(2, 0, 1)  # → [L, B, d_model]

# Expand style to match time dimension
s = style.expand(x.shape[0], x.shape[1], -1)  # [L, B, style_dim]
```

#### Step 2: Concatenate Style

```python
x = torch.cat([x, s], axis=-1)  # [L, B, d_model + style_dim]
x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
```

#### Step 3: Process LSTM/AdaLayerNorm Blocks

```python
for block in self.lstms:
    if isinstance(block, AdaLayerNorm):
        # AdaLayerNorm processing
        x = x.transpose(-1, -2)  # [B, features, L] → [B, L, features]
        x = block(x, style)       # Style-conditioned normalization
        x = x.transpose(-1, -2)   # → [B, features, L]

        # Re-concatenate style
        x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)

    else:  # LSTM
        x = x.transpose(-1, -2)  # [B, features, L] → [B, L, features]

        block.flatten_parameters()

        # Explicit state initialization
        batch_size = x.shape[0]
        h0 = torch.zeros(2, batch_size, self.d_model // 2, ...)
        c0 = torch.zeros(2, batch_size, self.d_model // 2, ...)

        x, _ = block(x, (h0, c0))
        x = x.transpose(-1, -2)  # → [B, features, L]
```

**Pattern**: Alternates between LSTM processing and AdaLayerNorm with style conditioning

### Inputs

- `x`: Input features `[B, d_model, L]`
- `style`: Style embedding `[B, style_dim]`
- `text_lengths`: Sequence lengths `[B]`
- `m`: Padding mask `[B, L]`

### Output

- `x`: Encoded features `[B, L, d_model]`

---

## 3. SineGenDeterministic

**Location**: Lines 294-325

**Purpose**: Generate deterministic harmonic sine waves for F0 (pitch) modeling

**Original Problem**: Original `SineGen` uses random phase initialization, causing non-deterministic output

### Architecture

```
Input F0: [B, L]
  ↓
Generate Harmonics: f0 * [1, 2, 3, ..., dim]
  ↓
Convert to Radians: fn / sampling_rate
  ↓
Add Random Phases at t=0
  ↓
Cumulative Phase Accumulation
  ↓
Wrap Phases: (phase % 1) * 2π
  ↓
Generate Sine Waves
  ↓
Apply UV (voiced/unvoiced) Mask
  ↓
Output: [B, L, dim]
```

### Key Components

#### Initialization

```python
def __init__(self, original_sine_gen):
    self.sine_amp = original_sine_gen.sine_amp
    self.harmonic_num = original_sine_gen.harmonic_num
    self.dim = original_sine_gen.dim
    self.sampling_rate = original_sine_gen.sampling_rate
    self.voiced_threshold = original_sine_gen.voiced_threshold
```

### Forward Pass Details

#### Step 1: UV Detection

```python
# Smooth voiced/unvoiced transition
uv = torch.sigmoid((f0 - self.voiced_threshold) * 0.5)
```

**Purpose**: Determines which frames are voiced (have pitch) vs unvoiced (noise)

**Why Sigmoid**: Smooth transition instead of hard threshold prevents discontinuities

#### Step 2: Harmonic Generation

```python
# Generate harmonic frequencies: f0, 2*f0, 3*f0, ..., dim*f0
harmonic_nums = torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype)
fn = f0 * harmonic_nums.view(1, 1, -1)  # [B, L, dim]
```

**Example**:
```
f0 = 100 Hz, dim = 8
fn = [100, 200, 300, 400, 500, 600, 700, 800] Hz
```

#### Step 3: Phase Accumulation

```python
# Convert frequency to radians per sample
rad_values = (fn / self.sampling_rate)  # [B, L, dim]

# Apply random phases at initial timestep
rad_values[:, 0, :] = rad_values[:, 0, :] + random_phases.squeeze(1)

# Cumulative phase (unwrapped)
phase_accum = torch.cumsum(rad_values, dim=1)
```

**Key**: `random_phases` input makes this deterministic and controllable

#### Step 4: Phase Wrapping

```python
# Wrap to [0, 2π] to prevent numerical overflow
phase_wrapped = (phase_accum - torch.floor(phase_accum)) * 2 * np.pi
```

**Why Wrap**: Prevents phase values from growing unbounded over long sequences

#### Step 5: Sine Wave Generation

```python
sine_waves = torch.sin(phase_wrapped) * self.sine_amp * uv
```

**Shape**: `[B, L, dim]` - batch of harmonic sine waves with UV masking

### Inputs

- `f0`: Fundamental frequency `[B, L]` float32 (in Hz)
- `random_phases`: Initial phases `[B, 1, dim]` or `[B, dim]` float32

### Output

- `sine_waves`: Harmonic components `[B, L, dim]` float32

---

## 4. SourceModuleHnNSFDeterministic

**Location**: Lines 330-347

**Purpose**: Combine harmonic (sine) and noise source signals

**Original Problem**: Non-deterministic noise generation

### Architecture

```
Input F0: [B, L]
  ↓
SineGenDeterministic → Harmonic Waves [B, L, dim]
  ↓
Linear Layer
  ↓
Tanh Activation → sine_merge
  ↓
Deterministic Noise: sin(f0 * 100) * amp/3
  ↓
Output: sine_merge (+ noise implicit)
```

### Key Components

```python
def __init__(self, original_source):
    self.sine_amp = original_source.sine_amp
    self.l_sin_gen = SineGenDeterministic(original_source.l_sin_gen)
    self.l_linear = original_source.l_linear
    self.l_tanh = original_source.l_tanh
```

### Forward Pass

```python
def forward(self, x, random_phases):
    # Generate harmonics
    sine_wavs = self.l_sin_gen(x, random_phases)  # [B, L, dim]

    # Linear transform + activation
    sine_merge = self.l_tanh(self.l_linear(sine_wavs))

    # Deterministic noise (F0-dependent)
    noise = torch.sin(x * 100) * self.sine_amp / 3

    return sine_merge
```

**Note**: `noise` is generated but not explicitly added in return (may be used in generator)

### Inputs

- `x`: F0 values `[B, L]` or `[B, L, 1]`
- `random_phases`: Phase initialization `[B, dim]`

### Output

- `sine_merge`: Source signal `[B, L, features]`

---

## 5. GeneratorDeterministic

**Location**: Lines 350-422

**Purpose**: Main vocoder - converts acoustic features to audio waveform

**Original Problem**: Non-deterministic harmonic generation

### Architecture

```
Inputs: features [B, C, F], style [B, 256], f0 [B, F], phases [B, 9]
  ↓
[Harmonic Source Generation]
  ↓
F0 Upsampling: [B, F] → [B, F*upsample_factor]
  ↓
SourceModule → harmonic source
  ↓
STFT → magnitude + phase
  ↓
[Upsampling Blocks]
  ↓
For each upsample layer:
  ├─ LeakyReLU
  ├─ Noise Convolution on harmonic
  ├─ Noise Residual (style-conditioned)
  ├─ Upsample main features
  ├─ Add harmonic + features
  └─ Residual Blocks (style-conditioned)
  ↓
[Final Processing]
  ↓
LeakyReLU
  ↓
Conv Post
  ↓
Split: magnitude [0:n_fft//2+1] | phase [n_fft//2+1:]
  ↓
Magnitude: exp(clamp(mag, -10, 10))
Phase: sin(phase)
  ↓
Inverse STFT
  ↓
Output Audio: [B, 1, T]
```

### Key Components

#### Initialization

```python
def __init__(self, original_generator):
    # Copy all original components
    self.num_kernels = original_generator.num_kernels
    self.num_upsamples = original_generator.num_upsamples
    self.noise_convs = original_generator.noise_convs     # ModuleList
    self.noise_res = original_generator.noise_res         # ModuleList
    self.ups = original_generator.ups                     # ModuleList
    self.resblocks = original_generator.resblocks         # ModuleList
    self.post_n_fft = original_generator.post_n_fft
    self.conv_post = original_generator.conv_post
    self.reflection_pad = original_generator.reflection_pad
    self.stft = original_generator.stft
    self.f0_upsamp = original_generator.f0_upsamp

    # Replace with deterministic version
    self.m_source = SourceModuleHnNSFDeterministic(original_generator.m_source)
```

### Forward Pass Details

#### Step 1: Harmonic Source Generation

```python
# Upsample F0 to match audio frame rate
f0_up = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, 1, F] → [B, F*factor]

# Generate harmonic source
har_source = self.m_source(f0_up, random_phases)     # [B, F*factor, features]
har_source = har_source.transpose(1, 2).squeeze(1)   # [B, T]

# STFT to get frequency representation
har_spec, har_phase = self.stft.transform(har_source)
har = torch.cat([har_spec, har_phase], dim=1)        # [B, n_fft+2, F']
```

#### Step 2: Upsampling Loop

```python
for i in range(self.num_upsamples):
    x = F.leaky_relu(x, negative_slope=0.1)

    # Process harmonic source
    x_source = self.noise_convs[i](har)
    x_source = self.noise_res[i](x_source, s)  # Style conditioning

    # Upsample main features
    x = self.ups[i](x)

    # Reflection padding for last layer
    if i == self.num_upsamples - 1:
        x = self.reflection_pad(x)

    # Dimension matching
    if x_source.shape[2] != x.shape[2]:
        if x_source.shape[2] < x.shape[2]:
            x_source = F.pad(x_source, (0, x.shape[2] - x_source.shape[2]))
        else:
            x_source = x_source[:, :, :x.shape[2]]

    # Combine main + harmonic
    x = x + x_source

    # Residual blocks
    xs = None
    for j in range(self.num_kernels):
        if xs is None:
            xs = self.resblocks[i * self.num_kernels + j](x, s)
        else:
            xs += self.resblocks[i * self.num_kernels + j](x, s)
    x = xs / self.num_kernels
```

**Key Points**:
- Each upsampling layer processes both main features and harmonic source
- Style conditioning applied via `noise_res` and `resblocks`
- Dimension matching ensures tensors align before addition
- Multiple residual blocks averaged together

#### Step 3: Final Audio Generation

```python
x = F.leaky_relu(x)
x = self.conv_post(x)  # Final convolution

# Split into magnitude and phase
x_mag = x[:, :self.post_n_fft // 2 + 1, :]
x_mag = torch.clamp(x_mag, min=-10, max=10)
spec = torch.exp(x_mag)

phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

# Inverse STFT to time domain
audio = self.stft.inverse(spec, phase)  # [B, T]

return audio
```

**Why Clamp**: Prevents numerical instability in exp() operation

### Inputs

- `x`: Acoustic features `[B, channels, F]`
- `s`: Style embedding `[B, style_dim]`
- `f0`: Fundamental frequency `[B, F]`
- `random_phases`: Phase initialization `[B, 9]`

### Output

- `audio`: Raw waveform `[B, T]` float32

---

## 6. KokoroCompleteCoreML

**Location**: Lines 453-578

**Purpose**: End-to-end model wrapper combining all components

### Architecture Overview

```
Inputs: input_ids, ref_s, random_phases, attention_mask
  ↓
[Text Encoding]
BERT → BertEncoder → d_en [B, 256, L]
  ↓
[Duration Prediction]
TextEncoderPredictorFixed → LSTM → duration_proj
→ pred_dur [B, L]
  ↓
[Alignment]
create_alignment_matrix → pred_aln_trg [1, L, F]
  ↓
[Feature Encoding]
d_en @ alignment → en [B, 256, F]
TextEncoderFixed → asr @ alignment → asr_features [B, C, F]
  ↓
[Prosody Prediction]
F0Ntrain(en) → F0_pred, N_pred [B, F]
  ↓
[Pre-Generator]
F0_conv + N_conv + asr → encoded [B, C, F]
  ↓
[Decoder Blocks] (4x)
decode_block(encoded, asr_res, F0, N) → x [B, C, F]
  ↓
[Generator]
GeneratorDeterministic → audio [B, 1, T]
  ↓
Outputs: audio, audio_length, pred_dur
```

### Key Components

#### Initialization

```python
def __init__(self, model, bert, bert_encoder, predictor, device="cpu", samples_per_frame=600):
    self.model = model
    self.bert = bert
    self.bert_encoder = bert_encoder
    self.predictor = predictor

    # Fixed versions
    self.predictor_text_encoder = TextEncoderPredictorFixed(predictor.text_encoder)
    self.text_encoder = TextEncoderFixed(model.text_encoder)

    # Constants
    self.samples_per_frame = 600  # 24kHz / 40fps

    # Frontend
    self.F0_conv = model.decoder.F0_conv
    self.N_conv = model.decoder.N_conv
    self.encode = model.decoder.encode
    self.asr_res = model.decoder.asr_res

    # Decoder
    self.decode_blocks = model.decoder.decode  # 4 blocks

    # Generator
    self.generator = GeneratorDeterministic(model.decoder.generator)
```

### Critical Method: create_alignment_matrix

**Location**: Lines 482-511

**Purpose**: Map variable-length phoneme durations to fixed-length acoustic frames

**Problem**: CoreML doesn't support in-place assignment or dynamic loops

**Solution**: Pure broadcasting-based implementation

```python
def create_alignment_matrix(self, pred_dur, device, max_frames):
    """
    Input: pred_dur [L] - duration per phoneme in frames
    Output: alignment [1, L, F] - binary matrix
    """

    # Ensure 1D, clamp to >= 0 (PADs can stay 0)
    if pred_dur.dim() == 0:
        pred_dur = pred_dur.unsqueeze(0)
    pred_dur = torch.round(pred_dur).clamp(min=0)
    pred_dur_int = pred_dur.to(torch.int32)

    # Cumulative endpoints
    cum = torch.cumsum(pred_dur_int, dim=0)        # [L]
    starts = torch.nn.functional.pad(cum[:-1], (1, 0), value=0)  # [L]

    # Determine max frames
    if max_frames is None:
        max_frames = int(cum[-1].item())

    # Create frame grid [0, 1, 2, ..., max_frames-1]
    frame_grid = torch.arange(max_frames, device=device, dtype=torch.int32).unsqueeze(0)  # [1, F]

    # Broadcast comparison
    starts = starts.to(torch.int32).unsqueeze(1)   # [L, 1]
    ends = cum.to(torch.int32).unsqueeze(1)        # [L, 1]

    # mask[i, j] = True if frame j belongs to phoneme i
    mask = (frame_grid >= starts) & (frame_grid < ends)  # [L, F]

    # Only include frames up to total duration
    total = torch.clamp(cum[-1], max=max_frames).to(torch.int32)
    valid_cols = (frame_grid < total.unsqueeze(0))
    mask = mask & valid_cols

    return mask.to(torch.float32).unsqueeze(0)  # [1, L, F]
```

**Example**:

```
pred_dur = [3, 2, 4, 0]  # Last is PAD
cum = [3, 5, 9, 9]
starts = [0, 3, 5, 9]

Frame:  0 1 2 3 4 5 6 7 8
Ph 0:   1 1 1 0 0 0 0 0 0  (frames 0-2)
Ph 1:   0 0 0 1 1 0 0 0 0  (frames 3-4)
Ph 2:   0 0 0 0 0 1 1 1 1  (frames 5-8)
Ph 3:   0 0 0 0 0 0 0 0 0  (PAD, duration=0)
```

### Forward Pass

#### Phase 1: Text Encoding

```python
# Build attention mask
attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.int32)
text_mask_bool = (attention_mask == 0)  # True where PAD
input_lengths = attention_mask.sum(dim=1)

# BERT encoding
bert_output = self.bert(input_ids, attention_mask=attention_mask)
d_en = self.bert_encoder(bert_output).transpose(-1, -2)  # [B, 256, L]

# Extract style
style = ref_s[:, 128:]  # Last 128 dims
```

#### Phase 2: Duration Prediction

```python
# Predictor text encoder
d = self.predictor_text_encoder(d_en, style, input_lengths, text_mask_bool)

# LSTM for duration
lstm_layers = self.predictor.lstm.num_layers * 2  # Bidirectional
h0 = torch.zeros(lstm_layers, batch_size, hidden_size, ...)
c0 = torch.zeros(lstm_layers, batch_size, hidden_size, ...)
x, _ = self.predictor.lstm(d, (h0, c0))

# Duration projection
duration = self.predictor.duration_proj(x)
duration = torch.sigmoid(duration).sum(axis=-1) / speed  # [B, L]
pred_dur = torch.round(duration).clamp(min=1)

# Zero out PADs
valid = attention_mask.to(dtype=pred_dur.dtype)
pred_dur = pred_dur * valid  # PADs → 0
```

#### Phase 3: Alignment and Feature Encoding

```python
# Calculate total frames and audio length
total_frames = pred_dur.sum(dim=1)  # [B]
audio_length_samples = (total_frames * self.samples_per_frame).to(torch.int32)

# Create alignment (batch=1 path)
total_frames_int = int(torch.clamp(total_frames[0], min=1).item())
pred_aln_trg = self.create_alignment_matrix(pred_dur[0], device, total_frames_int)

# Align features
en = d.transpose(-1, -2) @ pred_aln_trg  # [1, channels, F]

# F0 and N prediction
F0_pred, N_pred = self.predictor.F0Ntrain(en, style)

# ASR features
t_en = self.text_encoder(input_ids, input_lengths, text_mask_bool)
asr = t_en @ pred_aln_trg  # [1, C, F]
```

#### Phase 4: Pre-Generator

```python
ref_s_style = ref_s[:, :128]  # First 128 dims

F0_processed = self.F0_conv(F0_pred.unsqueeze(1))  # [1, c, F]
N_processed = self.N_conv(N_pred.unsqueeze(1))     # [1, c, F]

x = torch.cat([asr, F0_processed, N_processed], dim=1)
x_encoded = self.encode(x, ref_s_style)
asr_res = self.asr_res(asr)  # Residual connection
```

#### Phase 5: Decoder

```python
x_current = x_encoded
for decode_block in self.decode_blocks:  # 4 blocks
    x_input = torch.cat([x_current, asr_res, F0_processed, N_processed], dim=1)
    x_current = decode_block(x_input, ref_s_style)
```

#### Phase 6: Audio Generation

```python
audio = self.generator(x_current, ref_s_style, F0_pred, random_phases)
return audio, audio_length_samples, pred_dur
```

### Inputs

- `input_ids`: Phoneme tokens `[B, L]` int32
- `ref_s`: Reference style `[B, 256]` float32
- `random_phases`: Phase init `[B, 9]` float32
- `attention_mask`: Valid tokens `[B, L]` int32 (1=valid, 0=pad)

### Outputs

- `audio`: Waveform `[B, 1, T_fixed]` float32
- `audio_length_samples`: Actual length `[B]` int32
- `pred_dur`: Phoneme durations `[B, L]` float32

---

## Key Design Decisions

### 1. Explicit LSTM State Initialization

**Why**: CoreML tracing requires explicit tensor shapes

```python
# Required for CoreML
h0 = torch.zeros(num_layers * num_directions, batch, hidden_size, ...)
c0 = torch.zeros(num_layers * num_directions, batch, hidden_size, ...)
output, (hn, cn) = lstm(input, (h0, c0))
```

### 2. Deterministic Components

**Why**: Reproducible output, easier debugging, consistent testing

**Approach**: Replace random phase generation with controlled `random_phases` input

### 3. Broadcasting-Only Operations

**Why**: CoreML doesn't support dynamic loops or in-place assignment

**Example**: `create_alignment_matrix` uses only broadcasting comparisons

### 4. Fixed Input Shapes

**Why**: CoreML requires static shapes for model compilation

**Solution**: Pad to `MAX_TOKENS`, use attention masks to mark valid positions

### 5. Style Conditioning Throughout

**How**: Style vector concatenated or passed to AdaLayerNorm at each stage

**Purpose**: Maintains speaker identity and prosodic characteristics

---

## Model Sizes and Configurations

| Component | Parameters | Output Shape |
|-----------|-----------|--------------|
| BERT | ~82M | [B, L, 768] |
| TextEncoder | ~2M | [B, 256, L] |
| Predictor | ~1M | [B, L] durations |
| Decoder (4 blocks) | ~10M | [B, C, F] |
| Generator | ~15M | [B, T] audio |
| **Total** | ~**110M** | - |

---

## Data Flow Summary

```
Text "hello world"
  ↓ (G2P)
Phonemes [HH, AH, L, OW, W, ER, L, D]
  ↓ (Tokenize)
IDs [23, 45, 12, 67, 89, 34, 12, 56]
  ↓ (BERT + Encoder)
Features [B, 256, 8]
  ↓ (Duration Predictor)
Durations [3, 4, 2, 3, 2, 4, 2, 3] frames
  ↓ (Alignment Matrix)
Acoustic Features [B, 256, 23]  # sum(durations)=23 frames
  ↓ (F0/N Predictors)
Pitch: [B, 23], Energy: [B, 23]
  ↓ (Decoder Blocks)
Encoded: [B, C, 23]
  ↓ (Generator + Harmonics)
Audio: [B, 13800]  # 23 frames * 600 samples/frame
```

---

## References

- Original Kokoro: `hexgrad/Kokoro-82M`
- StyleTTS2 paper: [arXiv:2306.07691](https://arxiv.org/pdf/2306.07691)
- Related: `problems_encountered.md`, `kokoro_coreml_fix.patch`
