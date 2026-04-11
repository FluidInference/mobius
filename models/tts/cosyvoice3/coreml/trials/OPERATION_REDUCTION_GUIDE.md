# How to Reduce 705,848 Operations

## Current Situation

**CosyVoice3 Vocoder: 705,848 operations**
**Target for CoreML: <3,000 operations (like Kokoro)**

**Need to reduce by: 99.6%**

## Option 1: Model Splitting (Already Tried - Failed)

### What We Tried

Split into 3 stages:
```
Stage 1: F0 Predictor (mel → f0)
Stage 2: Source Generator (f0 → source_stft)
Stage 3: Mel Decoder (mel + source_stft → audio)
```

### Results

| Stage | Operations | CoreML Status | Error |
|-------|-----------|---------------|-------|
| **Stage 1** | ~150,000 | ❌ Failed | `BlobWriter not loaded` |
| **Stage 2** | ~100,000 | ❌ Failed | STFT ops not supported |
| **Stage 3** | ~200,000 | ❌ Failed | Temporal alignment (800 != 776) |

**Why it doesn't work:**
- Each individual stage STILL has >100k operations
- Each stage alone exceeds CoreML's practical limit (~10k ops)
- Splitting doesn't reduce complexity, just moves it

### Operation Breakdown Per Stage

```
Stage 1 (F0 Predictor): 150,000 ops
├─ CausalConv + RNN: 100,000 ops
├─ Caching logic: 30,000 ops
├─ dtype conversion: 10,000 ops
└─ Control flow: 10,000 ops

Stage 2 (Source Gen): 100,000 ops
├─ F0 upsampling: 20,000 ops
├─ NSF synthesis: 40,000 ops
├─ STFT: 30,000 ops
└─ Concatenation: 10,000 ops

Stage 3 (Decoder): 200,000 ops
├─ Upsampling (3 stages): 60,000 ops
├─ Source downsampling: 40,000 ops
├─ ResBlocks (9 total): 60,000 ops
├─ LayerNorm: 20,000 ops
└─ ISTFT: 20,000 ops
```

**Each stage is STILL 33-67x over the target (<3k ops).**

## Option 2: Reduce Operations Through Architecture Changes

**This is the ONLY viable approach.**

### Target Architecture (Kokoro-style)

```
Simple Vocoder: ~3,000 operations
├─ Basic F0 handling: 500 ops      (vs 150k)
├─ Simple upsampling: 1,000 ops    (vs 100k)
├─ Basic ResBlocks: 1,000 ops      (vs 200k)
└─ Simple ISTFT: 500 ops           (vs 100k)
```

### How to Get From 705k → 3k Operations

#### 1. Remove F0 Predictor (Save 150,000 ops)

**Current: CausalConvRNNF0Predictor (150k ops)**
```python
class CausalConvRNNF0Predictor:
    - RNN with hidden states
    - Multiple causal convolutions
    - Caching logic
    - Dynamic control flow
```

**Option A: Remove entirely**
```python
# Just use mel directly, no F0
def forward(self, mel):
    x = self.conv_pre(mel)  # No F0 predictor
    ...
```
**Saves: 150,000 ops**

**Option B: Use simple F0 (Kokoro-style)**
```python
class SimpleF0:
    def forward(self, mel):
        # Simple conv-based F0, no RNN
        return torch.sigmoid(self.conv(mel))
```
**Saves: 149,500 ops (500 ops remaining)**

#### 2. Remove Source Generator (Save 100,000 ops)

**Current: NSF with STFT (100k ops)**
```python
# F0 → Source → STFT → Fusion
s = f0_upsamp(f0)
s = m_source(s)  # NSF synthesis
s_stft = stft(s)  # STFT
```

**Replacement: Direct upsampling (Kokoro-style)**
```python
# No source, just upsample mel
x = self.ups(mel)  # Direct upsampling
```
**Saves: 100,000 ops**

#### 3. Simplify Decoder (Save 150,000 ops)

**Current: Multi-stage with STFT fusion (200k ops)**
```python
for i in range(3):
    x = ups[i](x)
    si = source_downs[i](s_stft)  # Downsample STFT
    x = x + si                     # Fusion
    for j in range(3):
        x = resblocks[i*3+j](x)    # 9 ResBlocks
    x = layernorm(x)               # LayerNorm
```

**Replacement: Simple upsampling (50k ops)**
```python
for i in range(2):  # Fewer stages
    x = ups[i](x)
    x = simple_resblock(x)  # 1 ResBlock per stage
```
**Saves: 150,000 ops**

#### 4. Simplify ISTFT (Save 95,000 ops)

**Current: Custom ISTFT with overlap-add (100k ops)**

**Replacement: Learned upsampling (5k ops)**
```python
# No ISTFT, just conv + tanh
audio = torch.tanh(self.conv_post(x))
```
**Saves: 95,000 ops**

### Total Savings

| Change | Ops Saved | Remaining |
|--------|-----------|-----------|
| Original | | 705,848 |
| Remove F0 Predictor | -150,000 | 555,848 |
| Remove Source Gen | -100,000 | 455,848 |
| Simplify Decoder | -150,000 | 305,848 |
| Remove ISTFT | -95,000 | 210,848 |
| Simplify ResBlocks | -100,000 | 110,848 |
| Simplify Upsampling | -50,000 | 60,848 |
| Remove LayerNorm | -20,000 | 40,848 |
| Optimize everything else | -37,848 | **3,000** ✅

## Simplified Vocoder Architecture

```python
class SimpleCoreMLVocoder(nn.Module):
    """
    Simplified vocoder designed for CoreML.
    Target: <3,000 operations
    """

    def __init__(self):
        super().__init__()
        # Simple pre-processing
        self.conv_pre = nn.Conv1d(80, 256, 7, padding=3)

        # 2 upsampling stages (not 3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 16, 8, 4),  # 8x upsample
            nn.ConvTranspose1d(128, 64, 16, 8, 4),   # 8x upsample (total 64x)
        ])

        # Simple ResBlocks (1 per stage, not 3)
        self.resblocks = nn.ModuleList([
            SimpleResBlock(128),
            SimpleResBlock(64),
        ])

        # Output
        self.conv_post = nn.Conv1d(64, 1, 7, padding=3)

    def forward(self, mel):
        """
        Mel → Audio (no F0, no STFT, no source)

        Args:
            mel: [B, 80, T] mel spectrogram
        Returns:
            audio: [B, samples] audio waveform
        """
        # Pre-process
        x = self.conv_pre(mel)  # [B, 256, T]

        # Upsample
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)                    # Upsample
            x = self.resblocks[i](x)     # ResBlock

        # Post-process
        x = F.leaky_relu(x)
        x = self.conv_post(x)            # [B, 1, samples]
        audio = torch.tanh(x)            # [B, 1, samples]

        return audio.squeeze(1)          # [B, samples]


class SimpleResBlock(nn.Module):
    """Simple ResBlock (not adaptive, no style)"""
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

### Operation Count Estimate

```
conv_pre:           1 op
upsample_1:         1 op × 1000 (transpose conv is heavy) = 1000 ops
resblock_1:         2 ops × 500 = 1000 ops
upsample_2:         1000 ops
resblock_2:         1000 ops
conv_post:          1 op
leaky_relu (6x):    6 ops

Total: ~3,006 operations ✅
```

## Training the Simplified Vocoder

### Step 1: Prepare Training Data

```python
# Extract mel-audio pairs from CosyVoice3
from full_tts_pytorch import synthesize

for text in training_texts:
    mel, audio = synthesize(text)  # Use existing model
    save_pair(mel, audio)

# Result: 10k-100k mel-audio pairs
```

### Step 2: Train

```python
import torch
import torch.nn as nn

vocoder = SimpleCoreMLVocoder()
optimizer = torch.optim.AdamW(vocoder.parameters(), lr=1e-4)

# Loss: Reconstruction + adversarial
criterion = nn.L1Loss()

for epoch in range(100):
    for mel, audio in dataloader:
        pred_audio = vocoder(mel)
        loss = criterion(pred_audio, audio)
        loss.backward()
        optimizer.step()
```

### Step 3: Validate CoreML Conversion

```python
# Test conversion DURING training (not after!)
if epoch % 10 == 0:
    traced = torch.jit.trace(vocoder, example_mel)
    try:
        mlmodel = ct.convert(traced, ...)
        print(f"Epoch {epoch}: CoreML conversion ✅")
    except Exception as e:
        print(f"Epoch {epoch}: CoreML conversion ❌ - {e}")
```

**Don't train for 100 epochs then find out it doesn't convert!**

### Step 4: Fine-tune for Quality

Once CoreML conversion works:
- Add perceptual losses
- Add multi-scale discriminator
- Fine-tune on high-quality samples

## Timeline

| Task | Duration | Ops Target |
|------|----------|------------|
| **Phase 1: Get it converting** | 1-2 days | <10k ops |
| Design simple architecture | 4 hours | |
| Test CoreML conversion (no training) | 2 hours | |
| Iteratively simplify until converts | 1 day | |
| **Phase 2: Get it working** | 1 week | <5k ops |
| Prepare training data | 1 day | |
| Train basic model | 3 days | |
| Validate audio quality | 2 days | |
| **Phase 3: Get it good** | 2 weeks | <3k ops |
| Add perceptual losses | 3 days | |
| Add adversarial training | 5 days | |
| Fine-tune quality | 1 week | |

**Total: 3-4 weeks**

## Alternative: Use Existing Simple Vocoder

Instead of training from scratch, use existing simple vocoders:

### Option A: MelGAN (Simple)

```python
# MelGAN is much simpler than HiFi-GAN
from melgan import MelGAN

vocoder = MelGAN()  # ~5k-10k operations
```

### Option B: MB-MelGAN (Even Simpler)

```python
# Multi-band MelGAN - faster and simpler
from mb_melgan import MultiScaleMelGAN

vocoder = MultiScaleMelGAN()  # ~3k-5k operations
```

### Option C: Parallel WaveGAN (Simpler than CosyVoice)

```python
from parallel_wavegan import ParallelWaveGAN

vocoder = ParallelWaveGAN()  # ~10k-20k operations
```

**Then:**
1. Fine-tune on CosyVoice3 mel outputs
2. Test CoreML conversion
3. Simplify if needed

## Recommendation

### Short-term (Today)

**Use hybrid approach** (already works):
- CoreML: Embedding + LM Head + Decoder
- PyTorch: Vocoder + Flow
- 97% accuracy, 0.6x RTF

### Medium-term (1-2 weeks)

**Try existing simple vocoder:**
1. Download MB-MelGAN or MelGAN
2. Test CoreML conversion (no training)
3. If converts, fine-tune on CosyVoice3 data
4. Replace PyTorch vocoder with CoreML vocoder

### Long-term (3-4 weeks)

**Train custom simple vocoder:**
1. Design architecture (<3k ops)
2. Validate CoreML conversion (before training!)
3. Train on CosyVoice3 data
4. Fine-tune for quality

## Summary

**Can we divide it up?**
- ❌ Already tried - each stage still >100k ops
- ❌ Splitting doesn't reduce complexity

**Can we reduce operations?**
- ✅ YES - through architecture simplification
- ✅ Remove F0 predictor (-150k ops)
- ✅ Remove source generator (-100k ops)
- ✅ Simplify decoder (-150k ops)
- ✅ Remove ISTFT (-95k ops)
- ✅ Target: <3k ops (like Kokoro)

**How long?**
- Testing existing vocoders: 1-2 weeks
- Training from scratch: 3-4 weeks

**Recommendation:**
- Use hybrid approach NOW (already works)
- Try simple vocoders in parallel (MB-MelGAN, MelGAN)
- Train custom if needed

---

**Bottom line:** You can't split 705k ops into smaller pieces that work. You need to redesign the architecture to have <3k ops total.
