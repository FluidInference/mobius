# Kokoro TTS CoreML Implementation

## Quick Links

### Documentation
- **[Model Architecture Deep Dive](doc/v21_conversion_script_outline.md)** - Complete technical breakdown of all model classes and CoreML modifications
- **[Problems Encountered](doc/problems_encountered.md)** - Comprehensive issue tracking, solutions, and performance benchmarks
- **[TTS Concepts Guide](doc/tts_concepts.md)** - Foundational TTS concepts (prosody, F0, alignment, etc.)

### External Resources
- **Model Hub**: [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)
- **CoreML Models**: [FluidInference/kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml)
- **Main Repository**: [FluidInference/mobius](https://github.com/FluidInference/mobius)
- **StyleTTS2 Paper**: [arXiv:2306.07691](https://arxiv.org/pdf/2306.07691)
- **StyleTTS2 GitHub**: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- **StyleTTS2 Deep Dive**: [deepwiki.com/yl4579/StyleTTS2](https://deepwiki.com/yl4579/StyleTTS2)


## Repository Structure

```
tts/
├── README.md                              # This file
├── doc/
│   ├── v21_conversion_script_outline.md   # Model architecture documentation
│   ├── problems_encountered.md            # Issue tracking and solutions
│   └── tts_concepts.md                    # TTS fundamentals (if created)
├── v21.py                                 # Conversion script
├── v21.ipynb                              # Main Interactive notebook version
├── kokoro_coreml_fix.patch                # Required Kokoro library patches
└── uv.lock                                # Python dependencies
```

---

### Patch Kokoro Library

The original Kokoro library requires patches for CoreML compatibility:

```bash
# Apply patch to your Kokoro installation
cd /path/to/kokoro
patch -p1 < /path/to/mobius/models/tts/kokoro_coreml_fix.patch
```

See [kokoro_coreml_fix.patch](kokoro_coreml_fix.patch) for details.

### Download Pre-converted Models

```bash
# 5s model (streaming, short responses)
huggingface-cli download FluidInference/kokoro-82m-coreml kokoro_21_5s.mlmodelc

# 15s model (long-form synthesis)
huggingface-cli download FluidInference/kokoro-82m-coreml kokoro_21_15s.mlmodelc
```

---

## How It Works

### Pipeline Overview

```
Text Input
  ↓
[1. Phoneme Conversion]
  G2P (Grapheme-to-Phoneme)
  ↓
[2. BERT Encoding]
  Contextualized phoneme embeddings
  ↓
[3. Duration Prediction]
  How long each phoneme lasts
  ↓
[4. Alignment Matrix]
  Map phonemes → acoustic frames
  ↓
[5. Prosody Prediction]
  F0 (pitch) and energy
  ↓
[6. Style Conditioning]
  Reference voice embedding
  ↓
[7. Decoder Blocks]
  Feature refinement (4 blocks)
  ↓
[8. Generator]
  Harmonic synthesis + vocoder
  ↓
Audio Output (24kHz waveform)
```

### Core Components

| Component | Purpose | Input Shape | Output Shape |
|-----------|---------|-------------|--------------|
| **BERT** | Phoneme encoding | [B, L] tokens | [B, L, 768] features |
| **TextEncoder** | Linguistic features | [B, L] tokens | [B, 256, L] features |
| **Duration Predictor** | Timing estimation | [B, 256, L] | [B, L] durations |
| **Alignment Matrix** | Phoneme→Frame mapping | [L] durations | [L, F] mask |
| **F0 Predictor** | Pitch contour | [B, 256, F] | [B, F] pitch |
| **Decoder (4x)** | Feature refinement | [B, C, F] | [B, C, F] |
| **Generator** | Audio synthesis | [B, C, F] + F0 | [B, T] audio |

**Legend**: `B`=batch, `L`=sequence length, `F`=frames, `T`=samples, `C`=channels

For detailed architecture breakdowns, see [Model Architecture Documentation](doc/v21_conversion_script_outline.md).

---

## Model Variants

We provide three model configurations optimized for different use cases:

| Model | Target Duration | MAX_TOKENS | File Size | Use Case | Compilation Time |
|-------|-----------------|------------|-----------|----------|------------------|
| **5s** | ~5 seconds | 249 | ~80MB | Streaming, short responses, real-time chat | 1-3s |
| **10s** | ~10 seconds | 400 | ~100MB | Balanced, general purpose | 8s (new) / 145s (old devices) |
| **15s** | ~15 seconds | 512 | ~120MB | Long-form, complex sentences | 60-90s (ANE) |

### Selection Guide

- **Conversational AI / Chat**: Use 5s model for low latency
- **Audiobook / Article Reading**: Use 15s model for quality
- **Mixed Use**: Use 10s model as compromise
- **Streaming**: Use 5s model with sentence chunking

**Note**: Longer models degrade quality on short text. Use 5s model for "Hello" but 15s for paragraphs.

---

## Key TTS Concepts

### Prosody - The Music of Speech

Prosody encompasses pitch, rhythm, loudness, and timing that convey meaning beyond words:

**Examples**:
- "I can't believe it!" (excited: rising pitch, faster, louder)
- "I can't believe it..." (disappointed: falling pitch, slower, quieter)
- "I CAN'T believe it" (emphasis: pitch spike on stressed word)
- "Really?" (question: rising intonation) vs "Really." (statement: falling)

**Emotional Patterns**:
- **Anger**: Faster rate, higher intensity, clipped consonants
- **Sadness**: Slower rate, lower pitch, reduced variation
- **Surprise**: Sudden pitch jumps, extended vowels ("Whaaaat?")
- **Sarcasm**: Exaggerated pitch contours, deliberate pacing

### Speaker Style (Acoustic Signature)

The unique voice characteristics that make a speaker recognizable:

- **Timbre**: Deep/high, breathy/clear, gravelly/smooth
- **Vocal tract resonances**: Formant structure
- **Habitual patterns**: Speaking pace, articulation habits

**Examples**:
- Morgan Freeman: Deep, resonant, slow-paced, gravelly
- David Attenborough: Soft, breathy, measured pace, British accent
- News anchor: Clear articulation, neutral accent, steady pace

In Kokoro, this is captured in the **reference style embedding** (`ref_s`).

### F0 (Fundamental Frequency / Pitch)

The perceived pitch of speech, critical for:
- Questions vs statements (rising vs falling)
- Emphasis and stress patterns
- Emotional expression
- Speaker identity (average pitch)

### Alignment Matrix

Maps variable-length phonemes to fixed-length acoustic frames:

```
Phonemes: ["HH", "E", "L", "O"]
Durations: [3,    2,   4,   3] frames

Alignment Matrix:
       Frame: 0 1 2 3 4 5 6 7 8 9 10 11
    "HH":    1 1 1 0 0 0 0 0 0 0  0  0
    "E":     0 0 0 1 1 0 0 0 0 0  0  0
    "L":     0 0 0 0 0 1 1 1 1 0  0  0
    "O":     0 0 0 0 0 0 0 0 0 1  1  1
```

Each phoneme's features are repeated across its duration frames.

---

## CoreML Modifications

### Why Modifications Were Needed

Original Kokoro (PyTorch) uses features incompatible with CoreML:
- `pack_padded_sequence` (not supported in CoreML)
- Dynamic loops and in-place assignment
- Random phase initialization (non-deterministic)
- Variable-length sequences without masking

### Key Changes

| Component | Original Issue | Solution |
|-----------|----------------|----------|
| **TextEncoderFixed** | Uses `pack_padded_sequence` | Explicit LSTM states + masking |
| **TextEncoderPredictorFixed** | Same as above | Explicit states + AdaLayerNorm handling |
| **SineGenDeterministic** | Random phases | Controlled `random_phases` input |
| **GeneratorDeterministic** | Non-deterministic noise | Deterministic F0-based noise |
| **Alignment Matrix** | In-place assignment | Pure broadcasting operations |

### Kokoro Library Patch

Required changes to `kokoro/istftnet.py`:

```python
# Before (incompatible with CoreML)
fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

# After (CoreML compatible)
harmonic_range = torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device)
harmonic_range = harmonic_range.view(1, 1, -1)
fn = f0 * harmonic_range

# Also required
out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2.0, dtype=torch.float32))
```

Full patch: [kokoro_coreml_fix.patch](kokoro_coreml_fix.patch)

---

### Project Structure

```
tts/
├── v21.py                 # Main conversion script
│   ├── TextEncoderFixed               (Lines 197-281)
│   ├── TextEncoderPredictorFixed      (Lines 115-194)
│   ├── SineGenDeterministic           (Lines 294-325)
│   ├── SourceModuleHnNSFDeterministic (Lines 330-347)
│   ├── GeneratorDeterministic         (Lines 350-422)
│   └── KokoroCompleteCoreML           (Lines 453-578)
├── main.py               # Quick test script
├── requirements.txt      # Dependencies
└── kokoro_coreml_fix.patch  # Kokoro library patch
```

---

## License

This project builds upon:
- **Kokoro-82M**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **StyleTTS2**: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
