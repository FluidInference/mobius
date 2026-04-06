# Cohere Transcribe CoreML (INT8, 35-Second Window)

CoreML models for [Cohere Transcribe (March 2026)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), quantized to INT8 for 50% smaller size with preserved quality.

## Quick Start

```bash
# Download Q8 models
huggingface-cli download FluidInference/cohere-transcribe-03-2026-coreml \
  q8 --local-dir ./models/q8

# Install & run
cd models/q8
pip install -r requirements.txt
python quickstart.py audio.wav

# Multi-language
python example_inference.py audio.wav --language ja  # Japanese
```

**First load:** ~20s (ANE compilation), then cached for instant reuse (~1s)

## Model Specifications

| Component | Size | Format | Quantization |
|-----------|------|--------|--------------|
| Encoder | 1.8 GB | ML Program (.mlpackage) | INT8 (49.2% reduction) |
| Decoder | 146 MB | ML Program (.mlpackage) | INT8 (49.8% reduction) |
| Vocabulary | 331 KB | JSON (16,384 tokens) | - |

**Total:** 2.0 GB INT8 (was 3.9 GB FP16)

### Architecture
- **Type:** Encoder-decoder (Conformer + Transformer)
- **Languages:** 14 (en, es, fr, de, it, pt, pl, nl, sv, tr, ru, zh, ja, ko)
- **Window:** 35 seconds (3500 frames @ 10ms)
- **Output:** Up to 108 tokens (~15-25 seconds of speech)
- **Cache:** GPU-resident stateful KV cache
- **Quantization:** W8A16 (INT8 weights, FP16 activations)

## Quality Metrics

Tested on LibriSpeech test-clean (10 samples):
- **Average WER:** 11.42% (punctuation-normalized)
- **Perfect matches:** 90% (WER < 5%)
- **Performance:** 0.28x RTFx (faster than real-time)
- **Quality vs FP16:** Identical (90% perfect match rate)

**Known limitation:** ~10% of samples fail due to encoder training bias (quiet/high-pitched voices).

## Files

```
cohere_encoder.mlpackage         # 1.8 GB - Encoder (INT8)
cohere_decoder_stateful.mlpackage # 146 MB - Stateful decoder (INT8)
vocab.json                       # 331 KB - Vocabulary
cohere_mel_spectrogram.py        # Audio preprocessor (pure Python)
example_inference.py             # Complete CLI example
quickstart.py                    # Minimal 50-line example
requirements.txt                 # pip dependencies
pyproject.toml + uv.lock         # uv dependencies
```

## Platform Requirements

- **macOS:** 15.0+ (Sequoia) / **iOS:** 18.0+
- **Hardware:** Apple Silicon (M1/M2/M3/M4 or A-series)
- **RAM:** 6 GB minimum (8 GB recommended for Q8)
- **Python:** 3.10-3.13 recommended

**Note:** Stateful decoder requires macOS 15+ / iOS 18+ for CoreML State API.

## Usage

### Python (Minimal)

```python
from cohere_mel_spectrogram import CohereMelSpectrogram
import coremltools as ct
import soundfile as sf
import numpy as np
import json

# Load models
encoder = ct.models.MLModel("cohere_encoder.mlpackage")
decoder = ct.models.MLModel("cohere_decoder_stateful.mlpackage")
vocab = {int(k): v for k, v in json.load(open("vocab.json")).items()}

# Load and preprocess audio
audio, _ = sf.read("audio.wav", dtype="float32")
mel = CohereMelSpectrogram()(audio)
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, max(0, 3500 - mel.shape[2]))))[:, :, :3500]

# Encode
encoder_out = encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([min(mel.shape[2], 3500)], dtype=np.int32)
})

# Decode (see example_inference.py for complete loop)
# ...

print(text)
```

See `example_inference.py` for the complete implementation.

### Swift (FluidAudio)

See [FluidAudio integration guide](https://github.com/FluidInference/FluidAudio) for Swift usage.

## Performance

Tested on MacBook Pro M3 Max:

| Component | ANE % | Latency |
|-----------|-------|---------|
| Encoder (first load) | - | ~20s (compilation) |
| Encoder (cached) | 95% | ~800ms |
| Decoder (per token) | 85% | ~15ms |

**Total:** ~2-3 seconds for 30 seconds of audio (after initial compilation)

**Note:** INT8 quantization provides same performance as FP16 with 50% smaller model size.

## Quantization Details

- **Method:** Linear symmetric quantization (per-channel)
- **Format:** W8A16 (INT8 weights, FP16 activations)
- **Quality:** Preserved (90% perfect matches, same as FP16)
- **Size reduction:** 49-50% smaller
- **Speed:** Same as FP16 (no performance degradation)

The quantization uses CoreML's `linear_quantize_weights` with symmetric quantization:
- Encoder: 3.58 GB → 1.82 GB
- Decoder: 0.28 GB → 0.14 GB

## Model Format

These models use **ML Program** format (not neural network format). ML Program models:
- ✅ Must be in `.mlpackage` format (only supported format)
- ✅ Support advanced operations (better accuracy/performance)
- ✅ First load compiles to ANE, then cached
- ❌ Cannot be pre-compiled to `.mlmodelc` (not supported for ML Program)

The compilation happens automatically on first load and is cached by macOS for subsequent loads.

## Known Limitations

### Encoder Training Bias
~10% of samples fail due to encoder training data bias:
1. **Quiet speakers** (RMS < 0.03, 64% quieter than normal)
2. **High-pitched voices** (frequency > 1000 Hz, 62% higher than normal)

**Note:** This is a model training issue, not a CoreML conversion issue. INT8 and FP16 produce identical results.

### Audio Length
| Duration | Status | Notes |
|----------|--------|-------|
| < 35s | ✅ Supported | Single-pass processing |
| 35-70s | ⚠️ Chunking | Process in 2× 35s segments with overlap |
| > 70s | ⚠️ Chunking | Process in multiple 30-35s segments |

The decoder max 108 tokens (~15-25s speech). For dense speech or long audio, chunking is required.

## Technical Details

### Encoder Architecture
- **Layers:** 24 Conformer layers
- **Subsample ratio:** ~8x (3500 frames → 438 outputs)
- **Projection:** 1024 → 1024 encoder-decoder projection
- **Parameters:** 1.9B (INT8 quantized)

### Decoder Architecture
- **Layers:** 8 transformer decoder layers
- **Attention:** 8 heads × 128 head_dim
- **Cache:** GPU-resident KV cache (CoreML State API)
- **Max sequence:** 108 tokens

### Vocabulary
- **Type:** SentencePiece BPE
- **Size:** 16,384 tokens
- **Special tokens:** BOS (13764), EOS (3), PAD (0)

## License

Same as the original [Cohere Transcribe model](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (Apache 2.0).

## Citation

```bibtex
@misc{cohere-transcribe-2026,
  title={Cohere Transcribe},
  author={Cohere},
  year={2026},
  url={https://huggingface.co/CohereLabs/cohere-transcribe-03-2026}
}
```

## Links

- **Model Repository:** https://huggingface.co/FluidInference/cohere-transcribe-03-2026-coreml
- **Original Model:** https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- **FluidAudio (Swift):** https://github.com/FluidInference/FluidAudio
- **CoreML Conversion:** https://github.com/FluidInference/mobius
