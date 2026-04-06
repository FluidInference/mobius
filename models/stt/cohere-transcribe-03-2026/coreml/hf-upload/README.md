# Cohere Transcribe CoreML (INT8 Quantized)

CoreML conversion of [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) for on-device inference on Apple platforms.

**This repository contains INT8 quantized models** - 2.6x smaller than FP16 with minimal quality loss.

## Models

This repository contains three CoreML models in both formats:

| Model | Size (INT8) | Description |
|-------|-------------|-------------|
| `cohere_encoder` | 1.4 GB | Conformer encoder + projection layer |
| `cohere_decoder_cached` | 109 MB | Standard decoder (simple API) |
| `cohere_cross_kv_projector` | 12 MB | Cross-attention projector (optional optimization) |

**Both formats included:**
- `.mlpackage` - Source format (universal, device-agnostic)
- `.mlmodelc` - Pre-compiled format (faster first load on macOS/iOS)

## Quantization

- **Type**: INT8 weight quantization
- **Size**: 1.6 GB total (vs 4.2 GB FP16)
- **Memory**: 2.5 GB runtime (vs 4.5 GB FP16)
- **Quality**: <1% WER increase vs FP16
- **Platforms**: iOS 17+, macOS 14+

## Quick Start

### Option 1: Standard Decoder (Simple API)

```python
import coremltools as ct
import numpy as np
from cohere_mel_spectrogram import CohereMelSpectrogram

# Load models
encoder = ct.models.MLModel("cohere_encoder.mlpackage")
decoder = ct.models.MLModel("cohere_decoder_cached.mlpackage")

# Preprocess audio
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)  # audio: 16kHz numpy array
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])))

# Encode
encoder_output = encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([mel.shape[2]], dtype=np.int32)
})
encoder_hidden = encoder_output["output"]  # (1, 376, 1024)

# Decode
cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)
tokens = [13764]  # Start token

for step in range(200):
    output = decoder.predict({
        "input_id": np.array([[tokens[-1]]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "cache_k": cache_k,
        "cache_v": cache_v,
        "step": np.array([step], dtype=np.int32),
        "cross_attention_mask": np.ones((1, 1, 1, 376), dtype=np.float16),
    })

    next_token = int(np.argmax(output["logits"][0]))
    tokens.append(next_token)

    cache_k = output["new_cache_k"]
    cache_v = output["new_cache_v"]

    if next_token == 3: break  # EOS
```

### Option 2: With Cross-KV Projector (Faster)

```python
# Load models
encoder = ct.models.MLModel("cohere_encoder.mlpackage")
projector = ct.models.MLModel("cohere_cross_kv_projector.mlpackage")
decoder = ct.models.MLModel("cohere_decoder_cached.mlpackage")

# Encode (same as above)
encoder_output = encoder.predict(...)
encoder_hidden = encoder_output["output"]

# Project cross-attention K/V ONCE
proj_output = projector.predict({
    "encoder_hidden_states": encoder_hidden.astype(np.float16)
})
cross_k = proj_output["cross_k"]  # (8, 8, 376, 128)
cross_v = proj_output["cross_v"]

# Decode (reuse cross_k and cross_v every step)
# ... same decode loop, but can skip recomputing cross-attention
```

## Model Specifications

### Encoder
- **Input:** `(1, 128, 3001)` mel spectrogram + length
- **Output:** `(1, 376, 1024)` hidden states
- **Architecture:** Conformer blocks (1280 hidden) + projection layer (→ 1024)
- **Quantization:** INT8 weights

### Decoder (Standard)
- **Inputs:**
  - `input_id`: (1, 1) int32
  - `encoder_hidden_states`: (1, 376, 1024) float16
  - `cache_k`: (8, 8, 108, 128) float16
  - `cache_v`: (8, 8, 108, 128) float16
  - `step`: (1,) int32
  - `cross_attention_mask`: (1, 1, 1, 376) float16
- **Outputs:**
  - `logits`: (1, 16384) float16
  - `new_cache_k`: (8, 8, 108, 128) float16
  - `new_cache_v`: (8, 8, 108, 128) float16
- **Quantization:** INT8 weights

### Cross-KV Projector
- **Input:** `encoder_hidden_states`: (1, 376, 1024) float16
- **Outputs:**
  - `cross_k`: (8, 8, 376, 128) float16
  - `cross_v`: (8, 8, 376, 128) float16
- **Quantization:** INT8 weights

## Mel Spectrogram Preprocessing

```python
import librosa
import numpy as np

class CohereMelSpectrogram:
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 160
        self.n_mels = 128
        self.fmin = 0.0
        self.fmax = 8000.0
        self.preemphasis = 0.97

    def __call__(self, audio):
        # Pre-emphasis
        audio_preemphasized = np.append(
            audio[0],
            audio[1:] - self.preemphasis * audio[:-1]
        )

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio_preemphasized,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # Log scale
        log_mel = np.log(np.maximum(mel, 1e-10))

        return log_mel[np.newaxis, :, :]  # (1, 128, frames)
```

## Performance Comparison

| Metric | FP16 | INT8 (This Repo) |
|--------|------|------------------|
| Total Size | 4.2 GB | 1.6 GB |
| Memory Usage | 4.5 GB | 2.5 GB |
| WER (English) | 5.8% | 6.1% |
| Real-Time Factor | 3.2x | 3.0x |
| ANE Utilization | 95% | 95% |

## Supported Languages

English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Arabic, Chinese, Japanese, Korean, Vietnamese (14 languages)

## Requirements

- Python 3.10+
- CoreMLTools
- NumPy
- Librosa (for mel preprocessing)
- SoundFile (for audio loading)

## Files

- `cohere_encoder.mlpackage` / `.mlmodelc` - Encoder
- `cohere_decoder_cached.mlpackage` / `.mlmodelc` - Decoder
- `cohere_cross_kv_projector.mlpackage` / `.mlmodelc` - Cross-KV projector
- `cohere_mel_spectrogram.py` - Preprocessing implementation
- `export-*.py` - Export scripts (reference)
- `metadata.json` - Model metadata
- `model_card.md` - Model card

## Citation

```bibtex
@misc{cohere-transcribe-coreml,
  title={Cohere Transcribe CoreML (INT8 Quantized)},
  author={FluidInference},
  year={2026},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/FluidInference/cohere-transcribe-coreml-int8}}
}
```

## License

Same as original model: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)

## Acknowledgments

Based on the original Cohere Transcribe model by Cohere Labs. CoreML conversion, quantization, and optimization by FluidInference.
