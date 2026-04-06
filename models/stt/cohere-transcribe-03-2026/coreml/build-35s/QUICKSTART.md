# Quick Start Guide

Models are now live on HuggingFace at:
**https://huggingface.co/FluidInference/cohere-transcribe-03-2026-coreml**

## Installation

### 1. Download Models

```bash
# Download FP16 models (compiled .mlmodelc included for instant loading)
huggingface-cli download FluidInference/cohere-transcribe-03-2026-coreml \
  f16 --local-dir ./models/f16
```

This downloads ~7.7 GB to `./models/f16/`

### 2. Install Dependencies

```bash
cd models/f16

# Option 1: Fast install with uv (recommended)
uv sync

# Option 2: Standard pip
pip install -r requirements.txt
```

## Usage

### Minimal Example (50 lines)

```bash
cd models/f16
python quickstart.py audio.wav
```

### Complete Example (Multi-language support)

```bash
cd models/f16

# English (default)
python example_inference.py audio.wav

# Japanese
python example_inference.py audio.wav --language ja

# Spanish with longer output
python example_inference.py audio.wav --language es --max-tokens 256

# All options
python example_inference.py --help
```

## Performance

With compiled `.mlmodelc` models:
- **Load time:** ~1 second (instant!)
- **Encoding:** ~800ms for 30s audio (on M3 Max)
- **Decoding:** ~15ms per token

**Total:** ~2-3 seconds for 30 seconds of audio

## Supported Languages (14)

English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Dutch (nl), Swedish (sv), Turkish (tr), Russian (ru), Chinese (zh), Japanese (ja), Korean (ko)

## Python API

```python
from cohere_mel_spectrogram import CohereMelSpectrogram
import coremltools as ct
import soundfile as sf
import numpy as np
import json

# Load models (compiled .mlmodelc loads instantly)
encoder = ct.models.MLModel("cohere_encoder.mlmodelc")
decoder = ct.models.MLModel("cohere_decoder_stateful.mlmodelc")
vocab = {int(k): v for k, v in json.load(open("vocab.json")).items()}

# Load audio
audio, _ = sf.read("audio.wav", dtype="float32")

# Preprocess
mel = CohereMelSpectrogram()(audio)
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, max(0, 3500 - mel.shape[2]))))[:, :, :3500]

# Encode
encoder_out = encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([min(mel.shape[2], 3500)], dtype=np.int32)
})

# Decode (see example_inference.py for complete loop)
# ...

# Result
print(text)
```

See `example_inference.py` for the complete decoding implementation.

## Troubleshooting

### "No module named 'coremltools'"
```bash
pip install coremltools numpy soundfile
```

### Model not found
Make sure you're in the `models/f16/` directory where you downloaded the models.

### "Unable to load libmodelpackage"
You're using system Python 3.14 which has CoreML issues. Use Python 3.10-3.12:
```bash
brew install python@3.12
python3.12 -m pip install -r requirements.txt
python3.12 example_inference.py audio.wav
```

### Slow first load
If models take ~20 seconds to load:
- You're loading `.mlpackage` instead of `.mlmodelc`
- Make sure the examples point to `.mlmodelc` (they should by default)

### Platform Requirements
- macOS 15.0+ (Sequoia) or iOS 18.0+
- Apple Silicon (M1/M2/M3/M4 or A-series)
- 8 GB RAM minimum (16 GB recommended)

## What's Included

```
f16/
├── cohere_encoder.mlmodelc         # 3.6 GB - Encoder (compiled, instant load)
├── cohere_encoder.mlpackage        # 3.6 GB - Encoder (source)
├── cohere_decoder_stateful.mlmodelc # 291 MB - Decoder (compiled)
├── cohere_decoder_stateful.mlpackage # 291 MB - Decoder (source)
├── vocab.json                      # Vocabulary
├── cohere_mel_spectrogram.py       # Audio preprocessor
├── example_inference.py            # Complete CLI example
├── quickstart.py                   # Minimal 50-line example
├── requirements.txt                # pip dependencies
├── pyproject.toml / uv.lock        # uv dependencies
└── README.md                       # Full documentation
```

## Known Limitations

- 36% of samples may fail due to encoder training bias (quiet/high-pitched voices)
- Max 35 seconds per audio chunk (longer audio needs chunking)
- Max 108 output tokens (~15-25 seconds of speech)

See full documentation: https://huggingface.co/FluidInference/cohere-transcribe-03-2026-coreml/blob/main/f16/README.md

## Links

- **Model Repository:** https://huggingface.co/FluidInference/cohere-transcribe-03-2026-coreml
- **Original Model:** https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- **FluidAudio (Swift):** https://github.com/FluidInference/FluidAudio
