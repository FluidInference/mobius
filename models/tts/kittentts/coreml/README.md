# KittenTTS Nano CoreML

Convert [KittenTTS Nano](https://huggingface.co/KittenML/kitten-tts-nano-0.1) (15M param distilled Kokoro/StyleTTS2) from ONNX to CoreML for on-device inference on iOS and macOS.

**Features**: 15M params (tiny) | 24kHz audio | CPU-optimized | FP32 CoreML | Single model

---

## Quick Start

### Prerequisites

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install torch coremltools onnx onnxruntime numpy scipy phonemizer
```

Also requires `espeak-ng` for phonemization:
```bash
brew install espeak-ng
```

### Convert

```bash
# 5-second model (70 max tokens)
python convert_kittentts.py --seconds 5 --output kittentts_5s.mlpackage

# 10-second model (140 max tokens)
python convert_kittentts.py --seconds 10 --output kittentts_10s.mlpackage

# Verify weights only (no conversion)
python convert_kittentts.py --verify-only
```

### Inference (Python)

```python
import numpy as np
import coremltools as ct

model = ct.models.MLModel("kittentts_5s.mlpackage")

# Phonemize text with espeak
import phonemizer, re
backend = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
phonemes = backend.phonemize(["Hello world"])[0]
tokens = ' '.join(re.findall(r"\w+|[^\w\s]", phonemes))

# Build input_ids from vocab (see convert_kittentts.py for full vocab string)
input_ids = np.zeros((1, 70), dtype=np.int32)
# ... fill with token indices ...

# Load voice
voices = np.load("voices.npz")
ref_s = voices["expr-voice-2-m"].reshape(1, -1).astype(np.float32)

out = model.predict({
    "input_ids": input_ids,
    "ref_s": ref_s,
    "random_phases": np.random.randn(1, 9).astype(np.float32),
    "attention_mask": attention_mask,
    "source_noise": np.random.randn(1, 120000, 9).astype(np.float32),
})

audio = out["audio"].flatten()
length = int(out["audio_length_samples"].flatten()[0])
audio = audio[:length]  # tail is already zeroed
```

---

## Model Architecture

```
Text -> Phonemes -> ALBERT -> Duration -> Alignment -> F0/Energy -> Style -> Decoder -> Generator -> Audio
```

KittenTTS Nano is a distilled version of Kokoro/StyleTTS2 with the same architecture but smaller dimensions:

| Component | Kokoro-82M | KittenTTS Nano |
|-----------|-----------|----------------|
| ALBERT hidden | 768 | 768 |
| ALBERT embed | 128 | 128 |
| Style dim | 128 | 128 |
| Text encoder | 512 | 512 |
| Decoder blocks | 4 | 4 |
| Generator channels | 256->128->64 | 256->128->64 |
| Parameters | 82M | 15M |
| Quantization | None | INT8 (ONNX) |

### Key Components

- **ALBERT Encoder**: Shared-weight transformer for phoneme context (4 repeats of 1 layer)
- **Predictor**: Duration, F0, and energy prediction with bidirectional LSTMs
- **Decoder**: 4 AdaIN decode blocks with style conditioning
- **Generator**: ISTFTNet vocoder with Snake activations, harmonic source module

---

## Conversion Details

The ONNX model uses INT8 quantization (ConvInteger, MatMulInteger, DynamicQuantizeLSTM). The conversion pipeline:

1. **Extract & dequantize** ONNX weights (INT8 * scale + zero_point -> FP32)
2. **Reconstruct** PyTorch model architecture from ONNX graph analysis
3. **Load** dequantized weights into PyTorch model (561/573 parameters)
4. **Trace** with `torch.jit.trace` using fixed input shapes
5. **Convert** to CoreML mlprogram format (FP32, iOS 17+)

### Bugs Fixed During Conversion

| Bug | Impact | Fix |
|-----|--------|-----|
| LSTM gate order ONNX [i,o,f,c] vs PyTorch [i,f,g,o] | Silent wrong output | Reorder gate weights during loading |
| BERT weight mapping (embedding_hidden_mapping_in swapped) | Wrong text encoding | Swap weight assignment |
| BatchNorm1d instead of LayerNorm in TextEncoder | Different normalization | Replace with LayerNorm + LeakyReLU |
| LeakyReLU instead of Snake activation in resblocks | Robotic audio | Implement Snake: x + (1/a)*sin^2(a*x) |
| Resblock dilations (1,1,1) instead of (1,3,5) | ~2x volume loss | Set convs1 dilations to (1,3,5) |
| NoiseResBlock missing dilations | Degraded noise path | Add dilations parameter (1,3,5) |
| reflection_pad (3,3) instead of (1,0) | Wrong padding | Fix to nn.ReflectionPad1d((1,0)) |
| conv_post missing padding=3 | Frequency response error | Add padding=3 to Conv1d |
| Phase accumulation fp32 drift in CoreML | Robotic harmonics | Chunked cumsum with periodic wrapping |

### CoreML-Specific Fixes

- **Phase accumulation**: `torch.cumsum` over 42k steps causes fp32 precision drift between CoreML and PyTorch runtimes. Higher harmonics (9th at 1800Hz) lose correlation (0.79). Fix: reshape into 300-step frames, cumsum per frame, carry wrapped inter-frame phase.
- **Fixed frame count**: Model uses `fixed_total_frames` to avoid dynamic shape issues in traced graph.
- **Tail zeroing**: Audio buffer is zeroed past `audio_length_samples` so consumers don't need to trim.

---

## Inputs & Outputs

### Inputs

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `input_ids` | [1, N] | INT32 | Phoneme token IDs (0-padded) |
| `ref_s` | [1, 256] | FLOAT32 | Voice style vector (from voices.npz) |
| `random_phases` | [1, 9] | FLOAT32 | Initial harmonic phases |
| `attention_mask` | [1, N] | INT32 | 1=valid token, 0=padding |
| `source_noise` | [1, T, 9] | FLOAT32 | Stochastic noise for unvoiced regions |

N = max tokens (e.g. 70 for 5s model). T = max audio samples (e.g. 120000 for 5s).

### Outputs

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `audio` | [1, 1, T+20] | FLOAT32 | Audio waveform (24kHz), zeroed past valid length |
| `audio_length_samples` | [1] | INT32 | Number of valid audio samples |
| `pred_dur` | [1, N] | FLOAT32 | Predicted duration per token (frames) |

---

## Verification Results

Comparison with ONNX reference (same text, matched frame count):

| Metric | Value |
|--------|-------|
| CoreML vs PyTorch correlation | 0.963 |
| RMS ratio (CoreML/ONNX) | 0.99 |
| Whisper transcription match | Identical |
| Parameters loaded | 561/573 (12 use defaults) |

The 12 unloaded parameters are `predictor.text_encoder.lstms.{1,3,5,7,9,11}.norm.{weight,bias}` — LayerNorm layers that default to weight=1, bias=0, matching the ONNX constants.

---

## Source Model

- **Model**: [KittenML/kitten-tts-nano-0.1](https://huggingface.co/KittenML/kitten-tts-nano-0.1)
- **Format**: ONNX (INT8 quantized, 23.8 MB)
- **Sample rate**: 24kHz
- **Voices**: 6 voices in `voices.npz`
- **Architecture**: Distilled Kokoro/StyleTTS2

---

## Files

```
coreml/
├── convert_kittentts.py          # Conversion script (model architecture + weight loading + CoreML export)
├── README.md                     # This file
├── kitten_tts_nano_weights.npz   # Extracted dequantized weights (numpy)
└── kitten_tts_nano_weights.pt    # Extracted weights (PyTorch state dict)
```

---

**Requires**: iOS 17+ / macOS 14+ | Python 3.10+ | coremltools 9.0+
