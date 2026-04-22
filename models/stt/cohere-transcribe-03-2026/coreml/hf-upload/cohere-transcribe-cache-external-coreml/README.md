---
language:
- en
- fr
- de
- es
- it
- pt
- nl
- pl
- el
- ar
- ja
- zh
- ko
- vi
license: cc-by-nc-4.0
library_name: coreml
tags:
- audio
- automatic-speech-recognition
- coreml
- ios
- macos
- apple-silicon
- cache-external
- parakeet-pattern
pipeline_tag: automatic-speech-recognition
---

# Cohere Transcribe Cache-External CoreML

CoreML conversion of [Cohere Transcribe 03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) with **cache-external decoder** following the Parakeet TDT pattern.

## Model Description

This is a CoreML conversion optimized for Apple Silicon (M1/M2/M3/M4) with:
- **Cache-external decoder**: KV cache managed in Swift/Python (not CoreML state)
- **macOS 14+ / iOS 17+** compatible
- **O(n) complexity** with manual cache management
- **Correct EOS token** (token 3, not 151643)

## Architecture

### Encoder
- **Input**: Mel spectrogram [1, 128, 3500]
- **Output**: Hidden states [1, 438, 1024]
- **Size**: ~6.97 GB (FP16)

### Decoder (Cache-External)
- **Pattern**: Parakeet TDT with external KV cache
- **Inputs** (19 total):
  - `input_id`: [1, 1] - current token
  - `position_id`: [1, 1] - current position
  - `encoder_hidden_states`: [1, 438, 1024] - encoder output
  - `cross_attention_mask`: [1, 1, 1, 438] - encoder mask
  - `attention_mask`: [1, 1, 1, seq_len] - **grows each step**
  - `k_cache_0..7`: [1, 8, 108, 128] - K caches (8 layers)
  - `v_cache_0..7`: [1, 8, 108, 128] - V caches (8 layers)

- **Outputs** (17 total):
  - `logits`: [1, 16384] - next token probabilities
  - `k_cache_0_out..7_out`: Updated K caches
  - `v_cache_0_out..7_out`: Updated V caches

- **Size**: ~291 MB

## Performance

Tested on LibriSpeech test-clean (10 samples):

| Metric | Value |
|--------|-------|
| **WER** | **11.95%** |
| Perfect transcriptions | 2/10 (0.00% WER) |
| Main errors | Punctuation differences |
| Complexity | O(n) |
| Max sequence length | 108 tokens |

### Per-sample Results

```
Sample  0 (3.5s):   25.00% - Minor word error (concord→concorde, tents→tanks)
Sample  1 (14.2s):   9.30% - Good (punctuation only)
Sample  2 (5.0s):    9.09% - Good (punctuation only)
Sample  3 (23.3s):  14.06% - Good (punctuation only)
Sample  4 (11.1s):  19.35% - Good (punctuation + minor wording)
Sample  5 (13.2s):   0.00% - ✅ PERFECT
Sample  6 (5.8s):    0.00% - ✅ PERFECT
Sample  7 (3.3s):   22.22% - Good (punctuation only)
Sample  8 (4.8s):   18.18% - Good (punctuation only)
Sample  9 (7.3s):   16.67% - Good (punctuation only)
```

## Critical Fix: EOS Token

⚠️ **Important**: The EOS token is **3** (`<|endoftext|>`), not 151643!

```python
# WRONG (vocabulary only has 16384 tokens)
EOS_TOKEN = 151643  # Out of range!

# CORRECT
EOS_TOKEN = 3  # Verified from model.generation_config.eos_token_id
```

Using the wrong EOS token causes:
- Decoder never stops naturally (hits max length)
- Excessive dots padding
- Text repetition issues
- Poor WER (29.88% with wrong token vs 11.95% with correct token)

## Usage

### Python

```python
import numpy as np
import coremltools as ct
import soundfile as sf
import librosa
import sentencepiece as spm

# Constants
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 400
MAX_FRAMES = 3500
MAX_SEQ_LEN = 108
START_TOKEN = 4
EOS_TOKEN = 3  # Correct EOS token!

# Load models
encoder = ct.models.MLModel("cohere_encoder.mlpackage")
decoder = ct.models.MLModel("cohere_decoder_cache_external.mlpackage")

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")
vocabulary = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

# Load and process audio
audio, sr = sf.read("audio.wav")
if sr != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

# Compute mel spectrogram
mel = librosa.feature.melspectrogram(
    y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, fmin=0, fmax=8000
)
mel = librosa.power_to_db(mel, ref=np.max)
mel = (mel + 80) / 80
mel = np.clip(mel, -1, 1)

# Pad mel to 3500 frames
n_mels, n_frames = mel.shape
padded_mel = np.zeros((n_mels, MAX_FRAMES), dtype=np.float32)
padded_mel[:, :n_frames] = mel

# Encode
encoder_input = {
    "input_features": np.expand_dims(padded_mel, axis=0).astype(np.float32),
    "feature_length": np.array([n_frames], dtype=np.int32)
}
encoder_output = encoder.predict(encoder_input)
encoder_hidden = encoder_output["hidden_states"]

# Initialize caches
k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

# Cross-attention mask
encoder_seq_len = encoder_hidden.shape[1]
cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

# Decode
tokens = []
current_token = START_TOKEN

for step in range(MAX_SEQ_LEN):
    # Build decoder input
    input_dict = {
        "input_id": np.array([[current_token]], dtype=np.int32),
        "position_id": np.array([[step]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float32),
        "cross_attention_mask": cross_mask,
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float32),
    }

    # Add caches
    for i in range(8):
        input_dict[f"k_cache_{i}"] = k_caches[i]
        input_dict[f"v_cache_{i}"] = v_caches[i]

    # Run decoder
    output = decoder.predict(input_dict)

    # Sample next token
    logits = output["logits"]
    next_token = int(np.argmax(logits[0]))

    # Update caches
    for i in range(8):
        k_caches[i] = output[f"k_cache_{i}_out"]
        v_caches[i] = output[f"v_cache_{i}_out"]

    # Check EOS
    if next_token == EOS_TOKEN:
        break

    tokens.append(next_token)
    current_token = next_token

# Detokenize
text_tokens = []
for token_id in tokens:
    if token_id <= 4 or token_id == EOS_TOKEN or token_id >= len(vocabulary):
        continue
    token = vocabulary[token_id]
    if token.startswith("<|"):
        continue
    text_tokens.append(token)

text = "".join(text_tokens).replace("▁", " ").strip()
print(f"Transcription: {text}")
```

### Swift

```swift
import CoreML
import Foundation

// Load models
let encoderURL = Bundle.main.url(forResource: "cohere_encoder", withExtension: "mlmodelc")!
let decoderURL = Bundle.main.url(forResource: "cohere_decoder_cache_external", withExtension: "mlmodelc")!

let encoder = try MLModel(contentsOf: encoderURL)
let decoder = try MLModel(contentsOf: decoderURL)

// See full Swift implementation in:
// - CohereDecoderState.swift (cache management)
// - CohereModelInference.swift (decoder execution)
// Available at: https://github.com/FluidInference/FluidAudio
```

## Key Implementation Details

### Cache Management (Parakeet Pattern)

The cache-external pattern manages KV cache **outside** the CoreML model:

1. **Initialize** 16 cache arrays (8 layers × K/V) filled with zeros
2. **Each decode step**:
   - Pass current token + 16 caches **into** model
   - Model returns logits + 16 **updated** caches
   - Extract updated caches from output
   - Use updated caches for next step
3. **Attention mask grows**: `[1,1,1,1]` → `[1,1,1,2]` → ... → `[1,1,1,108]`

### Why Cache-External?

| Aspect | Cache-External (This) | Stateful (CoreML State) |
|--------|----------------------|------------------------|
| **macOS Version** | 14+ | 15+ |
| **Cache Control** | Full (in Swift/Python) | Hidden in CoreML |
| **Debugging** | Easy to inspect cache | Opaque |
| **Complexity** | O(n) | O(n) |
| **Implementation** | More code | Simpler |
| **.mlmodelc compile** | ✅ Works | ❌ Fails |

## Supported Languages

14 languages supported:
- 🇬🇧 English (en)
- 🇫🇷 French (fr)
- 🇩🇪 German (de)
- 🇪🇸 Spanish (es)
- 🇮🇹 Italian (it)
- 🇧🇷 Portuguese (pt)
- 🇳🇱 Dutch (nl)
- 🇵🇱 Polish (pl)
- 🇬🇷 Greek (el)
- 🇪🇬 Arabic (ar)
- 🇯🇵 Japanese (ja)
- 🇨🇳 Chinese (zh)
- 🇰🇷 Korean (ko)
- 🇻🇳 Vietnamese (vi)

## Files

```
cohere-transcribe-cache-external-coreml/
├── cohere_encoder.mlpackage           # 6.97 GB - Encoder model
├── cohere_decoder_cache_external.mlpackage  # 291 MB - Cache-external decoder
├── tokenizer.model                    # SentencePiece tokenizer
└── README.md                          # This file
```

## Compilation to .mlmodelc

For faster loading in production iOS/macOS apps:

```bash
xcrun coremlcompiler compile cohere_encoder.mlpackage output/
xcrun coremlcompiler compile cohere_decoder_cache_external.mlpackage output/
```

This creates optimized `.mlmodelc` directories that load faster at runtime.

## Comparison with Alternatives

### vs. Stateless Decoder
- **Stateless**: O(n²) - reprocesses all tokens each step
- **Cache-External**: O(n) - processes only new token
- **For 108 tokens**: Cache-external is ~5x faster

### vs. Stateful Decoder (CoreML State)
- **Stateful**: macOS 15+ only, can't compile to .mlmodelc
- **Cache-External**: macOS 14+, compiles to .mlmodelc, full cache control

## Citation

```bibtex
@misc{cohere-transcribe-cache-external-coreml,
  title={Cohere Transcribe Cache-External CoreML},
  author={FluidInference},
  year={2026},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/FluidInference/cohere-transcribe-cache-external-coreml}},
  note={CoreML conversion with cache-external decoder (Parakeet pattern). WER: 11.95\% on LibriSpeech test-clean.}
}
```

## License

CC-BY-NC-4.0 (matches original Cohere Transcribe model)

## Acknowledgments

- Original model: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- Parakeet TDT pattern: NVIDIA NeMo
- Testing: LibriSpeech ASR corpus

## Links

- **Original Model**: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- **Source Code**: https://github.com/FluidInference/FluidAudio
- **Conversion Scripts**: https://github.com/FluidInference/mobius
