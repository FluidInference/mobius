# Cohere Transcribe 03-2026 - CoreML (Quantized INT8)

## Overview
Quantized CoreML conversion of the Cohere Transcribe 03-2026 speech recognition model, optimized for Apple Neural Engine and Apple Silicon devices.

## Model Description

A 3-model ASR (Automatic Speech Recognition) pipeline consisting of:

1. **Audio Encoder** (1.4GB quantized) - Converts Mel spectrogram to hidden states
2. **Decoder** (109MB quantized) - Transformer with KV cache
3. **Cross-KV Projector** (12MB quantized) - Pre-computes cross-attention keys/values

## Quantization

- **Type**: INT8 weight quantization
- **Size Reduction**: ~2.6x smaller than FP16
- **Quality**: Minimal degradation (<1% WER increase)
- **Speed**: Faster inference on CPU, same ANE performance

## Supported Languages
English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish, Arabic, Chinese, Japanese, Korean, and Vietnamese (14 languages total)

## Model Formats

Both `.mlpackage` (source) and `.mlmodelc` (compiled) formats included:

- **mlpackage**: Universal format, device-agnostic
- **mlmodelc**: Pre-compiled for macOS/iOS, faster first load

## Quick Start Example

```swift
import FluidAudio

let manager = try await CohereAsrManager(language: .english)
let result = try await manager.transcribe(audioBuffer)
print(result.text)
```

## Hardware Requirements

- **Minimum**: M1 / A14+ chip
- **Recommended**: M2+ chip
- **Memory**: 2.5GB (reduced from 4.5GB)
- **Platform**: macOS 14+, iOS 17+

## Performance Metrics

| Metric | FP16 | INT8 (This Model) |
|--------|------|-------------------|
| Size | 4.2GB | 1.6GB |
| Memory Usage | 4.5GB | 2.5GB |
| WER (English) | 5.8% | 6.1% |
| Real-Time Factor | 3.2x | 3.0x |
| ANE Utilization | 95% | 95% |

*Tested on FLEURS benchmark samples*

## Base Model
- Original: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- License: MIT
- Conversion: FluidInference
