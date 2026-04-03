# Cohere Transcribe 03-2026 Conversion Notes

High-level notes on converting this model to CoreML for FluidAudio integration.

## Model Overview

- **Name**: Cohere Transcribe 03-2026
- **Size**: 2B parameters (2 billion)
- **Architecture**: Conformer encoder + Transformer decoder
- **HuggingFace**: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- **License**: Apache 2.0
- **Performance**: WER 5.42 (English ASR Leaderboard), RTFx ~524x

## Key Differences from Existing Models

### vs Parakeet TDT (0.6B)
- **3.3x larger** (2B vs 0.6B params)
- **Different architecture**: Conformer vs FastConformer
- **Different decoder**: Transformer vs RNN-T (TDT)
- **More languages**: 14 vs English-only

### vs Qwen3 ASR (0.6B)
- **3.3x larger** (2B vs 0.6B params)
- **Different encoder**: Conformer vs Qwen3 audio tower
- **Same decoder family**: Both use Transformer decoders
- **More languages**: 14 vs Chinese/multilingual

## Conversion Strategy

### Component Split

```
CohereAsrForConditionalGeneration
├── audio_encoder (Mel + Conformer)  → cohere_audio_encoder.mlpackage
├── decoder (Transformer)            → cohere_decoder.mlpackage
└── lm_head (Projection)             → cohere_lm_head.mlpackage
```

This mirrors the Qwen3 conversion approach.

### Expected Challenges

1. **Size**: 2B params may exceed ANE memory limits
   - Solution: Quantization (INT8/INT4)
   - Fallback: GPU/CPU execution

2. **Conformer ops**: May have ANE-incompatible operations
   - Solution: Profile with `coreml-cli --fallback`
   - Fallback: Accept CPU fallback for some layers

3. **Compilation time**: 5-10 min on first load
   - Solution: Document in README, show loading UI in VoiceInk
   - Similar to Parakeet v3-large behavior

4. **KV cache**: Decoder needs state management
   - Solution: Similar to Qwen3 stateful decoder
   - May need separate prefill/decode models

## Integration Plan

### mobius
- Conversion scripts in `mobius/models/stt/cohere-transcribe-03-2026/coreml/`
- Status tracking in `CONVERSION_STATUS.md`
- PR to mobius with conversion scripts and trial notes

### HuggingFace
- Repo: `FluidInference/cohere-transcribe-03-2026-coreml`
- Upload all `.mlpackage` files
- Model card with benchmarks and usage

### FluidAudio
- Manager: `Sources/FluidAudio/ASR/Cohere/CohereAsrManager.swift`
- Model registration in `ModelNames.swift`
- CLI command: `fluidaudiocli cohere-transcribe audio.wav`
- Tests: `Tests/FluidAudioTests/CohereAsrManagerTests.swift`

## Performance Expectations

Based on 2B params and Conformer architecture:

- **Compile time**: 5-10 minutes (first load on ANE)
- **Warm inference**: ~500ms for 30s audio (M-series)
- **RTFx**: ~60x (30s audio in ~0.5s)
- **Memory**: 4-6GB peak
- **Model size**:
  - FP32: ~8GB
  - FP16: ~4GB
  - INT8: ~2GB
  - INT4: ~1GB

## Language Support

14 languages supported:
- **European**: English, French, German, Italian, Spanish, Portuguese, Greek, Dutch, Polish
- **Asian**: Chinese (Mandarin), Japanese, Korean, Vietnamese
- **MENA**: Arabic

Language code must be specified (no auto-detection).

## Timeline Estimate

- **Conversion**: 1-2 days (including trials)
- **Validation**: 1 day
- **Profiling**: 1 day
- **HuggingFace upload**: 1 day
- **FluidAudio integration**: 2-3 days
- **Total**: ~1 week

## References

- Model card: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- Conformer paper: https://arxiv.org/abs/2005.08100
- Similar conversion: `mobius/models/stt/qwen3-asr-0.6b/coreml/`
- Integration example: `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrManager.swift`
