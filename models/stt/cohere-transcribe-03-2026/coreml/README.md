# Cohere Transcribe 03-2026 CoreML Conversion

Converts Cohere Transcribe 03-2026 (2B parameter Conformer-based ASR model) to CoreML for on-device Apple inference.

## Model Overview

- **Architecture**: Large Conformer encoder + lightweight Transformer decoder
- **Parameters**: 2 billion (2B)
- **Languages**: 14 (EN, FR, DE, IT, ES, PT, EL, NL, PL, ZH, JA, KO, VI, AR)
- **Input**: 16kHz mono audio → log-Mel spectrogram
- **Output**: Transcribed text with punctuation
- **Source**: [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- **License**: Apache 2.0

## Key Features

- **No timestamps**: Unlike Whisper, this model produces text-only output
- **No diarization**: Speaker identification not supported
- **Punctuation control**: Can toggle punctuation on/off
- **Multi-language**: Supports 14 languages (language code required, no auto-detection)
- **Long-form audio**: Handles audio >35s via automatic chunking

## Known Challenges

1. **Size**: 2B parameters → Large model for ANE, may require quantization
2. **Memory**: May exceed ANE limits on older devices (iPhone 12, M1)
3. **Compilation time**: Expect 5-10 min first load on ANE (similar to Parakeet v3-large)
4. **No VAD**: Model transcribes silence/noise (requires VAD preprocessing)

## Environment Setup

```bash
cd mobius/models/stt/cohere-transcribe-03-2026/coreml
uv sync
```

## Prerequisites

### HuggingFace Access (Required)

This model is **gated** on HuggingFace. You must:

1. **Request access**: Visit [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and click "Request Access"
2. **Authenticate**: Run `huggingface-cli login` and provide your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Login
huggingface-cli login
```

## Conversion

### Full conversion (all components)

```bash
uv run python convert-cohere-transcribe.py \
  --output-dir ./build/cohere-transcribe
```

### Component-specific conversion

```bash
# Audio encoder only (mel spectrogram + Conformer encoder)
uv run python convert-cohere-transcribe.py \
  --components audio_encoder \
  --output-dir ./build/cohere-transcribe

# Decoder only
uv run python convert-cohere-transcribe.py \
  --components decoder \
  --output-dir ./build/cohere-transcribe

# All components
uv run python convert-cohere-transcribe.py \
  --components audio_encoder,decoder,lm_head \
  --output-dir ./build/cohere-transcribe
```

## Validation

Compare PyTorch vs CoreML outputs to verify numerical parity:

```bash
uv run python compare-models.py \
  --audio-file /path/to/test.wav \
  --coreml-dir ./build/cohere-transcribe \
  --language en
```

## Model Components

The model is split into multiple CoreML packages:

1. **Audio Encoder** (`cohere_audio_encoder.mlpackage`)
   - Input: Raw audio waveform (16kHz)
   - Output: Acoustic embeddings
   - Contains: Mel spectrogram frontend + Conformer encoder

2. **Decoder** (`cohere_decoder.mlpackage`)
   - Input: Acoustic embeddings + text token IDs
   - Output: Hidden states
   - Contains: Transformer decoder with KV cache

3. **LM Head** (`cohere_lm_head.mlpackage`)
   - Input: Hidden states
   - Output: Logits over vocabulary
   - Contains: Final projection layer

## Expected Performance

Based on similar 2B models:

- **Compilation time**: 5-10 minutes first load (ANE optimization)
- **Warm inference**: ~500ms for 30s audio on M-series chips
- **RTFx**: ~60x (real-time factor, 30s audio in 0.5s)
- **Memory**: ~4-6GB peak during inference

## Quantization Options

To reduce model size for deployment:

```bash
# INT8 quantization (2x compression, minimal quality loss)
uv run python convert-cohere-transcribe.py \
  --output-dir ./build/cohere-transcribe-int8 \
  --quantize int8

# INT4 quantization (4x compression, some quality loss)
uv run python convert-cohere-transcribe.py \
  --output-dir ./build/cohere-transcribe-int4 \
  --quantize int4
```

## Profiling

Use `coreml-cli` to analyze device assignment and performance:

```bash
cd ../../tools/coreml-cli
uv sync

# Benchmark latency across compute unit configs
uv run coreml-cli ../../models/stt/cohere-transcribe-03-2026/coreml/build/cohere-transcribe/cohere_audio_encoder.mlmodelc

# Check ANE compatibility (CPU fallback ops)
uv run coreml-cli ../../models/stt/cohere-transcribe-03-2026/coreml/build/cohere-transcribe/cohere_audio_encoder.mlmodelc --fallback
```

## Integration with FluidAudio

After successful conversion and validation:

1. Upload converted models to HuggingFace: `FluidInference/cohere-transcribe-03-2026-coreml`
2. Register model in FluidAudio's `ModelNames.swift`
3. Implement inference manager in `Sources/FluidAudio/ASR/Cohere/`
4. Add CLI command to `fluidaudiocli`
5. Write tests and benchmarks

## Conversion Notes

### ✅ Conversion Status: SUCCESSFUL

**Date**: 2026-04-03
**Result**: All three components successfully converted to CoreML

| Component | Size | Conversion Time |
|-----------|------|-----------------|
| Audio Encoder | 3.6 GB | ~90 seconds |
| Decoder | 293 MB | ~85 seconds |
| LM Head | 32 MB | ~4 seconds |

**Total size**: 3.9 GB (FP32, unquantized)

### Critical Success Factor: Dependency Versions

The conversion **requires exact dependency versions** matching Parakeet v3:

```toml
requires-python = "==3.10.12"
dependencies = [
    "coremltools==9.0b1",      # NOT 9.0 (beta handles dynamic ops better)
    "torch==2.7.0",             # NOT 2.11.0
    "transformers==4.57.6",     # NOT older versions
    "scikit-learn==1.5.1",
]
```

**Key insight**: Using coremltools 9.0 (stable) instead of 9.0b1 (beta) causes dynamic shape tracing errors. The beta version has better handling of conditional operations.

### Trial 1: Gated model access (RESOLVED)

- **Error**: `OSError: You are trying to access a gated repo`
- **Solution**:
  1. Request access at HuggingFace model page
  2. Run `huggingface-cli login` with token
- **Time to resolve**: ~5 minutes (waiting for access approval)

### Trial 2: Missing dependencies (RESOLVED)

- **Error**: `ImportError: No module named 'sentencepiece'`
- **Solution**: Added `sentencepiece==0.2.0` to pyproject.toml
- **Root cause**: Custom tokenizer requires SentencePiece

### Trial 3: Wrong model attributes (RESOLVED)

- **Error**: `AttributeError: object has no attribute 'audio_encoder'`
- **Solution**: Inspected model structure, found correct names:
  - `encoder` (not `audio_encoder`)
  - `transf_decoder` (not `decoder`)
  - `log_softmax` (not `lm_head`)
- **Lesson**: Always inspect custom model structure before conversion

### Trial 4: Dynamic shape errors with wrong coremltools version (RESOLVED)

- **Error**: `TypeError: only 0-dimensional arrays can be converted to Python scalars`
- **Root cause**: Using coremltools 9.0 instead of 9.0b1
- **Solution**: Downgraded to Python 3.10.12, coremltools 9.0b1, torch 2.7.0
- **Key learning**: Version mismatch causes fundamental conversion failures
- **Credit**: User suggested checking Parakeet v3's uv.lock file

### Trial 5: Missing decoder positions parameter (RESOLVED)

- **Error**: `TypeError: forward() missing required argument 'positions'`
- **Solution**: Added `positions` tensor input to DecoderWrapper
- **Code fix**:
  ```python
  def forward(self, input_ids, positions, encoder_hidden_states):
      decoder_output, _ = self.transf_decoder(
          input_ids=input_ids,
          positions=positions,  # Required by TransformerDecoderWrapper
          encoder_hidden_states=encoder_hidden_states,
      )
      return decoder_output
  ```

### Trial 6: Decoder tuple return value (RESOLVED)

- **Error**: `RuntimeError: Only tensors can be output from traced functions`
- **Root cause**: Decoder returns `(hidden_states, past_key_values)` tuple
- **Solution**: Unpack tuple to extract tensor: `decoder_output, _ = self.transf_decoder(...)`

### Trial 7: Missing config attributes (RESOLVED)

- **Error**: `AttributeError: 'CohereAsrConfig' object has no attribute 'hidden_size'`
- **Root cause**: Config uses nested dictionaries, not flat attributes
- **Solution**: Use nested access:
  ```python
  encoder_hidden_size = config.encoder["d_model"]  # 1280
  decoder_hidden_size = config.transf_decoder["config_dict"]["hidden_size"]  # 1024
  lm_head_hidden_size = config.head["hidden_size"]  # 1024
  ```

### Community Success Reference

Discord member `love4cristiano` reported successful conversion with:
- **Performance**: 15-35x RTF on M3 Pro
- **Quantization**: 6-bit with minimal WER drop
- **Target**: GPU preferred over ANE (CPU 20x, ANE only 10x due to overhead)
- **Format**: FP16 better than INT8 for performance

### Recommendations Based on Conversion Experience

1. **Always use exact dependency versions** from known working conversions (check uv.lock files)
2. **Target GPU, not ANE** for this model (ANE overhead hurts performance)
3. **Use 6-bit or FP16 quantization** instead of INT8 for best speed/quality trade-off
4. **Inspect custom models thoroughly** before conversion (use `dir()`, `inspect.signature()`)
5. **Handle tuple returns** from decoder layers by unpacking
6. **Use fixed sequence lengths** during tracing to avoid dynamic shape issues

### Known Limitations

- **No KV cache**: Current conversion doesn't support stateful decoding with past_key_values
- **Fixed input shapes**: Audio must be preprocessed to exact shape (batch=1, n_mels=128, time=3000)
- **No dynamic batching**: Batch size is fixed at 1 during tracing
- **Large model size**: 3.9 GB may exceed limits on older devices (consider quantization)

### Next Steps

1. **Validation**: Compare CoreML vs PyTorch outputs for numerical parity
2. **Profiling**: Benchmark with coreml-cli (latency, device assignment)
3. **Quantization**: Try 6-bit quantization for deployment
4. **HuggingFace Upload**: Publish to FluidInference/cohere-transcribe-03-2026-coreml
5. **FluidAudio Integration**: Implement CohereAsrManager

## References

- Model card: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- Paper: [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- FluidAudio integration guide: `Documentation/ModelConversion.md`
