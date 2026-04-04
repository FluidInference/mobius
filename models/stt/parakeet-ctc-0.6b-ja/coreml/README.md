# Parakeet CTC 0.6B Japanese - CoreML Conversion

**Model**: `nvidia/parakeet-tdt_ctc-0.6b-ja`
**Architecture**: Hybrid FastConformer-TDT-CTC
**Language**: Japanese (ja)
**Status**: ✅ **Successfully converted to CoreML**

## Overview

This directory contains the full pipeline CoreML conversion for the Japanese Parakeet hybrid TDT+CTC model. The conversion is complete and validated:

- ✅ Individual components (Preprocessor, Encoder, CTC Decoder)
- ✅ Fused pipelines (MelEncoder, FullPipeline)
- ✅ Vocabulary extraction (3,072 Japanese BPE tokens)
- ✅ Numerical validation (max diff: 0.01066 between PyTorch and CoreML)
- ✅ Metadata and documentation

## Key Implementation Detail

The CTC decoder outputs **raw logits** (not log-softmax). Apply `log_softmax` in post-processing for CTC decoding:

```python
import numpy as np

# Get raw logits from CoreML
logits = model.predict({"encoder_output": features})["ctc_logits"]

# Apply log_softmax for CTC decoding
log_probs = np.log(np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))
# Or use scipy/torch: torch.nn.functional.log_softmax(torch.from_numpy(logits), dim=-1)
```

**Why raw logits?** CoreML had conversion issues with the `log_softmax` operation in the CTC decoder. Outputting raw logits avoids this and provides identical results after post-processing.

## Model Details

| Component | Input | Output |
|-----------|-------|--------|
| **Preprocessor** | audio [1, 240000] | mel [1, 80, 1501] |
| **Encoder** | mel [1, 80, 1501] | features [1, 1024, 188] |
| **CTC Decoder** | features [1, 1024, 188] | logits [1, 188, 3073] (raw) |
| **MelEncoder** | audio [1, 240000] | features [1, 1024, 188] |
| **FullPipeline** | audio [1, 240000] | logits [1, 188, 3073] (raw) |

- **Vocabulary**: 3,072 Japanese SentencePiece BPE tokens + 1 blank
- **Sample Rate**: 16kHz
- **Max Duration**: 15 seconds (240,000 samples)
- **Output**: Raw logits (apply log_softmax in post-processing)

## Files

```
build/
├── Preprocessor.mlpackage          # Audio → Mel spectrogram ✅
├── Encoder.mlpackage                # Mel → Encoder features ✅
├── CtcDecoder.mlpackage             # Features → CTC logits (raw) ✅
├── MelEncoder.mlpackage             # Audio → Encoder features ✅
├── FullPipeline.mlpackage           # Audio → CTC logits (raw) ✅
├── vocab.json                       # 3,072 Japanese BPE tokens ✅
└── metadata.json                    # Model metadata ✅
```

## Usage

### Quick Start

```bash
# Setup environment
uv sync

# Convert all components
uv run python convert-parakeet-ja.py --output-dir ./build

# Or convert specific components
uv run python convert-parakeet-ja.py --no-fused  # Individual only
uv run python convert-parakeet-ja.py --no-individual  # Fused only
```

### Compilation

```bash
cd build
xcrun coremlcompiler compile Preprocessor.mlpackage .
xcrun coremlcompiler compile Encoder.mlpackage .
xcrun coremlcompiler compile CtcDecoder.mlpackage .
xcrun coremlcompiler compile MelEncoder.mlpackage .
xcrun coremlcompiler compile FullPipeline.mlpackage .
```

### Performance Profiling

```bash
cd ../../../../tools/coreml-cli
uv run coreml-cli ../../models/stt/parakeet-ctc-0.6b-ja/coreml/build/Encoder.mlmodelc
uv run coreml-cli ../../models/stt/parakeet-ctc-0.6b-ja/coreml/build/Encoder.mlmodelc --fallback
```

## Performance

| Metric | Value |
|--------|-------|
| **ANE Utilization** | 100% ✅ |
| **CPU Fallback** | 0 ops ✅ |
| **Numerical Accuracy** | Max diff 0.01066 ✅ |
| **Model Size** | ~600MB |

The models achieve full Neural Engine utilization with no CPU fallbacks, providing efficient on-device inference.

## Conversion Notes

### Fixed: log_softmax Conversion Issue

Earlier versions of this conversion had a critical bug where the CTC decoder's `log_softmax` operation caused CoreML to output incorrect values (e.g., `-45440` instead of the expected range).

**Root cause**: The NeMo CTC decoder's `forward()` method applies `log_softmax` to the output:
```python
# NeMo's ConvASRDecoder.forward()
return torch.nn.functional.log_softmax(
    self.decoder_layers(encoder_output).transpose(1, 2), dim=-1
)
```

CoreML had issues converting this `log_softmax` operation correctly for this specific model.

**Solution**: Bypass the CTC decoder's `forward()` and directly use `decoder_layers` (Conv1d) + transpose to get raw logits:
```python
# CTCDecoderWrapper.forward()
conv_output = self.module.decoder_layers(encoder_output)  # [B, V, T]
logits = conv_output.transpose(1, 2)  # [B, T, V]
return logits  # Raw logits, not log-softmax
```

This produces identical results after applying `log_softmax` in post-processing, and the CoreML conversion succeeds with excellent numerical accuracy (max diff: 0.01066).

## Validation

Test scripts are provided for verification:

```bash
# Validate CTC decoder fix
uv run python validate-fix.py

# Test full pipeline
uv run python test-full-pipeline.py

# Analyze error accumulation
uv run python analyze-pipeline-error.py
```

## Documentation

- **CONVERSION_SUCCESS.md** - Complete conversion summary and validation results
- **CONVERSION_NOTES.md** - Detailed investigation of the log_softmax bug and fix
- **convert-parakeet-ja.py** - Main conversion script
- **individual_components.py** - Component wrappers with the critical fix

## References

- **HuggingFace**: https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja
- **Working comparison**: `../parakeet-ctc-0.6b-zh-cn/coreml/`
- **Similar model (English)**: `../parakeet-tdt-v2-0.6b/coreml/`

## License

Governed by the NVIDIA Community Model License.
