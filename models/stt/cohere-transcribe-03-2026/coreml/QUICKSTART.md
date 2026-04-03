# Quick Start Guide

Get the Cohere Transcribe conversion running in 5 minutes.

## Prerequisites

- Python 3.10+
- `uv` package manager ([install](https://github.com/astral-sh/uv))
- macOS 14+ or iOS 17+ for CoreML deployment
- **HuggingFace account** with access to gated model

## Step 0: Get HuggingFace Access

This model requires authentication:

1. Go to https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
2. Click "Request Access" (usually approved instantly)
3. Run `huggingface-cli login` and provide your token

```bash
pip install huggingface-hub
huggingface-cli login
```

## Step 1: Setup Environment

```bash
cd mobius/models/stt/cohere-transcribe-03-2026/coreml
uv sync
```

This installs all dependencies in an isolated environment.

## Step 2: Run Conversion

```bash
# Full conversion (all components)
uv run python convert-cohere-transcribe.py

# Or specify output directory
uv run python convert-cohere-transcribe.py --output-dir ./my-build
```

Expected output:
```
[1/3] Exporting Audio Encoder...
  Saved: build/cohere-transcribe/cohere_audio_encoder.mlpackage
[2/3] Exporting Decoder...
  Saved: build/cohere-transcribe/cohere_decoder.mlpackage
[3/3] Exporting LM Head...
  Saved: build/cohere-transcribe/cohere_lm_head.mlpackage
```

⏱️ **Time**: 10-30 minutes depending on hardware

## Step 3: Validate (Optional but Recommended)

```bash
# Download a test audio file or use your own
uv run python compare-models.py \
  --audio-file /path/to/test.wav \
  --coreml-dir ./build/cohere-transcribe \
  --language en
```

This compares PyTorch vs CoreML outputs to verify accuracy.

## Step 4: Profile Performance

```bash
cd ../../../../tools/coreml-cli
uv sync

# Benchmark audio encoder
uv run coreml-cli ../../models/stt/cohere-transcribe-03-2026/coreml/build/cohere-transcribe/cohere_audio_encoder.mlmodelc

# Check ANE compatibility
uv run coreml-cli ../../models/stt/cohere-transcribe-03-2026/coreml/build/cohere-transcribe/cohere_audio_encoder.mlmodelc --fallback
```

## Common Issues

### Issue: Model too large for tracing

**Error**: `RuntimeError: CUDA out of memory` or similar

**Solution**: The 2B model is large. Try:
```bash
# Use quantization during conversion
uv run python convert-cohere-transcribe.py --quantize int8
```

### Issue: Transformers version too old

**Error**: `ImportError: cannot import name 'CohereAsrForConditionalGeneration'`

**Solution**:
```bash
uv pip install 'transformers>=4.57.0' --upgrade
```

### Issue: Audio sample rate mismatch

**Error**: Validation shows large errors

**Solution**: Ensure your test audio is 16kHz:
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 output.wav
```

## Next Steps

1. **Test on device**: Deploy to iPhone/Mac and measure real performance
2. **Optimize**: Try different quantization levels (int8, int4)
3. **Integrate**: Add to FluidAudio (see main README.md)
4. **Upload**: Share on HuggingFace (see ModelConversion.md)

## Files Generated

```
build/cohere-transcribe/
├── cohere_audio_encoder.mlpackage    # Mel + Conformer encoder (~2GB)
├── cohere_decoder.mlpackage          # Transformer decoder (~500MB)
├── cohere_lm_head.mlpackage          # LM head (~100MB)
└── metadata.json                     # Model config
```

## Resources

- Full docs: `README.md`
- Conversion guide: `Documentation/ModelConversion.md`
- Model card: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
