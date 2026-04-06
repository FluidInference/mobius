# Package Contents - Complete Upload Package

## Total Size: 7.7 GB

All files ready for HuggingFace upload in this directory.

## CoreML Models (7.8 GB)

### Source Format (.mlpackage)
- **cohere_encoder.mlpackage** - 3.6 GB
  - Conformer encoder + projection layer
  - Input: (1, 128, 3500) mel spectrogram
  - Output: (1, 438, 1024) hidden states
  - First load: ~20 seconds (ANE compilation)

- **cohere_decoder_stateful.mlpackage** - 291 MB
  - Transformer decoder with stateful KV cache
  - GPU-resident cache via CoreML State API
  - Max sequence length: 108 tokens
  - First load: ~3 seconds (ANE compilation)

### Compiled Format (.mlmodelc) ⚡
- **cohere_encoder.mlmodelc** - 3.6 GB
  - Pre-compiled for instant loading
  - Loads in ~1 second (no compilation needed)
  - Identical inference to .mlpackage

- **cohere_decoder_stateful.mlmodelc** - 291 MB
  - Pre-compiled for instant loading
  - Loads in ~0.5 seconds (no compilation needed)
  - Identical inference to .mlpackage

**Why include both?**
- `.mlmodelc`: Production use (instant loading)
- `.mlpackage`: Development use (model inspection, debugging)

## Vocabulary
- **vocab.json** - 331 KB
  - 16,384 SentencePiece tokens
  - Multilingual: 14 languages
  - Format: `{"token_id": "token_string"}`

## Audio Preprocessing
- **cohere_mel_spectrogram.py** - 3.6 KB
  - Pure Python implementation
  - No transformers dependency
  - Exact match of Cohere's preprocessing
  - Config: 128 mel bins, 16kHz, 10ms hop

## Inference Examples

### Complete Example
- **example_inference.py** - 10 KB
  - Full-featured CLI tool
  - Multi-language support (14 languages)
  - Arguments: `--language`, `--max-tokens`, `--model-dir`
  - Audio loading via soundfile
  - Error handling and progress messages
  - Auto-detects .mlmodelc/.mlpackage

### Quick Start
- **quickstart.py** - 2.0 KB
  - Minimal 50-line example
  - Perfect for quick testing
  - Uses compiled .mlmodelc
  - Single file transcription

## Dependencies

### pip (Standard)
- **requirements.txt** - 170 B
  ```
  coremltools>=9.0
  numpy>=1.24.0
  soundfile>=0.12.0
  huggingface-hub>=0.20.0
  ```

### uv (Fast, Locked)
- **pyproject.toml** - 6.1 KB
  - Project metadata and dependencies
  - Compatible with uv and pip

- **uv.lock** - 404 KB
  - Locked dependency versions
  - Reproducible installs
  - Faster than pip
  - Usage: `uv sync`

## Documentation
- **README.md** - 7.5 KB
  - Complete model card
  - Quick start guide
  - Usage examples (Python & Swift)
  - Performance metrics
  - Known limitations
  - Platform requirements
  - License and citation

## File Summary

| File | Size | Type | Purpose |
|------|------|------|---------|
| cohere_encoder.mlpackage | 3.6 GB | Model (source) | Encoder, slow first load |
| cohere_encoder.mlmodelc | 3.6 GB | Model (compiled) | Encoder, instant load |
| cohere_decoder_stateful.mlpackage | 291 MB | Model (source) | Decoder, slow first load |
| cohere_decoder_stateful.mlmodelc | 291 MB | Model (compiled) | Decoder, instant load |
| vocab.json | 331 KB | Data | Vocabulary mapping |
| cohere_mel_spectrogram.py | 3.6 KB | Code | Audio preprocessor |
| example_inference.py | 10 KB | Code | Complete inference CLI |
| quickstart.py | 2.0 KB | Code | Minimal example |
| requirements.txt | 170 B | Config | pip dependencies |
| pyproject.toml | 6.1 KB | Config | uv project config |
| uv.lock | 404 KB | Config | Locked dependencies |
| README.md | 7.5 KB | Docs | Model documentation |

## Upload Verification Checklist

Before uploading, verify:
- ✅ All 12 files present
- ✅ No __pycache__ or .pyc files
- ✅ Models load successfully (test with quickstart.py)
- ✅ Total size: 7.7 GB
- ✅ README.md renders correctly
- ✅ Python files have valid syntax

## Post-Upload Testing

After uploading to HuggingFace:

```bash
# Download
huggingface-cli download FluidInference/cohere-transcribe-03-2026-coreml \
  --local-dir test-download

# Test compiled models (instant loading)
cd test-download
python quickstart.py sample.wav

# Test full example
python example_inference.py sample.wav --language en

# Verify load time (should be ~1 second with .mlmodelc)
time python -c "import coremltools as ct; ct.models.MLModel('cohere_encoder.mlmodelc')"
```

## Performance Benefits

### Loading Time Comparison

| Format | First Load | Subsequent Loads |
|--------|-----------|------------------|
| .mlpackage | ~20s (ANE compile) | ~20s (recompile after sleep) |
| .mlmodelc | ~1s | ~1s |

### Why This Matters

**Without .mlmodelc:**
- User waits 20 seconds on first transcription
- After Mac sleep, waits another 20 seconds
- Poor user experience

**With .mlmodelc:**
- User waits 1 second consistently
- No ANE recompilation needed
- Professional UX

## Upload Command

```bash
# From this directory (build-35s/)
huggingface-cli upload FluidInference/cohere-transcribe-03-2026-coreml . --repo-type model
```

**Estimated upload time:** 40-45 minutes (7.7 GB)

## What's NOT Included (Intentional)

- ❌ INT8 models (quality issues)
- ❌ Test scripts (not needed by users)
- ❌ Development tools (model comparison, debugging)
- ❌ Investigation docs (already summarized in README)
- ❌ __pycache__ (Python bytecode cache)

## License

All files inherit the license from the original Cohere Transcribe model (Apache 2.0).
