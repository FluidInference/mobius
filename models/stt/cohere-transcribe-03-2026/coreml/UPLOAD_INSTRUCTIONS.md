# HuggingFace Upload Instructions

## Ready to Upload: FP16 Models (35-Second Window)

The FP16 models are ready for upload to HuggingFace.

### Location
```
build-35s/
├── cohere_encoder.mlpackage           # 3.6 GB - Encoder (source format)
├── cohere_encoder.mlmodelc            # 3.6 GB - Encoder (compiled, instant loading)
├── cohere_decoder_stateful.mlpackage  # 291 MB - Decoder (source format)
├── cohere_decoder_stateful.mlmodelc   # 291 MB - Decoder (compiled, instant loading)
├── vocab.json                         # 331 KB - Vocabulary (16,384 tokens)
├── cohere_mel_spectrogram.py          # 3.6 KB - Audio preprocessor
├── example_inference.py               # 9.8 KB - Complete inference example
├── quickstart.py                      # 1.9 KB - Minimal 50-line example
├── requirements.txt                   # 170 B - Python dependencies (pip)
├── pyproject.toml                     # 6.1 KB - Project config (uv)
├── uv.lock                            # 404 KB - Locked dependencies (uv)
└── README.md                          # 6.7 KB - Model documentation
```

**Note:** Both `.mlpackage` and `.mlmodelc` are included:
- `.mlmodelc` (compiled): Loads instantly (~1 second), recommended for production
- `.mlpackage` (source): Slower first load (~20 seconds compilation), but allows model inspection

### Quality Verified
- ✅ Average WER: 23.76%
- ✅ Perfect matches: 64%
- ✅ Correct 35-second window (3500 frames → 438 encoder outputs)
- ✅ Encoder/decoder dimensions compatible
- ✅ Stateful decoder with GPU-resident cache

## Upload Steps

### 1. Create HuggingFace Repository

```bash
# Login to HuggingFace
huggingface-cli login

# Create repository (or use existing one)
huggingface-cli repo create cohere-transcribe-03-2026-coreml --type model
```

### 2. Upload Files

```bash
cd build-35s

# Upload encoder (large file, may take 10-20 minutes)
huggingface-cli upload FluidInference/cohere-transcribe-03-2026-coreml \
  cohere_encoder.mlpackage \
  --repo-type model

# Upload decoder
huggingface-cli upload FluidInference/cohere-transcribe-03-2026-coreml \
  cohere_decoder_stateful.mlpackage \
  --repo-type model

# Upload vocabulary
huggingface-cli upload FluidInference/cohere-transcribe-03-2026-coreml \
  vocab.json \
  --repo-type model

# Upload README
huggingface-cli upload FluidInference/cohere-transcribe-03-2026-coreml \
  README.md \
  --repo-type model
```

**Note:** Replace `FluidInference/cohere-transcribe-03-2026-coreml` with your actual repository name.

### 3. Alternative: Upload via Git LFS

If you prefer Git LFS:

```bash
# Clone repository
git clone https://huggingface.co/FluidInference/cohere-transcribe-03-2026-coreml
cd cohere-transcribe-03-2026-coreml

# Copy files
cp -r ../build-35s/cohere_encoder.mlpackage .
cp -r ../build-35s/cohere_decoder_stateful.mlpackage .
cp ../build-35s/vocab.json .
cp ../build-35s/README.md .

# Track large files with LFS
git lfs track "*.mlpackage/**"
git add .gitattributes

# Commit and push
git add .
git commit -m "Add Cohere Transcribe CoreML FP16 models (35s window)"
git push
```

## What About INT8 Models?

The INT8 models are **not recommended for upload** at this time:

### INT8 Quality Issues
- ✅ Average WER: 25.2% (acceptable)
- ❌ Perfect matches: 0% (need 60%+)
- ❌ Catastrophic failure on 23s sample (110% WER)
- ⚠️ Unstable on long audio

### INT8 Location (if needed later)
```
build-35s-int8/
├── cohere_encoder_int8.mlpackage           # 1.82 GB
└── cohere_decoder_stateful_int8.mlpackage  # 145.8 MB
```

### Future INT8 Improvements
If you want to revisit INT8 quantization later:
1. Try `per_tensor` granularity (more stable)
2. Quantize encoder only, keep decoder FP16
3. Use different calibration data
4. Test on more diverse audio samples

## Repository Metadata

Add to your HuggingFace model card:

```yaml
---
language:
- en
- es
- fr
- de
- it
- pt
- pl
- nl
- sv
- tr
- ru
- zh
- ja
- ko
license: apache-2.0
library_name: coreml
tags:
- audio
- speech
- asr
- coreml
- apple-silicon
pipeline_tag: automatic-speech-recognition
---
```

## Post-Upload

After uploading, verify:

1. **Test download:**
   ```bash
   huggingface-cli download FluidInference/cohere-transcribe-03-2026-coreml \
     cohere_encoder.mlpackage
   ```

2. **Update FluidAudio integration:**
   - Update model URLs in FluidAudio codebase
   - Test model loading from HuggingFace
   - Update documentation

3. **Announce:**
   - Update project README
   - Post on Discord/Twitter
   - Link from original Cohere Transcribe model page

## File Checksums

Verify file integrity before upload:

```bash
cd build-35s
shasum -a 256 vocab.json
find cohere_encoder.mlpackage -type f -exec shasum -a 256 {} \; | shasum -a 256
find cohere_decoder_stateful.mlpackage -type f -exec shasum -a 256 {} \; | shasum -a 256
```

## Questions?

If you encounter upload issues:
- Large file timeout: Use Git LFS instead of CLI
- Authentication: Run `huggingface-cli login` again
- Repo permissions: Ensure you have write access
