# Audio Window Length Fix

## Issue Discovered

Our encoder was exported with **3001 frames** (30.01 seconds), but the official Cohere model uses **3500 frames** (35 seconds).

## Calculation

```
Sample rate: 16000 Hz
Hop length: 160 samples
Time per frame: 160 / 16000 = 0.01 seconds (10ms)

BEFORE (incorrect):
  3001 frames × 10ms = 30.01 seconds ❌

AFTER (correct):
  3500 frames × 10ms = 35.00 seconds ✅
```

## Official Config Confirmation

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained('CohereLabs/cohere-transcribe-03-2026', trust_remote_code=True)
# config.max_audio_clip_s: 35
# config.max_seq_len: 1024
```

## Impact

**Before fix:**
- We were **truncating 5 seconds** of audio
- Audio > 30s was being silently cut off
- Longer utterances couldn't be fully processed

**After fix:**
- Full 35-second audio window supported
- Matches official model capabilities
- No silent truncation

## Changes Made

### 1. Encoder Export
**File:** `export-encoder.py`
- Line 79: Changed `max_frames = 3001` → `max_frames = 3500`
- Added comment: `# Official: 35 seconds at 10ms/frame`

### 2. Test Scripts
Updated all 16 test scripts that referenced 3001:
- `analyze-audio-properties.py`
- `compare-encoder-pytorch-coreml.py`
- `compare-full-pytorch-coreml.py`
- `compare-full-pytorch-coreml-simple.py`
- `compare-stateful-stateless-long.py`
- `debug-encoder-outputs.py`
- `investigate-failing-samples.py`
- `test-10s-samples.py`
- `test-audio-length-sweep.py`
- `test-full-reference-pipeline.py`
- `test-librispeech.py`
- `test-long-audio.py`
- `test-our-encoder-reference-decoder.py`
- `test-pytorch-long-audio-simple.py`
- `test-stateful-decoder.py`
- `test-stateless-coreml.py`

### 3. Model Re-export
- Re-exported encoder to: `build/cohere_encoder.mlpackage`
- New input shape: `(1, 128, 3500)` instead of `(1, 128, 3001)`

## Testing

```bash
# Test with 35s audio
uv run python tests/test-stateful-decoder.py

# Verify encoder accepts 3500 frames
uv run python -c "
import coremltools as ct
import numpy as np
encoder = ct.models.MLModel('build/cohere_encoder.mlpackage')
mel = np.random.randn(1, 128, 3500).astype(np.float32)
output = encoder.predict({'input_features': mel, 'feature_length': np.array([3500], dtype=np.int32)})
print(f'✓ Encoder accepts 3500 frames, output shape: {list(output.values())[0].shape}')
"
```

## Updated Limitations

### Audio Length Support

| Duration | Status | Notes |
|----------|--------|-------|
| < 35s | ✅ Fully supported | Single-pass processing |
| 35-70s | ⚠️ Requires chunking | 2× 35s chunks with overlap |
| > 70s | ⚠️ Multiple chunks | Process in 30-35s segments |

### Hard Limits

1. **Encoder input: 3500 frames (35 seconds)**
   - Before: 3001 frames (30 seconds) ❌
   - After: 3500 frames (35 seconds) ✅

2. **Decoder output: 108/256/1024 tokens**
   - Default: 108 tokens (~15-25s speech)
   - Extended: 256 tokens (~40-60s speech)
   - Official: 1024 tokens (~150-200s speech) - not yet exported

## Next Steps

### Optional: Export 1024-token Decoder

The official model supports up to 1024 tokens. To match this:

```bash
uv run python export-decoder-stateful.py --max-seq-len 1024 --output-dir build
# Creates: cohere_decoder_stateful_1024.mlpackage
```

**Trade-offs:**
- ✅ Matches official model
- ✅ Handles very dense speech
- ❌ Higher memory usage
- ❌ Slightly slower inference

## Verification

After re-export, verify:

```python
# Check encoder shape
import coremltools as ct
encoder = ct.models.MLModel('build/cohere_encoder.mlpackage')
print(encoder.get_spec().description.input[0].type.multiArrayType.shape)
# Should show: [1, 128, 3500]

# Test with 35s audio
from datasets import load_dataset
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
for sample in dataset:
    duration = len(sample['audio']['array']) / 16000.0
    if 30 <= duration <= 35:
        print(f"Found {duration:.2f}s sample - testing...")
        # Run full pipeline test
        break
```

## Credits

- Identified by: User question about official 35-second window
- Fixed: Updated encoder export and all test scripts
- Verified: Re-exported encoder with correct frame limit
