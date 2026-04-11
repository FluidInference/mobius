# Testing Pre-trained MB-MelGAN Quality

**Goal:** Evaluate pre-trained MB-MelGAN quality with CosyVoice3 mel spectrograms before fine-tuning.

## Quick Test (No CosyVoice3)

**Test with synthetic mels:**
```bash
python test_pretrained_quality.py --no-cosyvoice
```

**Result:** ✅ Working!
```
Output: mbmelgan_quality_test/
  - test_1_synthetic.wav
  - test_2_synthetic.wav
  - test_3_synthetic.wav
```

**Note:** Synthetic mels produce noise, not intelligible speech. This just confirms the model works.

## Full Quality Test (With CosyVoice3)

### 1. Download CosyVoice3 Model

**Option A: Using script (recommended)**
```bash
./download_cosyvoice3.sh
```

**Option B: Manual download**
```bash
# Install git-lfs
brew install git-lfs
git lfs install

# Clone model
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice3-0.5B.git pretrained_models/Fun-CosyVoice3-0.5B
```

**Download time:** 10-30 minutes (depends on connection)

### 2. Run Quality Test

**Test with real CosyVoice3 mels:**
```bash
python test_pretrained_quality.py
```

**What this does:**
1. Loads pre-trained MB-MelGAN weights (VCTK 24kHz)
2. Loads CosyVoice3 model
3. Generates 3 test audio samples with CosyVoice3
4. Extracts mel spectrograms from CosyVoice3 audio
5. Runs mels through MB-MelGAN
6. Saves both versions for comparison

**Output:**
```
mbmelgan_quality_test/
├── test_1_original.wav   # CosyVoice3 original audio
├── test_1_mbmelgan.wav   # MB-MelGAN generated audio
├── test_2_original.wav
├── test_2_mbmelgan.wav
├── test_3_original.wav
└── test_3_mbmelgan.wav
```

### 3. Evaluate Quality

**Listen to both versions:**
```bash
# macOS
open mbmelgan_quality_test/test_1_original.wav
open mbmelgan_quality_test/test_1_mbmelgan.wav
```

**Expected results:**

| Aspect | Expected | Reason |
|--------|----------|--------|
| **Intelligibility** | May be lower | MB-MelGAN trained on VCTK, not CosyVoice3 |
| **Voice quality** | Different | Different training data (VCTK vs CosyVoice3) |
| **Prosody** | Similar | Mel spectrogram preserves prosody |
| **Artifacts** | Possible | Not fine-tuned on CosyVoice3 data |
| **Speech structure** | Preserved | Basic phonetic structure should be there |

**Quality decision matrix:**

| If quality is... | Then... |
|-----------------|---------|
| **Good enough** | ✅ Use pre-trained as-is! Deploy immediately |
| **Recognizable but imperfect** | ⚡ Fine-tune for 5-10 epochs (1-2 hours) |
| **Poor/unintelligible** | 🔄 Fine-tune for 20+ epochs (6-12 hours) |
| **Completely broken** | ⚠️ Debug mel spectrogram extraction |

## Evaluation Criteria

### Good Quality ✅
- Speech is intelligible
- Words are clear
- Prosody is natural
- Minimal artifacts
- **Action:** Use pre-trained, skip fine-tuning!

### Acceptable Quality ⚡
- Speech is mostly intelligible
- Some words are unclear
- Prosody is decent
- Some artifacts present
- **Action:** Quick fine-tune (5-10 epochs, 1-2 hours)

### Poor Quality 🔄
- Speech is hard to understand
- Many words are unclear
- Prosody is unnatural
- Many artifacts
- **Action:** Full fine-tune (20 epochs, 6-12 hours)

### Broken ⚠️
- No intelligible speech
- Just noise
- Nothing recognizable
- **Action:** Debug mel extraction, check model loading

## Next Steps Based on Results

### If Quality is Good ✅

**Deploy immediately:**
```python
import coremltools as ct

# Load pre-trained CoreML model
vocoder = ct.models.MLModel("mbmelgan_pretrained_coreml.mlpackage")

# Use with CosyVoice3
mel = extract_mel_from_cosyvoice3(text)
bands = vocoder.predict({"mel_spectrogram": mel})
audio = pqmf_synthesis(bands)  # TODO: Implement PQMF
```

**No fine-tuning needed!**

### If Quality is Acceptable ⚡

**Quick fine-tune:**
```bash
# Generate 200 samples (30 min)
python generate_training_data.py --num-samples 200

# Quick fine-tune (1-2 hours)
python train_mbmelgan.py --epochs 10 --batch-size 8

# Test again
python test_pretrained_quality.py
```

**Improvement expected:**
- Better voice match
- Fewer artifacts
- Clearer speech
- More natural prosody

### If Quality is Poor 🔄

**Full fine-tune:**
```bash
# Generate 1,000 samples (2 hours)
python generate_training_data.py --num-samples 1000

# Full fine-tune (6-12 hours CPU, 1 hour GPU)
python train_mbmelgan.py --epochs 20 --batch-size 8

# Test again
python test_pretrained_quality.py
```

**Significant improvement expected!**

### If Quality is Broken ⚠️

**Debug steps:**
1. Check mel spectrogram extraction
2. Verify mel shape: `[1, 80, frames]`
3. Check mel range: typically `[-10, 2]` in log scale
4. Compare with CosyVoice3's actual vocoder input
5. Verify pre-trained weights loaded correctly

**Debug script:**
```python
# Check mel extraction
mel = compute_mel_spectrogram(audio)
print(f"Mel shape: {mel.shape}")  # Should be [1, 80, frames]
print(f"Mel range: [{mel.min():.2f}, {mel.max():.2f}]")  # Should be ~[-10, 2]

# Compare with CosyVoice3's mel
cosyvoice_mel = extract_cosyvoice3_internal_mel(audio)
print(f"Difference: {(mel - cosyvoice_mel).abs().max():.4f}")  # Should be small
```

## Technical Details

### Mel Spectrogram Parameters

**CosyVoice3 vocoder expects:**
```python
{
    'sample_rate': 24000,
    'n_fft': 2048,
    'hop_length': 300,
    'n_mels': 80,
    'f_min': 80,
    'f_max': 7600,
}
```

**VCTK MB-MelGAN was trained with:**
```python
{
    'sample_rate': 24000,  # ✅ Same!
    'hop_size': 300,       # ✅ Same!
    'num_mels': 80,        # ✅ Same!
    'fmin': 80,            # ✅ Same!
    'fmax': 7600,          # ✅ Same!
}
```

**Perfect match!** This is why the pre-trained model should work.

### Why Pre-trained Might Work Well

**Reasons for optimism:**
1. ✅ Same sample rate (24kHz)
2. ✅ Same mel parameters (80 bins, 300 hop, etc.)
3. ✅ Multi-speaker training (VCTK has 109 speakers)
4. ✅ English language overlap
5. ✅ High-quality training (1M steps)

**Reasons for concern:**
1. ⚠️ Different dataset (VCTK vs CosyVoice3)
2. ⚠️ Different speaker characteristics
3. ⚠️ CosyVoice3 may have unique mel characteristics

**Most likely:** Works reasonably well, fine-tuning improves it further.

## Files

**Test scripts:**
- `test_pretrained_quality.py` - Quality evaluation script
- `download_cosyvoice3.sh` - Download CosyVoice3 model

**Outputs:**
- `mbmelgan_quality_test/*.wav` - Test audio files

**Documentation:**
- `TESTING_GUIDE.md` - This file
- `MBMELGAN_FINETUNING.md` - Fine-tuning guide
- `MBMELGAN_SUCCESS.md` - Pre-trained model results

## Summary

**Current status:**
- ✅ Pre-trained MB-MelGAN downloaded (99.26 MB, VCTK 24kHz)
- ✅ CoreML conversion tested (202 ops, 4.50 MB)
- ✅ Synthetic test working (produces audio)
- ⏳ Quality test pending (need CosyVoice3)

**Next steps:**
1. Download CosyVoice3: `./download_cosyvoice3.sh` (10-30 min)
2. Run quality test: `python test_pretrained_quality.py`
3. Listen and evaluate
4. Decide: deploy as-is, quick fine-tune, or full fine-tune

**Timeline:**
- Download + test: 30-60 min
- If good → Deploy immediately! (0 hours)
- If acceptable → Quick fine-tune (1-2 hours)
- If poor → Full fine-tune (6-12 hours)

**Recommended approach:**
1. Run quality test first (30 min)
2. Make decision based on actual results
3. Only fine-tune if needed

**Best case:** Pre-trained works well, deploy in 1 hour! 🚀
