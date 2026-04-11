# Insights from john-rocky/CoreML-Models Repository

**Repository:** https://github.com/john-rocky/CoreML-Models

This repository is a treasure trove of CoreML conversion examples, particularly relevant for audio models.

## 🎯 Most Relevant Models

### 1. **Kokoro-82M TTS** ✨ (Directly Relevant!)

**What it is:**
- 82M-parameter TTS by hexgrad
- StyleTTS2 architecture (BERT + duration predictor + iSTFTNet vocoder)
- 24kHz speech in 9 languages
- **First CoreML port with on-device bilingual (English + Japanese) text input**

**Architecture:**
- **Predictor model:** BERT + LSTM duration head + text encoder
  - Input: `input_ids [1, T≤256]` + `ref_s_style [1, 128]`
  - Output: `duration [1, T]` + `d_for_align [1, 640, T]` + `t_en [1, 512, T]`
  - Size: 75 MB

- **Decoder model (3 buckets):** iSTFTNet vocoder
  - Buckets: 128 / 256 / 512 frames
  - Input: `en_aligned [1, 640, frames]` + `asr_aligned [1, 512, frames]` + `ref_s [1, 256]`
  - Output: Audio @ 24kHz
  - Size: 238-246 MB per bucket

**Key Conversion Techniques:**

```python
# 1. Model Splitting Strategy
# Split into Predictor + Decoder because duration creates dynamic length
class PredictorWrapper(nn.Module):
    def __init__(self, kmodel):
        self.bert = kmodel.bert
        self.predictor = kmodel.predictor
        # Extract only predictor components

    def forward(self, input_ids, ref_s_style):
        # Returns: duration, d_for_align, t_en
        # Duration used to align features in Swift

# 2. Bucketed Decoder Strategy
DECODER_BUCKETS = [128, 256, 512]  # Pick smallest >= predicted frames
# At runtime: predict duration → choose bucket → pad → decode → trim

# 3. Flexible Input Length (RangeDim)
flex_len = ct.RangeDim(lower_bound=1, upper_bound=MAX_PHONEMES, default=MAX_PHONEMES)
pred_ml = ct.convert(
    traced_pred,
    inputs=[ct.TensorType(name="input_ids", shape=(1, flex_len), dtype=np.int32)],
    ...
)

# 4. Patched CoreML ops for shape operations
def _patched_int(context, node):
    # Custom int op for shape computations
    ...
_ct_ops._TORCH_OPS_REGISTRY.register_func(_patched_int, torch_alias=["int"], override=True)
```

**Download Links:**
- [Predictor.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/download/kokoro-v1/Kokoro_Predictor.mlpackage.zip) (75 MB)
- [Decoder_128/256/512.mlpackage.zip](https://github.com/john-rocky/CoreML-Models/releases/tag/kokoro-v1)
- [Sample App: KokoroDemo](https://github.com/john-rocky/CoreML-Models/tree/master/sample_apps/KokoroDemo)
- [Conversion Script](https://github.com/john-rocky/CoreML-Models/blob/master/conversion_scripts/convert_kokoro.py)

---

### 2. **OpenVoice V2** (Voice Conversion)

**What it is:**
- Zero-shot voice conversion
- Record source and target voice, convert on-device

**Models:**
- **SpeakerEncoder.mlpackage:** 1.7 MB
  - Input: Spectrogram `[1, T, 513]`
  - Output: 256-dim speaker embedding

- **VoiceConverter.mlpackage:** 64 MB
  - Input: Spectrogram + speaker embeddings
  - Output: Waveform audio (22050 Hz)

**Links:**
- [Download](https://github.com/john-rocky/CoreML-Models/releases/tag/openvoice-v1)
- [Sample App](https://github.com/john-rocky/CoreML-Models/tree/master/sample_apps/OpenVoiceDemo)

---

### 3. **HTDemucs** (Audio Source Separation)

**What it is:**
- Hybrid Transformer Demucs
- Separates music into 4 stems: drums, bass, vocals, other

**Model:**
- Size: 80 MB (FP32)
- Input: Audio waveform `[1, 2, 343980]` @ 44.1kHz
- Output: 4 stems (stereo)

**Links:**
- [Download](https://github.com/john-rocky/CoreML-Models/releases/tag/demucs-v1)
- [Sample App](https://github.com/john-rocky/CoreML-Models/tree/master/sample_apps/DemucsDemo)

---

### 4. **pyannote segmentation-3.0** (Speaker Diarization)

Relevant to our FluidAudio diarization work!

---

## 🔑 Key Patterns Applicable to CosyVoice3

### 1. **Model Splitting for Dynamic Lengths**

**Problem:** CosyVoice3 has dynamic-length outputs (like Kokoro's duration predictor)

**Solution:** Split into fixed-shape models
- **Model 1 (Predictor):** Flexible input → predicted length
- **Model 2 (Decoder):** Fixed output buckets

```python
# CosyVoice3 could use similar approach:
# 1. LLM → predict token count
# 2. Flow → predict mel frame count
# 3. Vocoder buckets: [125, 250, 500] frames (like we already did!)
```

### 2. **Bucketed Decoder Strategy**

**Our MB-MelGAN already uses this!**

```python
# We implemented:
ct.EnumeratedShapes(shapes=[(1, 80, 125), (1, 80, 250), (1, 80, 500)])

# Similar to Kokoro's approach:
DECODER_BUCKETS = [128, 256, 512]
```

### 3. **RangeDim for Flexible Inputs**

**Kokoro uses:**
```python
flex_len = ct.RangeDim(lower_bound=1, upper_bound=256, default=256)
```

**We could use for MB-MelGAN:**
```python
# Instead of EnumeratedShapes, use RangeDim:
ct.RangeDim(lower_bound=50, upper_bound=500, default=125)
# More flexible than 3 fixed buckets!
```

### 4. **Disable Complex Operations**

**Kokoro:**
```python
model = KModel(repo_id='hexgrad/Kokoro-82M', disable_complex=True)
```

**Our CosyVoice3:**
- Already disabled complex STFT operations
- Using real-valued alternatives

### 5. **Operation Patching**

**Kokoro patches int() ops for shape operations**

Could be useful if we hit shape computation issues in CosyVoice3 LLM/Flow models.

---

## 💡 Action Items for CosyVoice3

### Immediate (MB-MelGAN):
- ✅ Already using bucketed approach (EnumeratedShapes)
- ⚡ **Try RangeDim instead** - more flexible than 3 fixed buckets
  ```python
  ct.TensorType(
      name="mel_spectrogram",
      shape=(1, 80, ct.RangeDim(50, 500, default=125))
  )
  ```

### Future (Full Pipeline):
1. **Study Kokoro's predictor/decoder split**
   - Apply to CosyVoice3 LLM (predict token count → bucket selection)
   - Apply to Flow (predict mel frames → bucket selection)

2. **On-device G2P**
   - Kokoro has bilingual G2P without Python dependencies
   - Could inspire CosyVoice3 text preprocessing

3. **Swift Integration Patterns**
   - Check KokoroDemo sample app for Swift integration
   - Bucket selection logic
   - Audio trimming/padding

---

## 📚 Other Useful Models in Repo

- **Stable Diffusion variants** - conversion patterns for large models
- **Florence-2** - vision-language model split into 3 CoreML models
- **Real-ESRGAN** - super-resolution (similar complexity to vocoders)
- **Basic Pitch** - music transcription (audio → MIDI)

---

## 🔗 Resources

- **Repo:** https://github.com/john-rocky/CoreML-Models
- **Kokoro Sample App:** https://github.com/john-rocky/CoreML-Models/tree/master/sample_apps/KokoroDemo
- **Conversion Scripts:** https://github.com/john-rocky/CoreML-Models/tree/master/conversion_scripts
- **All Releases:** https://github.com/john-rocky/CoreML-Models/releases

---

## 🎯 Next Steps

1. **Immediate:** Test RangeDim for MB-MelGAN (more flexible than EnumeratedShapes)
2. **Review:** Kokoro conversion script for additional patterns
3. **Study:** KokoroDemo Swift app for integration patterns
4. **Consider:** Similar model splitting for CosyVoice3 LLM/Flow components

This repository proves that **complex TTS models CAN be fully converted to CoreML**! 🎉
