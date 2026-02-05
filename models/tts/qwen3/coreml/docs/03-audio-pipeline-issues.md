# Audio Pipeline Issues

Issues related to audio output quality, silence handling, and end-to-end pipeline validation.

---

## 1. Leading Silence in Generated Audio

**Problem:** English Swift output had ~1.5s of near-silence before speech started. Chinese Swift output had ~2.0s leading silence (exactly 48,000 samples = 25 codec frames).

**Root Cause:** Temperature sampling for the initial CB0 tokens can produce codec codes that decode to low-energy audio. This is stochastic -- different random seeds produce different lead-in durations. PyTorch output typically had 0.5s lead-in vs Swift's 1.5-2.0s.

**Fix:** Added `trimSilence()` post-processing in `Qwen3TtsSynthesizer.swift`:
- Scans forward in 20ms windows until RMS exceeds threshold (0.02)
- Scans backward with 200ms windows to skip tiny trailing blips
- Keeps 50ms padding on each side for natural onset/offset

Results after trimming: EN 5.92s -> 4.40s, ZH 4.16s -> 3.50s.

---

## 2. Trailing Blip After Speech Ends

**Problem:** Chinese audio had speech at 0-5.5s, silence at 5.5-7s, then a small artifact (blip) at ~9s before the audio ended at 10s.

**Root Cause:** The audio decoder processes all 125 frames even though only ~47 contain valid speech. Padding frames occasionally produce non-zero but meaningless audio. The blip at 9s corresponded to a codec frame near the end of the padded region that happened to decode to a brief impulse.

**Fix:** The frame-count trim (cutting audio to `actualFrames * 1920` samples) removes the blip since it occurs well after the last valid frame. The `trimSilence` end-detection uses 200ms windows specifically to skip such small blips.

---

## 3. PyTorch Chinese Reference Producing Silent Audio

**Problem:** When generating PyTorch reference audio for Chinese comparison, the output was completely silent (RMS=0.0000).

**Root Cause:** Used `do_sample=False` for `model.generate()`. Chinese text requires temperature sampling to produce valid audio. With greedy decoding, the code predictor produces degenerate codebook patterns for Chinese input specifically.

**Fix:** Changed to `do_sample=True, temperature=0.9, top_k=50` (the model's default config), which produces correct Chinese speech.

---

## 4. Garbled Audio Recognized as Chinese/Japanese Gibberish

**Problem:** Early V9+V3 pipeline output was transcribed by Whisper as Chinese/Japanese/French gibberish instead of the expected English.

**Root Cause:** The code predictor in V3 was producing different codebook tokens than the official model. This was traced to incorrect model version pairing and missing speaker embedding, which caused the embedding sum to diverge from the reference, producing garbled speech-like audio.

**Fix:** Proper pairing of V9 prefill with correct decode version, using `speaker_embedding_official.npy`, and enabling `do_sample=True` for the code predictor.

---

## 5. Chinese "shijie" vs "shijian" Transcription Ambiguity

**Problem:** Chinese Swift output was transcribed by Whisper as containing "事件" (shijian, "event") instead of "世界" (shijie, "world"). PyTorch and CoreML Python outputs transcribed correctly.

**Root Cause:** Not a pipeline bug. The two words are phonetically similar (shijie vs shijian) and the stochastic sampling produces slightly different prosody each run. Whisper's recognition is sensitive to these subtle tonal differences. Multiple runs of the same pipeline sometimes transcribe correctly and sometimes don't.

**Resolution:** Accepted as expected behavior for stochastic TTS. Spectral cosine similarity between Swift and PyTorch outputs was 0.73 (reasonable for temperature-sampled TTS), confirming the audio quality is comparable.

---

## 6. RMS Display Showing "0.0" for Quiet Audio

**Problem:** Test output showed "Audio RMS: 0.0" even though audio contained speech.

**Root Cause:** Python format string `f"{0.035:.1f}"` rounds to "0.0", making quiet (but valid) audio appear silent.

**Fix:** Changed format from `:.1f` to `:.4f` for sufficient precision.

---

## 7. Spectral Similarity Expectations for Stochastic TTS

**Problem:** Initial expectation was that Swift output should closely match PyTorch reference. Spectral cosine similarity was only 0.73-0.76.

**Root Cause:** Even two runs of the same PyTorch model with different seeds produce spectral cosine similarity of only ~0.12. TTS with temperature sampling is inherently stochastic -- the audio sounds correct but the waveforms are completely different.

**Resolution:** Established correct expectations:
- Same model, same seed: ~0.92 cosine similarity
- Same model, different seed: ~0.12 cosine similarity
- CoreML vs PyTorch (different seeds): 0.73-0.92 cosine similarity
- All three pipelines (Swift, CoreML Python, PyTorch) produce correct ASR transcriptions
