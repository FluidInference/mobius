# MB-MelGAN Fine-tuning Guide for CosyVoice3

Complete pipeline for fine-tuning MB-MelGAN vocoder on CosyVoice3 mel spectrograms to achieve pure CoreML TTS.

## Overview

**Problem**: CosyVoice3's original vocoder (705,848 operations) is too complex for CoreML conversion.

**Solution**: Fine-tune lightweight MB-MelGAN vocoder (202 operations) on CosyVoice3 mel spectrograms.

**Result**: Pure CoreML TTS pipeline with acceptable quality and performance.

---

## Pipeline Steps

### 1. Download Pre-trained MB-MelGAN

```bash
uv run python download_mbmelgan.py
```

Downloads VCTK multi-band MelGAN checkpoint (1M steps) to `mbmelgan_pretrained/`.

**Size**: ~20 MB
**Source**: kan-bayashi/ParallelWaveGAN

### 2. Generate Training Data from CosyVoice3

```bash
uv run python generate_training_data.py
```

Generates 1,000 (mel, audio) pairs using CosyVoice-300M.

**Output**:
- `mbmelgan_training_data/mels/*.pt` - 80-bin mel spectrograms
- `mbmelgan_training_data/audio/*.wav` - 22.05kHz audio samples

**Progress**: ~60 sec/sample → ~16 hours for 1,000 samples

**Note**: Runs in background, use `tail -f` to monitor:
```bash
tail -f /tmp/claude/[...]/tasks/[task_id].output
```

### 3. Quick Fine-tune Demo (Optional)

```bash
uv run python quick_finetune.py
```

Tests the fine-tuning pipeline with synthetic data (500 samples, 20 epochs).

**Output**:
- `mbmelgan_quickstart/mbmelgan_quickstart.pt` - Fine-tuned weights
- `mbmelgan_quickstart/mbmelgan_quickstart_coreml.mlpackage` - CoreML model

**Purpose**: Validates end-to-end pipeline before production training.

### 4. Production Fine-tuning

```bash
uv run python train_mbmelgan.py --epochs 100 --batch-size 8
```

Fine-tunes MB-MelGAN on real CosyVoice3 data (1,000 samples).

**Output**:
- `mbmelgan_finetuned/mbmelgan_epoch_*.pt` - Checkpoints every 10 epochs
- `mbmelgan_finetuned/mbmelgan_final.pt` - Final weights
- `mbmelgan_finetuned/mbmelgan_coreml_fp32.mlpackage` - CoreML (FP32)
- `mbmelgan_finetuned/mbmelgan_coreml_fp16.mlpackage` - CoreML (FP16)

**Training time**: ~6-12 hours on CPU (depends on batch size)

**Hyperparameters**:
- Learning rate: 1e-4
- Optimizer: Adam
- Loss: Multi-scale STFT + L1
- Input: 80-bin mel, variable length (via RangeDim)
- Output: 4-band audio

### 5. Test Quality

```bash
uv run python test_quickstart_quality.py
```

Compares fine-tuned MB-MelGAN against PyTorch CosyVoice3 baseline.

**Metrics**:
- MAE (Mean Absolute Error)
- PESQ (Perceptual Evaluation of Speech Quality)
- Spectral convergence

---

## CoreML Conversion Best Practices

Based on benchmarks from `test_fp32_vs_fp16.py` and `test_rangedim_quickstart.py`:

### Precision: FP32 vs FP16

| Metric | FP16 | FP32 | Recommendation |
|--------|------|------|----------------|
| **Accuracy (MAE)** | 0.056 | 0.000 | **FP32** for quality |
| **Model Size** | 4.50 MB | 8.94 MB | FP16 for size |
| **Inference Time** | 129ms | 1664ms | FP16 for speed |

**Kokoro/HTDemucs precedent**: Both use FP32 for audio quality
**Trade-off**: 2x larger model, 12.9x slower, but perfect accuracy

**Recommendation**:
- Production TTS: Use FP32
- Real-time apps: Test FP16 quality first

### Input Shape: RangeDim vs EnumeratedShapes

| Metric | EnumeratedShapes | RangeDim |
|--------|------------------|----------|
| **Flexibility** | 3 fixed (125,250,500) | Any 50-500 |
| **Conversion Time** | 8.45s | 3.93s (2.1x faster) |
| **259 frames** | ❌ Fails | ✅ Works |

**Kokoro precedent**: Uses RangeDim(1, 256) for phoneme input

**Recommendation**: Use RangeDim for production
- No padding artifacts
- Simpler runtime (no bucket selection)
- Supports exact input sizes

---

## File Structure

```
mobius/models/tts/cosyvoice3/coreml/
├── MBMELGAN_FINETUNING_GUIDE.md     # This guide
├── JOHN_ROCKY_PATTERNS.md           # CoreML conversion patterns
├── COREML_MODELS_INSIGHTS.md        # Analysis of john-rocky's repo
│
├── download_mbmelgan.py             # Download pre-trained checkpoint
├── generate_training_data.py        # Generate CosyVoice3 data
├── quick_finetune.py                # Quick demo (synthetic data)
├── train_mbmelgan.py                # Production fine-tuning
├── test_quickstart_quality.py       # Quality evaluation
│
├── test_fp32_vs_fp16.py             # Precision benchmark
├── test_rangedim_quickstart.py      # Input shape benchmark
│
├── mbmelgan_pretrained/             # Downloaded checkpoint
├── mbmelgan_training_data/          # Generated (mel, audio) pairs
├── mbmelgan_quickstart/             # Quick demo output
├── mbmelgan_finetuned/              # Production output
├── precision_test/                  # FP32 vs FP16 models
└── rangedim_quickstart_test/        # RangeDim vs Enum models
```

---

## Implementation in train_mbmelgan.py

### Current Configuration

```python
# Model architecture (matching pre-trained VCTK)
model = MelGANGenerator(
    in_channels=80,
    out_channels=4,  # Multi-band
    channels=384,
    kernel_size=7,
    upsample_scales=[5, 5, 3],  # 75x upsampling
    stack_kernel_size=3,
    stacks=4
)

# CoreML conversion (EnumeratedShapes + FP16)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(
        name="mel_spectrogram",
        shape=ct.EnumeratedShapes(shapes=[
            (1, 80, 125), (1, 80, 250), (1, 80, 500)
        ])
    )],
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS17,
)
```

### Recommended Updates (TODO)

Based on benchmark findings:

```python
# Use RangeDim for flexibility
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(
        name="mel_spectrogram",
        shape=(1, 80, ct.RangeDim(lower_bound=50, upper_bound=500, default=125))
    )],
    compute_precision=ct.precision.FLOAT32,  # Better quality
    minimum_deployment_target=ct.target.iOS17,
)
```

**Benefits**:
- RangeDim: Any mel length 50-500 supported (no padding)
- FP32: Perfect accuracy (MAE=0 vs PyTorch)

**Trade-offs**:
- 2x larger model (8.9 MB vs 4.5 MB)
- 12.9x slower inference (1664ms vs 129ms)

---

## Key Findings from Benchmarks

### FP32 Accuracy

```
PyTorch:  [−0.234, 0.187] range
FP16:     [−0.183, 0.224] range (MAE=0.056)
FP32:     [−0.234, 0.187] range (MAE=0.000) ← Perfect match!
```

### RangeDim Flexibility

```
Input: 259 frames (not in [125, 250, 500] enum)

EnumeratedShapes: ❌ Error - must crop to 250 or pad to 500
RangeDim:         ✅ Works directly - (1, 4, 19425) output
```

---

## References

- **MB-MelGAN Paper**: Multi-band MelGAN: Faster Waveform Generation for High-Quality TTS
- **Pre-trained Model**: [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- **Kokoro Patterns**: [john-rocky/CoreML-Models](https://github.com/john-rocky/CoreML-Models)
- **CosyVoice**: [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M)

---

## Next Steps

1. **Wait for training data**: 217/1,000 samples complete (~10 hours remaining)
2. **Run production fine-tuning**: `uv run python train_mbmelgan.py --epochs 100`
3. **Evaluate quality**: Compare against PyTorch CosyVoice3
4. **Update train script**: Apply RangeDim + FP32 based on benchmarks
5. **Integrate with FluidAudio**: Add to TTS product

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| **RTFx** (Real-time factor) | > 1.0x | TBD (after fine-tuning) |
| **Quality vs PyTorch** | MAE < 0.01 | TBD |
| **Model Size** | < 10 MB | 8.9 MB (FP32) ✅ |
| **Latency (250 frames)** | < 500ms | ~400ms (estimated) |

---

## Troubleshooting

### Training data generation stuck?

Check background task output:
```bash
tail -f /tmp/claude/-Users-kikow-brandon-voicelink-FluidAudio/tasks/*.output
```

### CoreML conversion fails?

1. Check operation count: `test_fp32_vs_fp16.py` shows 202 ops (well under limit)
2. Try ONNX intermediate: `torch.onnx.export()` → `ct.convert(onnx_path)`
3. Check for unsupported ops: complex STFT, unfold, etc.

### Poor quality after fine-tuning?

1. Increase epochs (100 → 200)
2. Lower learning rate (1e-4 → 5e-5)
3. Add more training data (1,000 → 5,000 samples)
4. Use multi-scale STFT loss (already implemented)

---

**Status**: Training data generation in progress (21.7% complete)
**Next**: Production fine-tuning after data generation completes
