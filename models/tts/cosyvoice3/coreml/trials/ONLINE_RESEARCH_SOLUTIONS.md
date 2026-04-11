# Online Research: Solutions for Complex Vocoder CoreML Conversion

Research conducted: 2026-04-10

**Problem:** CosyVoice3 vocoder has 705,848 operations (235x more than Kokoro's ~3,000) and fails to convert to CoreML.

**Research goal:** Find solutions beyond the hybrid CoreML + PyTorch approach.

---

## TL;DR - Viable Solutions

| Solution | Feasibility | Implementation Effort | Quality Impact | Speed Impact |
|----------|-------------|----------------------|----------------|--------------|
| **1. Knowledge Distillation** | ✅ High | 2-4 weeks | Minimal (95%+ quality) | 3-5x faster |
| **2. Model Compression Pipeline** | ✅ High | 1-2 weeks | Minimal | 10-50x smaller |
| **3. Replace with Lightweight Vocoder** | ✅ High | 1 week | Medium (90-95% quality) | 5-10x faster |
| **4. iOS 18/macOS 15 Native Features** | ⚠️ Medium | Immediate | None | 3-5x faster on ANE |
| **5. Hybrid (current)** | ✅ Already works | 0 weeks | None | 0.6x RTF proven |

---

## Solution 1: Knowledge Distillation (Recommended)

### Overview

Train a lightweight student model that mimics CosyVoice3 vocoder's behavior but with vastly simpler architecture.

### Research Evidence

**Nix-TTS** ([Nix-TTS: Lightweight and End-to-End Text-to-Speech](https://ar5iv.labs.arxiv.org/html/2203.15643)):
- Achieved **89.34% parameter reduction** from teacher model
- **3.04× inference speedup**
- Only **5.23M parameters** in final student model
- Uses **module-wise distillation** - can distill encoder and decoder independently

**Spiking Vocos** ([Spiking Vocos: An Energy-Efficient Neural Vocoder](https://arxiv.org/html/2509.13049v1)):
- Uses **self-architectural distillation** for knowledge transfer
- Achieves **ultra-low energy consumption**
- Matches teacher quality with significantly reduced operations

**Transformer TTS Distillation** ([Knowledge distillation for Transformer-based TTS](https://www.isca-archive.org/ssw_2025/henriksson25_ssw.pdf)):
- Knowledge distillation enables **significant model size reduction** while **fully replicating teacher performance**
- Can directly optimize to CFG-balanced probabilities, removing CFG at inference (faster)

### Implementation Plan

```python
# 1. Design student vocoder (target: <3k operations)
class StudentVocoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple architecture (like Kokoro)
        self.conv_pre = nn.Conv1d(80, 256, 7, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 16, 8, 4),  # 8x
            nn.ConvTranspose1d(128, 64, 16, 8, 4),   # 8x (total 64x)
        ])
        self.resblocks = nn.ModuleList([
            SimpleResBlock(128),
            SimpleResBlock(64),
        ])
        self.conv_post = nn.Conv1d(64, 1, 7, padding=3)

    def forward(self, mel):
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = self.resblocks[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x).squeeze(1)

# 2. Prepare training data
teacher = CausalHiFTGenerator(...)  # Full CosyVoice3 vocoder
student = StudentVocoder()

# Extract mel-audio pairs
for text in training_texts:
    mel, audio = cosyvoice.inference_cross_lingual(text, prompt_wav)
    pairs.append((mel, audio))

# 3. Train with distillation loss
def distillation_loss(student_output, teacher_output, ground_truth):
    # Reconstruction loss
    l1_loss = F.l1_loss(student_output, ground_truth)

    # Distillation loss (match teacher's output)
    distill_loss = F.l1_loss(student_output, teacher_output)

    # Perceptual loss (optional)
    perceptual_loss = mel_loss(student_output, ground_truth)

    return l1_loss + 0.5 * distill_loss + 0.1 * perceptual_loss

for epoch in range(100):
    for mel, audio in dataloader:
        student_audio = student(mel)
        with torch.no_grad():
            teacher_audio = teacher(mel)

        loss = distillation_loss(student_audio, teacher_audio, audio)
        loss.backward()
        optimizer.step()

    # Validate CoreML conversion every 10 epochs
    if epoch % 10 == 0:
        test_coreml_conversion(student)
```

### Expected Results

- **Parameters:** 5-10M (vs 21M original)
- **Operations:** <3,000 (vs 705,848)
- **Quality:** 95%+ of teacher quality
- **Speed:** 3-5x faster inference
- **CoreML:** ✅ Should convert successfully

### Timeline

- Week 1: Design student architecture, prepare training data
- Week 2-3: Train with distillation, validate quality
- Week 4: Fine-tune, validate CoreML conversion

---

## Solution 2: Model Compression Pipeline

### Overview

Apply state-of-the-art compression techniques to reduce vocoder complexity.

### Research Evidence

**Apple CoreML Compression** ([Use Core ML Tools for machine learning model compression](https://developer.apple.com/videos/play/wwdc2023/10047/)):
- **Palettization:** Discretize weights using lookup tables (1,2,3,4,6,8-bit precision)
- **INT4/INT8 Quantization:** For weights and activations
- **W8A8 mode:** 8-bit activation + weight quantization on A17 Pro/M4 leverages faster int8-int8 compute path on Neural Engine
- **INT4 per-block quantization:** Works well for models using GPU on Mac

**Comprehensive Compression Pipeline** ([Model Compression Techniques Guide](https://createbytes.com/insights/model-compression-techniques-guide)):
1. **Prune** network to remove structural redundancy
2. **Apply knowledge distillation**
3. **Quantize** the resulting model
4. **Result:** 10x, 50x, or even 100x compression rates

**Quantization + Pruning** ([Integrating Pruning with Quantization](https://arxiv.org/html/2509.04244v1)):
- Combined approach yields better results than either technique alone
- Pruning reduces operations count
- Quantization reduces memory footprint

### Implementation Plan

```python
import coremltools as ct

# 1. Prune the vocoder
import torch.nn.utils.prune as prune

def prune_vocoder(model, amount=0.5):
    """Remove 50% of weights with lowest magnitude"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    return model

vocoder = CausalHiFTGenerator(...)
vocoder = prune_vocoder(vocoder, amount=0.5)

# 2. Trace and convert
traced = torch.jit.trace(vocoder, example_mel)

# 3. Apply aggressive quantization
model = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=example_mel.shape)],
    compute_precision=ct.precision.FLOAT16,  # Start with FP16
    minimum_deployment_target=ct.target.iOS18,  # Use latest features
)

# 4. Apply post-conversion compression
import coremltools.optimize as cto

# Palettization (4-bit weights)
config = cto.coreml.OpPalettizerConfig(
    mode="kmeans",
    nbits=4,
)
compressed_model = cto.coreml.palettize_weights(model, config)

# 5. Test
compressed_model.save("vocoder_compressed.mlpackage")
```

### Expected Results

- **Size reduction:** 10-50x smaller
- **Operation reduction:** 50-70% fewer ops
- **Quality:** 90-95% of original (lossy compression)
- **CoreML:** ⚠️ May still be too complex, but worth trying

### Timeline

- Day 1-2: Apply pruning to original vocoder
- Day 3-4: Test CoreML conversion with pruned model
- Day 5-7: Apply quantization/palettization
- Week 2: Fine-tune if quality degrades

---

## Solution 3: Replace with Proven Lightweight Vocoder

### Overview

Replace CosyVoice3's HiFi-GAN vocoder with a proven lightweight alternative, then fine-tune on CosyVoice3's mel outputs.

### Research Evidence

**FARGAN** ([Ultra-Lightweight Neural Differential DSP Vocoder](https://arxiv.org/html/2401.10460v1)):
- **600 MFLOPS** complexity (1/5 of LPCNet, 1/20 of original LPCNet)
- Better quality than LPCNet
- LPCNet development has **stopped** - users encouraged to switch to FARGAN

**Multi-band MelGAN** ([Basis-MelGAN: Efficient Neural Vocoder](https://www.researchgate.net/publication/353067773_Basis-MelGAN_Efficient_Neural_Vocoder_Based_on_Audio_Decomposition)):
- Only **1.91M parameters**
- Reduced computational complexity from **5.85 to 0.95 GFLOPS** (6x reduction)
- Maintains high quality

**Basis-MelGAN**:
- **7.95 GFLOPs** vs HiFi-GAN V1's **17.74 GFLOPs** (2.2x reduction)
- Comparable high-quality audio

**Bunched LPCNet** ([Bunched LPCNet: Vocoder for Low-Cost TTS](https://www.researchgate.net/publication/343599352_Bunched_LPCNet_Vocoder_for_Low-cost_Neural_Text-To-Speech_Systems)):
- **2.19x improvement** over baseline run-time on mobile device
- Less than **0.1 decrease** in TTS mean opinion score

### Implementation Plan

```python
# Option A: Use FARGAN (recommended)
from fargan import FARGAN

# 1. Download pre-trained FARGAN
vocoder = FARGAN()

# 2. Fine-tune on CosyVoice3 data
teacher = CausalHiFTGenerator(...)  # Original vocoder

for epoch in range(20):
    for text in training_texts:
        # Generate mel from CosyVoice3
        mel, target_audio = cosyvoice.inference_cross_lingual(text, prompt_wav)

        # Train FARGAN to match
        pred_audio = vocoder(mel)
        loss = F.l1_loss(pred_audio, target_audio)
        loss.backward()
        optimizer.step()

# 3. Test CoreML conversion
traced = torch.jit.trace(vocoder, example_mel)
mlmodel = ct.convert(traced, ...)  # Should work - FARGAN is lightweight

# Option B: Use Multi-band MelGAN
from mb_melgan import MultiScaleMelGAN

vocoder = MultiScaleMelGAN()
# Fine-tune as above...
```

### Expected Results

- **FARGAN:** 600 MFLOPS, should convert to CoreML ✅
- **MB-MelGAN:** 0.95 GFLOPS, likely converts ✅
- **Quality:** 90-95% of CosyVoice3 (after fine-tuning)
- **Speed:** 5-10x faster than hybrid approach

### Timeline

- Day 1-2: Download and test FARGAN/MB-MelGAN
- Day 3-5: Prepare CosyVoice3 training data
- Week 2: Fine-tune on CosyVoice3 outputs
- Week 3: Validate quality and CoreML conversion

---

## Solution 4: iOS 18 / macOS 15 Native CoreML Improvements

### Overview

Leverage new CoreML features in iOS 18+ and macOS 15+ for better large model support.

### Research Evidence

**New CoreML APIs** ([GitHub - ggml-org/llama.cpp Neural Engine Discussion](https://github.com/ggml-org/llama.cpp/discussions/336)):
- New CoreML APIs in **macOS 15+ and iOS 18+** allow allocating tensors directly
- Can apply operations efficiently using Neural Engine
- **Available only from Swift** (not Objective-C/C++)

**Neural Engine Performance** ([Core ML Integration in iOS and macOS Apps](https://applemagazine.com/core-ml-integration-02tr)):
- Geekbench 6 AI benchmarks show **3-5X faster inference** on ANE vs CPU
- Matrix multiplication shows **4x speedup** on Mac M2 using ANE (217ms vs 1316ms GPU)

**A17 Pro / M4 Optimizations** ([Use Core ML Tools for compression](https://developer.apple.com/videos/play/wwdc2023/10047/)):
- Quantizing both activations and weights to **int8** leverages **optimized compute on Neural Engine**
- Can improve runtime latency in compute-bound models
- **W8A8 mode** (8-bit activation + weight) on newer hardware

**Neural Engine Palettization** ([Core ML Overview](https://developer.apple.com/machine-learning/core-ml/)):
- Neural Engine accelerates models with **low-bit palettization: 1, 2, 4, 6 or 8 bits**
- For memory-bound models, can lead to **inference gains**

### Implementation Plan

```python
# 1. Target iOS 18+ / macOS 15+ explicitly
model = ct.convert(
    traced_vocoder,
    inputs=[ct.TensorType(shape=mel_shape)],
    minimum_deployment_target=ct.target.iOS18,  # Latest target
    compute_precision=ct.precision.FLOAT16,
)

# 2. Apply aggressive int8 quantization (A17 Pro / M4 optimization)
import coremltools.optimize as cto

config = cto.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int8",  # Both weights and activations
)
quantized_model = cto.coreml.linear_quantize_weights(model, config)

# 3. Apply palettization for Neural Engine
palette_config = cto.coreml.OpPalettizerConfig(
    mode="kmeans",
    nbits=4,  # 4-bit palettization
)
final_model = cto.coreml.palettize_weights(quantized_model, palette_config)

# 4. Save and test on iOS 18+ / macOS 15+ device
final_model.save("vocoder_ios18.mlpackage")
```

### Expected Results

- **Speed:** 3-5x faster on ANE vs CPU/GPU
- **Size:** Significantly reduced via quantization
- **Compatibility:** ⚠️ iOS 18+ / macOS 15+ only
- **CoreML conversion:** ⚠️ Still may fail if graph too complex

### Limitations

- **Still may not work:** Graph complexity (705k ops) is the fundamental issue
- **New APIs don't solve operation count:** They just optimize what exists
- **Platform requirement:** Requires cutting-edge OS versions

### Timeline

- Immediate: No additional development needed
- Test on iOS 18+ / macOS 15+ devices with new quantization settings

---

## Solution 5: Successful Reference Implementation

### Overview

Study and replicate the **kokoro-coreml** approach that successfully converted a vocoder to CoreML.

### Research Evidence

**kokoro-coreml** ([GitHub - mattmireles/kokoro-coreml](https://github.com/mattmireles/kokoro-coreml)):
- Successfully exported Kokoro TTS vocoder to CoreML
- Achieves **30-50% speedup** through Apple Neural Engine optimization
- **Two-stage architecture** with fixed shapes
- **Swift-side alignment** to avoid CoreML dynamic-shape pitfalls

**Key implementation details:**
- Fixed input shapes (no dynamic dimensions)
- Custom STFT implementation that works in CoreML
- ~3,000 operations (vs CosyVoice3's 705k)

### What Makes Kokoro Work

**From previous analysis:**
- **Simple F0 handling:** No complex CausalConvRNNF0Predictor
- **Basic source generation:** No NSF (Neural Source Filter)
- **Optimized STFT:** Custom implementation designed for CoreML
- **Fewer upsampling stages:** 2-3 vs CosyVoice3's complex multi-stage
- **Simpler ResBlocks:** No adaptive normalization or style conditioning

### Implementation Plan

```python
# Study the key differences and apply to CosyVoice3

# 1. Remove complex components
class SimplifiedCosyVoice3Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        # REMOVE: CausalConvRNNF0Predictor
        # REMOVE: SourceModuleHnNSF
        # REMOVE: Multi-stage STFT fusion

        # KEEP: Simple upsampling path (Kokoro-style)
        self.conv_pre = nn.Conv1d(80, 256, 7, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 16, 8, 4),
            nn.ConvTranspose1d(128, 64, 16, 8, 4),
        ])
        self.resblocks = nn.ModuleList([
            SimpleResBlock(128),  # Simplified, no AdaIN
            SimpleResBlock(64),
        ])
        self.conv_post = nn.Conv1d(64, 1, 7, padding=3)

    def forward(self, mel):
        # Direct mel → audio (no F0, no source, no STFT fusion)
        x = self.conv_pre(mel)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = self.resblocks[i](x)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x).squeeze(1)

# 2. Train from scratch (or distill from original)
# See Solution 1 for training approach

# 3. Use Kokoro's STFT if needed
from kokoro.istftnet import TorchSTFT
self.stft = TorchSTFT(n_fft=16, hop_length=4)
```

### Expected Results

- **Operations:** ~3,000 (like Kokoro)
- **CoreML:** ✅ Should convert successfully
- **Quality:** 85-95% (depends on training)
- **Speed:** 8x RTF (like Kokoro)

### Timeline

- Week 1: Simplify architecture to Kokoro-level
- Week 2-3: Train simplified model
- Week 4: Validate CoreML conversion and quality

---

## Comparison Matrix

| Solution | Feasibility | Effort | Quality | Speed | CoreML Success | Notes |
|----------|-------------|--------|---------|-------|----------------|-------|
| **Knowledge Distillation** | ✅ High | 2-4 weeks | 95%+ | 3-5x | ✅ High | Proven approach, best quality |
| **Compression Pipeline** | ⚠️ Medium | 1-2 weeks | 90-95% | 2-3x | ⚠️ Medium | May still be too complex |
| **Lightweight Vocoder** | ✅ High | 1 week | 90-95% | 5-10x | ✅ High | Fastest to implement |
| **iOS 18 Features** | ⚠️ Medium | Immediate | 100% | 3-5x | ⚠️ Low | Doesn't solve root cause |
| **Kokoro Approach** | ✅ High | 3-4 weeks | 85-95% | 8x | ✅ High | Proven reference |
| **Hybrid (current)** | ✅ Proven | 0 weeks | 97% | 0.6x RTF | N/A | Already works |

---

## Recommended Action Plan

### Immediate (1 week)

**Option A: Test FARGAN replacement**
- Download FARGAN pre-trained model
- Test quality on CosyVoice3 mel outputs (no fine-tuning)
- Attempt CoreML conversion
- **If successful:** Fine-tune for 1-2 weeks
- **If fails:** Move to Option B

**Option B: Apply compression pipeline**
- Prune existing vocoder (50% weights)
- Test CoreML conversion with iOS 18 quantization
- **If successful:** Deploy
- **If fails:** Move to knowledge distillation

### Medium-term (2-4 weeks)

**Knowledge Distillation (if Options A/B fail)**
- Design student architecture (<3k ops, Kokoro-style)
- Prepare CosyVoice3 training data (mel-audio pairs)
- Train with distillation loss
- Validate CoreML conversion every 10 epochs
- Fine-tune for quality

### Fallback

**Continue with hybrid approach**
- Already proven: 97% accuracy, 0.6x RTF
- 60% CoreML (embedding, lm_head, decoder)
- 40% PyTorch (vocoder, flow)
- Production-ready today

---

## Sources

### CoreML Optimization
- [CoreML Export for YOLO26 Models](https://docs.ultralytics.com/integrations/coreml/)
- [CoreML Model Variants | OmniZip-CVPR2026](https://deepwiki.com/adminasmi/OmniZip-CVPR2026/7.1-coreml-model-variants)
- [Core ML Tools Overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- [Use Core ML Tools for compression - WWDC23](https://developer.apple.com/videos/play/wwdc2023/10047/)
- [Model Intermediate Language (MIL)](https://deepwiki.com/apple/coremltools/5-model-intermediate-language-(mil))
- [Core ML Tools FAQs](https://apple.github.io/coremltools/docs-guides/source/faqs.html)
- [Performance Guide - Pruning](https://apple.github.io/coremltools/docs-guides/source/opt-pruning-perf.html)

### Lightweight Neural Vocoders
- [Ultra-Lightweight Neural Differential DSP Vocoder (FARGAN)](https://arxiv.org/html/2401.10460v1)
- [Ultra-Lightweight Neural DSP Vocoder - OpenReview](https://openreview.net/forum?id=gfb6KmY3dT)
- [Spiking Vocos: Energy-Efficient Neural Vocoder](https://arxiv.org/html/2509.13049v1)
- [Basis-MelGAN: Efficient Neural Vocoder](https://www.researchgate.net/publication/353067773_Basis-MelGAN_Efficient_Neural_Vocoder_Based_on_Audio_Decomposition)
- [Bunched LPCNet: Vocoder for Low-Cost TTS](https://www.researchgate.net/publication/343599352_Bunched_LPCNet_Vocoder_for_Low-cost_Neural_Text-To-Speech_Systems)
- [LPCNet GitHub - xiph/LPCNet](https://github.com/xiph/LPCNet)

### HiFi-GAN and CoreML Conversion
- [kokoro-coreml - GitHub](https://github.com/mattmireles/kokoro-coreml)
- [CoreML Models Zoo - GitHub](https://github.com/john-rocky/CoreML-Models)
- [Model Compression Techniques Guide](https://createbytes.com/insights/model-compression-techniques-guide)
- [Integrating Pruning with Quantization](https://arxiv.org/html/2509.04244v1)
- [Apple Neural Engine Transformers](https://github.com/apple/ml-ane-transformers)
- [Deploying Transformers on ANE - Apple ML Research](https://machinelearning.apple.com/research/neural-engine-transformers)

### iOS 18/macOS 15 Improvements
- [Core ML Overview - Apple Developer](https://developer.apple.com/machine-learning/core-ml/)
- [Core ML Integration in iOS and macOS Apps](https://applemagazine.com/core-ml-integration-02tr)
- [Neural Engine Support Discussion - llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/336)
- [Faster Stable Diffusion with Core ML](https://huggingface.co/blog/fast-diffusers-coreml)
- [Core ML Documentation](https://developer.apple.com/documentation/coreml)

### Knowledge Distillation for TTS
- [Nix-TTS: Lightweight End-to-End TTS via Distillation](https://ar5iv.labs.arxiv.org/html/2203.15643)
- [Nix-TTS GitHub](https://github.com/rendchevi/nix-tts)
- [Knowledge Distillation for Transformer TTS](https://www.isca-archive.org/ssw_2025/henriksson25_ssw.pdf)
- [Cross-Lingual Knowledge Distillation via Flow-Based Voice Conversion](https://link.springer.com/chapter/10.1007/978-981-99-8126-7_20)
- [NoreSpeech: Knowledge Distillation based Conditional Diffusion](https://www.semanticscholar.org/paper/c29d2e98fda1bf772139da11814e313836df3704)

---

## Conclusion

**Pure CoreML is achievable, but requires architecture redesign.**

**Best approach:**
1. **Try FARGAN first** (1 week) - fastest path to pure CoreML
2. **If FARGAN fails, use knowledge distillation** (2-4 weeks) - proven to work
3. **Fallback to hybrid** - already production-ready

**Hybrid approach remains the most practical solution for immediate deployment.**

The fundamental issue is not CoreML limitations, but CosyVoice3's architecture being designed for quality (705k ops) rather than mobile efficiency (3k ops).

All solutions require trading some quality for simplicity, except the hybrid approach which maintains full quality at the cost of PyTorch dependency.
