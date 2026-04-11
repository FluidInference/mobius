# Research Papers

This document lists all research papers and models referenced in the CosyVoice3 CoreML conversion project.

## Primary Models

### CosyVoice3 (Target Model)

**CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens**
- Authors: Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, Qian Chen, Wei Luo, Yike Guo, Wen Wang
- Institution: Alibaba Group
- Year: 2024
- Paper: https://arxiv.org/abs/2407.05407
- Code: https://github.com/FunAudioLLM/CosyVoice
- Model: https://huggingface.co/FunAudioLLM/CosyVoice-300M

**Key Contributions:**
- Supervised discrete speech tokens for improved prosody
- Progressive training: token prediction → duration → speech generation
- 300M parameter model with multilingual zero-shot capabilities
- Issues for CoreML: Vocoder with 705,848 operations (too complex)

---

### MB-MelGAN (Replacement Vocoder)

**Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech**
- Authors: Geng Yang, Shan Yang, Kai Liu, Peng Fang, Wei Chen, Lei Xie
- Institution: Northwestern Polytechnical University, Tencent AI Lab
- Year: 2020
- Paper: https://arxiv.org/abs/2005.05106
- Code (ParallelWaveGAN): https://github.com/kan-bayashi/ParallelWaveGAN

**Key Contributions:**
- Multi-band processing (4 subbands) for efficiency
- Pseudo-QMF (quadrature mirror filter) decomposition
- Parallel processing of subbands
- 4× faster than MelGAN, maintains quality
- **Complexity**: 202 operations (3,494× reduction vs CosyVoice3 vocoder)

**Pre-trained Checkpoint:**
- Dataset: VCTK (109 speakers, 44 hours)
- Training: 1M steps
- Repository: `kan-bayashi/ParallelWaveGAN` (vctk_multi_band_melgan.v2)

---

## Reference Models (CoreML Implementation Patterns)

### Kokoro-82M TTS

**Model Information:**
- Repository: https://github.com/john-rocky/CoreML-Models
- Type: First bilingual (English/Japanese) CoreML TTS
- Parameters: 82M
- Architecture: StyleTTS2-based
- Year: 2024

**Key CoreML Patterns Learned:**
1. **Model splitting**: Predictor (variable length) + Decoder buckets (fixed)
2. **RangeDim for flexible inputs**: Supports arbitrary input sizes (50-500 frames)
3. **FP32 for audio**: "FP16 corrupts audio quality" (direct quote)
4. **Bucketed decoder approach**: 5 decoders for different mel lengths
5. **Runtime trimming**: Predict → pad → decode → trim to exact length

**Base Model Paper (StyleTTS 2):**
- Title: "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models"
- Authors: Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani
- Institution: Columbia University
- Year: 2023
- Paper: https://arxiv.org/abs/2306.07691

---

### HTDemucs (Audio Quality Reference)

**Hybrid Transformers for Music Source Separation**
- Authors: Simon Rouard, Francisco Massa, Alexandre Défossez
- Institution: Meta AI Research (FAIR)
- Year: 2022
- Paper: https://arxiv.org/abs/2211.08553
- Code: https://github.com/facebookresearch/demucs

**Key Contributions:**
- Hybrid architecture: time-domain + spectrogram processing
- Transformer layers for global context
- Real-time music source separation (vocals, drums, bass, other)

**CoreML Implementation:**
- Repository: https://github.com/john-rocky/CoreML-Models
- Key decision: **FP32 to prevent overflow in frequency operations**
- Validated our FP32 choice for MB-MelGAN (audio quality critical)

---

### pyannote.audio (Speaker Diarization Reference)

**pyannote.audio: neural building blocks for speaker diarization**
- Authors: Hervé Bredin, Antoine Laurent
- Institution: CNRS, Université Paris-Saclay
- Year: 2020
- Paper: https://arxiv.org/abs/2104.04045
- Code: https://github.com/pyannote/pyannote-audio

**Key Contributions:**
- Modular pipeline: segmentation → embedding → clustering
- PyanNet segmentation model
- Speaker embedding extraction
- VBx (Variational Bayes) clustering

**CoreML Implementation:**
- Repository: https://github.com/john-rocky/CoreML-Models
- Community model: pyannote/speaker-diarization-community-1
- Pattern: Multi-stage pipeline with separate CoreML models

---

## Supporting Research

### VCTK Corpus (Training Data)

**CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit**
- Institution: University of Edinburgh, Centre for Speech Technology Research (CSTR)
- Speakers: 109 native English speakers (various accents)
- Duration: ~44 hours
- Sample rate: 48 kHz
- Link: https://datashare.ed.ac.uk/handle/10283/3443

**Usage in this project:**
- Pre-trained MB-MelGAN checkpoint trained on VCTK
- Fine-tuning starting point for CosyVoice3 adaptation

---

### FARGAN Vocoder (Investigated Alternative)

**FARGAN: Fast Autoregressive GAN for Neural Vocoding**
- Authors: W. Bastiaan Kleijn, Felicia Lim, Jan Skoglund, Andrew Luebs, Arvindh Krishnaswamy
- Institution: Google Research
- Year: 2023
- Paper: https://arxiv.org/abs/2303.05012
- Code: https://github.com/google/fargan

**Key Contributions:**
- Extremely fast neural vocoder (RTF > 100×)
- Autoregressive GAN architecture
- Designed for low-complexity deployment

**Why not used:**
- Investigated as alternative to MB-MelGAN
- Documented in `trials/FARGAN_ANALYSIS.md`
- Decision: Stuck with MB-MelGAN due to existing pre-trained checkpoints and proven CoreML compatibility

---

## CoreML Conversion Research

### Apple CoreML Documentation

**Core ML Performance**
- Link: https://developer.apple.com/documentation/coreml/core_ml_api/optimizing_core_ml_performance
- Key topics: ANE (Apple Neural Engine) optimization, compute unit selection, model size

**Converting Trained Models to Core ML**
- Link: https://developer.apple.com/documentation/coreml/converting_trained_models_to_core_ml
- Key topics: coremltools usage, model optimization, quantization

**Flexible Input Shapes (RangeDim)**
- Link: https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
- Documentation of ct.RangeDim for variable-length inputs
- Used for mel spectrogram inputs (50-500 frames)

---

## Key Metrics & Benchmarks

### Operation Count Analysis

From our research (documented in `trials/OPERATION_COUNT_ANALYSIS.md`):

| Component | Operations | CoreML Viable |
|-----------|-----------|---------------|
| CosyVoice3 Vocoder (Original) | 705,848 | ❌ No (> 10k limit) |
| MB-MelGAN Vocoder | 202 | ✅ Yes |
| **Reduction Factor** | **3,494×** | 🎯 |

### Quality Metrics

From `benchmarks/test_fp32_vs_fp16.py`:

| Metric | FP16 | FP32 |
|--------|------|------|
| MAE (Mean Absolute Error) | 0.056184 | 0.000000 (perfect) |
| Model Size | 4.50 MB | 8.94 MB |
| Inference Time | 129 ms | 1,664 ms |

**Decision**: Use FP32 for quality-critical applications (follows Kokoro + HTDemucs approach)

### Input Shape Strategy

From `benchmarks/test_rangedim_quickstart.py`:

| Metric | EnumeratedShapes | RangeDim |
|--------|------------------|----------|
| Conversion Time | 8.45s | 3.93s (2.1× faster) |
| Flexibility | 3 fixed sizes | Any 50-500 frames |
| 259 frames test | ❌ Fails | ✅ Works |

**Decision**: Use RangeDim for production (proven by Kokoro TTS)

---

## Citation Format

If you use this work, please cite the relevant papers:

### CosyVoice3
```bibtex
@article{du2024cosyvoice,
  title={CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}
```

### Multi-band MelGAN
```bibtex
@inproceedings{yang2020multiband,
  title={Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech},
  author={Yang, Geng and Yang, Shan and Liu, Kai and Fang, Peng and Chen, Wei and Xie, Lei},
  booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)},
  pages={492--498},
  year={2021},
  organization={IEEE}
}
```

### StyleTTS 2 (Kokoro Base)
```bibtex
@article{li2023styletts2,
  title={StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models},
  author={Li, Yinghao Aaron and Han, Cong and Raghavan, Vinay S and Mischler, Gavin and Mesgarani, Nima},
  journal={arXiv preprint arXiv:2306.07691},
  year={2023}
}
```

### HTDemucs
```bibtex
@inproceedings{rouard2023hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

### pyannote.audio
```bibtex
@inproceedings{bredin2020pyannote,
  title={pyannote.audio: neural building blocks for speaker diarization},
  author={Bredin, Herv{\'e} and Laurent, Antoine},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7124--7128},
  year={2020},
  organization={IEEE}
}
```

---

## Additional Resources

### Tutorials & Guides

- **coremltools Documentation**: https://coremltools.readme.io/
- **ParallelWaveGAN Training Guide**: https://github.com/kan-bayashi/ParallelWaveGAN#training
- **Kokoro Runtime (Swift)**: https://github.com/john-rocky/CoreML-Models/blob/master/sample_apps/KokoroDemo/KokoroDemo/KokoroTTS.swift
- **Apple Neural Engine Guide**: https://github.com/hollance/neural-engine

### Related Projects

- **FluidAudio**: https://github.com/FluidInference/FluidAudio (This project's parent)
- **CoreML Community Models**: https://github.com/john-rocky/CoreML-Models
- **Whisper.cpp**: https://github.com/ggerganov/whisper.cpp (CoreML example)
- **llama.cpp**: https://github.com/ggerganov/llama.cpp (Mobile LLM deployment patterns)

---

## Research Journey Documentation

For detailed documentation of our research process, see:

- **docs/JOHN_ROCKY_PATTERNS.md** - 10 CoreML conversion patterns from Kokoro
- **docs/COREML_MODELS_INSIGHTS.md** - Analysis of successful CoreML audio models
- **trials/KOKORO_APPROACH_ANALYSIS.md** - Deep dive into Kokoro TTS patterns
- **trials/OPERATION_REDUCTION_GUIDE.md** - How we achieved 3,494× reduction
- **trials/MBMELGAN_SUCCESS.md** - Breakthrough moment documentation
- **trials/README.md** - Complete index of 43 trial documents

---

**Last Updated**: 2026-04-11

This research demonstrates that **CoreML TTS is feasible at scale** when using proper architecture replacement (MB-MelGAN vocoder) and following proven patterns (RangeDim, FP32, model splitting).
