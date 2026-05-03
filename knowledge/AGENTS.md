# Knowledge Base Index - Audio Models & CoreML Documentation

A comprehensive knowledge repository combining cutting-edge speech processing research papers and Apple CoreML optimization tooling for on-device AI deployment.

---

## 📚 Knowledge Base Structure

```
knowledge/
├── audio/                          # Speech & audio model research papers (deep dives)
│   ├── AGENTS.md                   # Audio models index (see below)
│   ├── Canary-Parakeet-Multilingual-ASR/
│   ├── CoVoST-2-Multilingual-Speech-Translation/
│   ├── Fast-Conformer-Efficient-Speech-Recognition/
│   ├── Fast-Conformer-Linearly-Scalable-v6/
│   └── Token-Duration-Transducer-TDT/
│
├── papers/                         # Lightweight reading-list index (all papers)
│   └── INDEX.md                    # arXiv links + FluidAudio component map
│
├── coreml/                         # CoreML documentation & tooling
│   ├── AGENTS.md                   # CoreML snapshot index
│   ├── core-ml-on-device-llama.md  # LLM optimization guide
│   ├── coremltools/                # Apple coremltools 9.0b1 docs
│   └── neural-engine/              # Apple Neural Engine reference
│
└── AGENTS.md                        # This file
```

**Where to start:**
- For the **complete reading list** (every paper that underpins a
  shipped FluidAudio component, with arXiv links and a component map):
  → **[`papers/INDEX.md`](./papers/INDEX.md)**
- For **deep extracted notes** on a specific paper: → `audio/<Paper-Slug>/index.md`

---

## 🎯 Quick Navigation

### **Audio Models & Research**
For speech recognition, translation, and spoken language understanding:
→ See **[`audio/AGENTS.md`](./audio/AGENTS.md)**

Key topics:
- Fast Conformer: 2.8× faster speech recognition
- CoVoST 2: Massive multilingual speech translation (21→En, En→15)
- Token-Duration Transducer: Joint token & duration prediction
- Canary & Parakeet: Production-ready multilingual ASR (25 languages)

### **CoreML Optimization & Deployment**
For on-device LLM and neural network deployment:
→ See **[`coreml/AGENTS.md`](./coreml/AGENTS.md)**

Key topics:
- Llama-3.1 CoreML optimization (~33 tok/s on M1 Max)
- coremltools 9.0b1 API reference & conversion workflows
- Apple Neural Engine architecture & transformer deployment
- Quantization, pruning, palettization strategies

---

## 📖 Detailed Index

### Audio Models (`./audio/`)

Complete collection of state-of-the-art speech processing research with practical implementations.

#### 1. **Fast Conformer with Linearly Scalable Attention** (2305.05084 & v6)

**Papers**: Fast Conformer Efficient Speech Recognition (v1 & v6)
**Source**: NVIDIA Research
**Key Innovation**: Redesigned Conformer with 8× downsampling + limited context attention

| Aspect | Details |
|--------|---------|
| **Speed** | 2.8× faster than original Conformer |
| **Scalability** | Supports up to 1.1B parameters without architecture changes |
| **Long-form Audio** | Processes up to 11 hours with limited context + global attention token |
| **Accuracy** | State-of-the-art on LibriSpeech, MLS, Common Voice, WSJ |
| **Applications** | ASR, Speech Translation (1.66× speedup), Intent Classification |

**Performance Highlights**:
- LibriSpeech test-other: 4.99% WER (RNNT)
- Speech Translation: +1.66× speedup with competitive BLEU
- Long-form transcription: 675 minutes max duration on A100

**Best For**: General-purpose efficient speech recognition, long-form transcription, multilingual ASR

---

#### 2. **CoVoST 2: Massively Multilingual Speech-to-Text Translation** (2007.10310)

**Paper**: CoVoST 2 and Massively Multilingual Speech-to-Text Translation
**Source**: Facebook AI Research
**Key Innovation**: Largest open multilingual ST corpus to date

| Aspect | Details |
|--------|---------|
| **Languages** | 21 source → English, English → 15 targets (25 total) |
| **Data Volume** | 2,880 hours of speech (5+ languages, 78K speakers) |
| **Language Coverage** | European + low-resource pairs (Mongolian, Tamil, Latvian, etc.) |
| **Baselines** | Comprehensive monolingual, bilingual, multilingual models |
| **License** | CC0 (fully open) |

**Supported Languages**: French, German, Spanish, Catalan, Italian, Russian, Chinese, Portuguese, Persian, Estonian, Mongolian, Dutch, Turkish, Arabic, Swedish, Latvian, Slovenian, Tamil, Japanese, Indonesian, Welsh

**Best For**: Multilingual research, low-resource language pairs, many-to-one/one-to-many translation studies

---

#### 3. **Token-and-Duration Transducer (TDT)** (2304.06795)

**Paper**: Efficient Sequence Transduction by Jointly Predicting Tokens and Durations
**Source**: NVIDIA Research & Carnegie Mellon University
**Key Innovation**: Joint prediction of tokens AND frame durations for efficient transduction

| Aspect | Details |
|--------|---------|
| **Inference Speed** | 2.82× faster (ASR), 2.27× faster (ST), 1.28× faster (SLU) |
| **Accuracy** | Better accuracy than conventional Transducers across all tasks |
| **Mechanism** | Skips frames guided by predicted duration (frame-by-frame → variable-length) |
| **Tasks** | Speech Recognition, Speech Translation, Intent Classification & Slot Filling |

**Performance Gains**:
- Speech Recognition: Better accuracy + 2.82× speedup
- Speech Translation: +1 BLEU on MUST-C + 2.27× speedup
- Intent Classification: +1% intent accuracy + 1.28× speedup

**Best For**: Production systems requiring both speed and accuracy, variable-length sequence processing

---

#### 4. **Canary-1B-v2 & Parakeet-TDT-0.6B-v3** (NVIDIA)

**Paper**: Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST
**Source**: NVIDIA Research
**Key Innovation**: Production-ready, billion-parameter multilingual models optimized for real-world deployment

**Canary-1B-v2**:
| Aspect | Details |
|--------|---------|
| **Size** | 1 Billion parameters (FastConformer + Transformer) |
| **Languages** | 25 European languages |
| **Training Data** | 1.7M hours (Granary + NeMo ASR Set 3.0) |
| **Inference Speed** | 10× faster than Whisper-large-v3 |
| **Timestamps** | NeMo Forced Aligner (NFA) for reliable segment-level timestamps |
| **Accuracy** | Outperforms Whisper-large-v3 on English ASR |

**Parakeet-TDT-0.6B-v3**:
- Lightweight alternative: 600M parameters
- Uses Token-Duration-Transducer architecture
- 25-language support with smaller footprint
- Optimized for edge/mobile deployment

**Best For**: Production multilingual ASR, edge deployment, cost-effective inference

---

### CoreML Tooling & Optimization (`./coreml/`)

Curated documentation and knowledge snapshots for Apple CoreML model optimization and deployment.

#### 1. **On-Device Llama-3.1 with CoreML**
**Source**: Apple Machine Learning Research
**Published**: November 2024

**Key Techniques**:
- PyTorch → CoreML export with KV cache support
- Float16 + Int4 quantization for 16GB → smaller models
- GPU targeting for best compute/bandwidth ratio
- Context window: 2048 tokens on M1 Max

**Performance**: ~33 tokens/sec on M1 Max-class devices

**Best For**: Understanding LLM optimization patterns, on-device inference, GPU tuning

#### 2. **coremltools 9.0b1 Documentation**
**Source**: [apple/coremltools](https://github.com/apple/coremltools) (Release 9.0b1, Commit: 511e360f)
**License**: BSD (see knowledge/coreml/coremltools/LICENSE.txt)

**Contents**:
- **Conversion Workflows**: PyTorch, TensorFlow 1/2, ONNX interoperability
- **Optimization Tool (OPT)**: Quantization, pruning, palettization APIs
- **Advanced Topics**: Typed execution, stateful models, custom operators
- **Deployment**: Xcode integration, flexible inputs, performance profiling

**Highlights**:
- Int8 model inputs/outputs support (new in 9.0b1)
- iOS 26 / macOS 26 / watchOS 26 / tvOS 26 targets
- PyTorch 2.7 and ExecuTorch 0.5 coverage
- GPU low-precision accumulation hints

**Best For**: Model conversion workflows, quantization strategies, API reference

#### 3. **Neural Engine Architecture & Optimization**
**Source**: [hollance/neural-engine](https://github.com/hollance/neural-engine) (Vendored snapshot)
**Reference**: `neural-engine/AGENTS.md` for full index

**Key Documents**:
- `neural-engine-transformers.md`: 2022 Apple guide for ANE-optimized transformer export
- `distilbert` case study: Export, ANE kernel tuning, validation tips
- Architecture guide: Neural Engine capabilities, kernel patterns, performance bounds

**Best For**: Understanding Apple Silicon neural accelerator, transformer deployment optimization

---

## 🔗 Cross-Reference: Speech Models → CoreML

### Deploying Audio Models on Apple Silicon

**Fast Conformer → CoreML**:
1. Export encoder from NVIDIA NeMo
2. Reference: `coreml/coremltools/docs-guides/convert-pytorch-workflow.md`
3. Apply quantization: See `coreml/coremltools/docs-guides/opt-quantization-api.md`
4. Profile on Neural Engine: `coreml/neural-engine/neural-engine-transformers.md`

**LLM Chat System (Llama + Canary)**:
1. Optimize Canary ASR encoder using patterns from `coreml/core-ml-on-device-llama.md`
2. Deploy Llama-3.1 decoder with KV cache (LLM guide)
3. Combine via stateful model: `coreml/coremltools/docs-guides/stateful-models.md`

---

## 📊 Model Comparison Matrix

| Model | Parameters | Languages | Inference Speed | Best Use Case |
|-------|------------|-----------|------------------|---------------|
| **Fast Conformer** | 120M–1.1B | EN + DE | 2.8× baseline | General ASR, long-form |
| **CoVoST 2** | Varies | 21→EN, EN→15 | Baseline | Research, multilingual |
| **Token-Duration Transducer** | Varies | Multiple | 2.27–2.82× | ST, SLU, efficiency |
| **Canary-1B-v2** | 1B | 25 European | 10× Whisper-v3 | Production multilingual ASR |
| **Parakeet-TDT-0.6B** | 600M | 25 European | Fast | Edge/mobile ASR |
| **Llama-3.1-8B-Instruct (CoreML)** | 8B | English | ~33 tok/s (M1 Max) | On-device chat, inference |

---

## 🛠️ Workflows & Recipes

### Speech Recognition + LLM on macOS

1. **ASR Pipeline**:
   - Input audio → Canary-1B-v2 (multilingual)
   - Output: Transcript + timestamps (via NFA)

2. **LLM Processing**:
   - Transcript → Llama-3.1 CoreML (optimized for M1 Max)
   - Output: Response text

3. **Deployment**:
   - Bundle both models in app
   - Use stateful models for streaming (kv-cache)
   - Reference: `coreml/coremltools/docs-guides/stateful-models.md`

### Multilingual Speech Translation

1. **Encoder**: Fast Conformer (generic speech features)
2. **Decoder**: CoVoST 2 multilingual (21 source → English)
3. **Optimization**: Apply TDT duration prediction for 2.27× speedup
4. **Deployment**: CoreML via coremltools conversion + quantization

### Low-Resource Language ASR

1. **Baseline**: CoVoST 2 provides data + baselines for 15+ low-resource pairs
2. **Fine-tuning**: Use Canary-1B-v2 as pre-trained encoder
3. **Optimization**: Token-Duration Transducer for faster inference
4. **Quantization**: Reference `coreml/coremltools/docs-guides/opt-quantization-api.md`

---

## 📝 Knowledge Base Maintenance

### Adding New Research Papers

1. Place paper documentation in `audio/{descriptive-name}/`
2. Create or update `audio/{descriptive-name}/index.md` with:
   - Paper metadata (title, authors, arxiv/DOI)
   - Abstract summary
   - Key innovations & contributions
   - Performance metrics & benchmarks
   - Applications & use cases
   - Related work & references
3. Update `audio/AGENTS.md` with new entry in the model index
4. Link to this parent `AGENTS.md` if relevant

### Adding CoreML Documentation Snapshots

1. Follow vendoring checklist in `coreml/AGENTS.md`
2. Copy documentation-only assets (no build artifacts)
3. Retain upstream LICENSE/NOTICE files
4. Create colocated `AGENTS.md` with:
   - Upstream source, version, commit hash
   - Document layout & summaries
   - Re-vendoring steps
   - Key highlights & breaking changes
5. Update `coreml/AGENTS.md` with new snapshot info

---

## 🔍 Key Insights

### Speech Models (Audio)
1. **Efficiency Gains**: Modern architectures achieve 2–10× speedups without accuracy loss
2. **Multilingual Scale**: 15–25 language support is now standard for production models
3. **Attention Innovation**: Limited context + global tokens enable 11+ hour audio processing
4. **Joint Modeling**: Token-duration joint prediction improves both speed & accuracy
5. **Data Benefits**: Billion-parameter models show continued improvements with 1.7M+ hours training

### CoreML Deployment
1. **Quantization Wins**: Int4/Int8 quantization with minimal accuracy drop (critical for on-device)
2. **KV Cache**: Essential for LLM inference optimization (stateful models)
3. **GPU Targeting**: Best for memory-bandwidth-limited models (e.g., transformers on Apple Silicon)
4. **Timestamp Management**: Use Forced Aligner or dual-model approaches for segment-level precision
5. **Testing**: Always benchmark on target hardware (M1 vs M3 vs Neural Engine)

---

## 🔗 External References

**Research Communities**:
- [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo) — Open-source speech models
- [Apple Machine Learning Research](https://machinelearning.apple.com/) — CoreML research & optimization
- [Apple coremltools](https://github.com/apple/coremltools) — Official conversion & optimization SDK

**Related Datasets**:
- [CoVoST 2](https://github.com/facebookresearch/covost) — Multilingual speech translation corpus
- [LibriSpeech](http://www.openslr.org/12/) — Large-scale speech recognition benchmark
- [Common Voice](https://commonvoice.mozilla.org/) — Crowdsourced multilingual speech data

**Upstream Papers**:
- Conformer (Gulati et al., 2020): Base architecture for Fast Conformer
- LongFormer (Beltagy et al., 2020): Limited context attention mechanism
- Wav2Vec 2.0: Self-supervised speech representation learning
- MuST-C: Multilingual speech translation corpus (Di Gangi et al., 2019)

---

## 📅 Last Updated

**October 24, 2025**

---

**Questions or Contributions?** See `../CONTRIBUTING.md` for workflow guidelines.
