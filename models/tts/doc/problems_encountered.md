# Problems Encountered

## Overview
This document outlines the key challenges and issues encountered during the development and implementation of the TTS (Text-to-Speech) pipeline for StyleTTS2 & Kokoro.

---
## Component-Specific Issues

### 1. Generator Class

**Problem**: Core audio generation and length determination

- **Description**: This is the primary location where the pipeline generates audio and determines its length. There was various issues trying to get it satup from the frame count, ensuring proper token count, etc

---

### 2. SineGenDeterministic

**Problem**: Incorrect sine wave production for prosodic audio in SineGenDeterministic()

- **Description**: Issues with prosodic audio processing, specifically in sine wave generation. The sine generator helps us make the kokoro TTS audio sound more human and with energy. Without it or incorrectly configuring this would cause the audio to be incredibly robotic and unusable for user experience.

---

### 3. create_alignment_matrix Function

**Problem**: Frame count alignment and phoneme duration mapping

- **Location**: `KokoroCompleteCoreML.create_alignment_matrix`
- **Description**: Challenging task of mapping phoneme duration to audio frames
- **Technical Details**:
  - Difficulty getting frame counts correct
  - Mapping each phoneme's features to the correct time frames in audio
  - Misalignment can cause timing issues and unnatural speech patterns

---

### 4. Text Processing & Chunking Issues

**Problem**: Apostrophe handling and text preprocessing edge cases

- **Description**: Multiple text processing issues during chunking and phoneme conversion
  - Apostrophes going missing during chunking (e.g., "you're" → "you re")
  - This caused incorrect pronunciation where "isn't it" became "isn't he it" with "HE" being added
  - Contractions not being properly handled ("you're" should expand to "you are")
  - Apple's NL Framework sentence tokenization causing edge cases
  - Periods being removed from numbers (e.g., "I need 12.3g of apples" → "I need 12 of apples")
- **Resolution**: Fixed text preprocessing pipeline to handle contractions and numerical edge cases

---

### 5. CoreML Optimization Challenges

**Problem**: GPU vs ANE execution and dynamic input limitations

- **Description**: Performance optimization difficulties with CoreML implementation
- **Technical Details**:
  - Most model operations executing on GPU instead of ANE (Apple Neural Engine)
  - CoreML doesn't support dynamic inputs well or makes it extremely difficult
  - Streaming mode support is challenging due to CoreML's static input requirements
- **Impact**: Limits real-time performance and prevents efficient streaming synthesis
- **Status**: 🟡 Partial workaround with fixed-size models

---

### 6. Audio Quality vs Chunk Duration

**Problem**: Model duration optimization trade-offs

- **Description**: Different model variants needed for different audio duration ranges
- **Technical Details**:
  - 15s model produces poor quality audio for chunks under 3 seconds
  - 5s model performs better for short utterances but has size constraints
  - Quality degradation when using longer-duration models for shorter text
  - Example: "He lives in the USA" - 15s model has noticeably worse quality than 5s model
- **Current Solution**: Provide multiple model variants (5s, 10s, 15s) for different use cases
- **Consideration**: Average sentence length is ~8 seconds, but edge cases exist at both extremes
- **Model Size Trade-off**: v2 5s model is 325MB, v1 is 72MB (reverted to v1)

---

### 7. Pop/Click Artifacts

**Problem**: Audio artifacts at chunk boundaries for short durations

- **Status**: ✅ Resolved
- **Description**: Pop/click sounds occurring at chunk boundaries, especially for audio under 5 seconds
- **Occurrence**: Present across all model types (5s, 10s, 15s) when synthesis duration is shorter than model target
- **Resolution**: Fixed through improved boundary handling and crossfading

---

### 8. Testing & Quality Assurance Challenges

**Problem**: Manual testing requirements and edge case discovery

- **Description**: Difficulty in systematic quality testing of TTS output
- **Challenges**:
  - Manual listening required for each audio output
  - Hard to identify all edge cases without extensive testing
  - Unknown edge cases causing paranoia about release quality
  - No automated quality metrics for naturalness
- **Impact**: Stressful development cycle, difficult to ensure comprehensive coverage

---

### 9. Compute Unit Trade-offs (GPU vs ANE)

**Problem**: Performance and memory trade-offs between GPU and Neural Engine execution

- **Description**: Significant differences in performance and memory usage depending on compute unit configuration
- **Platform Differences**:
  - **macOS (Desktop)**: GPU is optimal
    - `cpuAndGPU`: Lower RAM (0.8-1GB), better performance
    - `cpuAndNeuralEngine`: Higher RAM (1.5GB), slower performance
  - **iOS (Mobile)**: Neural Engine is optimal for memory
    - GPU not directly supported, relies on ANE and CPU
    - `cpuAndNeuralEngine`: Lower RAM (~500MB), acceptable performance
    - `cpuAndGPU`: Higher RAM (1-2GB), better performance but memory intensive
- **Model Compilation Time Issues**:
  - 5s model: 1-3s to compile (acceptable)
  - 10s model: 8s on new devices, **145s on iPhone 13 Pro Max** with ANE
  - 15s model: **60-90s compilation time** on ANE
  - GPU compilation is much faster (1s for all models)
- **Device-Specific Behavior**:
  - iPhone 17 Pro: Fast compilation, handles both GPU/ANE well
  - iPhone 13 Pro Max: Severe compilation delays with ANE
  - M2/M3 Macs: GPU preferred for performance
- **Final Decision**: Provide flexibility, let developers choose based on device capabilities
  - Desktop: Use GPU (better performance, acceptable RAM on desktop)
  - Mobile (≥8GB RAM): Use GPU for performance
  - Mobile (<8GB RAM): Use ANE to conserve memory

---

### 10. Model Quantization Experiments (INT8)

**Problem**: Quantization reduced model size but degraded performance and RAM usage

- **Status**: ❌ Abandoned - Reverted to FP32
- **Description**: INT8 quantization experiments showed mixed results
- **Results**:
  - ✅ Model size: 300MB → 80MB (73% reduction)
  - ❌ RAM usage: Same or worse than FP32
  - ❌ Performance: Slower inference (RTF dropped from 12x to 9x on GPU)
  - ❌ Compile time: No improvement
  - ⚠️ Audio quality: No noticeable degradation initially detected
- **INT8 Performance Comparison** (30s audio on iPhone 17 Pro):
  - FP32 GPU: Peak 2.2GB RAM, RTF 12x, 1s load time
  - INT8 CPU: Peak 1.8GB RAM, RTF 9x, 1s load time
  - INT8 runs entirely on CPU, explaining performance degradation
- **Compression Experiments**:
  - Pruning + Palettization (4-bit) showed promise
  - Reduced peak RAM on ANE (700MB → 460MB)
  - But compilation time for 10s model still problematic on older devices
- **Final Decision**: Stick with FP32 v21 models (5s & 15s)
  - Better performance across all metrics
  - Simpler deployment (no quantization edge cases)
  - Model size acceptable for modern devices

---

### 11. Model Version Iterations (v21 → v24)

**Problem**: Performance regression in newer model versions

- **Description**: Attempted performance improvements in v22/v24 resulted in RAM explosion
- **Version Comparison** (66s audio generation on iPhone 17 Pro):
  - **v21 (Final Choice)**:
    - `cpuAndNeuralEngine`: 336MB RAM, RTF 7.26x, synthesis time 6.77s
    - `cpuAndGPU`: 625MB RAM, RTF 8.80x, synthesis time 5.16s
    - Fast compilation (1-6s)
  - **v24 (Rejected)**:
    - `cpuAndNeuralEngine`: 631MB RAM, RTF 2.35x (much slower)
    - `cpuAndGPU`: 1.14GB RAM, RTF 4.34x
    - Initial compilation: **70s on ANE**, 18s on GPU
- **Root Cause**: Optimization attempts in v24 introduced inefficiencies
- **Lesson**: Performance improvements must be validated across all compute units and devices
- **Final Implementation**: v21 models (5s & 15s) in FP32

---

### 12. Dictionary and Lexicon Management

**Problem**: Vocabulary loading dictionary support

- **Status**: ✅ Partially Resolved
- **Description**: Dictionary merging and loading caused slow startup times and multi-language complexity
- **Issues**:
  - Initial builds took long time to merge `us_gold.json` and `us_silver.json` on first run
  - Separate US and UK dictionaries needed (different pronunciations)
  - Other languages lack dictionary support (rely on ESpeakNG)
  - ESpeakNG OOV (out-of-vocabulary) effectiveness unclear for non-English
  - Constrained by Kokoro's 114-token vocabulary
- **Resolution**:
  - Pre-merged `us_gold.json` and `us_silver.json` into `us_lexicon_cache.json`
  - No more build-upon-run delays

---

### 13. Sentence Segmentation and Chunking

**Problem**: Proper sentence boundary detection and abbreviation handling

- **Status**: 🟡 Improved with custom algorithms
- **Description**: Accurate sentence segmentation critical for natural TTS output
- **Challenges**:
  - Apple's NL Framework doesn't handle all edge cases
  - Text without punctuation is difficult to segment
  - Long, poorly formed sentences common in real-world input
- **Solutions Explored**:
  - Sentence segmentation models: `segment-any-text`, `oliverguhr/fullstop-punctuation-multilang-large`
  - Custom Swift algorithm with abbreviation handling
  - Community contribution: KokoroTokenizer with Misaki G2P support
- **Final Implementation**:
  - Enhanced KokoroChunker with support for:
    - Currency: "$5.23" → "5 dollars and 23 cents"
    - Time: "2:30" → "2 thirty"
    - Decimals: "3.14" → "3 point 1 4"
    - Manual override: `[Mr. Smith](Mister Smith)`
    - Direct phonetic replacement: `[tomato](/təmˈɑːtQ/)`
  - Tested with comprehensive unit tests from MLX implementation
- **Importance**: Full sentences with proper punctuation critical for prosody
  - TTS performs best with complete contextual sentences

---

### 14. ESpeakNG Integration and Deployment

**Problem**: Bundling and deploying ESpeakNG for iOS

- **Status**: ✅ Resolved
- **Description**: ESpeakNG xcframework integration for phoneme generation
- **Solution**: Copied ESpeakNG bundle as part of the iOS app
- **Integration Points**:
  - Used for G2P (Grapheme-to-Phoneme) conversion
  - Fallback for words not in lexicon dictionary
  - Critical for multi-language support
  - Limited testing beyond English

---

### 15. Performance Metrics and Benchmarking

**Problem**: Measuring and optimizing real-time performance

- **Status**: ✅ Established benchmarking methodology
- **Description**: Standardized performance measurement across devices and configurations
- **Key Metrics**:
  - **RTF (Real-Time Factor)**: Ratio of synthesis speed to audio length
    - Example: RTF 10x = 6 min audio generated in 36s
    - Higher is better (8-12x typical for v21)
  - **Time to First Audio**: Latency before audio generation starts (critical for UX)
    - v21: 0.36-0.53s
  - **Peak Memory Usage**: Maximum RAM during inference
    - Varies widely: 336MB (ANE) to 2.2GB (GPU)
  - **Model Load/Compilation Time**: Time to prepare model for inference
    - Critical bottleneck: 1s to 145s depending on model size and compute unit
- **Real-World Performance** (iPhone 16 Pro):
  - ~6 min audio generated in 46s
  - RTF ~8x
  - Streaming support with chunking
- **Benchmarking Tools**:
  - Xcode Instruments for memory profiling
  - CoreML MPS graph debugging
  - Custom Swift benchmark harness: `swift run fluidaudio tts --benchmark`

---
