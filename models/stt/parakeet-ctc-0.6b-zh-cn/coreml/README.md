# Parakeet CTC 0.6B - Simplified Chinese (zh-CN) - CoreML

CoreML conversion of NVIDIA's Parakeet-CTC-0.6B Mandarin Chinese (Simplified) + English ASR model.

## Model Overview

- **Source**: `nvidia/riva/parakeet-ctc-riva-0-6b-unified-zh-cn:trainable_v3.0`
- **Architecture**: Hybrid RNNT+CTC (FastConformer encoder + CTC decoder head)
- **Languages**: Mandarin Chinese (Simplified, zh-CN) + English code-switching
- **Training Data**: 17,000+ hours of Mandarin and English speech
- **Vocabulary**: 7,000 SentencePiece BPE tokens + 1 blank token

## Converted Component

This conversion extracts **only the CTC decoder head** (linear projection layer):

```
Input:  encoder_output [1, 1024, T] - Encoder features
Output: ctc_logits     [1, T, 7001] - CTC log-probabilities
```

The encoder is not included (use the existing English Parakeet encoder or convert separately).

## Performance

Benchmarked on Apple M2 (16GB RAM, macOS 26.5):

| Compute Unit         | CPU%  | GPU% | ANE% | Latency |
|---------------------|-------|------|------|---------|
| **ALL** (default)   | 0.0%  | 0.0% |100.0%| 1.02ms  |
| cpu_and_neural_engine| 0.0% | 0.0% |100.0%| 1.04ms  |
| cpu_only            |100.0% | 0.0% | 0.0% | 2.03ms  |
| cpu_and_gpu         | 0.0%  |100.0%| 0.0% | 2.34ms  |

**Key Metrics:**
- ✅ 100% ANE utilization (no CPU fallback)
- ✅ 1.02ms inference time (sub-millisecond)
- ✅ 251ms cold compile time
- ✅ Runs entirely on Neural Engine (optimal efficiency)

## Directory Structure

```
conversion/                   # Model conversion scripts
├── export-ctc-zh-cn.py      # CTC head export (decoder only)
├── export-full-pipeline.py  # Full pipeline export (preprocessor + encoder + decoder)
├── quantize-encoder-advanced.py  # Post-training quantization
└── individual_components.py # PyTorch wrappers for tracing

validation/                   # Validation scripts
├── validate-ctc-zh-cn.py    # Single-sample validation (NeMo vs CoreML)
└── benchmark-cer.py         # Multi-sample validation on FLEURS

benchmark/                    # Benchmark scripts
├── benchmark-full-pipeline.py  # Full CoreML pipeline benchmark
└── text_normalizer.py       # Chinese text normalization utilities

build/                        # Generated output directory (not in repo)
├── CtcHeadZhCn.mlpackage    # Uncompiled CoreML model (generated)
├── CtcHeadZhCn.mlmodelc/    # Compiled CoreML model (generated)
├── vocab.json                # 7000-token BPE vocabulary (generated)
└── ctc_head_metadata.json    # Model metadata (generated)
```

**Note:** The `build/` directory is created when you run the conversion scripts and is not included in this repository.

## Setup

```bash
# Install dependencies
uv sync

# Environment is ready (Python 3.10, NeMo, CoreML Tools)
```

## Usage

### 1. Profile the Model

Measure performance on your device:

```bash
cd ../../../../tools/coreml-cli
uv run coreml-cli ../../models/stt/parakeet-ctc-0.6b-zh-cn/coreml/build/CtcHeadZhCn.mlmodelc
```

Check ANE optimization:

```bash
uv run coreml-cli ../../models/stt/parakeet-ctc-0.6b-zh-cn/coreml/build/CtcHeadZhCn.mlmodelc --fallback
```

### 2. Validate Conversion

Test against the original NeMo model:

```bash
# Download test audio
uv run python download-test-audio.py --output-dir ./test_audio --num-samples 5

# Validate (requires .nemo checkpoint)
uv run python validation/validate-ctc-zh-cn.py \
  --audio-file test_audio/aishell_test_000.wav \
  --nemo-path ../parakeet-ctc-riva-0-6b-unified-zh-cn_vtrainable_v3.0/*.nemo \
  --coreml-dir ./build
```

Expected output:
```
✓ Transcriptions match!
✓ Numerical accuracy: EXCELLENT (< 1e-3)
```

### 3. Benchmark on AISHELL-1

Coming soon: Full AISHELL-1 test set evaluation.

## Conversion Process

```bash
# Export CTC head to CoreML
uv run python conversion/export-ctc-zh-cn.py \
  --nemo-path ../parakeet-ctc-riva-0-6b-unified-zh-cn_vtrainable_v3.0/*.nemo \
  --output-dir ./build \
  --compute-units CPU_AND_NE

# Compile (automatic)
xcrun coremlcompiler compile build/CtcHeadZhCn.mlpackage build/
```

## Integration with FluidAudio

To use in Swift:

```swift
import CoreML

let model = try CtcHeadZhCn(configuration: MLModelConfiguration())
let input = CtcHeadZhCnInput(encoder_output: encoderFeatures)
let output = try model.prediction(input: input)
let ctcLogits = output.ctc_logits  // [1, T, 7001]
```

Decode CTC output using greedy or beam search decoder from FluidAudio's ASR module.

## Model Details

### Input Specification
- **Name**: `encoder_output`
- **Type**: Float32
- **Shape**: `[1, 1024, 188]`
- **Description**: Output from FastConformer encoder (hidden dim=1024, time steps=188 for 15s audio)

### Output Specification
- **Name**: `ctc_logits`
- **Type**: Float32
- **Shape**: `[1, 188, 7001]`
- **Description**: CTC log-probabilities (7000 tokens + 1 blank)

### Vocabulary
- **Format**: SentencePiece BPE
- **Size**: 7,000 tokens
- **Languages**: Chinese characters, English words, punctuation
- **Special Tokens**: Blank token at index 7000

## Benchmarking

### Standard Datasets

1. **AISHELL-1** (recommended)
   - 178 hours, 400 speakers
   - Standard Mandarin Chinese benchmark
   - Download: https://www.openslr.org/33/

2. **WenetSpeech Test Sets**
   - `ws_net`: Internet domain (podcasts, videos)
   - `ws_meeting`: Meeting domain (conference calls)
   - Download: https://github.com/wenet-e2e/WenetSpeech

### Quick Test

```bash
# Download 5 AISHELL-1 samples
uv run python download-test-audio.py --num-samples 5

# Run validation
for audio in test_audio/*.wav; do
    uv run python validate-ctc-zh-cn.py --audio-file "$audio" --nemo-path ../*.nemo
done
```

## Known Limitations

- **Encoder not included**: You need the FastConformer encoder to process raw audio
- **Fixed time steps**: Input shape is fixed at 188 time steps (~15 seconds of audio at 16kHz)
- **No language model**: This is the acoustic model only (add LM for better accuracy)
- **Simplified Chinese only**: Traditional Chinese (zh-TW) model is not available as .nemo

## References

- [NGC Model Card](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/models/parakeet-ctc-riva-0-6b-unified-zh-cn)
- [Mandarin-English Collection](https://catalog.ngc.nvidia.com/orgs/nvidia/collections/parakeet-ctc-0.6b-zh-cn)
- [AISHELL-1 Dataset](https://www.openslr.org/33/)
- [WenetSpeech Dataset](https://github.com/wenet-e2e/WenetSpeech)
- [Fast Conformer Paper](https://arxiv.org/abs/2305.05084)

## License

Governed by the NVIDIA Community Model License.
