# Kokoro TTS CoreML Implementation

High-quality neural Text-to-Speech for Apple devices. Convert Kokoro-82M (StyleTTS2-based) models to CoreML for on-device inference on iOS and macOS.

**Features**: Multi-speaker synthesis • Natural prosody • 8-12x real-time • On-device privacy • English optimized

---

## Quick Links

**Documentation**
- [Model Architecture](doc/v21_conversion_script_outline.md) - Technical deep dive
- [Problems & Solutions](doc/problems_encountered.md) - Issue tracking and benchmarks
- [TTS Concepts](doc/tts_concepts.md) - Prosody, F0, alignment explained

**Resources**
- [Kokoro-82M Model](https://huggingface.co/hexgrad/Kokoro-82M)
- [Pre-converted CoreML Models](https://huggingface.co/FluidInference/kokoro-82m-coreml)
- [StyleTTS2 Paper](https://arxiv.org/pdf/2306.07691) | [GitHub](https://github.com/yl4579/StyleTTS2)

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/FluidInference/mobius.git
cd mobius/models/tts

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Apply Kokoro Patch

The original Kokoro library needs patches for CoreML compatibility:

```bash
# Install Kokoro first
pip install kokoro-onnx

# Apply patch
cd $(python -c "import kokoro; import os; print(os.path.dirname(kokoro.__file__))")
patch -p1 < /path/to/mobius/models/tts/kokoro_coreml_fix.patch
```

See [kokoro_coreml_fix.patch](kokoro_coreml_fix.patch) for details.

### 3. Run Inference (Pre-converted Models)

Download and use pre-converted models:

```bash
# Download models
huggingface-cli download FluidInference/kokoro-82m-coreml kokoro_21_5s.mlmodelc
huggingface-cli download FluidInference/kokoro-82m-coreml kokoro_21_15s.mlmodelc

# Run inference in Python
python -c "
from v21 import run_coreml_with_pt_inputs

run_coreml_with_pt_inputs(
    mlpackage_path='kokoro_21_5s.mlmodelc',
    text='Hello world, how are you today?',
    voice='af_heart',
    out_wav='output.wav'
)
"
```

Or use as a notebook in VS Code/Jupyter:

```bash
# Open v21.py as interactive notebook (uses # %% cell markers)
code v21.py  # VS Code
# or
jupyter lab v21.py  # Jupyter
```

### 4. Convert Your Own Model

Run the conversion script interactively:

```python
# Open v21.py in VS Code or Jupyter Lab
# Run cells sequentially:
# 1. Load Kokoro model
# 2. Prepare example inputs
# 3. Trace with torch.jit
# 4. Convert to CoreML
# 5. Save .mlpackage

# Or run programmatically:
from v21 import KokoroCompleteCoreML
import torch
import coremltools as ct

# ... (see v21.py for full conversion code)
```

---

## Model Variants

| Model | Duration | Size | Use Case | Compilation | RAM (ANE) |
|-------|----------|------|----------|-------------|-----------|
| **5s** | ~5 sec | 80MB | Streaming, chat | 1-3s | 336MB |
| **10s** | ~10 sec | 100MB | Balanced | 8s | 460MB |
| **15s** | ~15 sec | 120MB | Long-form | 60-90s | 500MB |

**Selection Guide**:
- Chat/Real-time → 5s model
- Audiobooks → 15s model
- General use → 10s model

**Note**: Longer models degrade on short text ("Hello" → use 5s not 15s)

---

## How It Works

```
Text → Phonemes → BERT → Duration → Alignment → F0/Energy → Style → Decoder → Audio
```

**Pipeline**:
1. **G2P**: Text → phonemes (`"hello"` → `[HH, AH0, L, OW1]`)
2. **BERT**: Contextualized phoneme embeddings
3. **Duration Predictor**: How long each phoneme lasts
4. **Alignment Matrix**: Map phonemes to acoustic frames
5. **Prosody Predictors**: F0 (pitch) and energy
6. **Style Encoder**: Reference voice embedding
7. **Decoder**: 4 refinement blocks
8. **Generator**: Harmonic synthesis + vocoder → 24kHz audio

See [Architecture Doc](doc/v21_conversion_script_outline.md) for class-level details.

---

## Performance

**iPhone 17 Pro** (v21 5s):
- GPU: 625MB RAM, RTF 8.8x, 0.36s latency
- ANE: 336MB RAM, RTF 7.26x, 0.53s latency

**M3 Max**:
- GPU: 0.86GB RAM (best)
- ANE: 1.54GB RAM (slower)

**RTF** = Real-Time Factor (10x = generate 10 seconds of audio in 1 second)

Full benchmarks: [Problems Encountered - Performance](doc/problems_encountered.md#15-performance-metrics-and-benchmarking)

---

## Usage Examples

### Basic Synthesis

```python
from v21 import run_coreml_with_pt_inputs

# Generate speech
run_coreml_with_pt_inputs(
    "kokoro_21_5s.mlmodelc",
    "Hello world!",
    "af_heart",
    "output.wav"
)
```

### Prosody Control

```python
# Questions vs statements
run_coreml_with_pt_inputs("model.mlpackage", "Really?", "af_heart", "question.wav")
run_coreml_with_pt_inputs("model.mlpackage", "Really.", "af_heart", "statement.wav")

# Emphasis (use caps for stress)
run_coreml_with_pt_inputs("model.mlpackage", "I CAN'T believe it", "af_heart", "emphasis.wav")

# Pauses (use ellipsis)
run_coreml_with_pt_inputs("model.mlpackage", "Wait... what?", "af_heart", "pause.wav")
```

### Batch Processing

```python
texts = ["First sentence.", "Second sentence.", "Third sentence."]

for i, text in enumerate(texts):
    run_coreml_with_pt_inputs(
        "model.mlpackage", text, "af_heart", f"output_{i}.wav"
    )
```

### Compute Unit Configuration

```python
import coremltools as ct

# macOS/Desktop - use GPU
model = ct.models.MLModel("model.mlpackage",
                          compute_units=ct.ComputeUnit.cpuAndGPU)

# iOS (low RAM) - use ANE
model = ct.models.MLModel("model.mlpackage",
                          compute_units=ct.ComputeUnit.cpuAndNeuralEngine)
```

---

## CoreML Modifications

Original Kokoro (PyTorch) incompatible with CoreML. Key fixes:

| Issue | Solution |
|-------|----------|
| `pack_padded_sequence` | Explicit LSTM states + masking |
| Random phases | Deterministic `random_phases` input |
| Dynamic loops | Broadcasting-only operations |
| In-place ops | Pure functional transforms |

**Modified Classes**:
- `TextEncoderFixed` - LSTM without pack_padded_sequence
- `TextEncoderPredictorFixed` - Duration predictor with explicit states
- `SineGenDeterministic` - Controlled harmonic generation
- `GeneratorDeterministic` - F0-based deterministic noise
- `KokoroCompleteCoreML` - End-to-end wrapper

Details: [Model Architecture Doc](doc/v21_conversion_script_outline.md)

---

## Troubleshooting

**Robotic audio?**
- Use complete sentences with punctuation
- Check model version (use v21, not v22/v24)
- Verify reference voice loaded correctly

**High memory?**
- Switch to ANE: `compute_units=ct.ComputeUnit.cpuAndNeuralEngine`
- Use 5s model instead of 15s

**Long compilation (60s+)?**
- Use 5s model (1-3s compilation)
- Or use GPU instead of ANE (if RAM allows)

**Text issues (apostrophes, numbers)?**
- Apply text preprocessing:
  - "you're" → "you are"
  - "$5.23" → "5 dollars and 23 cents"
  - "2:30" → "2 thirty"

See [Problems Encountered](doc/problems_encountered.md) for detailed solutions.

---

## Project Structure

```
tts/
├── v21.py                    # Main conversion script (notebook-compatible)
├── doc/
│   ├── v21_conversion_script_outline.md
│   └── problems_encountered.md
├── kokoro_coreml_fix.patch   # Required Kokoro patches
├── requirements.txt
└── pyproject.toml
```

**v21.py** uses `# %%` cell markers - open in VS Code/Jupyter as notebook.

**Key Classes** (see [Architecture Doc](doc/v21_conversion_script_outline.md)):
- `TextEncoderFixed` (Lines 197-281)
- `TextEncoderPredictorFixed` (Lines 115-194)
- `SineGenDeterministic` (Lines 294-325)
- `SourceModuleHnNSFDeterministic` (Lines 330-347)
- `GeneratorDeterministic` (Lines 350-422)
- `KokoroCompleteCoreML` (Lines 428-578)

---

## Known Limitations

- **English optimized** - Other languages experimental (Mandarin quality varies)
- **Fixed input shapes** - Must pad to MAX_TOKENS (CoreML limitation)
- **No batching** - Batch size = 1 only
- **No true streaming** - Workaround: chunk at sentence boundaries
- **ANE compilation** - Slow on older devices (60-90s for 15s model)

Deferred features: INT8 quantization, dynamic shapes, batch>1

---

## Contributing

- **Bug Reports**: [GitHub Issues](https://github.com/FluidInference/mobius/issues)
- **Testing**: Share benchmarks from different devices
- **Pull Requests**: Welcome! Include tests and documentation

**Community Contributors**:
- @croqueteer - iOS testing and optimization
- @jamshaidali102 - Text processing improvements

---

## License

Builds upon:
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [StyleTTS2](https://github.com/yl4579/StyleTTS2)

See component licenses for details.

---

## Citation

```bibtex
@software{kokoro_coreml_2024,
  title={Kokoro TTS CoreML Implementation},
  author={FluidInference Team},
  year={2024},
  url={https://github.com/FluidInference/mobius}
}

@article{li2023styletts2,
  title={StyleTTS 2: Towards Human-Level Text-to-Speech},
  author={Li, Yinghao and Han, Cong and Raghavan, Nima},
  journal={arXiv preprint arXiv:2306.07691},
  year={2023}
}
```

---

**Version**: v21 (FP32) | **Updated**: 2024-10-06 | **Requires**: iOS 17+, macOS 14+
