# Adding New Model Conversions

Step-by-step guide for converting a new model to CoreML and shipping it in mobius. Intended for contributors and coding agents.

## Overview

Each new model conversion in mobius is one stage of a three-stage pipeline:

1. **mobius** (this repo) — Convert the source model (PyTorch/ONNX) to CoreML, validate numerical parity, smoke-test inference
2. **[HuggingFace](https://huggingface.co/FluidInference)** — Upload and host the converted model artifacts
3. **[FluidAudio](https://github.com/FluidInference/FluidAudio)** — Register the model, write Swift inference code, add CLI command, write tests, run benchmarks

Every new model should reference all three:

| Item | Example |
|------|---------|
| mobius PR | [`FluidInference/mobius#25`](https://github.com/FluidInference/mobius/pull/25) (conversion scripts, validation, trial notes) |
| HuggingFace repo | [`FluidInference/parakeet-tdt-0.6b-v3-coreml`](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) |
| FluidAudio PR | [`FluidInference/FluidAudio#315`](https://github.com/FluidInference/FluidAudio/pull/315) |

For the FluidAudio integration steps, see [`FluidAudio/Documentation/ModelConversion.md`](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/ModelConversion.md).

---

## Before You Start

Conversions rarely work on the first attempt. Common issues include tracing failures, shape mismatches, unsupported ops in CoreML, numerical drift between PyTorch and CoreML outputs, and ANE compatibility problems. Document what you tried, what failed, and why, so the next person doesn't repeat the same dead ends.

### Key constraints

- **Trace with `.CpuOnly`** — CoreML tracing must target CPU-only compute units
- **Target iOS 17+ / macOS 14+** — most users are on these versions
- **Use `uv`** — each conversion target has its own `pyproject.toml` for reproducible environments
- **Fixed shapes** — CoreML requires static input/output shapes; no dynamic dimensions
- **No in-place ops** — PyTorch in-place operations (`x.add_()`, `x[i] = ...`) must be replaced with functional equivalents
- **Deterministic operations** — replace random/stochastic ops with deterministic alternatives or external inputs

---

## Step 1: Create the conversion directory

Each conversion target is self-contained under `models/{class}/{model-name}/{target}/`:

```
models/
  stt/
    parakeet-tdt-v3-0.6b/
      coreml/
        convert-coreml.py        # Conversion script
        individual_components.py # Torch modules for tracing (optional)
        compare-components.py    # Parity validation (optional)
        quantize_coreml.py       # Quantization sweep (optional)
        pyproject.toml           # Python deps (uv-managed)
        uv.lock                  # Pinned dependencies
        README.md                # Conversion notes, source links, known issues
        TRIALS.md                # What was tried, what failed, what worked (optional)
        audio/                   # Sample audio for tracing/testing (optional)
        context/                 # Architecture docs, conversion plans (optional)
        doc/                     # Deep dives, problems encountered (optional)
```

**Classes:** `stt`, `vad`, `speaker-diarization`, `tts`, `emb`, `segment-text`

Use lowercase-kebab-case for directories and filenames. Mirror upstream model names where possible.

---

## Step 2: Set up the environment

Create a `pyproject.toml` with pinned dependencies. The recommended Python version is 3.10.12, but other versions may work.

```toml
[project]
name = "my-model-coreml"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "coremltools>=7.2",
    "torch>=2.4",
    "numpy",
    # Model-specific deps (NeMo, transformers, etc.)
]
```

Then:

```bash
uv sync
```

This creates a local `.venv` pinned by `pyproject.toml` and `uv.lock`. All commands should be run through `uv run` to keep resolutions reproducible.

---

## Step 3: Write the conversion script

The conversion script should:

1. **Load the source model** — from PyTorch checkpoint, NeMo, HuggingFace, ONNX, etc.
2. **Wrap into traceable modules** — extract stateful components (LSTM states, caches) into explicit inputs/outputs. Create dedicated `nn.Module` wrappers if the original model isn't directly traceable.
3. **Replace incompatible ops** — see [Common CoreML incompatibilities](#common-coreml-incompatibilities) below.
4. **Trace with `torch.jit.trace`** — using representative inputs at the target shape.
5. **Convert with `coremltools`**:

```python
import coremltools as ct

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="audio", shape=(1, 80, 4096))],
    outputs=[ct.TensorType(name="logits")],
    minimum_deployment_target=ct.target.iOS17,
    convert_to="mlprogram",
)
```

6. **Set metadata** — author, version, description.
7. **Save as `.mlpackage`**:

```python
coreml_model.save("Model.mlpackage")
```

8. **Compile to `.mlmodelc`** (optional, can also be done via `xcrun`):

```bash
xcrun coremlcompiler compile Model.mlpackage output_dir/
```

### Multi-component models

Most speech models have multiple components (preprocessor, encoder, decoder, joint network). Export each as a separate `.mlpackage`. See `models/stt/parakeet-tdt-v3-0.6b/coreml/` for an example with 6 components.

If components can be fused for better performance (e.g., mel + encoder), export both fused and separate variants and benchmark them.

### Common CoreML incompatibilities

| PyTorch Pattern | CoreML Fix |
|-----------------|------------|
| `pack_padded_sequence` | Explicit LSTM states + masking |
| `torch.rand()` / `torch.randn()` | Pass random values as model inputs |
| Dynamic shapes / loops | Fixed shapes with padding |
| In-place operations (`x.add_()`) | Functional equivalents (`x = x + y`) |
| `scaled_dot_product_attention` | Manual attention implementation (for iOS 17 compat) |
| Complex indexing (`x[mask]`) | Gather/scatter or broadcasting |

---

## Step 4: Validate numerical parity

Compare PyTorch and CoreML outputs on the same input to verify the conversion preserved accuracy. This is critical — small numerical differences can compound through multi-component pipelines.

```bash
uv run python compare-components.py compare \
  --output-dir model_coreml \
  --runs 10 --warmup 3
```

**What to check:**

- **Max absolute error** — how far off is the worst prediction?
- **Max relative error** — are near-zero values inflating relative error?
- **Distribution alignment** — do top-k predictions match?
- **End-to-end output** — does the final transcription/diarization/audio match?

Report results in the README. Include plots if the conversion has a comparison script.

See `models/stt/parakeet-tdt-v3-0.6b/coreml/` for a thorough example with per-component parity checks, L2 error plots, and latency measurements.

---

## Step 5: Smoke-test inference

Run a basic inference pass to confirm the converted model loads and produces reasonable output. This is a sanity check, not a formal benchmark — full benchmarks (WER, DER, RTFx on standard datasets, etc.) happen in [FluidAudio](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/ModelConversion.md#36-run-benchmarks).

```bash
uv run python test.py audio/sample.wav
```

**What to verify:**

- Model loads without errors on CPU and ANE
- Output is non-garbage (transcription makes sense, audio sounds right, diarization labels are plausible)
- Latency is in the right ballpark (not 10x slower than expected)

If the conversion has a comparison script, the PyTorch-vs-CoreML latency numbers from Step 4 already cover this.

### TTS models: ASR verification required

**For TTS (Text-to-Speech) conversions only**: You must verify the converted model using automatic speech recognition (ASR) to ensure semantic accuracy. Traditional verification methods (spectral similarity, mel-spectrogram comparison) can pass even when generated audio contains incorrect words, missing phonemes, or garbled speech that "looks" spectrally similar but is unintelligible.

**Requirements:**
1. Generate audio samples using both PyTorch (reference) and CoreML (converted) models
2. Transcribe all generated audio using an ASR model (Whisper, FluidAudio Parakeet, or equivalent)
3. Calculate Word Error Rate (WER) for both outputs compared to input text
4. Verify WER < 10% and PyTorch vs CoreML transcription difference < 2%
5. Document results with sample transcriptions in README

**Example verification:**
```python
import whisper
from jiwer import wer

test_texts = ["Your test sentences here...", "At least 20-30 diverse samples..."]
asr_model = whisper.load_model("base")

for text in test_texts:
    # Generate audio with PyTorch and CoreML
    pt_audio = generate_pytorch(text)
    cm_audio = generate_coreml(text)

    # Transcribe and compare
    pt_transcription = asr_model.transcribe(pt_audio)["text"]
    cm_transcription = asr_model.transcribe(cm_audio)["text"]

    print(f"PyTorch WER: {wer(text, pt_transcription):.2%}")
    print(f"CoreML WER:  {wer(text, cm_transcription):.2%}")
```

Use diverse test samples covering different lengths, phonetic variety, and edge cases (numbers, punctuation, proper nouns). Report aggregate WER in your README.

---

## Step 6: Explore quantization (optional)

Quantization reduces model size and can improve latency on some compute units. Common approaches:

- **INT8 linear** (per-channel) — ~2x smaller, minimal quality loss
- **INT8 linear** (per-tensor symmetric) — can cause large quality drops, test carefully
- **Palettization** (4-bit, 6-bit) — aggressive compression, quality varies

```bash
uv run python quantize_coreml.py \
  --input-dir model_coreml \
  --output-root model_coreml_quantized \
  --compute-units ALL --runs 10
```

Check that quantized output quality is acceptable relative to the FP32 baseline (1 - normalized L2 error). Note size reductions.

---

## Step 7: Document trials, errors, and architecture

This is one of the most valuable parts of a conversion. The next person converting a similar model — or debugging a regression — will rely on what you wrote down. Document as you go, not after the fact.

### What to document

**Trials and errors** — what you tried, what failed, and why. This prevents others from repeating dead ends.

- Failed tracing attempts (monolithic vs split components)
- Ops that didn't convert and what replaced them
- Bugs found during validation (wrong output shapes, silent numerical drift, garbage output)
- Workarounds for CoreML/ANE limitations
- Quantization experiments that degraded quality

**Architecture context** — how the source model works, links to papers and upstream repos.

- Link to the original model (HuggingFace, GitHub, paper)
- Model topology: what each component does, how they connect
- Input/output shapes and data flow
- Key design decisions in the source model that affect conversion
- Relevant research papers (architecture, training, evaluation)

**Platform-specific issues** — things that behave differently across devices or OS versions.

- ANE vs GPU vs CPU behavior differences
- iPhone vs Mac ANE dimension limits
- iOS Simulator limitations
- Float16 precision issues on ANE
- Model compilation time differences across devices

**What worked** — so future conversions can reuse successful patterns.

- Which tracing strategy worked (and why others didn't)
- Compute unit recommendations per component
- Successful fusing strategies
- Quantization variants that maintained quality

### Where to put it

Use whatever format fits the complexity. Existing conversions use several patterns:

| Format | When to use | Example |
|--------|-------------|---------|
| **`TRIALS.md`** | Chronological log of attempts | `models/tts/pocket_tts/coreml/TRIALS.md` |
| **`doc/problems_encountered.md`** | Categorized issue tracker | `models/tts/kokoro/coreml/doc/problems_encountered.md` |
| **`context/` directory** | Architecture docs, conversion plans | `models/stt/parakeet-tdt-v3-0.6b/coreml/context/` |
| **`IOS_COREML_ISSUES.md`** | Platform-specific bugs | `models/tts/pocket_tts/coreml/IOS_COREML_ISSUES.md` |
| **Integration report** | Debugging methodology write-up | `models/stt/canary-1b-v2/coreml/coreml_integration_report.md` |
| **Model analysis** | Deep dive on why something does/doesn't work | `models/vad/silero-vad/coreml/silero_vad_model_analysis.md` |
| **README section** | Brief notes (for simple conversions) | `models/stt/qwen3-asr-0.6b/coreml/README.md` |

For simple conversions, a "Known Issues" section in the README is enough. For complex ones, use dedicated files — Kokoro has 3 docs totaling 1,200+ lines, and that context has been referenced many times since.

---

## Step 8: Profile with coreml-cli

After validating numerical parity, profile the converted model to understand compute device assignment and latency:

```bash
cd tools/coreml-cli && uv sync   # one-time setup
uv run coreml-cli path/to/build/model.mlmodelc
```

This shows, for each compute-unit configuration (`all`, `cpu_only`, `cpu_and_gpu`, `cpu_and_neural_engine`):
- **Device assignment** — % of operations on CPU, GPU, and ANE
- **Cold compile time** — measured once per model by bypassing the compilation cache. Reflects what users experience the first time the model runs on their device — if this is too high (e.g., several seconds), the model may not be usable in practice.
- **Compile time** — cached load time per compute unit config. This is the cost paid on every app launch.
- **Predict latency** — median inference time

Use `--ops` to see which specific operations fall back to CPU (common with unsupported ANE ops). Use `--detailed` for private API data including backend support per op and why certain backends were rejected.

Include the profiling results in your README and PR description — reviewers want to know ANE utilization, compile times, and latency.

---

## Step 9: Write the README

Every conversion directory needs a README documenting:

1. **What the model does** — one-line description
2. **Source model** — link to the original (HuggingFace, NeMo, GitHub, paper)
3. **Layout** — directory structure and what each file does
4. **Environment setup** — `uv sync` instructions
5. **Conversion commands** — how to run the export
6. **Validation results** — parity checks, inference smoke test
7. **Known issues** — what didn't work, ANE quirks, accuracy caveats (or link to dedicated docs)
8. **Usage with FluidAudio** — how the exported models integrate (if applicable)
9. **Acknowledgements** — credit upstream authors, contributors, and relevant papers

See existing READMEs for reference:
- Detailed: `models/stt/parakeet-tdt-v3-0.6b/coreml/README.md`
- Multi-variant: `models/speaker-diarization/sortformer-streaming/README.md`
- TTS: `models/tts/kokoro/coreml/README.md`
- Minimal: `models/stt/qwen3-asr-0.6b/coreml/README.md`

---

## Step 10: Upload to HuggingFace

Upload the converted models to the [`FluidInference`](https://huggingface.co/FluidInference) organization.

### Get access

You need write access to the [`FluidInference`](https://huggingface.co/FluidInference) organization. Sign up at [huggingface.co](https://huggingface.co) and request access through the org page or contact a maintainer.

### Create the repository

Naming convention: `{model-name}-coreml` (e.g., `FluidInference/parakeet-tdt-0.6b-v3-coreml`)

### Upload artifacts

- `.mlmodelc` bundles (compiled CoreML models)
- `.mlpackage` files if applicable
- Supporting files: vocab JSON, embeddings, constants, metadata
- If the repo has variants (frame sizes, precisions), use subdirectories: `160ms/`, `320ms/`, `f32/`, `int8/`

### Update the model card

- Source attribution (link to original model)
- License
- Input/output shapes and compute unit recommendations

---

## Step 11: Open the mobius PR

The PR should include:

- Conversion script(s) and `pyproject.toml`
- README with source model link, conversion notes, validation results, known issues
- Link to the HuggingFace repo (once uploaded)
- Any helper modules or test scripts

**PR description format** (follow existing PRs):

- Model name and source
- What the conversion does (components exported, target shapes)
- Validation evidence (parity results, inference smoke test)
- Known limitations or follow-up work
- Link to the FluidAudio PR (if integration work has started)

---

## Checklist

### Conversion

- [ ] Directory created at `models/{class}/{name}/coreml/`
- [ ] `pyproject.toml` with pinned dependencies
- [ ] `uv.lock` generated via `uv sync`
- [ ] Conversion script exports `.mlpackage` files
- [ ] Models traced with `.CpuOnly`
- [ ] Minimum deployment target: iOS 17 / macOS 14
- [ ] Fixed input/output shapes (no dynamic dimensions)

### Validation

- [ ] Numerical parity checked against PyTorch baseline
- [ ] End-to-end output verified (transcription, diarization, audio, etc.)
- [ ] Inference smoke test passes on Apple Silicon
- [ ] Profiled with `coreml-cli` — ANE utilization and latency documented

### Documentation

- [ ] README with source model link, conversion steps, validation results, known issues
- [ ] Directory layout documented
- [ ] Link to source model paper / architecture docs
- [ ] Trials and errors documented (what failed, why, and what worked)
- [ ] Platform-specific issues noted (ANE quirks, device differences, iOS vs macOS)

### HuggingFace

- [ ] Repository created at `FluidInference/{model-name}-coreml`
- [ ] `.mlmodelc` bundles uploaded
- [ ] Supporting files included (vocab, embeddings, constants)
- [ ] Model card updated with source attribution and license

### PR

- [ ] PR opened with model description, parity validation, and HuggingFace link
- [ ] Link to FluidAudio PR (if integration started)

---

## Reference: Existing Conversions

| Class | Model | Directory | HuggingFace | Status |
|-------|-------|-----------|-------------|--------|
| stt | Parakeet TDT v3 (0.6B) | `models/stt/parakeet-tdt-v3-0.6b/coreml/` | [parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml) | Shipped |
| stt | Parakeet TDT v2 (0.6B) | `models/stt/parakeet-tdt-v2-0.6b/coreml/` | [parakeet-tdt-0.6b-v2-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml) | Shipped |
| stt | Parakeet TDT-CTC (110M) | `models/stt/parakeet-tdt-ctc-110m/coreml/` | [parakeet-ctc-110m-coreml](https://huggingface.co/FluidInference/parakeet-ctc-110m-coreml) | Shipped |
| stt | Parakeet EOU (120M) | `models/stt/parakeet-realtime-eou-120m/coreml/` | [parakeet-realtime-eou-120m-coreml](https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml) | Shipped |
| stt | Qwen3-ASR (0.6B) | `models/stt/qwen3-asr-0.6b/coreml/` | — | Shipped |
| stt | Nemotron Streaming (0.6B) | `models/stt/nemotron-speech-streaming-0.6b/coreml/` | — | Evaluated |
| stt | Canary-1B v2 | `models/stt/canary-1b-v2/coreml/` | — | Research |
| vad | Silero VAD | `models/vad/silero-vad/coreml/` | [silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml) | Shipped |
| speaker-diarization | Pyannote Community | `models/speaker-diarization/pyannote-community-1/coreml/` | [speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml) | Shipped |
| speaker-diarization | Sortformer Streaming | `models/speaker-diarization/sortformer-streaming/` | [diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml) | Shipped |
| tts | Kokoro (82M) | `models/tts/kokoro/coreml/` | [kokoro-82m-coreml](https://huggingface.co/FluidInference/kokoro-82m-coreml) | Shipped |
| tts | PocketTTS | `models/tts/pocket_tts/coreml/` | [pocket-tts-coreml](https://huggingface.co/FluidInference/pocket-tts-coreml) | Shipped |
| emb | CAM++ | `models/emb/cam++/coreml/` | — | Shipped |
| segment-text | SaT | `models/segment-text/coreml/` | — | Shipped |

## Reference: Documentation Examples

The best way to understand what good conversion documentation looks like is to read the existing ones. Here are the standouts, organized by what they do well:

### Trials and error tracking

| File | What it covers |
|------|---------------|
| `models/tts/pocket_tts/coreml/TRIALS.md` | Chronological log: failed monolithic tracing → split architecture → 5 bugs found with symptoms and fixes |
| `models/tts/kokoro/coreml/doc/problems_encountered.md` | 15 categorized problem areas with root causes and solutions (audio artifacts, text processing, quantization regressions, compute unit tradeoffs) |
| `models/stt/canary-1b-v2/coreml/coreml_integration_report.md` | Debugging methodology: monkeypatch ground-truth capture to find missing projection layer, wrong EOS token, prompt format bugs |

### Architecture and model context

| File | What it covers |
|------|---------------|
| `models/tts/kokoro/coreml/doc/v21_conversion_script_outline.md` | 900+ line deep dive: complete class hierarchy, forward pass walkthrough, data flow, CoreML compatibility patterns |
| `models/stt/parakeet-tdt-v3-0.6b/coreml/context/parakeet_tdt_v3_architecture.md` | Model topology, encoder/decoder/joint specs, fixed 15s window contract |
| `models/stt/parakeet-tdt-v3-0.6b/coreml/context/coreml_conversion_plan.md` | Export strategy, validation methodology, known caveats |
| `models/tts/pocket_tts/coreml/CONVERSION.md` | 4-model pipeline architecture, constants export, generation pipeline, Swift porting guidance |

### Platform-specific issues

| File | What it covers |
|------|---------------|
| `models/tts/pocket_tts/coreml/IOS_COREML_ISSUES.md` | 7 iOS-specific bugs: zero-length tensor crash, ANE float16 beeping, simulator silent audio, compilation time |
| `models/stt/parakeet-tdt-v3-0.6b/coreml/context/mel_encoder_ane_behavior.md` | iPhone 13 ANE compilation failure due to W=240000 exceeding ANE's 16384 dimension cap |
| `models/vad/silero-vad/coreml/silero_vad_model_analysis.md` | Why ANE doesn't speed up lightweight models, layer-by-layer operation breakdown, CPU vs ANE vs GPU comparison |

### Conversion step-by-step

| File | What it covers |
|------|---------------|
| `models/tts/pocket_tts/coreml/CONVERSION.md` | Complete guide: architecture overview, per-component specs, constants export, zero-PyTorch generation pipeline |
| `models/speaker-diarization/pyannote-community-1/coreml/README.md` | 3-script pipeline (convert → compare → quantize), agent-oriented and manual workflows, Python wrapper usage |

## Reference: Knowledge Base

The `knowledge/` directory contains curated research papers and platform documentation useful during conversions. Consult these when working on a new model in the same family or hitting ANE/CoreML issues.

**`knowledge/audio/`** — Speech model architecture papers:
- Fast Conformer (v1 & v6) — encoder architecture used by Parakeet, Canary, Nemotron
- Token-Duration Transducer (TDT) — decoding strategy used by Parakeet TDT models
- CoVoST 2 — multilingual speech translation benchmark
- Canary/Parakeet model cards — production ASR architecture details

**`knowledge/coreml/`** — Apple platform optimization:
- Core ML on-device Llama — optimization patterns for transformer models on ANE (~33 tok/s on M1 Max)
- coremltools 9.0 docs — conversion workflows, quantization, pruning, palettization APIs
- Neural Engine architecture — Apple Silicon accelerator capabilities, dimension limits, supported ops
