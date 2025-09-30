# pyannote-coreml

This Core ML port of the Hugging Face `pyannote/speaker-diarization-community-1` pipeline was produced primarily by the Mobius coding agent. The directory is laid out so another agent can pick it up and run end-to-end, while still giving power users a clear manual path through the convert → compare → quantize toolchain.

## What Lives Here

- `convert-coreml.py`, `compare-models.py`, `quantize-models.py` — scripted pipeline for export, parity checks, and post-export optimizations.
- `coreml_models/` — default output folder for `.mlpackage` bundles plus resource JSON.
- `docs/` — background notes (`docs/plda-coreml.md`, conversion guides, optimization results).
- `coreml_wrappers.py`, `embedding_io.py`, `plda_module.py` — importable helpers for wrapping Core ML bundles inside PyTorch pipelines.
- `pyproject.toml`, `uv.lock` — reproducible Python 3.10.12 environment pinned to Torch 2.4, coremltools 7.2, pyannote-audio 4.0.0.
- Sample clips (`yc_first_10s.wav`, `yc_first_minute.wav`, `../../../../longconvo-30m*.wav`) for smoke tests and benchmarking.

## Agent-Oriented Workflow

Mobius (or any compatible coding agent) can operate this toolkit by chaining three scripts:

1. `convert-coreml.py` exports FBANK, segmentation, embedding, and PLDA components to Core ML (with optional selective FP16).
2. `compare-models.py` runs PyTorch vs Core ML parity tests, reports timing, DER/JER metrics, and refreshes plots under `plots/`.
3. `quantize-models.py` generates INT8/INT4/palettized variants, benchmarks latency and memory, and emits comparison charts.

All scripts write machine-readable summaries to disk so an agent can decide what to ship or flag regressions. Automation typically runs them in that order inside this directory with `uv run`.

## Manual Pipeline

Prerequisites: macOS 14+, Xcode 15+, [uv](https://github.com/astral-sh/uv), access to the gated Hugging Face repo. Accept the user agreement on [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) before attempting to download the checkpoints, then fetch the assets into `pyannote-speaker-diarization-community-1/` (run `git lfs pull` if necessary).

```bash
# 1. Create or refresh the local environment
uv sync

# 2. Convert PyTorch checkpoints to Core ML
uv run python convert-coreml.py --model-root ./pyannote-speaker-diarization-community-1 \
    --output-dir ./coreml_models
# Optional: add --selective-fp16 for mixed precision exports

# 3. Compare PyTorch vs Core ML outputs, generate plots/metrics
uv run python compare-models.py --audio-path ../../../../longconvo-30m-last5m.wav \
    --model-root ./pyannote-speaker-diarization-community-1 \
    --coreml-dir ./coreml_models

# 4. Produce quantized variants and benchmark them (uses convert+compare outputs)
uv run python quantize-models.py --audio-path ../../../../longconvo-30m.wav \
    --coreml-dir ./coreml_models
# Add --skip-generation to benchmark existing variants only
```

Key artifacts land under `coreml_models/` (FP32/FP16 exports, PLDA Core ML bundle, resource JSON files) and `plots/` (latency and accuracy reports). The scripts emit timing summaries and DER/JER results directly to stdout for quick inspection.

## Using the Wrappers from Python

`coreml_wrappers.py` exposes helpers to drop the converted models into an existing pyannote pipeline. The snippet below loads the FBANK and embedding bundles, mirrors the PyTorch interface, and emits embeddings for a local clip.

```python
from pathlib import Path

import coremltools as ct
import torch
import torchaudio
from pyannote.audio import Model

from coreml_wrappers import CoreMLEmbeddingModule
from embedding_io import SEGMENTATION_FRAMES

root = Path(__file__).resolve().parent
embedding_ml = ct.models.MLModel(root / "coreml_models" / "embedding-community-1.mlpackage")
fbank_ml = ct.models.MLModel(root / "coreml_models" / "fbank-community-1.mlpackage")
prototype = Model.from_pretrained(str(root / "pyannote-speaker-diarization-community-1" / "embedding"))

wrapper = CoreMLEmbeddingModule(embedding_ml, fbank_ml, prototype, output_key="embedding")

waveform, _ = torchaudio.load(root / "yc_first_10s.wav")
waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform
weights = torch.ones(1, SEGMENTATION_FRAMES)
embedding = wrapper(waveform.unsqueeze(0), weights)
print(embedding.shape)
```

Call `wrap_pipeline_with_coreml` to swap the segmentation and embedding stages inside a full PyTorch diarization pipeline while keeping the VBx/PLDA logic on-device.

## Status & Known Limitations

- ✅ Conversion, comparison, and quantization scripts are in place and agent friendly.
- ✅ PLDA parameters now ship as a Core ML model (`plda-community-1.mlpackage`) with precise dtype handling (see `docs/plda-coreml.md`).
- ⚠️ Fixed 5 s embedding windows introduce mild oscillations around speaker transitions versus the variable-length PyTorch baseline (DER ~0.017–0.018). Plots under `plots/` illustrate the difference.
- 🔍 Further tuning ideas: adjust VBx thresholds, add post-processing to merge short segments, investigate weighted pooling exports once coremltools supports variable-length inputs.

## References

- Hugging Face pipeline: `pyannote/speaker-diarization-community-1`
- VBx clustering background: [VBx: Variational Bayes HMM Clustering](https://arxiv.org/abs/2012.14952)
- Additional notes and deep dives live in `docs/` (start with `docs/plda-coreml.md` and `ANE_OPTIMIZATION_RESULTS.md`).
