# Embedding Normalisation Notes

The Community-1 diarization pipeline expects L2-normalised speaker embeddings before applying the PLDA transform used by VBx. This document explains where that normalisation happens in the current code base and how to verify it.

## What The Base Model Emits

`pyannote.audio.models.embedding.wespeaker.WeSpeakerResNet34` returns 256-dim vectors whose norms hover around 0.7–0.8 for typical speech clips. You can check this quickly:

```bash
UV_CACHE_DIR=$(pwd)/.uv-cache MPLCONFIGDIR=$(pwd)/.matplotlib \
uv run python - <<'PY'
from pathlib import Path
import torch
from pyannote.audio import Model

model_root = Path('pyannote-speaker-diarization-community-1')
model = Model.from_pretrained(str(model_root / 'embedding')).eval()
with torch.inference_mode():
    emb = model(torch.randn(1, 1, 160000))
print('L2 norm:', emb.norm(p=2, dim=-1))
PY
```

The raw vectors are *not* unit length, so any downstream consumer must normalise before PLDA.

## Where We Enforce Normalisation

1. **Conversion wrapper** – `WeSpeakerTraceableBackend.forward` (in `convert-coreml.py`) finishes with:

```python
norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
norms = torch.clamp(norms, min=1e-12)
return embeddings / norms
```

   The traced TorchScript module—and therefore the Core ML export—always outputs unit-norm embeddings. This keeps the Core ML bundle aligned with Pyannote’s runtime expectation.

2. **PLDA modules** – `PLDATransformModule` in `plda_module.py` performs two `_l2_normalize` calls: once after subtracting `mean1` and once after subtracting `mean2`. The NumPy reference loader in `compare-models.py` mirrors the same behaviour (`_l2_norm`). Either path guarantees that embeddings are normalised before the LDA and PLDA projections.

3. **Comparison tooling** – `compare-models.py` prints the incoming norms when running `compare_plda(...)`. If the numbers are not ≈1.0, the log explicitly calls that out so we can double-check the preprocessing chain. Normalisation is not duplicated there because the PLDA transform already handles it.

## Action Items For Callers

- **Torch pipeline** – continue to rely on the Pyannote `Pipeline` class; it normalises internally via PLDA.
- **Core ML runtime** – use `CoreMLEmbeddingModule` from `coreml_wrappers.py`. It runs the FBANK frontend on CPU, feeds the backend Core ML model (already normalised), and returns `torch.Tensor` objects to upstream Pyannote code.
- **External consumers** – if you bypass both wrappers and call the exported PLDA bundles directly (e.g., from Swift), normalise any custom embeddings first to match the on-device expectation.

Keep this note handy when updating documentation. Any future change that removes the normalisation step in the wrapper or PLDA transform must be reflected here so validation scripts remain trustworthy.
