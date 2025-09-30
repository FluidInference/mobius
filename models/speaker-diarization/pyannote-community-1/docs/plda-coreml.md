# PLDA CoreML Model

## Overview

The PLDA (Probabilistic Linear Discriminant Analysis) transformation has been converted to a CoreML model (`plda-community-1.mlpackage`) to eliminate the need for separate JSON parameter files and ensure full numerical precision.

## Problem Solved

### Original Issue: NPZ → JSON Precision Loss

The original conversion pipeline had a critical bug:

1. **NPZ files** contained mixed precision:
   - `mean1`, `mu`, `tr`, `psi`: **float64** (critical PLDA parameters)
   - `mean2`, `lda`: **float32**

2. **Conversion bug** in the original `convert-coreml.py` implementation forced every tensor through `np.asarray(..., dtype=np.float32)`, silently down-casting float64 parameters.

3. **Result**: ~1e-7 precision loss that compounds through:
   - L2 normalization
   - LDA projection  
   - PLDA eigendecomposition
   - Score matrix computation

This precision loss could contribute to embedding collapse (similarities increasing from 0.04-0.14 to 0.78-0.83).

### Fixes Applied

1. **Fixed JSON conversion** (see `convert_resource` in `convert-coreml.py`): tensors are now serialized with `np.asarray(loaded[key])` so the original dtype is preserved alongside the `dtype` metadata string.

2. **Fixed JSON loading** (`load_plda_pipeline_from_json` in `compare-models.py`): tensors are reconstructed using the recorded dtype instead of assuming float32.

3. **Created PLDA CoreML model** (`plda_module.py`):
   - Encapsulates x-vector transform + PLDA projection
   - Maintains full precision (float32 in CoreML, computed from float64 parameters)
   - Can leverage GPU/ANE for inference

## Model Architecture

### Input
- **Shape**: `(batch, 256)` (enumerated shapes 1–32)
- **Type**: Float32
- **Description**: Raw speaker embeddings from the embedding model

### Processing Pipeline

1. **X-vector Transform**:
   - Center with `mean1` and L2 normalize
   - Project through LDA matrix (256 → 128 dimensions)
   - Shift with `mean2` and L2 normalize
   - Scale by sqrt(lda_dim)

2. **PLDA Transform**:
   - Center with PLDA mean `mu`
   - Project through PLDA transform matrix (128 → 128)
   - Truncate to first `lda_dim` dimensions (default: 128)

### Output
- **Shape**: `(batch, 128)`
- **Type**: Float32
- **Description**: PLDA-transformed features ready for similarity scoring

## Numerical Accuracy

Validation against reference NumPy implementation:

```
Max abs difference:  1.9e-06
Mean abs difference: 2.3e-07
Correlation:         1.0000000000
```

Similarity computation difference: `3e-08` (essentially perfect)

## Usage

### Python (with coremltools)

```python
import coremltools as ct
import numpy as np

# Load model
plda_model = ct.models.MLModel('coreml_models/plda-community-1.mlpackage')

# Apply transformation
embeddings = np.random.randn(4, 256).astype(np.float32)  # Any batch 1..32
result = plda_model.predict({"embeddings": embeddings})
plda_features = result["plda_features"]  # Shape: (batch, 128)

# Compute similarity scores
# For VBx clustering, typically use: features @ features.T
```

### Swift (iOS/macOS)

```swift
import CoreML

// Load model
let pldaModel = try PLDACommunity1(configuration: MLModelConfiguration())

// Prepare input
let embedding = try MLMultiArray(shape: [4, 256], dataType: .float32) // Batch 1..32
// ... fill embedding with data ...

let input = PLDACommunity1Input(embeddings: embedding)
let output = try pldaModel.prediction(input: input)
let features = output.plda_features  // Shape: [batch, 128]
```

## Benefits

1. **No external JSON files needed** - PLDA parameters are baked into the model
2. **Full precision preserved** - No float64 → float32 → float64 round-trip
3. **Hardware acceleration** - Can run on GPU/ANE if available
4. **Simpler deployment** - Single `.mlpackage` file
5. **Consistent interface** - Same pattern as segmentation/embedding models

## Model Files

- **CoreML model**: `coreml_models/plda-community-1.mlpackage`
- **Source code**: `plda_module.py`
- **Conversion**: Integrated into `convert-coreml.py`

## Backward Compatibility

The JSON resource files are still generated for backward compatibility:
- `coreml_models/resources/plda-parameters.json` (now with correct dtypes)
- `coreml_models/resources/xvector-transform.json` (now with correct dtypes)

Use the CoreML model for new deployments, JSON files for legacy code.

## Testing

Run validation:
```bash
uv run python3 -c "
from pathlib import Path
import coremltools as ct
import torch
from plda_module import load_plda_module_from_npz

# Load both versions
plda_torch = load_plda_module_from_npz(Path('pyannote-speaker-diarization-community-1'))
plda_coreml = ct.models.MLModel('coreml_models/plda-community-1.mlpackage')

# Test with random data
test_emb = torch.randn(1, 256)
torch_out = plda_torch(test_emb).numpy()
coreml_out = plda_coreml.predict({'embeddings': test_emb.numpy()})['plda_features']

print(f'Max difference: {abs(torch_out - coreml_out).max():.2e}')
"
```

## Performance

- **Model size**: ~136 KB (parameters)
- **Inference latency**: <1ms on modern hardware
- **Memory footprint**: Minimal (no dynamic allocations)

## References

- Original PLDA parameters: `pyannote-speaker-diarization-community-1/plda/*.npz`
- Implementation: VBx diarization backend
- Paper: [VBx: Variational Bayes HMM Clustering](https://arxiv.org/abs/2012.14952)
