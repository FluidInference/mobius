# VBx PLDA Bug Fix

## Problem

The VBx clustering was exhibiting aggressive merging behavior, reducing 4+ speakers down to 1-2 speakers. This was caused by using the **wrong PLDA CoreML model**.

## Root Cause

The `PLDA.mlpackage` file was actually a copy of `plda-community-1.mlpackage`, which outputs **`plda_features`** instead of **`rho`**.

### The Difference

VBx clustering requires the **rho** values for computing similarity scores:

```python
# Reference implementation (see `PLDAPipeline.rho` in compare-models.py)
def rho(self, embeddings: np.ndarray) -> np.ndarray:
    fea = self.transform(embeddings)
    return fea * np.sqrt(self.phi[: self.lda_dim])  # ← Critical sqrt(phi) scaling!
```

**Without sqrt(phi) scaling**:
- Similarity scores are incorrect
- All embeddings appear more similar than they should
- VBx aggressively merges distinct speakers

**With sqrt(phi) scaling**:
- Similarity scores match the PLDA model expectations
- VBx correctly distinguishes between speakers
- Clustering behavior matches PyTorch reference

### Model Comparison

| Model | Output Name | Contains sqrt(phi) scaling | Use for VBx |
|-------|-------------|---------------------------|-------------|
| `plda-community-1.mlpackage` | `plda_features` | ❌ No | ❌ No |
| `plda_rho-community-1.mlpackage` | `rho` | ✅ Yes | ✅ Yes |

## The Fix

```bash
cd coreml_models
rm -rf PLDA.mlpackage
cp -R plda_rho-community-1.mlpackage PLDA.mlpackage
```

If your deployment can consume `plda_rho-community-1.mlpackage` directly, prefer pointing to that bundle instead of keeping the legacy `PLDA.mlpackage` alias.

## Verification

```python
import coremltools as ct

model = ct.models.MLModel('coreml_models/PLDA.mlpackage')
spec = model.get_spec()

# Should output: rho (not plda_features)
for outp in spec.description.output:
    print(f'Output: {outp.name}')
```

## Swift Code Expectations

The Swift code already expects the correct output name:

```swift
// PLDARhoModel.swift line 22
guard let rho = output.featureValue(for: "rho")?.multiArrayValue else {
    throw DiarizerError.invalidVBxResource("PldaRho model output missing 'rho'")
}
```

With the wrong model, this code would fail at runtime with the error message. However, if an older version was silently outputting `plda_features` and the code was accessing it differently, the sqrt(phi) scaling would be missing.

## Impact

- **Before**: 4 speakers merged into 1-2
- **After**: Correct speaker count maintained
- **Clustering quality**: Matches PyTorch reference implementation

## Related Files

- `plda_module.py` - Defines PLDARhoModule with sqrt(phi) scaling
- `convert-coreml.py` - Exports both plda and plda_rho models
- `compare-models.py` - Reference implementation showing correct rho computation
- `VBxResources.swift` - Loads the PLDA model in Swift

## Testing

After applying the fix, test with:

```bash
cd /Users/brandonweng/code/FluidAudio
swift test_vbx_full.swift
```

Expected behavior: Should detect 2-4 speakers (depending on audio) instead of merging to 1-2.
