# Architecture: Weight Interpolation in Speaker Diarization Pipeline

## Overview

The pyannote community-1 speaker diarization pipeline uses two pretrained models (segmentation and embedding) that operate at different temporal resolutions. This document explains why weight interpolation is necessary and how it's implemented.

## The Frame Count Mismatch

### Segmentation Model
- **Input**: 160,000 samples (10 seconds @ 16 kHz)
- **Output**: 589 frames of speaker activity probabilities
- **Frame rate**: ~58.9 frames/second
- **Architecture**: PyanNet with specific convolution strides and padding
- **Purpose**: Predict which speakers are active in each time frame

### Embedding Model
- **Input**: 160,000 samples (10 seconds @ 16 kHz)
- **ResNet processing**: Features are downsampled through stride-2 convolutions
- **Pooling layer input**: 125 frames (after 8x downsampling)
- **Frame rate**: ~12.5 frames/second
- **Architecture**: WeSpeaker ResNet34 with statistics pooling
- **Purpose**: Extract speaker embeddings weighted by activity

### Why They Differ

Both models are **pretrained** with fixed architectures:

1. **Segmentation (589 frames)**: Uses specific convolution parameters (kernel size, stride, padding) that produce 589 output frames for 160k samples. This cannot be changed without retraining.

2. **Embedding (125 frames)**: The ResNet has stride-2 convolutions that progressively downsample features by 8x total. For 160k samples, the features at the pooling layer are 125 frames. This is baked into the trained weights.

These frame rates are **architectural constraints** - changing them would require retraining the models from scratch with different convolution parameters.

## The Interpolation Solution

Since the models have fixed frame rates, we must **resample** the segmentation weights to match the embedding's pooling layer:

```
Segmentation output (589 frames)
        ↓
   Interpolation (linear resampling)
        ↓
Embedding weights (125 frames)
```

### Why 10-Second Fixed Windows

The pipeline processes audio in fixed 10-second chunks:
- **Consistent frame counts**: 160k samples always → 589 segmentation frames, pooling layer frame count detected at conversion
- **No dynamic checks needed**: The frame counts are constant for all inputs of the same duration
- **Simpler model tracing**: No conditional logic required for CoreML conversion

### Implementation Approach

**Inside the CoreML Model:**
The interpolation happens inside the traced embedding model using PyTorch's `F.interpolate`:

```python
def forward(self, waveforms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    fbank = self.compute_fbank(waveforms)

    # Interpolate 589 → target_frames using precomputed gather indices/weights
    left = torch.gather(weights, 1, self._interp_left_idx.expand(weights.size(0), -1))
    right = torch.gather(weights, 1, self._interp_right_idx.expand(weights.size(0), -1))
    weights = (left * self._interp_left_weight) + (right * self._interp_right_weight)

    embeddings = self.resnet(fbank, weights=weights)[1]
    return embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
```

**Key design decisions:**
- **No conditionals**: Since inputs are fixed size, we always interpolate unconditionally
- **Inside the model**: The interpolation is part of the traced computation graph
- **Deterministic mapping**: Precomputed gather indices/weights reproduce SciPy’s `align_corners=True` behavior within float32 precision
- **Simple path**: One deterministic execution path for reliable CoreML conversion

### Why Not in the Runtime Wrapper?

While interpolation could be done in the Python runtime wrapper using `scipy.ndimage.zoom`, keeping it inside the model has benefits:

1. **Single computation graph**: All preprocessing is in CoreML, optimized together
2. **Fewer Python↔CoreML transitions**: Less overhead from data marshaling
3. **Cleaner API**: Runtime just passes segmentation output directly to embedding
4. **Platform optimization**: CoreML can optimize the entire graph for the target hardware

## Technical Details

### Frame Count Calculation

**Segmentation (589 frames):**
```
Frames = conv1d_num_frames(160000, kernel_size, stride, padding)
       ≈ 589 frames
```

**Embedding (pooling layer frames):**
```
Initial fbank frames ≈ 1000 frames
After ResNet layers (stride-2 convolutions) → frame count detected at conversion
Typical: ~125 frames (8× downsampling for this architecture)
```

### Interpolation Quality

Linear interpolation with `align_corners=True`:
- **Method**: Linear interpolation between adjacent frame weights
- **Preserve endpoints**: First and last frames are preserved exactly
- **Smooth transition**: Gradual weight changes between frames
- **Proven accuracy**: Achieved 0% DER on test benchmarks

### Why Interpolation Works

Speaker activity changes gradually over time, so:
- Resampling from 589→125 frames preserves the overall activity pattern
- The weighted pooling uses these weights to emphasize speaker-active regions
- The 4.7× downsampling doesn't lose critical information for this task

## Alternative Approaches (Not Used)

### 1. Retrain Models to Match Frame Rates
- **Pros**: No interpolation needed
- **Cons**: Requires full retraining, loses pretrained performance

### 2. Separate Interpolation Model
- **Pros**: Modularity
- **Cons**: Extra model overhead, unnecessary for such a simple operation

### 3. Runtime Interpolation (scipy.ndimage.zoom)
- **Pros**: Flexible, proven to work
- **Cons**: Python overhead, missed CoreML optimization opportunities

## Conclusion

The 589→125 frame interpolation is a necessary bridge between two pretrained models with different temporal resolutions. By implementing it inside the embedding model with unconditional logic, we achieve a clean, efficient, and CoreML-optimized solution.
