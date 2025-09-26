# NeMo Joint Network Fuse Mode

## Overview

The NeMo RNNT Joint network has two operational modes controlled by the `fuse_loss_wer` flag. Understanding this is crucial for proper validation and inference.

## Fuse Mode Details

### Normal Mode (`fuse_loss_wer = False`)

**Purpose**: Standard inference and model export
**Interface**: `joint(encoder_outputs, decoder_outputs)`
**Returns**: Raw logits `[B, T, U, V + 1]`
**Use case**:
- Model inference
- Model export (CoreML, ONNX, etc.)
- Validation and comparison

```python
# Simple joint computation
logits = model.joint(encoder_outputs=encoder_out, decoder_outputs=decoder_out)
```

### Fused Mode (`fuse_loss_wer = True`)

**Purpose**: Memory-optimized training with integrated loss/WER computation
**Interface**: Requires additional parameters
**Returns**: Loss and WER metrics
**Use case**: Training with memory constraints

**Required parameters when fused**:
- `encoder_outputs` (mandatory)
- `decoder_outputs` (optional, required for loss)
- `encoder_lengths` (required)
- `transcripts` (optional, required for WER)
- `transcript_lengths` (required)

**Memory optimization**: Processes batch in sub-batches to reduce memory usage

```python
# Fused mode - more complex interface
result = model.joint(
    encoder_outputs=encoder_out,
    decoder_outputs=decoder_out,
    encoder_lengths=enc_lengths,
    transcripts=transcripts,        # Ground truth tokens
    transcript_lengths=trans_lengths
)
```

## Why We Disable Fused Mode for Validation

### 1. Interface Simplicity
- Fused mode requires ground truth transcripts that we don't have during validation
- Normal mode only needs encoder/decoder outputs, matching CoreML model interface

### 2. Export Compatibility
```python
def _prepare_for_export(self, **kwargs):
    self._fuse_loss_wer = False  # Automatically disabled for export
    self.log_softmax = False
```
NeMo automatically disables fused mode during model export, indicating this is the correct mode for inference.

### 3. Validation Purpose
- We want raw logits for comparison, not loss/WER calculations
- Fused mode is training-oriented, not inference-oriented

### 4. CoreML Conversion
CoreML models are converted from the non-fused mode, so validation should use the same mode to ensure fair comparison.

## Implementation in compare-components.py

```python
# Disable fused mode for proper inference comparison
asr_model.joint.set_fuse_loss_wer(False)

# Now we can use the simple interface that matches CoreML
logits_ref = asr_model.joint(
    encoder_outputs=encoder_ref,
    decoder_outputs=decoder_ref
)
```

## Key Takeaway

For validation and inference, always use `fuse_loss_wer = False` to get the standard joint network behavior that matches exported models and enables fair comparison with CoreML implementations.