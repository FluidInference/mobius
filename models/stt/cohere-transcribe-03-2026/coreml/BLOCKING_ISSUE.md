# RESOLVED: CoreML Conversion via Standard Tracing

## Status: ✅ RESOLVED

Cohere Transcribe 03-2026 **has been successfully converted to CoreML** using standard `torch.jit.trace()`.

The initial blocking issue was caused by **dependency version mismatch**, not fundamental incompatibility.

## Solution

The conversion succeeded after matching exact dependency versions from Parakeet v3:

```toml
requires-python = "==3.10.12"
dependencies = [
    "coremltools==9.0b1",      # NOT 9.0
    "torch==2.7.0",             # NOT 2.11.0
    "transformers==4.57.6",     # NOT 4.51.3
    "scikit-learn==1.5.1",
]
```

## What Worked

### Audio Encoder Conversion
```python
class AudioEncoderWrapper(nn.Module):
    def __init__(self, model, fixed_length: int):
        super().__init__()
        self.encoder = model.encoder
        self.fixed_length = fixed_length

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        batch_size = input_features.shape[0]
        length = torch.full((batch_size,), self.fixed_length, dtype=torch.int64)
        encoder_output, _ = self.encoder(
            input_features=input_features,
            length=length
        )
        return encoder_output
```

**Key fix**: Use fixed length parameter to avoid dynamic shape issues during tracing.

### Decoder Conversion
```python
class DecoderWrapper(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,  # Required parameter
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)
        decoder_output, _ = self.transf_decoder(  # Returns tuple
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
        )
        return decoder_output
```

**Key fixes**:
1. Include `positions` parameter (required by TransformerDecoderWrapper)
2. Unpack tuple return value (decoder returns `(hidden_states, None)`)

### Metadata Extraction
```python
metadata = {
    "encoder_hidden_size": config.encoder["d_model"],  # 1280
    "decoder_hidden_size": config.transf_decoder["config_dict"]["hidden_size"],  # 1024
    "lm_head_hidden_size": config.head["hidden_size"],  # 1024
    "vocab_size": config.vocab_size,  # 16384
}
```

**Key fix**: CohereAsrConfig stores hidden sizes in nested dicts, not top-level attributes.

## Conversion Results

Successfully exported all three components:

| Component | Size | Description |
|-----------|------|-------------|
| `cohere_audio_encoder.mlpackage` | 3.6 GB | 2B param Conformer encoder |
| `cohere_decoder.mlpackage` | 293 MB | Transformer decoder |
| `cohere_lm_head.mlpackage` | 32 MB | Token classifier head |

**Total size**: ~3.9 GB (FP32, unquantized)

## Performance (Conversion Time)

- Audio encoder: ~90 seconds (5614 ops, 95 MIL passes)
- Decoder: ~85 seconds (slower transformer attention)
- LM head: ~4 seconds (simple linear projection)

## What We Learned

### Critical Insight
The initial error (`TypeError: only 0-dimensional arrays can be converted to Python scalars`) was **not due to fundamental incompatibility** with CoreML, but rather:

1. **Wrong coremltools version** (9.0 instead of 9.0b1)
2. **Wrong torch version** (2.11.0 instead of 2.7.0)
3. **Wrong transformers version** (4.51.3 instead of 4.57.6)

The beta version of coremltools (9.0b1) has better handling of dynamic shape operations.

### Community Success
Discord member `love4cristiano` reported successful conversion with:
- 15-35x RTF on M3 Pro
- 6-bit quantization with minimal WER drop
- GPU target preferred over ANE (CPU 20x, ANE only 10x)
- FP16 > INT8 for performance

Their approach likely also used proper dependency versions or alternative export methods.

## Previous Blocking Errors (Now Resolved)

### Error 1: Dynamic shape operations
```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```
**Resolution**: Fixed by using coremltools 9.0b1 and torch 2.7.0

### Error 2: Missing `positions` parameter
```
TypeError: TransformerDecoderWrapper.forward() missing 1 required positional argument: 'positions'
```
**Resolution**: Added positions tensor to DecoderWrapper signature

### Error 3: Tuple return value
```
RuntimeError: Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions
```
**Resolution**: Unpacked tuple return value `(hidden_states, None)` to extract tensor

### Error 4: Missing config attributes
```
AttributeError: 'CohereAsrConfig' object has no attribute 'hidden_size'
```
**Resolution**: Used nested config structure (e.g., `config.encoder["d_model"]`)

## Next Steps

1. ✅ **Conversion complete**
2. ⏭️ **Validation**: Compare CoreML vs PyTorch outputs for numerical parity
3. ⏭️ **Profiling**: Benchmark with coreml-cli (latency, ANE compatibility)
4. ⏭️ **Quantization**: Try 6-bit quantization (as per community success)
5. ⏭️ **HuggingFace Upload**: Publish to FluidInference/cohere-transcribe-03-2026-coreml
6. ⏭️ **FluidAudio Integration**: Create CohereAsrManager

## Recommendations

Based on conversion success and community feedback:

1. **Target GPU, not ANE** - ANE overhead makes GPU faster for this model
2. **Use FP16 or 6-bit quantization** - Better performance than INT8
3. **Expect 15-35x RTF** on M3 Pro class hardware
4. **Model is large (3.9 GB)** - Consider device storage constraints

## Files Updated

- ✅ `convert-cohere-transcribe.py` - Fixed wrappers and metadata extraction
- ✅ `pyproject.toml` - Corrected dependency versions
- ✅ `CONVERSION_STATUS.md` - Documented successful conversion
- ✅ `BLOCKING_ISSUE.md` - This file (marked as resolved)

## Conclusion

**Conversion is possible and SUCCESSFUL** using standard `torch.jit.trace()` approach with correct dependency versions.

The key was matching the exact environment that worked for Parakeet v3 conversion (Python 3.10.12, coremltools 9.0b1, torch 2.7.0).

**Status**: ✅ **RESOLVED** - Ready for validation and profiling
