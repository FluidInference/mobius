# Conversion Status

Track the progress of converting Cohere Transcribe 03-2026 to CoreML.

## Status: ✅ Conversion Complete - Ready for Validation

## Checklist

### Phase 1: Setup ✅
- [x] Create conversion directory structure
- [x] Write `pyproject.toml` with dependencies
- [x] Write conversion script `convert-cohere-transcribe.py`
- [x] Write validation script `compare-models.py`
- [x] Write README and QUICKSTART docs
- [x] Add `.gitignore`

### Phase 2: Initial Conversion ✅
- [x] Run `uv sync` to install dependencies
- [x] Verify transformers version supports `CohereAsrForConditionalGeneration`
- [x] Download model from HuggingFace (auto-downloaded on first run)
- [x] Run conversion: `uv run python convert-cohere-transcribe.py`
- [x] Document any errors encountered
- [x] Adjust conversion script if needed
- **Result**: All three components successfully converted to CoreML
  - Audio encoder: 3.6 GB (2B param Conformer)
  - Decoder: 293 MB
  - LM head: 32 MB

### Phase 3: Validation ✅
- [x] Obtain test audio file (16kHz WAV)
- [x] Run comparison: `uv run python compare-models.py --audio-file test.wav --coreml-dir ./build/cohere-transcribe`
- [x] Verify numerical parity
- [x] Document any discrepancies
- [x] Fix issues if numerical parity fails
- **Result**: ✅ PASSED with rtol=0.01, atol=0.02
  - Max absolute error: 0.011205
  - Mean absolute error: 0.000236
  - PyTorch and CoreML outputs match within acceptable tolerance

### Phase 4: Profiling (Blocked - Tool Incompatibility)
- [ ] Run `coreml-cli` to benchmark latency
- [ ] Check ANE compatibility with `--fallback` flag
- [ ] Document compile time, inference time, RTFx
- [ ] Test different compute unit configs (ALL, CPU_ONLY, CPU_AND_NE)
- **Status**: coreml-cli has compatibility issues with ML Program models (.mlpackage)
- **Workaround**: Manual profiling or wait for tool update
- **Community data**: 15-35x RTF on M3 Pro (GPU target)

### Phase 5: Optimization (Not Started)
- [ ] Try INT8 quantization
- [ ] Try INT4 quantization
- [ ] Compare quality/size trade-offs
- [ ] Select best quantization for deployment

### Phase 6: HuggingFace Upload (Not Started)
- [ ] Create HuggingFace repo: `FluidInference/cohere-transcribe-03-2026-coreml`
- [ ] Upload all `.mlpackage` files
- [ ] Upload `metadata.json`
- [ ] Write model card with:
  - Source attribution
  - License (Apache 2.0)
  - Input/output specs
  - Performance benchmarks
  - Usage examples

### Phase 7: FluidAudio Integration (Not Started)
- [ ] Register model in `ModelNames.swift`
- [ ] Create `CohereAsrManager` in `Sources/FluidAudio/ASR/Cohere/`
- [ ] Add CLI command to `fluidaudiocli`
- [ ] Write unit tests
- [ ] Write benchmark command
- [ ] Update FluidAudio README

### Phase 8: Documentation & PR (Not Started)
- [ ] Create mobius PR with conversion scripts
- [ ] Create FluidAudio PR with integration
- [ ] Link PRs to HuggingFace repo
- [ ] Update CLAUDE.md if needed

## Known Issues

### Issue 1: Model size may exceed ANE limits
- **Status**: Not yet tested
- **Impact**: May need quantization or may run on GPU instead of ANE
- **Workaround**: Start with INT8 quantization if ANE fails

### Issue 2: Conformer ops may not be ANE-compatible
- **Status**: Not yet tested
- **Impact**: Some layers may fall back to CPU/GPU
- **Workaround**: Profile with `coreml-cli --fallback` to identify issues

### Issue 3: Long compilation times expected
- **Status**: Expected based on 2B params
- **Impact**: 5-10 min first load on ANE
- **Workaround**: This is expected; document in README

## Trials

### Trial 0: Initial setup
- **Date**: 2026-04-03
- **Action**: Created conversion directory and scripts
- **Result**: ✅ Ready for conversion
- **Notes**: Following patterns from Qwen3 and Parakeet conversions

### Trial 1: First conversion attempt (Current)
- **Date**: 2026-04-03
- **Action**: Attempted to run conversion
- **Result**: ⚠️ Blocked - Model is gated
- **Error**: `OSError: You are trying to access a gated repo`
- **Solution**: User must:
  1. Request access at https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
  2. Run `huggingface-cli login` with their token
- **Notes**:
  - Updated conversion script to use `AutoModelForSpeechSeq2Seq` with `trust_remote_code=True`
  - Model uses custom classes that require remote code execution
  - Updated README and QUICKSTART to document gated access requirement

### Trial 2: Post-authentication conversion attempt
- **Date**: 2026-04-03
- **Action**: Retried conversion after HuggingFace authentication
- **Result**: ⚠️ Partial success - Model loaded, but CoreML conversion failed
- **Progress**:
  - ✅ Added `sentencepiece` dependency
  - ✅ Model loaded successfully with `trust_remote_code=True`
  - ✅ Identified correct model structure:
    - `encoder`: ConformerEncoder (not `audio_encoder`)
    - `transf_decoder`: TransformerDecoderWrapper
    - `encoder_decoder_proj`: Linear projection
    - `log_softmax`: TokenClassifierHead (LM head)
  - ✅ Updated wrappers with correct attribute names
  - ❌ CoreML conversion failed with `TypeError: only 0-dimensional arrays can be converted to Python scalars`
- **Root cause**: Model uses dynamic shape operations incompatible with CoreML tracing
- **Error location**: Integer cast operations in encoder's positional encoding and shape calculations
- **Notes**:
  - The Conformer encoder has conditional logic that depends on runtime tensor values
  - CoreML requires all shapes and control flow to be determined at trace time
  - This is a fundamental limitation, not a fixable bug

### Trial 3: Version mismatch investigation
- **Date**: 2026-04-03
- **Action**: User suggested checking Parakeet v3's `uv.lock` for working versions
- **Discovery**: Initial attempt used incompatible dependency versions:
  - ❌ Python 3.12.8 (too new)
  - ❌ coremltools 9.0 (should be 9.0b1)
  - ❌ torch 2.11.0 (too new)
  - ❌ transformers 4.51.3 (too old)
- **Solution**: Matched exact versions from Parakeet v3:
  - ✅ Python 3.10.12
  - ✅ coremltools 9.0b1
  - ✅ torch 2.7.0
  - ✅ transformers 4.57.6
  - ✅ scikit-learn 1.5.1
- **Notes**: This was the critical breakthrough that enabled successful conversion

### Trial 4: Successful conversion with fixed dependencies
- **Date**: 2026-04-03
- **Action**: Re-ran conversion with corrected dependency versions
- **Result**: ✅ SUCCESS - All three components converted
- **Fixes applied**:
  1. Fixed metadata save function to use correct config attributes:
     - `encoder["d_model"]` → encoder_hidden_size (1280)
     - `transf_decoder["config_dict"]["hidden_size"]` → decoder_hidden_size (1024)
     - `head["hidden_size"]` → lm_head_hidden_size (1024)
  2. Updated DecoderWrapper to include `positions` parameter
  3. Fixed decoder output unpacking (returns tuple of `(hidden_states, None)`)
- **Output**:
  - ✅ `cohere_audio_encoder.mlpackage` - 3.6 GB
  - ✅ `cohere_decoder.mlpackage` - 293 MB
  - ✅ `cohere_lm_head.mlpackage` - 32 MB
  - ✅ `metadata.json` - Model configuration
- **Performance**:
  - Audio encoder conversion: ~90 seconds (5614 ops, 95 MIL passes)
  - Decoder conversion: ~85 seconds (slower due to transformer attention)
  - LM head conversion: ~4 seconds (simple linear layer)
- **Next steps**: Validation and profiling

## Resources

- Model card: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- Similar conversion: `mobius/models/stt/qwen3-asr-0.6b/coreml/`
- Integration example: `Sources/FluidAudio/ASR/Qwen3/Qwen3AsrManager.swift`

## Timeline

- **Setup**: 2026-04-03 ✅
- **Initial conversion**: TBD
- **Validation**: TBD
- **HuggingFace upload**: TBD
- **FluidAudio integration**: TBD
- **Release**: TBD
