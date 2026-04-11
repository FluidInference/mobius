# CosyVoice3 CoreML Conversion Trials

This directory contains research documentation and trial results from the CosyVoice3 CoreML conversion project.

## Organization

### Key Research Documents

- **MBMELGAN_SUCCESS.md** - Breakthrough: MB-MelGAN vocoder conversion success
- **KOKORO_APPROACH_ANALYSIS.md** - Analysis of Kokoro TTS CoreML patterns
- **OPERATION_REDUCTION_GUIDE.md** - How we achieved 3,494× operation reduction
- **FINAL_RESOLUTION.md** - Final solution: vocoder replacement strategy

### Completed Trials

- **DECODER_COMPRESSION_SUCCESS.md** - LLM decoder layer compression attempts
- **SIMPLIFIED_VOCODER_SUCCESS.md** - Simplified vocoder architecture tests
- **LAYERNORM_FIX_SUCCESS.md** - LayerNorm conversion fixes

### Failed Approaches (Important Learnings)

- **COREML_STFT_ATTEMPT.md** - Why STFT operations don't work in CoreML
- **FRAME_BASED_VOCODER_FAILED.md** - Frame-by-frame vocoder approach failed
- **STATELESS_ONNX.md** - Stateless ONNX conversion attempts

### Analysis Documents

- **COMPLETE_ANALYSIS.md** - Comprehensive architecture analysis
- **OPERATION_COUNT_ANALYSIS.md** - Operation count breakdown
- **KOKORO_VS_COSYVOICE_COMPARISON.md** - Architecture comparison
- **FARGAN_ANALYSIS.md** - FARGAN vocoder investigation
- **CUSTOM_CODE_VS_ARCHITECTURE.md** - Code complexity vs architecture complexity

### Implementation Guides

- **IMPLEMENTATION_GUIDE.md** - Implementation strategies
- **SWIFT_INTEGRATION.md** - Swift integration patterns
- **TESTING_GUIDE.md** - Testing methodology

### Status Reports

- **PROGRESS.md** - Overall progress tracking
- **COMPLETE_STATUS.md** - Complete status summary
- **FINAL_STATUS.md** - Final project status
- **DEPLOYMENT_READY.md** - Deployment readiness assessment

### Issues & Solutions

- **VOCODER_COREML_ISSUE.md** - Vocoder conversion issues
- **SWIFT_LOADING_ISSUE.md** - Swift model loading problems
- **DEBUGGING_FINDINGS.md** - Debugging session results
- **RESBLOCKS_CRITICAL_FINDING.md** - ResBlock implementation issues

### Planning Documents

- **FULL_TTS_CONVERSION_PLAN.md** - Full pipeline conversion strategy
- **SOLUTION_PROPOSAL.md** - Proposed solutions
- **RECOMMENDED_SOLUTION.md** - Final recommended approach
- **FEASIBILITY.md** - Feasibility assessment

## Why These Trials Matter

These documents capture:

1. **Dead ends** - What doesn't work and why (saves future effort)
2. **Breakthroughs** - Key discoveries that led to success
3. **Architecture insights** - Understanding of CoreML limitations
4. **Research findings** - Analysis of successful projects (Kokoro, HTDemucs)

## Key Learnings Summary

1. **Operation count is critical** - > 10k ops = CoreML failure
2. **Architecture replacement > optimization** - 705k → 202 ops via vocoder swap
3. **STFT operations unsupported** - Need alternatives for frequency-domain work
4. **Model splitting essential** - Enables dynamic-length outputs
5. **FP32 for audio quality** - FP16 degrades audio (Kokoro/HTDemucs findings)
6. **RangeDim superiority** - More flexible than EnumeratedShapes

## Production Code

For the organized, production-ready code, see:
- `../docs/` - Comprehensive guides
- `../scripts/` - Training pipeline
- `../benchmarks/` - Performance tests
- `../README.md` - Master documentation

This `trials/` directory preserves the research journey that led to the final solution.
