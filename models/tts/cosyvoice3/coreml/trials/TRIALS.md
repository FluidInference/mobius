# CosyVoice3 CoreML Conversion Trials

Chronological log of CosyVoice3-0.5B CoreML conversion attempt.

---

## Phase 1: Model Analysis (2026-04-09)

### Repository Structure

Downloaded from `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`:

**ONNX Models (Ready to Convert):**
1. `campplus.onnx` (27 MB, 6.9M params)
   - Speaker embedding/verification
   - Input: `[batch, seq_len, 80]` (mel features)
   - Output: `[batch, seq_len]` (speaker embedding)
   - 3,206 nodes, mostly Conv/BatchNorm

2. `flow.decoder.estimator.fp32.onnx` (245 MB, 87M params)
   - Flow matching DiT decoder
   - Input: `x_t` (noised latent), `timestep`, `conditioning`
   - 22 transformer blocks with self-attention
   - Already in ONNX FP32 format

3. `speech_tokenizer_v3.onnx` / `speech_tokenizer_v3.batch.onnx`
   - Speech tokenizer (FSQ-based)
   - Converts audio to discrete tokens

**PyTorch Checkpoints (Need Conversion):**
1. `llm.pt` (1.9 GB, 508M params)
   - LLM component (CosyVoice3LM)
   - Autoregressive token prediction
   - **Challenge**: Not in ONNX format

2. `flow.pt` (967 MB, 332M params)
   - Full flow matching model
   - Includes the estimator (already have ONNX) + other components
   - Parameters: `input_embedding`, `pre_lookahead_layer`, `spk_embed_affine_layer`, `decoder.estimator.*`

3. `hift.pt` (79 MB, 20.8M params)
   - HiFi-GAN vocoder variant
   - Source-filter model with F0 prediction
   - Parameters: `m_source`, `conv_pre`, `ups`, `resblocks`, `conv_post`, `f0_predictor`

**Safetensors:**
- `CosyVoice-BlankEN/model.safetensors` - Unknown purpose, possibly LLM weights

### Architecture Summary

```
Text → [LLM] → Discrete Tokens → [Flow Matching] → Latent Features → [Vocoder] → Audio
        ↑                              ↑                                    ↑
    508M params                     332M params                         20.8M params
    (llm.pt)                        (flow.pt)                          (hift.pt)
```

**Speaker Embedding Pipeline:**
```
Reference Audio → [campplus.onnx] → Speaker Embedding (6.9M params)
                                            ↓
                                    [Injected into Flow/Vocoder]
```

### Total Parameters

| Component | Size (MB) | Parameters | Format | CoreML Status |
|-----------|-----------|------------|--------|---------------|
| LLM | 1,900 | 508M | PyTorch | ❌ Not started |
| Flow | 967 | 332M | PyTorch | 🟡 Decoder in ONNX (87M) |
| Vocoder | 79 | 20.8M | PyTorch | ❌ Not started |
| Speaker Embedding | 27 | 6.9M | ONNX | ✅ Ready |
| Speech Tokenizer | ? | ? | ONNX | ✅ Ready |
| **Total** | **~3 GB** | **~868M** | Mixed | **Partial** |

### Key Findings

1. **Partial ONNX availability**: 3 out of 5 components are already in ONNX
   - ✅ campplus.onnx - speaker embedding
   - ✅ speech_tokenizer_v3.onnx - tokenizer
   - ✅ flow.decoder.estimator.fp32.onnx - flow DiT decoder
   - ❌ LLM - only PyTorch checkpoint
   - ❌ Vocoder - only PyTorch checkpoint

2. **Model size discrepancy**: Advertised as "0.5B" but actual total is ~868M parameters
   - Likely counting only the LLM base (508M)
   - Flow + vocoder add another 353M

3. **Flow model redundancy**:
   - `flow.pt` (332M) contains the full flow model
   - `flow.decoder.estimator.fp32.onnx` (87M) is just the DiT decoder part
   - We may only need the ONNX decoder if we can reconstruct the wrapper

4. **Vocoder architecture**: HiFi-GAN variant with F0 conditioning
   - Should be convertible to CoreML (similar to existing TTS vocoders)
   - Uses weight normalization (parametrizations.weight.original0/1)

### Conversion Strategy

**Phase 1**: Convert existing ONNX models (easy wins)
1. ✅ campplus.onnx → CoreML (speaker embedding)
2. ✅ speech_tokenizer_v3.onnx → CoreML (tokenizer)
3. ✅ flow.decoder.estimator.fp32.onnx → CoreML (DiT decoder)

**Phase 2**: Convert PyTorch models
4. ❌ hift.pt → CoreML (vocoder) - reconstruct architecture, load weights, trace
5. ❌ llm.pt → CoreML (LLM) - **HARD** - 508M params, may not fit on ANE

**Phase 3**: Pipeline integration
6. ❌ Build inference pipeline connecting all components
7. ❌ Test end-to-end audio generation

### Open Questions

1. **LLM component**: How is llm.pt used? Need to find:
   - Original model architecture code
   - Input/output specifications
   - Inference loop structure

2. **Flow wrapper**: Can we use just the ONNX decoder or need full flow.pt?

3. **Text preprocessing**: Where is text normalization (CosyVoice-ttsfrd)?

4. **Token embeddings**: How do discrete tokens from LLM feed into flow decoder?

### Next Steps

1. ✅ **Start with ONNX conversions** (campplus, tokenizer, flow decoder)
   - These are ready to convert immediately
   - Will validate CoreML conversion pipeline

2. ❓ **Research LLM architecture**:
   - Find CosyVoice3LM implementation
   - Understand how llm.pt checkpoint is loaded
   - Determine if 508M params can run on ANE

3. ❓ **Vocoder conversion**:
   - Write PyTorch model wrapper for hift.pt
   - Load weights and trace
   - Convert to CoreML

---

## Phase 2: ONNX to CoreML Conversion

### Converting campplus.onnx (Speaker Embedding)

Status: **Not started**

Plan:
- Use `onnx_coreml` converter
- Input: mel spectrogram features [batch, seq_len, 80]
- Output: speaker embedding [batch, seq_len]
- Expected issues: dynamic shapes, batch processing

---

## Phase 3: PyTorch Model Conversion

### Converting hift.pt (Vocoder)

Status: **Not started**

Architecture hints from checkpoint keys:
- `m_source.l_linear` - harmonic source module (SourceModuleHnNSF)
- `conv_pre` - pre-convolution with weight normalization
- `ups.0/1/2` - 3 upsampling layers
- `source_downs.0/1/2` - source downsampling (for F0 path)
- `source_resblocks.0-2` - residual blocks for source path
- `resblocks.0-8` - main path residual blocks (9 total)
- `conv_post` - post-convolution
- `f0_predictor.condnet` - F0 prediction network

This matches a **source-filter HiFi-GAN** architecture similar to the one in KittenTTS.

---

## Phase 4: LLM Investigation

Status: **Not started**

Need to find:
1. CosyVoice3LM model definition
2. How to load llm.pt checkpoint
3. Inference code
4. Can it be exported to ONNX?

---

## Notes

- **ANE Compatibility Concerns**:
  - 508M param LLM unlikely to run efficiently on ANE
  - Flow DiT (87M) may have ANE issues (attention ops)
  - Vocoder (20.8M) should be ANE-compatible (mostly Conv ops)
  - Speaker embedding (6.9M) should be ANE-compatible

- **Memory Estimates**:
  - FP32 total: ~3.5 GB
  - FP16 quantized: ~1.75 GB
  - W8A16 quantized: ~1 GB (like Qwen TTS approach)

- **Comparison to Qwen TTS**:
  - Qwen TTS: 1.7B params, split into 6 models, W8A16 quantized → ~1 GB total
  - CosyVoice3: 868M params, need similar splitting strategy
