# CosyVoice3 CoreML Conversion - Status

**Date:** 2026-04-10

## ✅ Successfully Converted

### LLM (642M params → 1.2GB CoreML)
- cosyvoice_llm_embedding.mlpackage (260MB)
- cosyvoice_llm_lm_head.mlpackage (260MB)  
- decoder_layers/layer_0.mlpackage through layer_23.mlpackage (684MB)

### Vocoder (21M params → 83MB CoreML)
- hift.mlpackage (83MB)
- Working perfectly with LayerNorm fix

## ❌ Flow Model Blocked

**Root cause:** Missing `conformer` module dependency

The Flow decoder imports chain:
```
cosyvoice.flow.decoder
  → matcha.models.components.decoder
    → conformer (NOT FOUND)
```

**ONNX model exists:** flow.decoder.estimator.fp32.onnx (1.33 GB, works with ONNX Runtime)

**Attempted conversions:**
1. ONNX → CoreML (coremltools) - No ONNX frontend in v8.0+
2. ONNX → CoreML (onnx-coreml) - Incompatible versions
3. PyTorch → CoreML - Blocked by missing conformer

## To Complete Flow Conversion

Find conformer module (likely from wenet/espnet) and add to dependencies, then re-run convert_flow_final.py

## Summary

**Converted:** 1.28GB CoreML (LLM + Vocoder)
**Remaining:** 1.3GB ONNX (Flow - works but not CoreML)

**Blocker:** Single missing dependency (conformer module)
