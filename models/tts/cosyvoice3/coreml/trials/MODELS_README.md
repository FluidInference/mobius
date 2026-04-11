# CosyVoice3 CoreML Models

Total: 5 components (28 files, 1.3GB)

## Model Files

```
cosyvoice3/
├── cosyvoice_llm_embedding.mlpackage          260MB
├── cosyvoice_llm_lm_head.mlpackage            260MB
├── decoder_layers/                            684MB total
│   └── cosyvoice_llm_layer_[0-23].mlpackage   28MB each × 24
├── flow_decoder.mlpackage                      23MB
└── converted/
    └── hift_vocoder.mlpackage                  78MB
```

## Why 24 Decoder Layer Files?

**Technical reasons:**
- CoreML conversion limits (can't trace >1GB models in one pass)
- Memory efficiency during export/validation
- Individual layer optimization for ANE

**Runtime impact:** 
- ✅ All 24 loaded once at startup
- ✅ Stored as array in memory
- ✅ No performance difference vs single file
- ✅ Actually faster loading (parallel possible)

## Swift Usage

```swift
// Loads all 24 layers automatically
let tts = try CosyVoiceCoreML(modelDirectory: modelDir)

// User never sees the 24 files - just one API
let audio = try await tts.synthesize(text: "Hello!")
```

The complexity is hidden from the user by the `CosyVoiceCoreML` class.
