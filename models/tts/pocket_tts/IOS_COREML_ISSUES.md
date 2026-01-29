# PocketTTS iOS CoreML Issues

Document of issues encountered while bringing PocketTTS CoreML models to iOS.

---

## 1. Zero-Length Tensor Crash (Espresso / mimi_decoder)

**Status**: Fixed

**Symptom**: `mimi_decoder.mlpackage` crashes on iOS with `EXC_BAD_ACCESS (KERN_INVALID_ADDRESS at 0x0000000000000018)` in `Espresso::blob_cpu::__copy_to_host` on the `com.apple.CoreMLBatchProcessingQueue` thread.

**Root Cause**: The mimi_decoder model contained 3 state tensors with a zero-length dimension:
- `res0_conv1_prev`: shape `[1, 128, 0]`
- `res1_conv1_prev`: shape `[1, 64, 0]`
- `res2_conv1_prev`: shape `[1, 32, 0]`

These come from `StreamingConv1d` layers with `kernel_size=1` where the padding state size is `kernel - stride = 1 - 1 = 0`. The PyTorch model creates `torch.zeros(batch, channels, 0)` tensors that are identity pass-throughs (never read or written). CoreML's Espresso engine on iOS cannot handle zero-element blobs and crashes when attempting to copy them.

**Fix**:
1. **Model level**: Python script strips the 3 zero-length inputs, 3 zero-length outputs, and 3 identity ops from the MIL graph. The fixed model has 24 inputs / 24 outputs (was 27/27). Output is bit-for-bit identical to the original.
2. **Swift level**:
   - `mimiStateMapping` reduced from 26 to 23 entries (removed `res0_conv1_prev`, `res1_conv1_prev`, `res2_conv1_prev`)
   - `loadMimiInitialState` skips tensors with zero-length shapes via `guard !shapeArray.contains(0)`

**Affected file**: `mimi_decoder.mlpackage` on HuggingFace (`alexwengg/pocket-tts-coreml`) — replaced with fixed version.

**Lesson**: Always check for zero-length dimensions in CoreML model I/O. Espresso (CPU/GPU backend) crashes on zero-element blobs even though macOS ANE handles them fine.

---

## 2. iOS Simulator Produces Silent Audio

**Status**: Known limitation (not a bug)

**Symptom**: Full PocketTTS synthesis pipeline completes on the iOS Simulator without crashing, but the output WAV is entirely zeros (100% silence). All 4 models run and return outputs, but the computed values are all 0.0.

**Root Cause**: The iOS Simulator uses the Espresso CPU engine for CoreML inference. While it can load and execute the models without crashing (after the zero-length fix), it does not faithfully compute the full PocketTTS pipeline. This is a simulator limitation — the compute graph is too complex for the simulator's CPU-only Espresso backend to produce correct results.

**Workaround**: Test on a real iOS device. macOS CLI (`fluidaudiocli tts`) also produces correct audio and can be used for development verification.

---

## 3. Model Compilation Time on First Launch

**Status**: Expected behavior

**Symptom**: First launch on a new device takes significantly longer as CoreML compiles `.mlpackage` files to `.mlmodelc`. PocketTTS has 4 models totaling ~200MB of weights.

**Details**:
- `cond_step`, `flowlm_step`, `flow_decoder`, `mimi_decoder` each need compilation
- Compiled models are cached alongside the `.mlpackage` files
- Subsequent launches skip compilation

**Mitigation**: The `PocketTtsModelCache` actor caches compiled models and checks for existing `.mlmodelc` directories before recompiling.

---

## 4. Compute Unit Configuration

**Status**: Resolved

**Symptom**: Models may crash or produce incorrect results if run with `.all` compute units on the iOS Simulator (which has no ANE).

**Fix**: `PocketTtsModelCache` uses conditional compilation:
```swift
#if targetEnvironment(simulator)
config.computeUnits = .cpuAndGPU
#else
config.computeUnits = .all
#endif
```

This ensures the simulator uses CPU+GPU only, while real devices use ANE+CPU+GPU for maximum performance.

---

## 5. Float16 Output Handling (mimi_decoder)

**Status**: Fixed

**Symptom**: Mimi decoder outputs audio as float16 tensors on some devices. Using `dataPointer` with `Float.self` binding on float16 data produces garbage values.

**Fix**: `readFloatArray()` in `PocketTtsSynthesizer+Mimi.swift` checks `array.dataType` and uses the type-safe subscript accessor (`array[$0].floatValue`) for float16 data, which handles conversion automatically. Float32 data uses the fast direct memory access path.

---

## 6. AVAudioSession Configuration Required on iOS

**Status**: Fixed

**Symptom**: Audio playback produces no sound on iOS even with valid WAV data.

**Fix**: Must configure `AVAudioSession` before creating `AVAudioPlayer`:
```swift
try AVAudioSession.sharedInstance().setCategory(.playback)
try AVAudioSession.sharedInstance().setActive(true)
```

This is not required on macOS but mandatory on iOS for audio output.

---

## Summary of Model Changes

| Component | Original | Fixed | Change |
|-----------|----------|-------|--------|
| `mimi_decoder.mlpackage` inputs | 27 | 24 | Removed 3 zero-length tensor inputs |
| `mimi_decoder.mlpackage` outputs | 27 | 24 | Removed 3 zero-length tensor outputs |
| `mimiStateMapping` entries | 26 | 23 | Removed 3 zero-length state mappings |
| MIL identity ops | 3 | 0 | Removed dead pass-through operations |
| Computational output | Identical | Identical | Bit-for-bit verified on macOS |
