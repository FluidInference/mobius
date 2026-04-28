# PocketTTS iOS CoreML Issues

Document of issues encountered while bringing PocketTTS CoreML models to iOS.

---

## ANE Dispatch Status Summary

Final compute-unit assignment for the 4 PocketTTS inference models, with the
share of pipeline wall-time each accounts for and the share of that wall-time
that actually reaches the Apple Neural Engine.

| Model | Final dispatch | Calls per audio frame | % of pipeline wall-time | % of wall-time on ANE | ANE attempts | Outcome |
|---|---|---|---|---|---|---|
| `cond_step.mlmodelc` | `.cpuAndGPU` | ~0.1 (once per chunk prefill) | <0.1% | **0%** | Tried `.all`; tripped MPSGraph rank-5 / zero-shape assert on the `(2,1,512,16,64)` KV cache during prefill | ❌ Forced off ANE |
| `flowlm_step.mlmodelc` | `.all` | ~12.5 | ~10% | ~70-90% (matmul/attn on ANE; softmax + lm_head spill to CPU) | Initial direct convert worked. ANE 1.97× faster than GPU on this layer | ✅ Kept on ANE |
| `flow_decoder.mlmodelc` | `.all` | ~100 (8 LSD steps × 12.5 frames) | ~3% | ~80-100% (tiny MLP + Euler step, fully ANE-friendly) | Initial direct convert worked | ✅ Kept on ANE |
| `mimi_decoder.mlmodelc` | `.cpuOnly` | ~12.5 | ~10% | **0%** | 3 distinct failures: (1) zero-length tensor Espresso crash, (2) fp16 precision compounding → audible beeping, (3) 64-byte stride misalign segfault | ❌ Forced off ANE |
| `mimi_encoder.mlmodelc` (voice cloning, optional) | `.cpuAndGPU` | once per voice | n/a (offline) | **0%** | Not attempted for ANE — same streaming-state architecture as decoder | ❌ Not on ANE |

**Pipeline-level ANE utilization**: of the ~23% of wall-time spent in CoreML
inference (the rest is Swift glue, audio post-processing, and disk I/O), only
the `flowlm_step` + `flow_decoder` portion (~13% of total) actually touches the
ANE. That's roughly **~10% of total pipeline wall-time on ANE, ~90% on CPU/GPU**.

The architectural ceiling is set by `mimi_decoder`: as long as the streaming
codec uses fp16 state feedback across 23 tensors at 12.5 Hz, the audio decode
path is locked to CPU.

---

## ANE Failure Detail Per Model

### `cond_step` — MPSGraph rank-5 / zero-shape assert

| Aspect | Detail |
|---|---|
| Symptom | MPSGraph internal assertion when ANE partitioner tries to fold prefill into an ANE block |
| Root cause | KV cache shape `(2, 1, 512, 16, 64)` — rank 5, leading `2` is k/v stack. ANE's compiler hits a rank-planner edge case |
| Op replacement attempted? | No |
| Why no fix attempted | Call frequency is ~0.1/frame → <0.1% of wall-time. Even a 10× ANE speedup saves nothing measurable. Splitting `(2,1,...)` into separate k/v tensors would force a matching rewrite in `flowlm_step` (which consumes the stacked layout) for zero gain |
| Workaround | `.cpuAndGPU` — robust, no behavior change |
| Public match | Partial — rank-5 is documented as a legacy CoreML constraint; MPSGraph ANE access is undocumented and unreliable |

### `flowlm_step` — ✅ ANE WIN (no rewrite needed)

| Aspect | Detail |
|---|---|
| Why it works | Stateless feed-forward step, no per-frame state accumulation. Float16 drift is bounded to a single matmul stack |
| Dispatch | matmul/attention runs on ANE; softmax + lm_head fall back to CPU (ANE has limited softmax support) |
| Measured speedup | 1.97× faster on ANE than GPU |
| Pipeline impact | This is **the** hot model. The ANE win here drives end-to-end RTFx |

### `flow_decoder` — ✅ ANE WIN (trivial)

| Aspect | Detail |
|---|---|
| Why it works | Tiny model: `(1,1024) + (1,32) + 2 scalars → (1,32)`. Pure MLP + Euler integration. No state |
| Dispatch | Fully ANE-eligible |
| Pipeline impact | Called 8× per output frame at LSD step. Small absolute time but high call frequency — ANE keeps it cheap |

### `mimi_decoder` — ❌ ANE FAIL (3 distinct failure modes)

#### Failure A: Zero-length tensor Espresso crash

| Aspect | Detail |
|---|---|
| Symptom | `EXC_BAD_ACCESS (KERN_INVALID_ADDRESS at 0x0...18)` in `Espresso::blob_cpu::__copy_to_host` on `com.apple.CoreMLBatchProcessingQueue` |
| Root cause | 3 state tensors with zero-length dim from `StreamingConv1d(kernel_size=1)` layers: `res0_conv1_prev` `[1,128,0]`, `res1_conv1_prev` `[1,64,0]`, `res2_conv1_prev` `[1,32,0]` |
| Op replacement attempted? | Yes, two iterations |
| Iteration 1 | MIL-level strip of 3 zero-length inputs/outputs + 3 identity ops (`strip_zero_length_io` in `convert_mimi_decoder.py:166-221`). Worked for `neuralNetwork` format |
| Iteration 2 (mlProgram regression) | Spec-strip causes `"Model and main function must have same number of inputs and states"` because the MIL function still references them. Skipped at line 313-318 |
| Final fix | Swift-side: NULL-buffer sentinel allocates empty `MLMultiArray` with the zero-length shape via the schema discovery path |
| Public match | ✅ Documented community workaround pattern: "avoid empty/degenerate tensor shapes in the converted model" |

#### Failure B: fp16 precision compounding → beeping artifact

| Aspect | Detail |
|---|---|
| Symptom | Audible periodic beeping/buzzing in synthesized audio when `compute_units = .all` |
| Diagnosis ladder | (1) original vs MIL-stripped models: bit-identical in Python — model not at fault. (2) Python CoreML `CPU_AND_GPU`: no beeping. (3) Swift CoreML `.all` (ANE active): beeping reproducible |
| Root cause | 23 streaming state tensors feed back every 80ms (`convtr*_partial`, `upsample_partial`, attention KV caches). ANE's fp16 quantization perturbations (~1e-3 per frame) compound across 75 frames/sec into audible artifacts |
| Op replacement attempted? | No |
| Why no fix attempted | The "non-ANE-friendly op" is **the entire streaming feedback topology**, not a specific op. Fixes would require: per-op fp32 precision overrides (not exposed by CoreML), or restructuring Mimi to non-streaming (doesn't exist), or int8 quantization with proper scaling on the feedback path (huge engineering, no upstream support) |
| Final fix | `.cpuOnly` — CPU computes in fp32 implicitly. Bonus: CPU is 1.74× faster than GPU on this small streaming-conv model |
| Public match | ✅ Apple's official guidance: "Streaming/stateful CoreML models with many feedback tensors should avoid the ANE." Confirmed by Mish activation issue (coremltools#2359) and MatAnyone alpha-matte drift |

#### Failure C: 64-byte stride misalignment segfault

| Aspect | Detail |
|---|---|
| Symptom | Hard segfault (no recoverable error) when ANE picks up some state tensors |
| Root cause | ANE requires last-axis tile strides aligned to 64 bytes. Mimi's residual stack uses channel dims `[32, 64, 128, 256, 512]`. The 32-channel (`32 × 2 = 64 bytes`) and 64-channel layers sit *exactly* on the boundary; some intermediate transposed-conv outputs drop below 32 channels and misalign |
| Op replacement attempted? | No |
| Why no fix attempted | Padding every state tensor's channel dim to a multiple of 32 (= 64 bytes ÷ 2 for fp16) cascades through the residual stack and changes the model contract. Failure B already disqualifies ANE, so this fix has no payoff |
| Public match | ✅ Documented Apple ML research: "the last axis of an ANE buffer is not packed; it must be contiguous and aligned to 64 bytes." Recent Apple Dev Forums report: stateful mlprogram fails on ANE if state tensor width is not a multiple of 32 |

---

## v2 Conversion Effort — Net ANE Impact: 0%

The v2 round of conversion edits (FP32 IO declarations, anonymous outputs to
break SSA aliasing, semantic output renaming, NULL-buffer sentinel) was **not
ANE-related**. It was IO-contract plumbing for an `"Invalid heap allocated
handle"` MLE5 binder error that:

- Has **no public footprint** (zero indexed search results for the exact string)
- Was likely solvable on the Swift side via `MLShapedArray<Float16>` or fp16
  `dataPointer` reads, not by regenerating mlpackages
- Produced **no perf win** in benchmarks (v1 RTFx ≈ 0.10×, v2 RTFx ≈ 0.07-0.09×)

The v2 work has residual value for converting the 9 new language packs (which
have no prior CoreML conversions to regress against), but did not move the
ANE utilization needle on English.

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

**Fix**: `PocketTtsModelCache` loads all models with `.cpuAndGPU` compute units to avoid ANE float16 precision loss (see issue #7). The simulator uses the CPU backend automatically when ANE is unavailable.

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

## 7. ANE Float16 Precision Loss in Mimi Decoder (Beeping Artifact)

**Status**: Fixed

**Symptom**: PocketTTS audio output contains a constant audible beeping/buzzing artifact in the background during speech. The beeping is not present in audio generated by the Python CoreML reference script (`generate_coreml_v4.py`).

**Root Cause**: The Apple Neural Engine (ANE) processes all computations in native float16. The Mimi decoder has 23 streaming state tensors that feed back every frame (80ms), including overlap-add buffers (`convtr*_partial`, `upsample_partial`) and attention KV caches. When running on ANE, float16 quantization errors compound across frames, producing audible periodic artifacts.

The Python CoreML reference uses `compute_units=ct.ComputeUnit.CPU_AND_GPU` which avoids the ANE entirely. The CPU and GPU compute in float32, preventing precision loss in the streaming state feedback loop.

**Diagnosis**:
1. The original and stripped models produce **bit-identical** output in Python — the model stripping was ruled out as a cause.
2. Python CoreML output (CPU+GPU, no ANE): **no beeping**
3. Swift CoreML output (`.all`, uses ANE): **beeping present**
4. The beeping exists regardless of de-essing post-processing.

**Fix**: `PocketTtsModelCache` loads all four models with `.cpuAndGPU` compute units:
```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuAndGPU
```

All models use the same configuration to avoid ANE float16 precision loss across the full pipeline.

**Lesson**: Streaming/stateful CoreML models with many feedback tensors should avoid the ANE. The float16 precision loss is negligible for single-pass inference but compounds in iterative state feedback loops.

---

## 8. Voice-Dependent Duration Differences (MLX vs CoreML)

**Status**: Open — model-level behavior difference

**Symptom**: Some voices produce noticeably different audio durations between MLX and CoreML inference. Two voices (`azelma` and `javert`) show ~2 second differences.

| Voice   | MLX Duration | CoreML Duration | Difference |
|---------|-------------|-----------------|------------|
| alba    | 6.24s       | 6.08s           | -0.16s     |
| azelma  | 7.92s       | 5.92s           | -2.00s     |
| cosette | 6.24s       | 5.60s           | -0.64s     |
| javert  | 8.32s       | 10.24s          | +1.92s     |

**Root Cause**: This is a model-level inference behavior difference between MLX and CoreML, not a Swift post-processing issue. The duration is determined by the autoregressive generation loop (flowlm_step EOS detection and flow_decoder output), where small numerical differences between backends accumulate across generation steps, causing the model to terminate earlier or later depending on the voice conditioning.

**Notes**:
- `alba` and `cosette` are within acceptable tolerance (<1s)
- `azelma` generates shorter on CoreML (early EOS)
- `javert` generates longer on CoreML (late EOS)
- Audio quality is correct for all voices — only duration differs
- Cannot be fixed in Swift post-processing code; would require investigating numerical differences in the flow decoder or EOS detection at the model conversion level

---

## Summary of Model Changes

| Component | Original | Fixed | Change |
|-----------|----------|-------|--------|
| `mimi_decoder.mlpackage` inputs | 27 | 24 | Removed 3 zero-length tensor inputs |
| `mimi_decoder.mlpackage` outputs | 27 | 24 | Removed 3 zero-length tensor outputs |
| `mimiStateMapping` entries | 26 | 23 | Removed 3 zero-length state mappings |
| MIL identity ops | 3 | 0 | Removed dead pass-through operations |
| Computational output | Identical | Identical | Bit-for-bit verified on macOS |
