# Magpie Swift Port — Findings & Platform Quirks

Notes from porting Magpie TTS Multilingual 357M from `generate_coreml.py` (Python
reference) to the FluidAudio Swift package. Documents what doesn't transfer
cleanly, ANE compile behavior, and the perf optimizations that materially
moved the needle. Future agents (Swift or Python side) — read before changing
the conversion scripts.

## TL;DR

1. **`decoder_step.mlmodelc` cannot use ANE in production.** ANE compile
   reliably fails at runtime (`MILCompilerForANE: ANECCompile() FAILED`)
   under real synth, even though it succeeds with zeroed dummy inputs.
   Pin Swift consumers to `.cpuAndGPU` for that model.
2. **`decoder_prefill.mlmodelc` gives a 3.8× end-to-end speedup over the
   110-step prefill fallback.** Worth keeping in the canonical HF artifact set.
3. **fp16 host non-determinism is real and compounds with depth.** Stage-by-
   stage Swift ↔ Python parity: text_encoder ~50 dB SNR, prefill L0=64 dB →
   L11=44 dB, AR replay ~40 dB. Below ~50 dB you get audible single-word drift.
4. **`MLComputePlan.load(...)` (macOS 14.4+) SIGBUSes on every Magpie
   `.mlmodelc`** — can't introspect device assignment via the public API.

## Per-model compute placement (verified end-to-end)

Measured on M-series, real synth (8-word EN sentence, warm), Swift consumer:

| Model              | `.cpuOnly` | `.cpuAndGPU` | `.cpuAndNeuralEngine` | Recommendation |
| ------------------ | ---------- | ------------ | ---------------------- | -------------- |
| `text_encoder`     | 42 ms      | 43 ms        | **12 ms**              | ANE            |
| `decoder_prefill`  | 56 ms      | 23 ms        | **18 ms**              | ANE            |
| `decoder_step`     | 31 ms\*    | **22 ms**    | 10 ms\*\*              | **GPU** (see below) |
| `nanocodec_decoder`| —          | —            | runs on ANE            | ANE            |

\* dummy-input single-call benchmark; real synth is 96 s warm.
\*\* dummy-input speedup is misleading — see decoder_step section.

### decoder_step ANE failure mode (the trap)

`coreml-cli` and the dummy-input benchmark both report ANE works on
`decoder_step`. **In real synth it does not.** Stack:

- Single-call benchmark with `position = 0`: ANE compile succeeds, runs in
  10 ms → looks 3× faster than CPU.
- Real synth (incrementing `position` 0…N, real KV cache state): ANE
  recompile is triggered per call and **fails** with `MILCompilerForANE:
  ANECCompile() FAILED` (visible in stderr), then falls back to CPU at
  hundreds-of-ms per call. End-to-end this is 7% slower than `.cpuAndGPU`
  (103 s vs 96 s warm) and 34% slower cold.

The likely cause is the rank-4 split-K/V scatter pattern (`cache_k_i` /
`cache_v_i` are `[1, 512, H, D]` fp16 with `position` advancing per step).
ANEF can compile the topology against a static input but bails when actual
gather indices vary at runtime.

**Action items if you revisit `convert_decoder_step.py`:**

- Try a single rank-3 K/V layout (`[512, H*D]` fp16) instead of split rank-4.
- Try `position` as `int32` instead of `float16`.
- Try eliminating the scatter by writing the new K/V row as a separate output
  and letting the host concatenate (already what Swift's `MagpieKvCache` does
  conceptually).
- Verify with **real incrementing positions**, not zeros.

If ANE remains broken, the current `.cpuAndGPU` pin is correct. Document
the failure in `coreml/convert_decoder_step.py` so the next person doesn't
chase the dummy-input ghost.

## decoder_prefill is essential, not optional

The repo README marks `decoder_prefill.mlmodelc` as "optional" with the
fallback being 110 sequential `decoder_step` calls. Reality:

| Path                             | Wall (warm) |
| -------------------------------- | ----------- |
| 110-step prefill fallback        | ~420 s      |
| `decoder_prefill.mlmodelc` (1×)  | ~110 s      |

That's a **3.8× speedup** from a single batched call. Without it the Swift
port is unshippable. Ensure `convert_decoder_prefill.py` runs in CI / the
canonical HF asset upload.

### Prefill output naming

`decoder_prefill.mlmodelc` emits 12 outputs as anonymous CoreML var IDs:

```
var_208, var_374, var_540, var_706, var_872, var_1038,
var_1204, var_1370, var_1536, var_1702, var_1868, var_1958
```

Each is `[2, 1, 512, H, D]` fp16 with axis 0 = `[K_stacked, V_stacked]`.
Swift slices them with two `memcpy`s into the per-layer K/V cache. **Don't
rename these without bumping the Swift port.** Or — better — explicitly name
them `prefill_kv_layer_{0..11}` in `convert_decoder_prefill.py` so the Swift
binding is robust to recompiles.

## fp16 host non-determinism: what we measured

Swift CoreML and Python+coremltools CoreML produce **bit-different** fp16
outputs on the same inputs, same `.mlmodelc`, same M-series host. This is
documented Apple behavior (CoreML doesn't guarantee bit-exact reproducibility
across processes / load configurations). Magnitude:

- **text_encoder**: SNR(Swift, Python) ≈ 50.6 dB
- **decoder_prefill** per layer: L0 SNR 64 dB → L11 SNR 44 dB
  (compounds geometrically through the 12-layer cache)
- **decoder_step AR loop**: post-12-layer cache → ~40 dB after 100 steps

40 dB SNR is below the threshold where the top-k=80 sampler trajectory
diverges, which manifests as a single trailing word in the audio (e.g.
"…seashore, **and**" instead of "…seashore."). 4/5 speakers are unaffected
because their sampler trajectories are more stable; speaker 0 in our test
set consistently hits the drift.

**This is not a Swift bug.** Verified by:

1. Python+CoreML matches NeMo PyTorch.
2. Swift+CoreML's text_encoder output already differs from Python+CoreML at
   50 dB (no Swift-side math involved — it's just `MLModel.prediction`).
3. Swift's LocalTransformer (Accelerate+BNNS) matches a fp64 NumPy reference
   to >120 dB in isolation — so the post-CoreML Swift path is clean.

**If you want bit-identical Swift↔Python parity**, the only paths are:
- Force fp32 weights in the `.mlpackage` (size + perf cost — probably not
  worth it for ~1 word at the end of an utterance).
- Accept perceptual parity (current state).

## MLComputePlan crashes on Magpie `.mlmodelc`

Tried `MLComputePlan.load(contentsOf: url, configuration: cfg)` on macOS 14.5
across all four Magpie models, all three compute units. Every call SIGBUSes
(exit 138). Cannot be used for device-assignment introspection. The Swift CLI
falls back to a timing-based probe:

```
swift run fluidaudiocli magpie compute-plan
```

— loads each model under `.cpuOnly` / `.cpuAndGPU` / `.cpuAndNeuralEngine`,
runs 1 warmup + 3 timed iters, infers ANE usage from the speedup ratio
(>1.3× cpuOnly → ANE active). Hacky but works.

**`coreml-cli` from `tools/coreml-cli/` may have the same issue.** Test
before relying on its `--fallback` analysis for Magpie models. If it
crashes, file a follow-up to check whether it's an Apple issue or a
property of the conversion (e.g. `coremltools` version, MIL ops used).

## Suggested mobius-side follow-ups

1. **`convert_decoder_step.py`**: experiment with rank-3 K/V layout +
   int32 position, validate ANE compile under real position values.
2. **`convert_decoder_prefill.py`**: name the 12 outputs explicitly.
3. **`prepare_hf_upload.py`**: ensure `decoder_prefill.mlmodelc` is always
   included (not gated on availability — Swift port treats it as required).
4. **`generate_coreml.py`**: add a `--dump-intermediates` flag that writes
   the per-stage tensors (`encoder_output`, the 12 prefill K/V outputs,
   per-step decoder hidden, sampled codes, audio) to an `.npz`. Used by the
   Swift `MagpieParityCommand` and `MagpieProbeCommand` for stage-by-stage
   parity. Currently a manual modification each time.
5. **Documentation**: add a "Known Limitations" section to the main README
   noting decoder_step ANE failure and the fp16 host drift.

## Verified Swift performance budget (post-optimization)

Reference: 8-word English sentence, M-series, warm process.

```
text_encoder         12 ms (ANE)        — 1×
decoder_prefill      18 ms (ANE)        — 1×
decoder_step      ~22 ms (GPU/MPS)      — ~80–200× per utterance (this is the AR loop)
nanocodec_decoder   ~50 ms (ANE)        — 1×
LocalTransformer   ~3-5 ms/step (CPU/Accelerate+BNNS)
─────────────────────────────────────
Wall (warm)        ~96 s for ~3 s of audio at 22 kHz
RTFx              ~0.04× (sub-realtime)
```

The bottleneck is the AR loop (`decoder_step` × ~120 + LocalTransformer
sample × 8 codebooks per step). To beat realtime we need either:

- ANE on `decoder_step` (blocked by the compile failure documented above).
- A drastically faster LocalTransformer (MLX backend candidate).
- Speculative decoding / parallel sampling (architectural change).

## File-level cross-reference (Swift side)

| Concern                          | Swift file                                                                              |
| -------------------------------- | --------------------------------------------------------------------------------------- |
| Per-model compute units          | `Sources/FluidAudio/TTS/Magpie/Assets/MagpieModelStore.swift`                           |
| Prefill batched call + KV seed   | `Sources/FluidAudio/TTS/Magpie/Pipeline/Synthesize/MagpiePrefill.swift`, `MagpieKvCache.swift` |
| Stage-by-stage parity probe      | `Sources/FluidAudioCLI/Commands/MagpieProbeCommand.swift`                               |
| Compute-device probe             | `Sources/FluidAudioCLI/Commands/MagpieComputePlanCommand.swift`                         |
| LocalTransformer (Accelerate)    | `Sources/FluidAudio/TTS/Magpie/LocalTransformer/MagpieLocalTransformer.swift`           |
| fp64 LocalTransformer reference  | `Sources/FluidAudio/TTS/Magpie/LocalTransformer/MagpieLocalTransformerDouble.swift`     |
| Documentation                    | `Documentation/TTS/Magpie.md`                                                           |
