# StyleTTS2 ANE re-cut — blocker rationale

7-graph CoreML re-cut that mirrors Kokoro-ANE's shape (PLBert → PostBert
→ Alignment → DiffusionStep → Prosody → Noise → Vocoder). 6 of 7 graphs
pin to the Neural Engine; only Noise stays on `.all` (fp32 SineGen phase).

This file is the canonical home for the conversion-side blockers and how
each one was resolved. The Swift consumer (FluidAudio's
`Sources/FluidAudio/TTS/StyleTTS2/`) does not duplicate this rationale.

## Five blockers

### 1. EnumeratedShapes triggers the `E5RT FlexibleShapeInfo` bug

**Symptom.** Compiling `f0n_energy` with `coremltools.EnumeratedShapes`
loaded fine but threw at predict-time:

```
E5RT: tensor_buffer has known strides while the model has FlexibleShapeInfo
```

In the legacy 4-graph backend this forced `f0n_energy` to pin to
`.cpuOnly`, defeating the whole point of converting it.

**Fix.** Drop EnumeratedShapes for the new Prosody / Noise / Vocoder
stages. Bake a single fixed shape at `T_a_max = 2000` and pad the input
on the Swift side. Same trick Kokoro-ANE uses.

### 2. BiLSTM is not ANE-native

**Symptom.** Tracing the PostBert stack with an off-the-shelf
`nn.LSTM(bidirectional=True)` makes Core ML drop the layer to
`.cpuAndGPU`, killing ANE residency for the entire downstream graph.

**Fix.** Use the LSTM-unroll trick from Kokoro's
`models/tts/kokoro-v1.0/coreml/convert-coreml.py:272-301`
(`CoreMLDurationEncoder`) — replace the BiLSTM with two unidirectional
LSTMs run forward + reversed in Python, concatenated. Output is bit-exact
to the BiLSTM and ANE compiles it.

### 3. Diffusion attention einsum with leading ellipsis

**Symptom.** The diffusion UNet's cross-attention used
`einsum("...nd,...md->...nm", q, k)`. CoreML's einsum lowering chokes on
the leading ellipsis when the rank is dynamic.

**Fix.** Patched in `scripts/_styletts2_lib.py:235-256` (carries over from
the legacy 4-graph backend). Rewrite to explicit `q @ k.transpose(-1,
-2)` with the rank made static.

### 4. SineGen phase saturation in fp16

**Symptom.** HiFi-GAN's SineGen accumulates harmonic phase via `cumsum ×
2π × hop=300`, reaching magnitudes around 4000 mid-frame. In fp16 the
accumulator collapses to a few discrete values; output is robotic /
metallic. See `coreml/PHASE6_FP16_DECODER.md` and `coreml/PRECISION.md`
for the failed fp16 decoder trial that motivated this split.

**Fix.** Keep SineGen in its own fp32 graph (Stage 6, `noise.mlmodelc`,
compute units `.all`). The Vocoder stage consumes the downconverted fp16
harmonic / noise outputs. This is the same split Kokoro-ANE uses for the
exact same reason.

### 5. Snake activation is not ANE-native

**Symptom.** `AdaINResBlock1`'s Snake activation
(`x + (1/α) sin²(αx)`) uses `sin` + division, both of which Core ML
rejects on ANE — the layer falls back to CPU.

**Fix.** Apply the cos-identity patch from Kokoro's
`models/tts/kokoro-v1.0/coreml/convert-coreml.py:40-52`
(`AdaINResBlock1.forward = _cos_resblock1_forward`), which rewrites to
`(1 - cos(2αx)) / (2α)`. Numerically identical, ANE compiles cleanly.

## Acceptance gate

End-to-end log-mel cosine vs. the legacy 4-graph reference must hit
≥ 0.99 in `99_e2e_validate.py` before a 7-graph bundle ships.

## Related

- `coreml/PRECISION.md` — mixed-precision recipe (mostly legacy 4-graph,
  but the SineGen fp32 reasoning carries over verbatim).
- `coreml/PHASE6_FP16_DECODER.md` — the original fp16-decoder trial that
  surfaced the SineGen phase issue.
- `coreml/TRIALS.md` — chronological log of all conversion trials.
- `models/tts/kokoro-v1.0/coreml/convert-coreml.py` — source of the
  BiLSTM unroll and Snake cos-identity patches.
