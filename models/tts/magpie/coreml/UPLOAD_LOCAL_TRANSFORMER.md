# HuggingFace Upload — `local_transformer` (Magpie LT fusion)

Optional CoreML asset that replaces the Swift `MagpieLocalSampler` on the AR
hot path. **Standalone** — does not modify any existing Magpie model.

## What to upload

Both the source mlpackage and the compiled mlmodelc go to
[`FluidInference/magpie-tts-multilingual-357m-coreml`](https://huggingface.co/FluidInference/magpie-tts-multilingual-357m-coreml)
at the **repo root** (alongside `decoder_step.mlmodelc`,
`decoder_prefill.mlmodelc`, etc.).

| Path | Size | Notes |
|---|---|---|
| `mobius/models/tts/magpie/coreml/build/local_transformer.mlpackage` | 17 MB | source artifact, optional |
| `mobius/models/tts/magpie/coreml/compiled/build/local_transformer.mlmodelc` | 17 MB | **runtime asset** — required for the fast path |

The Swift downloader in
`Sources/FluidAudio/TTS/Magpie/Assets/MagpieResourceDownloader.swift`
fetches the `local_transformer.mlmodelc` subdirectory:

```swift
try await DownloadUtils.downloadSubdirectory(
    .magpieTts,
    subdirectory: ModelNames.Magpie.localTransformerFile,  // "local_transformer.mlmodelc"
    to: repoDir
)
```

So the HF tree must contain `local_transformer.mlmodelc/` as a directory
with `model.mil`, `weights/`, `metadata.json`, `coremldata.bin` inside.

## Suggested commit message on HF

```
feat: add local_transformer.mlmodelc (LT fusion sampler)

Standalone CoreML graph that takes decoder_step's 1x768 hidden state and
emits 8 codebook ints in a single ANE dispatch. Replaces 8 sequential Swift
LT passes + per-codebook top-k softmax + categorical sampling.

- Inputs:  decoder_hidden (1, 768) fp32, uniforms (8,) fp32,
           forbid_eos (1,) fp32, temperature (1,) fp32
- Output:  codes (8,) int32

Optional asset: when absent, FluidAudio falls back to MagpieLocalSampler.
ANE residency: 73.9%; per-step latency: 1.78 ms on M2.

End-to-end TTFA win on M2 release build, seed 42, "Hello from Magpie.":
1.283 s -> 1.122 s, -161 ms (-12.5 %).

CFG (cfgScale != 1.0) is not implemented in this graph; the synthesizer
keeps using the Swift sampler in that case.
```

## Reproduce the artifact

```bash
cd mobius/models/tts/magpie/coreml
uv run python convert_local_transformer.py \
  --constants-dir ~/.cache/fluidaudio/Models/magpie-tts/constants \
  --output-mlpackage build/local_transformer.mlpackage \
  --output-mlmodelc compiled/build/local_transformer.mlmodelc
```

The script reads LT weights from `constants/local_transformer/` (already in
the HF repo), traces the unrolled 8-codebook sampling graph, and emits both
artifacts. It uses fp32 (`ct.precision.FLOAT32`) — fp16 was tried and
introduced bias in the cumsum-vs-uniform compare path.

## Sanity check before upload

```bash
# Verify mlmodelc loads + produces a valid 8-element int32 output
cd ../../../tools/coreml-cli
uv run coreml-cli ../../models/tts/magpie/coreml/compiled/build/local_transformer.mlmodelc
```

Expect: ANE residency around 70-75 %, per-step latency < 2 ms,
cpu_and_neural_engine config preferred.

## End-to-end audio test (post-upload)

Once the asset is on HF, blow away the local cache and let the downloader
fetch it:

```bash
rm -rf ~/.cache/fluidaudio/Models/magpie-tts/local_transformer.mlmodelc
swift run -c release fluidaudiocli magpie text \
  --text "Hello from Magpie." --stream --speaker 0 --seed 42 \
  --output /tmp/lt_fused.wav
# logs should contain: "Using fused local_transformer CoreML sampler (ANE path)"
```

## Rollback

If a regression is reported, the Swift fallback is automatic — users can
delete `~/.cache/fluidaudio/Models/magpie-tts/local_transformer.mlmodelc`
(or the upstream HF repo can revert the upload) and FluidAudio will log
"Optional model local_transformer.mlmodelc not present; skipping" and fall
back to `MagpieLocalSampler`.
