
  # Sortformer CoreML Models

  Streaming speaker diarization models converted from NVIDIA's Sortformer to CoreML for Apple Silicon.

  ## Model Variants

  | Variant | File | Latency | Use Case |
  |---------|------|---------|----------|
  | **Default** | `Sortformer.mlmodelc` | ~1.04s | Low latency streaming |
  | **NVIDIA Low** | `SortformerNvidiaLow.mlmodelc` | ~1.04s | Low latency streaming |
  | **NVIDIA High** | `SortformerNvidiaHigh.mlmodelc` | ~30.4s | Best quality, offline |

  ## Configuration Parameters

  | Parameter | Default | NVIDIA Low | NVIDIA High |
  |-----------|---------|------------|-------------|
  | chunk_len | 6 | 6 | 340 |
  | chunk_right_context | 7 | 7 | 40 |
  | chunk_left_context | 1 | 1 | 1 |
  | fifo_len | 40 | 188 | 40 |
  | spkcache_len | 188 | 188 | 188 |

  ## Model Input/Output Shapes

  **General**:

  | Input | Shape | Description |
  |-------|-------|-------------|
  | chunk | `[1, 8*(C+L+R), 128]` | Mel spectrogram features |
  | chunk_lengths | `[1]` | Actual chunk length |
  | spkcache | `[1, S, 512]` | Speaker cache embeddings |
  | spkcache_lengths | `[1]` | Actual cache length |
  | fifo | `[1, F, 512]` | FIFO queue embeddings |
  | fifo_lengths | `[1]` | Actual FIFO length |

  | Output | Shape | Description |
  |--------|-------|-------------|
  | speaker_preds | `[C+L+R+S+F, 4]` | Speaker probabilities (4 speakers) |
  | chunk_pre_encoder_embs | `[C+L+R, 512]` | Embeddings for state update |
  | chunk_pre_encoder_lengths | `[1]` | Actual embedding count |
  | nest_encoder_embs | `[C+L+R+S+F, 192]` | Embeddings for speaker discrimination |
  | nest_encoder_lengths | `[1]` | Actual speaker embedding count |

  Note: `C = chunk_len`, `L = chunk_left_context`, `R = chunk_right_context`, `S = spkcache_len`, `F = fifo_len`.

  **Configuration-Specific Shapes**:
  
  | Input | Default | NVIDIA Low | NVIDIA High |
  |-------|---------|------------|-------------|
  | chunk | `[1, 112, 128]` | `[1, 112, 128]` | `[1, 3048, 128]` |
  | chunk_lengths | `[1]` | `[1]` | `[1]` |
  | spkcache | `[1, 188, 512]` | `[1, 188, 512]` | `[1, 188, 512]` |
  | spkcache_lengths | `[1]` | `[1]` | `[1]` |
  | fifo | `[1, 40, 512]` | `[1, 188, 512]` | `[1, 40, 512]`
  | fifo_lengths | `[1]` | `[1]` | `[1]` |
  
  | Output | Default | NVIDIA Low | NVIDIA High |
  |--------|---------|------------|-------------|
  | speaker_preds | `[1, 242, 128]` | `[1, 390, 128]` | `[1, 609, 128]` |
  | chunk_pre_encoder_embs | `[1, 14, 512]` | `[1, 14, 512]` | `[1, 381, 512]` |
  | chunk_pre_encoder_lengths | `[1]` | `[1]` | `[1]` |
  | nest_encoder_embs | `[1, 242, 192]` | `[1, 390, 192]` | `[1, 609, 192]` |
  | nest_encoder_lengths | `[1]` | `[1]` | `[1]` |

  
  | Metric        | Default | NVIDIA High |
  |---------------|---------|-------------|
  | Latency       | ~1.12s  | ~30.4s      |
  | RTFx (M4 Max) | ~5.7x   | ~125.3x     |

  ## Usage with FluidAudio (Swift)

  ```swift
  import FluidAudio

  // Initialize with default config (auto-downloads from HuggingFace)
  let diarizer = SortformerDiarizer(config: .default)
  let models = try await SortformerModels.loadFromHuggingFace(config: .default)
  diarizer.initialize(models: models)

  // Streaming processing
  for audioChunk in audioStream {
      if let result = try diarizer.processSamples(audioChunk) {
          for frame in 0..<result.frameCount {
              for speaker in 0..<4 {
                  let prob = result.getSpeakerPrediction(speaker: speaker, frame: frame)
              }
          }
      }
  }

  // Or batch processing
  let timeline = try diarizer.processComplete(audioSamples)
  for (speakerIndex, segments) in timeline.segments.enumerated() {
      for segment in segments {
          print("Speaker \(speakerIndex): \(segment.startTime)s - \(segment.endTime)s")
      }
  }
```
  Performance

https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md

  Files

  Models

  - Sortformer.mlpackage / .mlmodelc - Default config (low latency)
  - SortformerNvidiaLow.mlpackage / .mlmodelc - NVIDIA low latency config
  - SortformerNvidiaHigh.mlpackage / .mlmodelc - NVIDIA high latency config

  Scripts

  - convert_to_coreml.py - PyTorch to CoreML conversion
  - streaming_inference.py - Python streaming inference example
  - mic_inference.py - Real-time microphone demo

  Source

  Original model: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1

  Credits & Acknowledgements

  This project would not have been possible without the significant technical contributions of https://huggingface.co/GradientDescent2718.

  Their work was instrumental in:

  - Architecture Conversion: Developing the complex PyTorch-to-CoreML conversion pipeline for the 17-layer Fast-Conformer and 18-layer Transformer heads.
  - Build & Optimization: Engineering the static shape configurations that allow the model to achieve ~120x RTF on Apple Silicon.
  - Logic Implementation: Porting the critical streaming state logic (speaker cache and FIFO management) to ensure consistent speaker identity tracking.

  This project was built upon the foundational work of the NVIDIA NeMo team.


## WebGPU/WASM port knowledge (browser)

Sortformer 4-spk v2.1 runs fully in-browser in
[fluidaudio-web](https://github.com/FluidInference/fluidaudio-web)
(`src/engines/diarization-sortformer/`) — ORT-free: the shared WGSL
FastConformer encoder (int8, 17 layers, d512, symmetric Parakeet-style
subsampling, full attention offline) → Sortformer head (encoder_proj →
18-layer transformer → single_hidden_to_spks → sigmoid) → per-frame
4-speaker probs (~12.5 fps) → threshold/merge into segments. mel =
per-feature CMVN. Parity vs ORT: fp32 maxΔ 1.79e-7, int8 2.3e-3. Weights:
`FluidInference/fluidaudio-web` → `sortformer/`.

Load-bearing facts:

- **Offline single-chunk (EMPTY spkcache/fifo) only works for short audio.**
  On long meetings it collapses to the dominant speaker (secondary speakers
  → 0.000) — the spkcache/fifo streaming state loop is NOT optional there.
  Valid long-form DER requires porting NeMo's streaming state update.
- **NeMo's own ONNX export of this model has dynamic-slice issues**;
  community mirrors (`cgus/diar_streaming_sortformer_4spk-v2.1-onnx`) work.
- **NeMo xscaling (√512) is a runtime Mul in this export**, not folded into
  weights — easy to drop when porting from the graph.
- The fp32 model is CPU-runnable (not precision-gated like the int-quant ASR
  encoders) → fully verifiable headless before touching a browser.
