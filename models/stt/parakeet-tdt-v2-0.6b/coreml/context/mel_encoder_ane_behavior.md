# MelEncoder ANE behavior and iPhone 13 issue

Summary of the failure seen on iPhone 13 (A15 / E5):

> ANE model load has failed for on-device compiled macho. Must re-compile the E5 bundle. Invalid layer: Tensor dimensions N1D1C1H1W240000 are not within supported range, N[1-65536] D[1-16384] C[1-65536] H[1-16384] W[1-16384].

## Root cause

- The fused MelEncoder model takes raw waveform input shaped `[1, 240000]` for the fixed 15 s window (16 kHz × 15 s).
- In Core ML’s internal layout this maps to `N1 D1 C1 H1 W240000`.
- Apple Neural Engine (ANE) enforces a per-dimension cap of ≤ 16384 for H and W. `W=240000` violates this, so any ANE partition that “sees” the waveform will fail validation on A15 (and other iPhone ANE generations).
- The failure is caused by the preprocessor stage inside the fused MelEncoder (waveform → STFT/mel). The standalone encoder (mel → encoder) uses inputs around `[1, 128, 1501]` and is ANE-safe.

## Why earlier models “worked”

- Prior exports either:
  - Avoided fusing the waveform preprocessor with the encoder; or
  - Exported a “mel→encoder” model only; or
  - Relied on Core ML partitioning that kept the preprocessor on CPU/GPU while the encoder ran on ANE.
- Even with `computeUnits = .cpuAndNeuralEngine`, Core ML will fall back to CPU for non-ANE-eligible subgraphs. With split models, the preprocessor silently ran on CPU and the encoder on ANE.
- With the new fused MelEncoder, the compiler tries to start an ANE partition earlier (due to the encoder being ANE-friendly). That exposes the 240000-wide waveform to the ANE plan and triggers the dimension error.

## Device differences (M‑series Macs vs iPhones)

- Inputs are identical across devices; what differs is how Core ML partitions the graph.
- On M‑series Macs, the waveform preprocessor commonly stays on CPU/GPU; the ANE (if present) or GPU is used for the encoder. No ANE error is logged.
- On iPhones (A14/A15/A16/A17), the ANE compiler (E-series) may attempt to include earlier elementwise ops, making the NE subgraph input be the waveform itself, which fails validation with `W>16384`.
- Treat this limitation as universal across iPhones with ANE; the precise logging/behavior can vary, but no iPhone ANE can accept a subgraph that sees `W=240000`.

## Current repo behavior and fix

- The exporter now sets CPU+GPU compute units for waveform‑in components to avoid ANE attempts while preserving the 15 s window:
  - `parakeet_preprocessor.mlpackage`
  - `parakeet_mel_encoder.mlpackage` (fused waveform→mel→encoder)
- See the changes in `convert-parakeet.py`:
  - `convert-parakeet.py:225` and `convert-parakeet.py:237` — preprocessor exported with `ct.ComputeUnit.CPU_AND_GPU`.
  - `convert-parakeet.py:289` and `convert-parakeet.py:301` — fused mel+encoder exported with `ct.ComputeUnit.CPU_AND_GPU`.

## Recommended runtime settings (iOS)

- Preprocessor (waveform→mel): set `MLModelConfiguration.computeUnits = .cpuAndGPU` to skip ANE attempts and use CPU/GPU efficiently.
- Encoder / Decoder / Joint: set `computeUnits = .cpuAndNeuralEngine` (or `.all`) to leverage ANE.
- Using `.cpuAndNeuralEngine` for all models also works with the split setup (preprocessor falls back to CPU), but setting `.cpuAndGPU` on the preprocessor avoids ANE compile warnings and extra load-time overhead.

## Using the split models (recommended)

- Pipeline:
  1. `parakeet_preprocessor.mlmodelc` (CPU+GPU): input `audio_signal [1, 240000]`, `audio_length [1]` → output `mel [1, 128, 1501]`, `mel_length [1]`.
  2. `parakeet_encoder.mlmodelc` (ANE): input `mel [1, 128, 1501]`, `mel_length [1]` → output `encoder [1, 1024, 188]`, `encoder_length [1]`.
  3. Decoder / Joint / Decision: keep on ANE.
- Shapes are captured in `parakeet_coreml/metadata.json`.

## If fused‑on‑ANE is required

- The current fused API (`[1, 240000]` waveform input) cannot compile to ANE on iPhones due to the hard ≤16384 per‑dimension cap.
- Viable approach: introduce a chunked fused variant with input like `[1, 15, 16000]` and implement overlap‑aware preemphasis + STFT + mel inside the model to preserve exact parity with global STFT (center padding, hop=160, win=400). Ensure no op sees a dimension >16384. This changes the model API and requires careful seam handling.
- Alternative: stream ≤1.024 s windows with state and accumulate outputs, but this changes runtime control flow.

## FAQ

- Q: Which model causes the iPhone 13 error?
  - A: The preprocessor stage inside the fused MelEncoder (not the standalone encoder).
- Q: Is this only iPhone 13?
  - A: No. Treat it as affecting all iPhone ANE generations; some may just silently partition or fall back differently.
- Q: Why does M4 work?
  - A: Likely because the preprocessor subgraph runs on CPU/GPU; the ANE never “sees” `W=240000` on macOS.

