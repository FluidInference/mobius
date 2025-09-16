# Silero VAD CoreML Reference (Techniques to Reuse)

This note distills patterns from `/models/vad/silero-vad/coreml` that we can reuse for Parakeet-TDT and highlights CoreML limits to plan around.

## Tooling Patterns Worth Copying

- **Dedicated CLI entry points**: `convert-coreml.py` performs all exports, while `compare-models.py` focuses solely on validation/plotting. Keeping convert vs. compare independent shortens iteration when only one side changes.
- **Module wrappers**: Lightweight `nn.Module` wrappers (`STFTModel`, `EncoderModel`, `DecoderModel`) map state-dict weights into export-friendly graphs. They surface state tensors explicitly (hidden/cell) and clamp activations where CoreML would otherwise overflow.
- **Shape contracts via CoreML types**: Inputs/outputs use `ct.TensorType` to document tensor layouts. Silero leans on `ct.RangeDim` for flexible windows; for Parakeet we can simplify by locking to the fixed 15 s window (240 k samples → 1501 mels → 188 encoder frames).
- **Metadata & publish scripts**: Conversion annotates author/version on each mlpackage and reports disk size. `compile_modelc.py` shows how to mass-compile mlpackages to `.mlmodelc` bundles via `xcrun`—handy when preparing release artifacts.
- **Multiple deployment variants**: The project exports both modular (STFT→Encoder→Decoder) and unified pipelines. The 256 ms unified variant demonstrates how to encode fixed iteration counts inside the traced graph (8 chunks + noisy-OR aggregation) for batch workflows.

## Techniques Directly Applicable to Parakeet-TDT

- **Separate convert & compare CLIs**
  - `convert_parakeet_components.py` should mirror the Silero convert script: accept output directory, compute-unit/precision knobs, optional fused graph exports (e.g., preprocessor+encoder).
  - A new `compare_parakeet_models.py` can follow the VAD wrapper pattern: run NeMo vs. CoreML components, collect per-layer diffs, and optionally chart mel/encoder/token probabilities over time.
- **Wrapper modules per component**
  - We already have `PreprocessorWrapper`, `EncoderWrapper`, `DecoderWrapper`, `JointWrapper`; extend them with deterministic input padding/casts just like Silero’s wrappers to avoid CoreML runtime surprises.
- **Fixed-window CoreML shapes**
  - Keep CoreML inputs fixed to the 15 s window. If we ever change the window length, re-export the models instead of relying on dynamic shape bounds.
- **Explicit state I/O**
  - Follow Silero’s practice of exposing LSTM states as inputs/outputs every call. For Parakeet, that means separate `h`/`c` arrays of shape `[num_layers, batch, 640]`, enabling streaming decode loops outside CoreML.
- **Release automation**
  - Borrow `compile_modelc.py` to generate `.mlmodelc`; consider a similar helper for quantization experiments once baseline parity is achieved.

## Parakeet Components to Export

1. **Audio front-end (Mel)** – same strategy as Silero STFT: wrap the NeMo preprocessor, trace with fixed 15 s audio (240 k samples), and export with constant shapes.
2. **Encoder (FastConformer)** – export with constant shapes for the 188 encoder frames corresponding to the 15 s window (transpose to time-major for CoreML compatibility).
3. **Decoder (RNNT Prediction Network)** – stateful LSTM; replicate Silero’s explicit state tensors but with `[2, 1, 640]` dims and integer targets. Provide zero-state helper for CoreML consumers.
4. **Joint network + TDT head** – trace head-only module; outputs `[B, T_enc, U, V+1+5]`. Document token vs. duration slices (see `tdt_decoding_notes.md`).
5. **Fused graphs** – export both:
   - `mel_encoder` (preprocessor+encoder) to reduce I/O.
   - `joint_decision` (joint + split/softmax/argmax) to avoid host-side post-processing on logits and return `token_id`, `token_prob`, and `duration` directly.

## Likely CoreML Pain Points

- **Joint tensor size**: At 15 s the joint produces ~188×U×(8192+6) floats. Even at `U=256`, that is >400 MB of fp32 activations. Plan for fp16 exports and/or tiled decoding (call joint per-step rather than full grid).
- **Dynamic loops**: RNNT greedy/beam search loops can’t be represented in a single CoreML graph (no `while` support). Expect to orchestrate decoding in Swift/Python by issuing repeated `decoder` and `joint` calls.
- **Large vocabulary softmax**: Full RNNT softmax over 8193 classes is CoreML-friendly but expensive. Avoid embedding beam-search logic in CoreML; keep it host-side.
- **Duration logits**: Splitting the final 5 logits inside CoreML is trivial, but combining them with token control flow should stay in host code (mirrors Silero’s noisy-OR aggregation done in Python before tracing).
- **Tokenizer & text post-processing**: SentencePiece decoding, punctuation fixes, and timestamp formatting remain in Python/Swift; CoreML returns logits only.
- **Feature normalization stats**: Stick to constant statistics baked into the wrapper as Silero does for STFT weights; do not rely on CoreML `LayerNormalization` with dynamic reductions over time.

## Python / Host Responsibilities

- Audio chunking, state caching, and greedy/beam decoding loops (leveraging exported decoder + joint).
- Token → text conversion (`sentencepiece`), duration bucket interpretation, punctuation/TDT alignment heuristics.
- Validation tooling (diff calculations, plotting) analogous to Silero’s `compare-models.py`.
- Optional post-processing (e.g., noise suppression, segmentation) that require dynamic control flow.

Reuse these patterns to keep Parakeet’s conversion pipeline maintainable while respecting CoreML’s graph restrictions.
