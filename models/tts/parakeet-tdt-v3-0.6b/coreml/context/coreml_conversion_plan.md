# CoreML Conversion Plan (Parakeet‑TDT‑0.6B‑v3)

This plan describes how we export Parakeet’s sub‑modules to CoreML, validate numerics, and prepare for on‑device decoding. The pipeline keeps a fixed 15‑second audio window.

## Goals

- Export preprocessor, encoder, decoder, and joint as separate mlpackages.
- Also export fused variants: `mel_encoder` (preprocessor+encoder) and `joint_decision` (joint + split/softmax/argmax).
- Preserve streaming decoder state I/O for on‑device greedy decoding.
- Validate component outputs against the NeMo reference on a 15‑second clip.

## Environment

- Use `uv venv` and run via `uv run` to ensure reproducible resolutions.
- Python 3.10.12, `torch 2.5.0`, `coremltools 8.3.0`, `nemo-toolkit 2.3.1`.

## Export settings

- `convert_to = "mlprogram"`
- `minimum_deployment_target` = `ct.target.iOS17` (we only target iOS 17+ for the CoreML deployment)
- `compute_units` = `.CPU_AND_GPU` (or `.ALL` on iOS)
- `compute_precision` = `ct.precision.FLOAT16` or `None` (start with fp32 for validation, then try fp16)
- Fixed window: 15 seconds at 16 kHz → shapes as in `context/coreml_component_io.md`. No variable-length support is required; CoreML graphs can assume the 15 s window.

## Steps

- Load the NeMo checkpoint with `ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")`.
- Extract modules: preprocessor, encoder, decoder, joint.
- Wrap modules with `PreprocessorWrapper`, `EncoderWrapper`, `DecoderWrapper`, `JointWrapper`; derive `MelEncoderWrapper` and `JointDecisionWrapper` for fused exports.
- Trace each wrapper with static 15‑second inputs (batch=1). Ensure outputs match I/O contracts in `coreml_component_io.md`.
- Define CoreML inputs with explicit names and fixed shapes (15 s audio → mel 1501 → encoder 188). Keeping the window fixed simplifies CoreML deployment; re‑export if window size changes.
- Convert via `ct.convert(...)` with the settings above.
- Save each mlpackage under the chosen `output_dir`, along with a `metadata.json` capturing shapes, sample rate, vocab size, tokenizer path, and export metadata (author, conversion commit, etc.).
- Emit fused graphs (`parakeet_mel_encoder.mlpackage` and `parakeet_joint_decision.mlpackage`) alongside standalone components to reduce host I/O and post‑processing.
- Provide a helper like `compile_modelc.py` to batch `xcrun coremlcompiler` invocations when packaging for release.

## CLI layout

- `convert_parakeet_components.py` → conversion-only entry point (no validation), parameterized by compute units, precision, and optional fused exports. Model inspection (encoder strides, etc.) should auto-populate config, inspired by Silero VAD’s converter.
- `compare_parakeet_models.py` → validation/plotting script that loads reference NeMo modules and CoreML packages, runs diffs per component, and reports max/mean errors plus RTF metrics. Use lightweight wrappers similar to Silero’s `PyTorchJITWrapper` / `CoreMLVADWrapper`.
- Future helpers: `compile_modelc.py` (reuse pattern from VAD project) and optional quantization experiments once parity is established.

## Validation

- Run inference on a known 16 kHz clip (trim/pad to 15 s):
  - Compare preprocessor mel and lengths (max abs/rel diff thresholds: atol=1e-4, rtol=1e-3).
  - Compare encoder outputs and lengths.
  - Compare decoder outputs for a fixed target sequence and initial zero states.
  - Compare joint logits on the same `(T_enc, U)` grid; split `[V+1]` token logits from the last 5 duration logits when computing metrics.
- Record per‑component diffs in `metadata.json` for auditability.

## Decoding (device)

- Implement greedy RNNT in app code calling CoreML:
  - Either: use modular `joint` and perform split/softmax/argmax on host.
  - Or: call fused `joint_decision` to receive `token_id`, `token_prob`, and `duration` directly.
  - Preprocess → Encode once per window (or call fused `mel_encoder`).
  - Maintain `(h, c)` across symbol steps.
  - Blank handling: `token_id == V` indicates blank.

## Known caveats

- RNNT joint logits are large: `[188 × U × ~8200]` per window; consider fp16 and/or tiling over `U` to reduce memory.
- Length maths: mel frames use `center=True`, yielding 1501 frames for exactly 240,000 samples; encoder length computed via ceil divide by 8.
- For accurate timestamping, use the original NeMo decoder on server to validate any device‑side greedy implementation.
