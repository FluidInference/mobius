# Parakeet-TDT-CTC 110M — CoreML Export

This directory contains tools to export NVIDIA's `nvidia/parakeet-tdt_ctc-110m` hybrid RNNT/CTC model to CoreML.

The hybrid model has two decoder heads sharing one encoder:
- **TDT (Token Duration Transducer)**: Primary transcription head with duration prediction — used by FluidAudio for ASR
- **CTC**: Auxiliary head for keyword spotting and custom vocabulary support

## Layout

```text
mobius/models/stt/parakeet-tdt-ctc-110m/coreml
├── README.md                  # This file
├── convert-tdt-coreml.py      # TDT export: fused mel+encoder, RNNT decoder, joint decision
├── convert-coreml.py          # CTC export: fused mel+encoder+CTC head (for keyword spotting)
├── individual_components.py   # Shared torch.nn.Module wrappers for CoreML tracing
├── pyproject.toml             # Per-target Python environment (NeMo + coremltools)
└── audio/                     # Optional trace audio (15s 16kHz) for export
```

## Environment

From this directory:

```bash
uv sync
```

This will create/update a local environment pinned by `pyproject.toml` and `uv.lock` (Python 3.10.12, NeMo, coremltools, etc.).

## Export TDT (for FluidAudio ASR)

The `convert-tdt-coreml.py` script exports the TDT components used by FluidAudio:

```text
Preprocessor.mlpackage     — fused waveform → mel → encoder features
Decoder.mlpackage          — RNNT prediction network (LSTM)
JointDecision.mlpackage    — joint network (full T×U grid, with TDT duration)
JointDecisionSingleStep.mlpackage — single-step joint (for streaming)
vocab.json                 — SentencePiece vocabulary (array format)
metadata.json              — model dimensions and export configuration
```

Usage:

```bash
# From pretrained (downloads from HuggingFace)
uv run python convert-tdt-coreml.py \
  --output-dir parakeet_tdt_coreml \
  --audio-path audio/trace_15s.wav

# From local .nemo checkpoint
uv run python convert-tdt-coreml.py \
  --nemo-path ./parakeet-tdt-ctc-110m.nemo \
  --output-dir parakeet_tdt_coreml \
  --audio-path audio/trace_15s.wav

# Reuse a previously exported mel+encoder
uv run python convert-tdt-coreml.py \
  --reuse-encoder parakeet_tdt_coreml/Preprocessor.mlpackage \
  --output-dir parakeet_tdt_coreml_v2
```

Key differences from the 0.6B export:
- **Fused frontend**: mel spectrogram + encoder are a single `Preprocessor.mlpackage` (0.6B has separate Preprocessor + Encoder)
- **iOS 18 deployment target**: Required for int ops in the encoder's positional encoding
- **Smaller dimensions**: encoderDim=512, decoderHidden=640, decoderLayers=1, vocabSize=1024

### Using with FluidAudio

After export, compile the `.mlpackage` files to `.mlmodelc`:

```bash
xcrun coremlcompiler compile Preprocessor.mlpackage output_dir/
xcrun coremlcompiler compile Decoder.mlpackage output_dir/
xcrun coremlcompiler compile JointDecisionSingleStep.mlpackage output_dir/
# Rename to match FluidAudio's expected name
mv output_dir/JointDecisionSingleStep.mlmodelc output_dir/JointDecision.mlmodelc
cp vocab.json output_dir/
```

Then run with:

```bash
fluidaudiocli transcribe audio.wav --model-version tdt-ctc-110m --model-dir output_dir/
```

## Export CTC (for keyword spotting)

The `convert-coreml.py` script exports the CTC branch for keyword spotting:

```bash
uv run python convert-coreml.py convert \
  --model-id nvidia/parakeet-tdt_ctc-110m \
  --output-dir parakeet_ctc_coreml
```

This produces:
- `parakeet_ctc_mel_encoder.mlpackage` — waveform → encoder features
- `parakeet_ctc_decoder.mlpackage` — encoder → CTC log-probabilities

Note: The CTC head is trained as an auxiliary loss and produces blank-dominant outputs. It is not suitable for standalone greedy transcription — use the TDT export for that.
