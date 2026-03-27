# Sherpa-ONNX Zipformer2 → CoreML

Convert icefall Zipformer2 transducer checkpoints (used by Vosk and sherpa-onnx) to CoreML `.mlpackage` format. Converts from **original PyTorch `.pt` checkpoints**, not ONNX.

## Architecture

Standard RNN-T (transducer) with three components:

| Component | Input | Output |
|-----------|-------|--------|
| **Encoder** | `x` (1, T, 80) mel frames + `x_lens` (1,) | `encoder_out` (1, T', joiner_dim) + `encoder_out_lens` (1,) |
| **Decoder** | `y` (1, context_size) token IDs | `decoder_out` (1, joiner_dim) |
| **Joiner** | `encoder_out` (1, joiner_dim) + `decoder_out` (1, joiner_dim) | `logit` (1, vocab_size) |

Key differences from Parakeet TDT models:
- **Encoder takes mel frames** (80-dim), not raw audio — mel extraction is external
- **Stateless decoder** — embedding + Conv1d over a context window of token IDs (no LSTM)
- **Standard RNNT joiner** — `tanh(enc + dec) → logits`, no duration prediction

## Supported checkpoints

| Model | Checkpoint | Vocab | Causal |
|-------|-----------|-------|--------|
| vosk-model-en-0.62-atc | `epoch-56-avg-4.pt` | 500 BPE | No |

## Usage

```bash
cd models/stt/sherpa-onnx-zipformer/coreml
uv sync
```

### Convert

```bash
uv run python convert-coreml.py \
    --checkpoint /Volumes/hdd/models/vosk/vosk-model-en-0.62-atc/am/epoch-56-avg-4.pt \
    --tokens /Volumes/hdd/models/vosk/vosk-model-en-0.62-atc/lang/tokens.txt \
    --output-dir ./build/vosk-0.62-atc
```

Options:
- `--float16` — export with FP16 precision
- `--compute-units CPU_AND_GPU` — target GPU acceleration
- `--mel-frames 1495` — fixed encoder input size (default: ~15s of audio)

### Validate

```bash
uv run python compare-models.py \
    --checkpoint /Volumes/hdd/models/vosk/vosk-model-en-0.62-atc/am/epoch-56-avg-4.pt \
    --tokens /Volumes/hdd/models/vosk/vosk-model-en-0.62-atc/lang/tokens.txt \
    --coreml-dir ./build/vosk-0.62-atc \
    --audio-file sample_16khz.wav \
    --reference "expected transcription text"
```

Reports cosine similarity, max/mean absolute error for encoder outputs, and compares greedy RNNT transcriptions. Optionally computes WER against a reference.

### Quantize

```bash
uv run python quantize-coreml.py \
    --input-dir ./build/vosk-0.62-atc \
    --output-dir ./build/vosk-0.62-atc-int8
```

Applies int8 per-channel symmetric quantization to all components.

## Output structure

```
build/vosk-0.62-atc/
  encoder.mlpackage      # Zipformer2 encoder (mel → features)
  decoder.mlpackage      # Stateless prediction network
  joiner.mlpackage       # Joint network (enc + dec → logits)
  vocab.json             # BPE vocabulary (index = token ID)
  metadata.json          # Model configuration
```

## Mel spectrogram

The encoder expects 80-dim log-mel filterbank features (kaldi-compatible):
- Sample rate: 16 kHz
- Window: 25 ms (400 samples), hop: 10 ms (160 samples)
- Povey window, no dithering

See `mel.py` for a reference implementation using `torchaudio.compliance.kaldi.fbank`.

## Decoding

Standard greedy RNNT: step through encoder frames, query joiner with current decoder state, emit token if not blank, advance. See `rnnt_decode.py` for reference.

## Upstream

- Training: [icefall](https://github.com/k2-fsa/icefall) (k2/lhotse)
- Inference: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- Models: [Vosk](https://alphacephei.com/vosk/models)
