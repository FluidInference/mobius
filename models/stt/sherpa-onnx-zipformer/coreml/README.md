# Sherpa-ONNX Zipformer2 → CoreML

Convert icefall Zipformer2 transducer checkpoints (used by Vosk and sherpa-onnx) to CoreML `.mlpackage` format. Converts from **original PyTorch `.pt` checkpoints**, not ONNX.

## Architecture

Standard RNN-T (transducer) with three components:

| Component | Input | Output |
|-----------|-------|--------|
| **Preprocessor** (fused) | `audio_signal` (1, 239120) + `audio_length` (1,) | `encoder_out` (1, T', joiner_dim) + `encoder_out_lens` (1,) |
| **Decoder** | `y` (1, context_size) token IDs | `decoder_out` (1, joiner_dim) |
| **Joiner** | `encoder_out` (1, joiner_dim) + `decoder_out` (1, joiner_dim) | `logit` (1, vocab_size) |

Key differences from Parakeet TDT models:
- **Fused preprocessor** — kaldi fbank mel extraction + Zipformer2 encoder in one model, same `audio_signal` interface as Parakeet
- **Stateless decoder** — embedding + Conv1d over a context window of token IDs (no LSTM)
- **Standard RNNT joiner** — `tanh(enc + dec) → logits`, no duration prediction
- **blank_id = 0** (not 1024/8192)

## Supported checkpoints

Any icefall Zipformer2 transducer checkpoint (`.pt` with `model_avg` or `model` state dict). The model config (encoder_dim, num_layers, etc.) is read from the checkpoint metadata.

## Usage

```bash
cd models/stt/sherpa-onnx-zipformer/coreml
uv sync
```

### Convert (fused, recommended)

```bash
uv run python convert-coreml.py \
    --checkpoint /path/to/epoch-N-avg-M.pt \
    --tokens /path/to/tokens.txt \
    --output-dir ./build/my-model
```

This produces a `Preprocessor.mlpackage` that takes raw 16kHz audio — compatible with FluidAudio's `AsrModels.loadZipformer2(from:)`.

Options:
- `--float16` — export with FP16 precision (halves model size)
- `--compute-units CPU_AND_GPU` — target GPU acceleration
- `--no-fuse-mel` — export standalone encoder taking mel frames (for debugging)
- `--mel-frames 1495` — fixed encoder input size for `--no-fuse-mel` mode

### Validate

```bash
uv run python compare-models.py \
    --checkpoint /path/to/epoch-N-avg-M.pt \
    --tokens /path/to/tokens.txt \
    --coreml-dir ./build/my-model \
    --audio-file sample_16khz.wav \
    --reference "expected transcription text"
```

Reports cosine similarity, max/mean absolute error for encoder outputs, and compares greedy RNNT transcriptions. Optionally computes WER against a reference.

### Quantize

```bash
uv run python quantize-coreml.py \
    --input-dir ./build/my-model \
    --output-dir ./build/my-model-int8
```

Applies int8 per-channel symmetric quantization to all components (~3.4x compression).

### Debug mel spectrogram

```bash
uv run python debug-fbank.py --samples 240000
```

Step-by-step comparison of `fused_fbank.py` vs `torchaudio.compliance.kaldi.fbank` at every processing stage. Verifies full kaldi parity (cosine=1.000000 at each step).

## Output structure

```
build/my-model/
  Preprocessor.mlpackage   # Fused mel + encoder (audio → features)
  decoder.mlpackage        # Stateless prediction network
  joiner.mlpackage         # Joint network (enc + dec → logits)
  vocab.json               # BPE vocabulary (index = token ID)
  metadata.json            # Model configuration
```

## Mel spectrogram (fused)

The fused preprocessor includes a kaldi-compatible fbank extractor (verified at cosine=1.000000 against torchaudio reference):
- 80-dim log-mel filterbank
- Sample rate: 16 kHz, window: 25 ms, hop: 10 ms
- Povey window, preemphasis 0.97, DC offset removal
- HTK mel scale, low=20 Hz, high=Nyquist

## Decoding

Standard greedy RNNT: step through encoder frames, query joiner with current decoder state, emit token if not blank, advance. See `rnnt_decode.py` for reference.

## Upstream

- Training: [icefall](https://github.com/k2-fsa/icefall) (k2/lhotse)
- Inference: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- Models: [Vosk](https://alphacephei.com/vosk/models)
