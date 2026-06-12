# Parakeet-unified-en-0.6b — CoreML Conversion

Converts NVIDIA's [parakeet-unified-en-0.6b](https://huggingface.co/nvidia/parakeet-unified-en-0.6b)
(Unified FastConformer-RNNT, 600M) to CoreML for on-device offline **and**
streaming English ASR on Apple Silicon.

Requested in [FluidAudio#49](https://github.com/FluidInference/FluidAudio/issues/49#issuecomment-4526026299).

## Model Overview

| Feature | Value |
|---------|-------|
| Architecture | Unified FastConformer-RNNT (`chunked_limited_with_rc` attention + dynamic chunked conv) |
| Parameters | 600M |
| Sample rate | 16 kHz |
| Mel bins | 128 |
| Encoder | 24 layers, d=1024, 8× dw_striding subsampling (80 ms encoder frames) |
| Decoder | RNNT prediction net, 2-layer LSTM, hidden 640 |
| Vocab | 1024 BPE + blank (blank_idx = 1024) — **no TDT duration head** |
| Claimed WER | 1.63% test-clean / 3.11% test-other (offline), 6.52% @ 0.56 s latency |

Unlike the cache-aware Nemotron models, the unified model streams via
**chunked attention masks**: inference re-runs the encoder over a
`[left | chunk | right]` audio window each step and keeps only the chunk's
encoder frames (see NeMo `examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py`
with `att_context_size_as_chunk=true`). The encoder is *stateless*, which maps
directly onto fixed-shape CoreML.

## Environment

`chunked_limited_with_rc` / `conv_context_style=dcc` only exist on NeMo main —
the 2.7.3 PyPI release cannot load this checkpoint. Setup:

```bash
uv sync
# Overlay NeMo from git (its pyproject pins torch to the PyTorch CPU index,
# which conflicts with uv resolution, hence --no-deps):
uv pip install --no-deps --force-reinstall \
  "nemo_toolkit @ git+https://github.com/NVIDIA-NeMo/NeMo.git@95f92737cfb8ee0123bb328b07a2d24c6d859aff"
# Download the checkpoint (2.5 GB)
curl -L -o parakeet-unified-en-0.6b.nemo \
  "https://huggingface.co/nvidia/parakeet-unified-en-0.6b/resolve/main/parakeet-unified-en-0.6b.nemo"
```

**IMPORTANT: always run scripts with `uv run --no-sync`** — a plain `uv run`
re-syncs the environment and silently reverts the overlay to the 2.7.3 wheel
(symptom: `AttributeError`/`TypeError` about `att_chunk_context_size` when
restoring the checkpoint).

Known quirk: the released `.nemo` has no `validation_ds` config section and
NeMo's `transcribe()` dereferences it — scripts here inject an empty section
before calling `transcribe()`.

## Convert

```bash
uv run --no-sync python convert-coreml.py --output-dir ./build/parakeet_unified_coreml
```

Exports (FP16 mlprogram, iOS17+, traced CPU_ONLY):

| Package | Size | Notes |
|---------|------|-------|
| `parakeet_unified_preprocessor.mlpackage` | 0.6M | audio (≤15 s, RangeDim) → 128-mel |
| `parakeet_unified_encoder.mlpackage` | 1.1G | offline, fixed 15 s window (mel [1,128,1501] → [1,1024,188]) |
| `parakeet_unified_mel_encoder.mlpackage` | 1.1G | fused audio→encoder, fixed 15 s |
| `parakeet_unified_encoder_streaming_70_13_13.mlpackage` | 1.1G | chunked-attention mask baked in, 7.68 s window (mel [1,128,769] → [1,1024,96]) |
| `parakeet_unified_mel_encoder_streaming_70_13_13.mlpackage` | 1.1G | fused streaming variant |
| `parakeet_unified_decoder.mlpackage` | 14M | RNNT prediction net, single token + (h,c) state |
| `parakeet_unified_joint.mlpackage` | 3.3M | full-grid joint logits |
| `parakeet_unified_joint_decision_single_step.mlpackage` | 3.3M | argmax + prob + top-64 (no duration output) |

Streaming context is configurable in 80 ms encoder frames:
`--streaming-context left,chunk,right`. The default `70,13,13` =
5.6 s / 1.04 s / 1.04 s (2.08 s theoretical latency, best streaming WER per
model card). Trained chunk sizes: chunk ∈ {1,2,7,13}, right ∈ {0,1,2,3,4,7,13},
left = 70 (`att_chunk_context_size`). For ~1.12 s latency use `70,7,7`.

## Validate

```bash
uv run --no-sync python compare-models.py \
  --coreml-dir ./build/parakeet_unified_coreml --audio-file audio/yc_first_minute_16k.wav
```

Results on M-series (2026-06-12):

- **Offline (15 s)**: CoreML transcript **exactly matches** NeMo reference
  (greedy). Encoder fp16/ANE max_abs diff ≈ 1.66 on raw activations with no
  effect on the decoded token path.
- **Streaming (30 s, context 70,13,13)**: CoreML streamed transcript
  **word-for-word identical** to the NeMo *offline* transcript, punctuation
  included.

The streaming simulation mirrors NeMo's `StreamingBatchedAudioBuffer`: feed
`chunk+right` first, then `chunk` per step; run the streaming encoder on the
(zero-padded) window; decode all not-yet-decoded frames while holding back
`right` frames (re-encoded with more future context next step); RNNT decoder
LSTM state and last token persist across chunks.

### LibriSpeech test-clean WER (full CoreML chain, greedy)

```bash
uv run --no-sync python benchmark_wer.py --mode both
```

| Mode | WER | Files | RTFx (Python loop) |
|------|-----|-------|--------------------|
| Offline (15 s window) | **1.82%** | 2382 (238 files > 15 s skipped) | 117 |
| Streaming [70,13,13], 2.08 s latency | **2.15%** | all 2620 | 54 |

Reference: NVIDIA claims 1.63% offline on full-length audio (no 15 s cap).
Text normalization: strip punctuation, lowercase (same as the nemotron
benchmark). RTFx is bounded by the single-threaded Python decode loop, not
the models.

### Latency (median of 5, after warmup)

| Component | CPU+ANE | CPU+GPU |
|-----------|---------|---------|
| Streaming encoder (7.68 s window → one 1.04 s chunk) | 11.9 ms | 11.8 ms |
| Offline encoder (15 s window) | 25.3 ms | 16.0 ms |

≈87× real-time per streaming chunk on the encoder; decoder/joint steps are
single-digit ms on CPU.

Benign noise: `E5RT ... zero shape error` prints at process exit when the
RangeDim preprocessor has been loaded; predictions are unaffected.

## Stage for HuggingFace / Swift

```bash
uv run --no-sync python stage_hf.py   # → build/hf-staging/
```

Compiles each `.mlpackage` to `.mlmodelc`, exports `vocab.json`
(`{id: piece}`, the format FluidAudio's Swift `Tokenizer` reads), and copies
`metadata.json`. Upload of the staged directory to
`FluidInference/parakeet-unified-en-0.6b-coreml` is user-run.

## Files

- `convert-coreml.py` — export CLI (offline + streaming variants)
- `components.py` — torch wrappers (RNNT: no duration head, unlike TDT v3)
- `coreml_rnnt.py` — shared greedy RNNT decode + buffered streaming loop
- `compare-models.py` — parity + E2E greedy decode validation, offline & streaming
- `benchmark_wer.py` — LibriSpeech test-clean WER benchmark
- `stage_hf.py` — compile mlmodelc + vocab.json for HF upload / Swift host
- `inspect_model.py` — config dump + NeMo reference transcription

## Host integration notes (FluidAudio)

- Decode loop is classic greedy RNNT: per encoder frame, emit until blank
  (cap ~10 symbols/frame), advance decoder LSTM on each emitted token. No TDT
  frame-skipping — the host TdtDecoder is not reusable as-is.
- Streaming: maintain a 7.68 s rolling sample buffer; per 1.04 s chunk run the
  fused streaming mel+encoder, decode frames `[T-26, T-13)` of the 96-frame
  output (hold back right context), carry (h, c, last_token) across chunks.
- Vocab 1024 + blank=1024; tokenizer is the bundled SentencePiece model inside
  the `.nemo` (extract `*_tokenizer.model`).
