# Qwen3-ForcedAligner-0.6B → CoreML

CoreML conversion of [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) for on-device forced alignment on Apple platforms.

## Model Overview

Qwen3-ForcedAligner is a **non-autoregressive (NAR)** forced alignment model that takes audio + text and outputs per-word timestamps. It uses the same `Qwen3ASRForConditionalGeneration` architecture as Qwen3-ASR but runs inference differently — a single prefill pass instead of autoregressive decode.

- **Parameters:** 0.6B
- **Languages:** 11 (Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish)
- **Max audio:** 5 minutes
- **Timestamp resolution:** 80ms segments
- **License:** Apache 2.0

## Architecture

Two audio encoder approaches are available. The inference script auto-detects which
to use based on which `.mlpackage` files are present.

### Split Encoder (higher accuracy)

```
Qwen3ASRForConditionalGeneration
  └── thinker
        ├── audio_tower (24-layer Transformer, 1024 dim)
        │   ├── conv frontend                               → forced_aligner_audio_conv.mlpackage
        │   └── transformer + projection                    → forced_aligner_audio_transformer.mlpackage
        ├── model (28-layer Qwen3 decoder, 1024 dim)       → forced_aligner_decoder_prefill.mlpackage
        │   └── embed_tokens                                → forced_aligner_embedding.mlpackage
        └── lm_head (1024 → 5000)                          → forced_aligner_lm_head.mlpackage
```

The audio encoder is split into two CoreML models to preserve cross-chunk attention.
Conv runs per-chunk, then all conv outputs are concatenated and passed through the
transformer in a single call with full bidirectional attention across all frames.
This matches the native PyTorch behavior and gives the best accuracy (4.4ms AAS).

### Monolithic Encoder (faster, simpler)

```
Qwen3ASRForConditionalGeneration
  └── thinker
        ├── audio_tower (24-layer Transformer, 1024 dim)   → forced_aligner_audio_encoder.mlpackage
        ├── model (28-layer Qwen3 decoder, 1024 dim)       → forced_aligner_decoder_prefill.mlpackage
        │   └── embed_tokens                                → forced_aligner_embedding.mlpackage
        └── lm_head (1024 → 5000)                          → forced_aligner_lm_head.mlpackage
```

The entire audio encoder (conv + transformer + projection) is a single CoreML model
that processes each 100-frame mel chunk independently. This is faster (one model call
per chunk, no concatenation step) but each chunk's 13 output frames only see their own
context — no cross-chunk attention. This reduces accuracy (20.7ms AAS) but may be
acceptable depending on the use case.

### Which to use?

| | Split Encoder | Monolithic Encoder |
|---|---|---|
| Models | `audio_conv` + `audio_transformer` | `audio_encoder` |
| Size | ~1.1GB combined | ~604MB |
| AAS (mean boundary error) | **4.4 ms** | 20.7 ms |
| % within 20ms | **95.4%** | 90.7% |
| Cross-chunk attention | Yes | No |
| Model calls (audio) | N conv + 1 transformer | N encoder |
| Best for | Accuracy-critical alignment | Latency-sensitive / real-time |

The inference script (`run_coreml_inference.py`) checks for `audio_conv` + `audio_transformer`
first. If found, it uses the split approach. Otherwise it falls back to the monolithic
`audio_encoder` if present.

### Key Differences from Qwen3-ASR-0.6B

| | ASR-0.6B | ForcedAligner-0.6B |
|---|---|---|
| Audio encoder layers | 18 | **24** |
| Audio encoder dim | 896 | **1024** |
| Audio encoder heads | 14 | **16** |
| Vocab size | 151,936 | **152,064** |
| RoPE | standard | **interleaved mrope** |
| Inference | autoregressive | **NAR (single prefill)** |
| Output | text tokens | **ms timestamps** |

## Input/Output Shapes

### Split Encoder

#### Audio Conv (per-chunk)
```
Input:  mel_input     [1, 128, 100]       float32   (128 mel bins, 100 frames = 1 window)
Output: conv_features [1, 13, 1024]       float32   (13 frames after 8x conv downsampling)
```

#### Audio Transformer (all chunks concatenated)
```
Input:  features      [1, 256, 1024]      float32   (padded concatenated conv features)
Output: audio_embeddings [1, 256, 1024]   float32   (trim to actual frame count)
```

### Monolithic Encoder

#### Audio Encoder (per-chunk)
```
Input:  mel_input     [1, 128, 100]       float32   (128 mel bins, 100 frames = 1 window)
Output: audio_embeddings [1, 13, 1024]    float32   (13 frames, trim for short last chunk)
```

### Token Embedding
```
Input:  input_ids   [1, seq_len]          int32     (seq_len ∈ [1, 1024])
Output: embeddings  [1, seq_len, 1024]    float32
```

### Decoder Prefill (NAR)
```
Input:  hidden_states [1, 1024, 1024]     float32   (full sequence)
        position_cos  [1, 1024, 128]      float32   (RoPE cos)
        position_sin  [1, 1024, 128]      float32   (RoPE sin)
Output: output_hidden [1, 1024, 1024]     float32
```

### LM Head
```
Input:  hidden_states [1, seq_len, 1024]  float32   (seq_len ∈ [1, 1024])
Output: logits        [1, seq_len, 5000]  float32   (raw timestamp values, NOT vocab tokens)
```

> **Note:** The LM head output dim is 5000 (not vocab_size 152064). The ForcedAligner
> predicts raw timestamp values via argmax, where each value × 80ms = absolute time.
> 5000 × 80ms = 400s, covering up to ~6.7 minutes of audio.

## Inference Pipeline

Steps 1-3 differ depending on encoder approach. Steps 4-11 are shared.

### Split Encoder (steps 1-3)
```
1. Audio → Whisper mel spectrogram → [1, 128, T]
2. Chunk mel into 100-frame windows → Audio Conv (per-chunk) → conv features
3. Concatenate all conv features → pad to 256 → Audio Transformer → audio embeddings
```

### Monolithic Encoder (steps 1-3)
```
1. Audio → Whisper mel spectrogram → [1, 128, T]
2. Chunk mel into 100-frame windows → Audio Encoder (per-chunk) → embeddings
3. Concatenate per-chunk embeddings (trim last chunk to actual frames)
```

### Shared (steps 4-11)
```
4. Tokenize text with <timestamp> delimiters between words
5. Build input_ids: <audio_start> <audio_pad>... <audio_end> word1 <ts><ts> word2 <ts><ts> ...
6. Embed: audio embeddings + text token embeddings → concatenated sequence
7. Compute MRoPE cos/sin → Decoder prefill (single pass) → hidden states
8. LM head → logits
9. argmax at timestamp_token_id positions → raw ms values
10. Fix monotonicity (LIS algorithm) → final timestamps
11. Scale: ms = raw_value * 80 (timestamp_segment_time)
```

## Conversion

```bash
# Install dependencies
uv pip install torch coremltools transformers typer soundfile

# Clone Qwen3-ASR source (required for model classes)
git clone https://github.com/QwenLM/Qwen3-ASR.git /path/to/qwen3-asr

# Convert split encoder (default — higher accuracy)
uv run python convert-coreml.py

# Convert monolithic encoder (faster)
uv run python convert-coreml.py --components audio_encoder embedding decoder_prefill lm_head

# Convert all components (both encoder approaches)
uv run python convert-coreml.py --components audio_conv audio_transformer audio_encoder embedding decoder_prefill lm_head
```

## Benchmarking

```bash
# Generate PyTorch reference timestamps from cached test-clean
uv run python compare-models.py --num-files 10 --output results/pytorch_reference.json

# Single file mode
uv run python compare-models.py --audio-file audio.wav --text "hello world" --language English
```

### Parity Metrics (3 LibriSpeech test-clean samples, 54 word boundaries)

#### Split Encoder

| Metric | Value | Notes |
|--------|-------|-------|
| AAS (mean boundary error) | 4.4 ms | lower is better |
| Max boundary error | 160 ms | single position, 2 segments |
| % within 20ms | 95.4% | |
| % within 80ms (1 segment) | 99.1% | 80ms = 1 timestamp segment |
| % within 160ms (2 segments) | 100.0% | |
| PyTorch latency (avg) | ~4736 ms | CPU, includes first-run warmup |
| CoreML latency (avg) | ~2781 ms | ALL compute units |

Per-sample results:
- Long (28 words): 1.4ms AAS, 98.2% within 20ms
- Short (8 words): 10.0ms AAS, 87.5% within 20ms
- Medium (18 words): 6.7ms AAS, 94.4% within 20ms

#### Monolithic Encoder

| Metric | Value | Notes |
|--------|-------|-------|
| AAS (mean boundary error) | 20.7 ms | ~5x worse than split |
| % within 20ms | 90.7% | |
| % within 80ms (1 segment) | 92.6% | |
| % within 160ms (2 segments) | 96.3% | |

The accuracy gap is caused by each chunk's 13 frames only attending to themselves
in the transformer, missing cross-chunk context that the native PyTorch encoder provides.

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|audio_start\|>` | 151669 | Start of audio embeddings |
| `<\|audio_end\|>` | 151670 | End of audio embeddings |
| `<\|audio_pad\|>` | 151676 | Audio embedding placeholder |
| `<timestamp>` | 151705 | Timestamp prediction position |

## LM Head Architecture

The ForcedAligner's LM head is **not** the same as the ASR model's:

| | ASR LM Head | ForcedAligner LM Head |
|---|---|---|
| Output dim | 151,936 (vocab tokens) | **5,000** (raw timestamp values) |
| Purpose | Next-token prediction | Timestamp regression via argmax |
| Decoding | argmax → token ID → text | argmax → raw_value × 80ms → time |

The embedding table is still 152,064 tokens (shared architecture), but the LM head
projects to 5,000 outputs — enough for timestamps up to 400 seconds at 80ms resolution.

## Known Issues

See [problems_encountered.md](./problems_encountered.md) for detailed conversion journal.

## References

- **Model:** [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- **Paper:** [arXiv:2601.21337](https://arxiv.org/abs/2601.21337)
- **Source:** [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- **Community request:** [FluidAudio#49](https://github.com/FluidInference/FluidAudio/issues/49)
