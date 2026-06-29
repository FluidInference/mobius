---
license: openmdw-1.1
library_name: openvino
pipeline_tag: automatic-speech-recognition
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
base_model_relation: quantized
tags:
  - openvino
  - automatic-speech-recognition
  - streaming-asr
  - rnnt
  - fastconformer
  - cache-aware-streaming
  - multilingual
  - on-device
  - intel
language:
  - en
  - es
  - fr
  - de
  - it
  - pt
  - nl
  - pl
  - ru
  - uk
  - zh
  - ja
  - ko
  - ar
  - hi
---

# Nemotron-3.5-ASR-Streaming-Multilingual 0.6B — OpenVINO IR

OpenVINO IR export of NVIDIA's [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
a **cache-aware streaming FastConformer-RNNT** with prompt-conditioned multilingual decoding (40+ languages).
This is the OpenVINO sibling of [`FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML`](https://huggingface.co/FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML),
intended for the [eddy-audio](https://github.com/FluidInference/eddy-audio) runtime (OpenVINO, CPU / Intel NPU / iGPU).

## Contents

```
fp32/   FP32 IR        (~2.5 GB)  — graph the WER/CER table below was measured on
fp16/   FP16 IR        (~1.3 GB)  — identical transcripts, half the size, NPU-friendly
int8/   INT8-weight IR (~749 MB)  — weight-only encoder (per-channel sym); WER matches FP32
```

Each folder holds 4 IR pairs plus the tokenizer + metadata:

| file | role |
|---|---|
| `nemotron_preprocessor.xml/.bin` | 128-bin log-mel front end (16 kHz) |
| `nemotron_encoder.xml/.bin` | 24-layer cache-aware streaming FastConformer (d_model=1024) + language `prompt_kernel` |
| `nemotron_decoder.xml/.bin` | RNNT prediction network (2× LSTM @ 640) |
| `nemotron_joint.xml/.bin` | joint network → vocab 13088 (blank=13087) |
| `nemotron_vocab.json` | SentencePiece pieces (▁ word-boundary marker) |
| `metadata.json` | shapes, cache dims, `prompt_dictionary` (lang → prompt_id), lang-tag token ids |

## Streaming / cache-aware I/O

- Chunk: `chunk_mel_frames=112` (+`pre_encode_cache=9` → `total_mel_frames=121`), `att_context_size=[56,0]`.
- Encoder cache carried across chunks: `cache_channel [1,24,56,1024]`, `cache_time [1,24,1024,8]`, `cache_len [1]`.
- Language is selected per chunk via an integer `prompt_id` (see `prompt_dictionary` in `metadata.json`; `auto` = 101).
- The 39 language-tag token ids in `metadata.json["lang_tag_token_ids"]` are stripped from the decoded text.

## Accuracy (FLEURS, full test splits, FP32, greedy, forced language)

Measured on the OpenVINO IR (CPU). Reference column is the FluidAudio CoreML build of the same base model.

| Lang | Metric | OpenVINO | FluidAudio ref | Δ | n |
|------|--------|---------:|---------------:|------:|----:|
| en_us | WER | **11.78** | 12.09 | −0.31 | 647 |
| es_419 | WER | **6.99** | 9.01 | −2.02 | 908 |
| fr_fr | WER | **12.92** | 15.18 | −2.26 | 676 |
| cmn_hans_cn | CER | **21.05** | 24.54 | −3.49 | 945 |
| ja_jp | CER | **15.12** | 16.86 | −1.74 | 650 |
| **weighted** | mixed | **13.70** | 15.79 | **−2.09** | 3826 |

The export matches or beats the reference on every language — no conversion regression.
FP16 produces transcripts identical to FP32.

Throughput: **RTFx ≈ 3.66×** audio-weighted on a single CPU (Intel Xeon E5-2699 v4, 4 vCPU);
higher on Intel NPU / iGPU.

## INT8 weight-only (`int8/`)

Weight-only INT8 compression of the **encoder** (per-channel symmetric, data-free — mirrors
the CoreML `linear_quantize_weights` build). Only the weight constants become int8 with fp16
scales; activations stay fp16. The 24 Conformer relative-position projections
(`self_attn.linear_pos`) are kept FP16 — int8-compressing them trips an OpenVINO CPU compile
bug and they carry little weight mass. Decoder / joint / preprocessor stay FP16.

Measured on the **same FLEURS full test splits, forced language, greedy** — directly comparable
to the FP32 table above:

| Lang | Metric | FP32 | INT8-wo | Δ | n |
|------|--------|-----:|--------:|------:|----:|
| en_us | WER | 11.78 | **11.93** | +0.15 | 647 |
| es_419 | WER | 6.99 | **7.05** | +0.06 | 908 |
| fr_fr | WER | 12.92 | **12.99** | +0.07 | 676 |
| cmn_hans_cn | CER | 21.05 | **20.94** | −0.11 | 945 |
| ja_jp | CER | 15.12 | **15.15** | +0.03 | 650 |
| **weighted** | mixed | 13.70 | **13.73** | **+0.03** | 3826 |

Essentially lossless — the largest swing is +0.15 WER (within run-to-run noise) and Chinese
CER improves slightly.

**What INT8 buys you — and what it doesn't:**

| | FP32 | FP16 | INT8-wo |
|---|---:|---:|---:|
| encoder `.bin` | 2.5 GB | 1.25 GB | **732 MB** |
| folder size | ~2.5 GB | ~1.3 GB | **~749 MB** |
| peak RSS (CPU) | 5.8 GB | 3.9 GB | **2.1 GB** |
| RTFx (audio-wtd, 4 vCPU) | 3.66× | ≈3.66× | **2.96×** |

INT8 is a **memory / disk** win (~½ the RAM of FP16), not a speed win: on x86 CPU the int8
weights decompress to float per-op, so it runs *slower* than FP16. The speed payoff is on
**Intel NPU**, where native int8 execution applies. Choose `int8/` for memory-constrained or
NPU deployments; choose `fp16/` for fastest CPU inference.

## License

Inherits `openmdw-1.1` from the base model. See the base model card for terms.
