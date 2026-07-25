# NeuTTS-2E → CoreML

Conversion of [neuphonic/neutts-2e](https://huggingface.co/neuphonic/neutts-2e)
(emotional English TTS: Qwen3 236M backbone + [NeuCodec](https://huggingface.co/neuphonic/neucodec))
to CoreML mlpackages.

## Pipeline

```
text ──tokenizer──► prompt ids ──► LM-Prefill ──► logits + KV cache
                                        │
                             LM-Decode (top-k sampling loop, 50 codes/s)
                                        │  <|speech_N|> tokens
                             NeuCodec-Decoder ──► 24 kHz waveform
```

Prompt layout (BPE, no phonemizer):
`<|TEXT_PROMPT_START|>{ref_text}[<|EMOTION|>]{text}<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{ref codes}`,
generate until `<|SPEECH_GENERATION_END|>` (temp 1.0, top-k 50, min 50 new tokens).
The four fixed speakers ship as pre-encoded NeuCodec code sequences
(`samples/*.pt`), so only the codec **decoder** is needed on-device.
Upstream additionally applies a Perth watermark to the output waveform in the
host app; that is postprocessing, not part of these models.

## Models

| mlpackage | I/O | Size | Target |
|---|---|---|---|
| `LM-Prefill-T768-M2048-fp16` | ids [1,768] + len → last-pos logits [1,217232], kv_k/kv_v [28,1,4,2048,128] | 456 MB | macOS 14+ |
| `LM-Decode-M2048-fp16` | id [1,1] + kv in/out + cur_len → logits | 451 MB | macOS 14+ |
| `LM-Decode-M2048-fp16-stateful` | id [1,1] + cur_len → logits, KV as `StateType` | 451 MB | macOS 15+ |
| `NeuCodec-Decoder-fp16` | codes [1,T≤2000] (RangeDim) → audio [1,T·480] | 372 MB | macOS 14+ |

The tied embedding/LM-head weight ([217232, 512]) is used for both the input
gather and the output projection through one shared parameter, so CoreML
const-dedup stores it once per package. The stateful decode is bit-identical
to the pass-through variant and avoids round-tripping 117 MB of KV per step;
prefill KV seeds the state via `write_state` (float32 arrays only — the fp16
state converts internally).

## Commands

```bash
uv sync
uv run python gen-pytorch-ref.py                       # PyTorch reference (bf16 LM + fp32 codec)
uv run python convert-lm.py    --output-dir ./build/lm-fp16 --fp16 --stateful-decode
uv run python convert-codec.py --output-dir ./build/codec   --fp16
# End-to-end CoreML synthesis
uv run python inference.py --lm-dir ./build/lm-fp16 \
    --codec ./build/codec/NeuCodec-Decoder-fp16.mlpackage \
    --text "I can't believe it's finally here!" --speaker emily --emotion happy
# Deterministic parity replay of the PyTorch reference run
uv run python inference.py --lm-dir ./build/lm-fp16 \
    --codec ./build/codec/NeuCodec-Decoder-fp16.mlpackage --teacher-force
```

## Parity (M5 Pro, macOS 26)

- LM wrappers vs HF fp32: max|Δ| 2.8e-5 (prefill), 3.1e-5 (decode), argmax match.
- CoreML fp16 vs torch fp32: max|Δ| 0.024 (prefill), 0.081 (decode), argmax match;
  stateful ≡ pass-through exactly (two chained steps checked).
- Codec wrapper vs `neucodec.decode_code`: SNR 123.9 dB (fp32 torch);
  CoreML fp16: SNR 46.6 / 41.7 / 40.1 dB at T = 125 / 300 / 800 codes.
- Teacher-forced replay of the 301-token PyTorch reference: 98.7 % of
  reference tokens inside the CoreML top-50 sampling support; final audio
  SNR 41.5 dB vs the PyTorch waveform.
- Sampled end-to-end run (warm): decode 9.1 ms/token at M=2048, **7.0 ms/token
  (143 tok/s) at M=1024** (LM on GPU), prefill 33–40 ms, codec 12.7–27× RT on
  ANE. Compute units: LM decode best on ALL/CPU_AND_GPU (ANE rejects the
  decode graph outright, ANECCompile error -14, same as Qwen3-0.6B; CPU_ONLY
  still real-time at 58 tok/s); codec ~2× faster on CPU_AND_NE than GPU.
  `inference.py` defaults to this split.
- Streaming (`--stream`, upstream 25-frame windowed overlap-add over the
  flexible-length codec): with the M=1024 pair, TTFA ≈ 554 ms, steady-state
  inter-chunk ≈ 303 ms against the 500 ms budget, 1.6× RT overall; batch ≈
  2.0× RT. 0 % WER (transcribes identically to batch). M=1024 caps
  prompt+generation at 1024 tokens (~11 s of audio after the emily prompt) —
  hosts should pick the M=2048 pair for longer utterances.

### Speed dead-ends (measured, don't retry)

- **int8 weight quantization: zero speedup** (9.5 ms/tok either way) — the
  one-token step is dispatch-latency-bound (hundreds of small GPU ops), not
  weight-bandwidth-bound. int8 still halves disk (451→227 MB) at a small
  quality cost (48/50 top-50 overlap); `compress-lm.py` kept for that.
- **Pure fp16 (no fp32 op islands): catastrophically wrong** — fp16 RMSNorm
  overflows on Qwen3 activation outliers (top-50 overlap 0/50). The fp32
  pow/reduce_mean/rsqrt/softmax islands are load-bearing.
- **In-model top-k head: slower** (+1.0 ms/tok — topk over 217 232 costs more
  than shipping the 870 KB logits), and it exposed a CoreML kernel bug: topk
  over a >2^17-wide *intermediate* tensor returns indices modulo 131072
  (values correct; both fp16 and fp32; chunked two-stage topk gives the same
  corruption; the op is fine in a standalone model fed by an input, so it is
  layout-dependent on the big matmul output).

## Conversion gotchas (also see git history)

- **torch pin `>=2.7,<2.8`** — newer torch traces `x.shape` unpacking,
  `repeat_interleave`, and `shape[-1] // 2` into `aten::Int`/`floor_divide`
  nodes that coremltools 9.0 cannot fold ("only 0-dimensional arrays can be
  converted to Python scalars"). The wrappers additionally use static ints and
  expand+reshape for GQA so no shape-dependent ops remain. torchaudio must be
  pinned to the same minor as torch (ABI).
- **fp16 integer hazard** — FSQ digit math (`floor(code/basis) % 4`) breaks
  under the fp16 compute pass for codes > 2048; dequant is a precomputed
  [65536, 8] embedding gather instead.
- **NeuCodec RoPE quirk** — `bs_roformer5.Attention` feeds `[b, h, t, d]`
  into torchtune's rope (which expects `[b, s, h, d]`), so the pretrained
  model rotates by *head index*, constant over time. Replicated with fixed
  per-head cos/sin buffers; consequently the code-length axis stays flexible
  (RangeDim) with no positional tables.
- **ISTFT** — vocos "same"-padding ISTFT reimplemented as real IDFT 1×1 convs
  + `ConvTranspose1d` overlap-add with window-envelope division (verified vs
  the library to 1e-3 before conversion).

## Comparison vs FluidAudio TTS engines (M5 Pro, 2026-07-25)

Three text lengths — S: 5 words (~2 s), M: 14 words (~5 s), L: 53-word
paragraph (~23 s) — warm runs (mean of 3 per cell), WER via parakeet-tdt-v3
round-trip. FluidAudio engines via `fluidaudiocli tts --metrics` (release);
NeuTTS-2E via `inference.py` (LM on GPU, codec on ANE).

RTFx (inference speed vs audio duration):

| Engine | S | M | L | WER S/M/L | Disk | TTFA |
|---|---|---|---|---|---|---|
| Supertonic-3 | 20× | 43× | 87× | 0 / 4.8 / 1.2 % | 284 MB | ≈ batch (fast) |
| KokoroAne | 1.9× | 6.5× | 19× | 0 / 0 / 0 % | 782 MB | — |
| StyleTTS2 | 4.6× | 11.3× | 18× | 80 / 0 / 12.7 % | 452 MB | — |
| PocketTTS | 3.4× | 5.7× | 6.2× | 0 / 0 / 0 % | 866 MB | streaming-capable |
| **NeuTTS-2E batch** | 1.3× | 1.5× (2.0× @M1024) | 1.7× | 6.7 / 0 / 0 % | 1.28 GB | n/a |
| **NeuTTS-2E stream** | 1.4× | 1.4× (1.6× @M1024) | 1.5× | 0 / 0 / 0 % | 1.28 GB | **650 ms (554 @M1024)** |

Notes:

- NeuTTS-2E is the only engine with emotion control, and (with KokoroAne and
  PocketTTS) one of three that stayed fully intelligible on the long
  paragraph. Its S-column WER is one sampled insertion in one of three seeds
  ("…finally here **and**") — autoregressive variance, not corruption.
- The streaming mode holds TTFA at ~650 ms independent of text length while
  batch latency grows with duration; steady-state chunk cadence 340 ms vs the
  500 ms real-time budget.
- StyleTTS2 initially failed all M/L texts with `corruptedModel`: the
  FluidAudio downloader materializes the sized `bert_fp16_t*` /
  `fused_diffusion_sampler_fp16_t*` mlmodelc **without `model.mil`** (the
  hosted files in `FluidInference/StyleTTS-2-coreml/iteration_3/compiled/`
  are complete — likely the HF listing pagination gap from the download
  refactor, issue #765). Benchmarked after curling the six missing
  `model.mil` files into the cache. Its 80 % WER at S ("Elon is Mahir") is
  the unsized short-window path, unrelated to that bug.
- Cross-host caveat: NeuTTS-2E timings are from the Python coremltools host;
  the others ran the release Swift CLI.

## Follow-ups

- LM decode is dispatch-latency-bound; the remaining levers are a Swift host
  (per-step Python overhead ~1–2 ms) and multi-token/speculative decode.
  ANE rejects the decode graph (-14) and int8 doesn't help speed (see above).
- Multifunction package (macOS 15+) to share weights between prefill and
  decode (saves ~450 MB on disk).
- Swift host port (tokenizer + sampling loop + Perth watermark) for FluidAudio.
