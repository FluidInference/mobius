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
- Sampled end-to-end run: 13 ms/token decode (≈77 tok/s vs 50 needed for
  real-time), prefill 169 ms, codec 2.7–9.6× RT (ComputeUnit.ALL).

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

## Follow-ups

- ANE profiling (`tools/coreml-cli`) and a fixed-shape codec variant if ANE
  residency is worth it; LM decode currently runs GPU-dominant.
- Weight compression: 111 M of the 236 M params are the tied vocab matrix —
  int8/palettized embedding would roughly halve both LM packages.
- Multifunction package (macOS 15+) to share weights between prefill and
  decode (saves ~450 MB on disk).
- Swift host port (tokenizer + sampling loop + Perth watermark) for FluidAudio.
