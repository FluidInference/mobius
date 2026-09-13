# MOSS-TTS-Nano → CoreML

Conversion of [OpenMOSS-Team/MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M)
(0.1B multilingual streaming TTS with zero-shot voice cloning, 20 languages, native 48 kHz stereo) and
its codec [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)
(22M, causal-transformer codec, 16×1024 RLFQ at 12.5 Hz) to CoreML. Apache-2.0 upstream.

## Pipeline

```
prompt wav ─CodecEncoder─► 16×T ref codes ─┐
text ─SentencePiece─► ids ──────────────────┴─► rows [T,17] ─Prefill─► hidden + KV
                                                                  │
                     ┌────────────────────────────────────────────┘
                     ▼   per 80 ms frame
                 Frame (1-layer local transformer + 16 heads, sampling in-graph) ─► 16 codes ─► CodecStep ─► 2×3840 samples
                     │                                                                ▲
                 Step (global GPT-2, KV update) ◄──── row [assistant_slot, 16 codes] ──┘
```

Rows are `[text_token, code_0 … code_15]`; the audio pad id (1024) marks unused code slots. Prompt layout
(voice-clone mode) is upstream's `build_inference_input_ids`: `<im_start>user … Reference: <audio_start>
{ref rows with user_slot=8} <audio_end> … Text: {text} </user_inst><im_end> <im_start>assistant <audio_start>`.
Generation stops when the text head picks `audio_end` (7) instead of `assistant_slot` (9).

## Models (`build/`)

| mlpackage | I/O | Notes |
|---|---|---|
| `MossNano-Prefill-T512-M1024-fp16` | rows [1,512,17] int32 + len → hidden [1,768], kv_k/kv_v [12,1,12,1024,64] | right-padded prompt, causal + key mask |
| `MossNano-Step-M1024-fp16` | row [1,1,17] + kv in/out + cur_len → hidden | one-hot KV write at `cur_len` |
| `MossNano-Frame-fp16` | hidden + text_u [1] + audio_u [1,16] + temps + top_p + rep_penalty + seen [1,16,1024] + greedy → should_continue, frame [1,16] | 17 sequential local passes unrolled; inverse-CDF sampling from host uniforms (top-k 50 text / 25 audio fixed) |
| `MossNano-CodecStep-fp16` | codes [16,1,1] + frame_index + 24 KV caches → audio [1,2,3840] + caches | streaming; caches [1,4,{500,800,1200,1600},64] per layer |
| `MossNano-CodecDecoder-fp16` | codes [16,1,T≤125] (RangeDim) → audio [1,2,T·3840] | batch decode ≤10 s (full T×T attention) |
| `MossNano-CodecEncoder-fp32` | audio [1,2,S≤188·3840] (RangeDim, S % 3840 == 0) → codes [16,1,S/3840] | prompt encode; fp16 loses 34 % of codes |

All targets macOS 14 / iOS 17 (no `StateType`). Weights: LM ~235 MB fp16 total, codec ~40 MB.

## Commands

`assets/en_2.wav` and `assets/zh_1.wav` are the upstream demo prompts (`MOSS-TTS-Nano/assets/audio/`,
gitignored here as `*.wav`); copy them in before running.

```bash
uv sync
uv run python verify_pytorch.py                       # upstream reference (sampled) → build/ref_pytorch.wav + tokens
uv run python verify_pytorch.py --do-sample 0 --output build/ref_greedy.wav   # deterministic parity target
uv run python convert_lm.py --fp16
uv run python convert_codec.py --fp16 --skip-encoder && uv run python convert_codec.py --skip-decoder
uv run python convert_codec_step.py --fp16
uv run python infer_coreml.py --replay 375            # teacher-forced greedy replay vs PyTorch
uv run python infer_coreml.py --codec-step build/codec/MossNano-CodecStep-fp16.mlpackage --text "…"
uv run python benchmark.py
```

## Parity (M5 Pro, macOS 26.7, coremltools 9.0, torch 2.7.0)

- Wrappers vs upstream fp32: prefill/step hidden max|Δ| 7e-6 / 5e-6; frame greedy tokens 16/16; codec
  decoder SNR 234 dB; encoder 1584/1584 codes; streaming step decoder vs full decode SNR 78.9 dB.
- CoreML fp16 vs wrappers: prefill hidden 4e-3, step hidden 1.1e-2; greedy replay of the 375-frame
  reference **370/375 frames token-exact** (five near-tie flips); codec full decoder SNR 43.5 dB (T=57)
  / 34.3 dB (T=125); streaming step SNR 56.6 dB on GPU, 40.6 dB CPU, 37.9 dB ANE; fp32 encoder exact.
- Intelligibility (Parakeet ASR via `fluidaudiocli tts-asr-verify --score-only`, two English phrases,
  voice-clone prompt `assets/en_2.wav`): CoreML chain (LM fp16 + streaming codec) macro WER 8.3 %
  (only "riverbank" → "river bank"), upstream PyTorch fp32 10.1 % (same split + one dropped word).
- Greedy decoding in upstream never emits the stop token (runs to max frames); sampling (defaults
  text T=1.5, audio T=1.7 / top-p 0.8 / top-k 25) is the quality mode, greedy is only a parity oracle.

## Latency (warm, ms per call)

| model | ALL | ANE | GPU | CPU |
|---|---|---|---|---|
| Frame | 5.3 | 5.7 | 5.5 | 4.1 |
| Prefill T512 | 10.9 | 117 | 11.2 | 52.9 |
| Step M1024 | 7.6 | 15.7 | 7.8 | 58.8 |
| CodecStep | 5.2 | 7.8 | 5.3 | 5.5 |
| CodecDecoder (57 frames) | 6.8 | 241 | 7.6 | 89.8 |
| CodecEncoder fp32 (7.9 s prompt) | 22.5 | 589 | 32.5 | 247 |

Per 80 ms frame the LM costs ≈ 13 ms (step + frame) and streaming codec ≈ 5 ms, i.e. ≈ 4.4× real time
with ≈ 30 ms compute to first audio after a ≈ 11 ms prefill. The Python driver measures ≈ 9 + 9 ms
(feed construction + 38 MB KV round trip per step); a stateful (`StateType`, iOS 18) step would remove
that copy. Prefill/Step/CodecDecoder fail ANE compilation (`ANECCompile FAILED`) and fall back — GPU is
the intended unit for those; Frame and CodecStep run on any unit.

## Known gaps / follow-ups

- Text normalisation: upstream runs WeTextProcessing + a robust normaliser before tokenising; not ported
  (FluidAudio already ships English normalisers + NeMo ITN that can cover this).
- Prompt encode needs fp32 (residual LFQ amplifies fp16 error in deep codebooks).
- `nq < 16` (lower-bitrate) decoding not exposed; frame graph is fixed at 16 codebooks.
- No stateful KV variant yet; M=1024 caps prompt+generation at 1024 rows (≈ 66 s after a 200-row prompt).
