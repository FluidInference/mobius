# Chatterbox Nano (110M, English) → CoreML — Trial Report

Conversion trial for [ResembleAI/chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano)
(`t3_nano_v1` + `s3gen_meanflow`), requested on Discord after the
Multilingual port — the 500M T3 + 10-step CFM was "still quite heavy for
edge devices, especially older iPhones".

## Architecture / export boundary

| Component | Upstream | CoreML export | Host (Swift) responsibility |
|---|---|---|---|
| T3 | GPT2-small (12L×768d×12H, 110M), batch 1 (no CFG), HF BPE tokenizer (50276 incl. paralinguistic tags) | `T3Nano-Prefill` (T=512, M=1536) + `T3Nano-Decode` (I/O KV) + stateful (MLState, iOS 18+); GPT2 `wpe` applied in-graph | BPE tokenization, embedding prep (tables exported), turbo sampling (temp/top-k/top-p/rep-penalty) |
| Alignment analyzer | none in Turbo/Nano | — | — |
| S3Gen flow | same UpsampleConformer encoder; distilled **meanflow** UNet, 2 plain Euler steps, no CFG, extra `r` (end-time) embedding | `FlowMean-N500` — encoder + 2-step Euler in-graph, host-supplied noise `z` | bucket pad/crop, seeded z |
| S3Gen vocoder | HiFTNet (identical class to MTL) | `HiFT-T1000` — matmul STFT/iSTFT, SineGen phase/noise as inputs | seeded phase/noise draws |
| Reference conditioning | VoiceEncoder LSTM + S3TokenizerV2 + CAMPPlus | NOT converted (this trial) | precomputed voices ship as tensors (`conds.pt` pattern); ref-wav encoding = follow-up |

Key deltas vs the MTL trial: GPT2 blocks (LayerNorm+bias, fused c_attn,
gelu_new, learned absolute positions — no RoPE), batch-1 everywhere,
`speech_head` has a bias, speech vocab 6563, cond prefix = 1 speaker row +
375 prompt-token rows = 376.

## Parity

| Check | Result |
|---|---|
| T3 wrappers vs stock GPT2 path (fp32 PyTorch) | TBD |
| T3 CoreML fp16 vs wrappers | TBD |
| Flow wrapper vs stock (padded bucket, captured z) | TBD |
| HiFT wrapper vs stock (zeroed SineGen randomness) | TBD |
| Flow CoreML fp16 | TBD |
| HiFT CoreML fp16 | TBD |
| e2e CoreML chain | TBD |

## Gotchas

- HF CDN throttles per-connection (~120 KB/s observed); `aria2c -x16` or
  `CHATTERBOX_NANO_CKPT` local-dir override in `src/nano_ckpt.py`.
- Nano support is not on PyPI (0.1.7) — `chatterbox-tts` is pinned to the
  git commit that ships `tts_turbo(nano=True)` + `S3Gen(meanflow=True)`.
- TBD

## Size

| Artifact | fp16 |
|---|---|
| T3Nano-Prefill | TBD |
| T3Nano-Decode (I/O or stateful) | TBD |
| FlowMean-N500 | TBD |
| HiFT-T1000 | TBD |
| embedding/head tables | TBD |
| **Total** | TBD |

## Performance

- TBD

## Follow-ups

- Voice cloning path (VoiceEncoder, S3TokenizerV2, CAMPPlus, 24 kHz mel).
- Weight sharing between prefill/decode packages; int4/palettization pass.
- Turbo (GPT2-medium, 350M) via the same wrappers — one-line config change.
- Swift port in FluidAudio reusing the Chatterbox MTL backend (PR #907).
