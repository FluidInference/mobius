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
| T3 wrappers vs stock GPT2 path (fp32 PyTorch, 20 cached steps) | logits 1.4e-05 |
| T3 CoreML fp16 I/O-KV (CPU_AND_GPU) vs wrappers | logits 2.2e-02 |
| T3 CoreML fp16 stateful (from-zero sequential feed, 391+20 steps) | logits 1.9e-02 |
| Flow wrapper vs stock (padded bucket, captured z) | mel 1.1e-05 |
| HiFT wrapper vs stock (zeroed SineGen randomness) | wav 1.1e-06 |
| Flow CoreML fp16 (CPU_AND_GPU) | mel max 5.9e-02, mean 1.8e-03 |
| HiFT CoreML fp16 (CPU_AND_GPU) | wav max 3.3e-03, mean 1.2e-04 |
| e2e CoreML chain | renders; Parakeet-v3 round-trip below |

### e2e ASR round-trip (Parakeet v3 via fluidaudiocli)

| Input | CoreML e2e transcript |
|---|---|
| plain sentence | "The quick brown fox jumps over the lazy dog near the riverbank." (verbatim; identical to the PyTorch baseline's transcript) |
| `[chuckle]` tag sentence | "Hi there, Sarah here from Mocha Phone, calling you back. Have you got one minute to chat about the billing issue?" — tag consumed, chuckle audible; the PyTorch baseline's chuckle was even transcribed as "Huh." |

Sampled decodes are not token-comparable across fp16 vs fp32 logits; the
comparison shows the CoreML chain reproduces upstream behavior, not
bit-identical audio.

## Gotchas

- HF CDN throttled per-connection (~120 KB/s observed); `aria2c -x16` or
  the `CHATTERBOX_NANO_CKPT` local-dir override in `src/nano_ckpt.py`.
  Careful: bare multi-URI `aria2c URL1 URL2` treats the URLs as **mirrors
  of one file** and silently interleaves different files — use `-i list`
  with per-entry `out=` (or `-Z`).
- The huggingface_hub Xet path stalled at 0 bytes indefinitely on this
  network; `HF_HUB_DISABLE_XET=1` alone didn't fix throughput, only aria2.
- Nano support is not on PyPI (0.1.7) — `chatterbox-tts` is pinned to the
  git commit that ships `tts_turbo(nano=True)` + `S3Gen(meanflow=True)`.
- Unlike MTL, the tokenizer emits no BOT/EOT wrapping and prefill ends with
  a **single** BOS speech embed (no double-BOS quirk). Don't copy the MTL
  host path.
- Otherwise clean: every wrapper hit parity on the first run — the MTL
  toolkit's padding re-zeroing and matmul-STFT machinery carried over
  unchanged.

## Size

| Artifact | fp16 |
|---|---|
| T3Nano-Prefill-T512-M1536 | 173 MB |
| T3Nano-Decode-M1536 (I/O) | 184 MB |
| T3Nano-Decode-M1536 (stateful) | 184 MB |
| FlowMean-N500 | 228 MB |
| HiFT-T1000 | 40 MB |
| FlowMean-N1000 (extended, opt-in) | 242 MB |
| HiFT-T2000 (extended, opt-in) | 40 MB |
| embedding/head tables + voice | ~88 MB |
| tokenizer | 1.4 MB |
| **Total** | 895 MB naive (all three T3 packages) / ~710 MB shipping one decode variant / ~530 MB with T3 weight sharing (vs ~1.9 GB fp32 checkpoint; MTL CoreML bundle is ~2.3 GB) |

The extended pair (`convert-s3gen-nano.py --bucket 1000`) exists because
the default buckets' *usable* budget is much smaller than the raw numbers
suggest (FluidAudio #924): the built-in voice's 250 prompt tokens + 3
silence tokens leave N500 with ≤247 generated tokens ≈ 9.9 s of audio per
call (N1000: ≤747 ≈ 29.9 s), and the 512 prefill holds 376 rows of voice
conditioning + 1 BOS, leaving ≤135 text BPE tokens. The T3 side needs no
re-export — M1536 already supports ~1020 generated tokens and the GPT2
checkpoint's `wpe` is [8196, 768]. N1000 fp16 parity: flow mel max|d|
1.5e-02 mean 1.7e-03, HiFT wav max|d| 5.5e-03 mean 1.4e-04 (same class as
N500).

## Performance

(M-series Mac, CPU_AND_GPU compute units, GPU shared with other work.)

- Stateful T3 decode: **3.3 ms median/step** ≈ 300 tok/s against the 25 Hz
  speech-token rate → **~12× real-time** for the AR stage (MTL: 16.8 ms).
  Context feed from zero state: 3.5 ms/pos — Swift should still seed
  MLState from prefill KV.
- I/O-KV decode (e2e runs): 9 ms/step (MTL: 38 ms).
- Prefill (T=512): 0.08 s.
- FlowMean-N500: **0.38 s** warm per call (MTL 10-step CFG flow: 4.4–4.7 s
  — the meanflow distillation kills the former RTF bottleneck).
- HiFT-T1000: 0.08–0.10 s.
- e2e compute (6.36 s audio, I/O-KV): 157×9 ms + 0.38 + 0.09 ≈ 1.9 s →
  **~3.4× real-time**; with the stateful decode this projects to ~1.0 s →
  **~6× real-time**. (Wall RTFx in the Python driver is 0.66–1.10 only
  because it reloads/compiles the MLModels per run; Swift keeps them
  resident.)
- PyTorch CPU reference: RTFx 3.1–3.6 (upstream's "3× faster than
  realtime on 8 cores" reproduces).

## Follow-ups

- Voice cloning path (VoiceEncoder, S3TokenizerV2, CAMPPlus, 24 kHz mel).
- Weight sharing between prefill/decode packages; int4/palettization pass.
- Turbo (GPT2-medium, 350M) via the same wrappers — one-line config change.
- Swift port in FluidAudio reusing the Chatterbox MTL backend (PR #907).
