# Chatterbox Multilingual (23-lang) → CoreML — Trial Report

Conversion trial for [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)
multilingual (`t3_mtl23ls_v2` + `s3gen`), requested in FluidAudio issue #49
("How about the Chatterbox multilanguage model?" — the 3.2 GB fp32 footprint
was the stated concern).

## Architecture / export boundary

| Component | Upstream | CoreML export | Host (Swift) responsibility |
|---|---|---|---|
| T3 | Llama-520M, CFG batch 2, learned pos embs, 23-lang tokenizer | `T3-Prefill` (T=256, M=1024) + `T3-Decode` (I/O KV) + `T3-Decode-stateful` (MLState, iOS 18+) | tokenization, embedding prep (tables exported), CFG combine, sampling, AlignmentStreamAnalyzer |
| Alignment analyzer | attention-hook heuristics forcing/suppressing EOS | 3 head rows (L12H15, L13H11, L9H2) exported as `align_attn` output | `verify/analyzer_port.py` is the port spec |
| S3Gen flow | CosyVoice2-style UpsampleConformer + causal UNet CFM, 10 Euler steps | `Flow-N500` — encoder + full CFG Euler loop in-graph, host-supplied noise `z` | bucket pad/crop, seeded z |
| S3Gen vocoder | HiFTNet (NSF + iSTFT), f0 ConvRNN | `HiFT-T1000` — matmul STFT/iSTFT, SineGen phase/noise as inputs | seeded phase/noise draws |
| Reference conditioning | VoiceEncoder LSTM + S3TokenizerV2 + CAMPPlus | NOT converted (this trial) | precomputed voices ship as tensors (`conds.pt` pattern); ref-wav encoding = follow-up |

## Parity

| Check | Result |
|---|---|
| T3 wrappers vs stock eager path (fp32 PyTorch, 12 cached steps) | logits 3.8e-05, align rows 0.0 |
| T3 CoreML fp16 (CPU_AND_GPU) vs wrappers | logits 2.6e-02 (range ±15), align 1.4e-03 |
| Flow wrapper vs stock (padded bucket) | mel 1.0e-05 after embed re-zeroing fix |
| HiFT wrapper vs stock (zeroed SineGen randomness) | wav 1.3e-05 |
| Flow CoreML fp16 | TBD |
| HiFT CoreML fp16 | TBD |
| e2e CoreML chain (en/de/fr) | TBD |

## Gotchas (chronological)

- `resemble-perth` imports `pkg_resources`; needs `setuptools<81`.
- Stock prefill context ends with **two** BOS embeds (`prepare_input_embeds`
  already appends one, `inference()` cats another). The analyzer's first
  chunk reads both query rows. Replicated, not "fixed".
- `CPU_ONLY` predict on the T3 packages hard-crashes the process
  (matches the Chatterbox-Flash CoreML README's warning). Use GPU/ANE units.
- `CPU_AND_NE` fails `ANECCompile` on the `[30,2,16,1024,64]` KV I/O
  tensors. Stateful MLState decode is the perf path; ANE prefill would need
  per-layer KV outputs (untried).
- UpsampleConformerEncoder is padding-sensitive: `embed`'s Linear+LayerNorm
  turns zero rows into a bias vector and the right-looking `pre_lookahead`
  conv reads it at the tail of the valid region (max mel error 0.86 before
  the fix). Wrapper re-zeroes padded positions after each embed stage →
  bit-exact vs the unpadded run.
- espnet `rel_shift` uses `view_as` (no MIL lowering) — patched to reshape.
- SineGen (non-causal HiFTNet variant) draws random phase + noise per call;
  both are graph inputs now, drawn host-side.

## Size

| Artifact | fp16 |
|---|---|
| T3-Prefill | 977 MB |
| T3-Decode (I/O or stateful) | 977 MB |
| Flow-N500 | TBD |
| HiFT-T1000 | TBD |
| embedding/pos/head tables | TBD |
| **Total** | TBD (vs 3.2 GB fp32 checkpoints) |

Only ONE T3 weight copy is needed on device if prefill/decode share a
palettized/linked artifact — not attempted in this trial; the naive bundle
ships both.

## Performance

TBD — stateful decode step time, flow, hift on M-series GPU.

## Follow-ups

- Voice cloning path: convert VoiceEncoder (LSTM), S3TokenizerV2, CAMPPlus,
  and the 24 kHz mel extractor, or keep ref encoding as an offline Python
  step per voice.
- Weight sharing between prefill/decode packages; int4/palettization pass.
- ANE residency experiments (per-layer KV I/O, fp32 pins for RMSNorm).
- `t3_mtl23ls_v3` / `s3gen_v3` checkpoints (newer revs on the HF repo).
- Swift port in FluidAudio (`ChatterboxManager`), reusing the cosyvoice3
  stateful-decode host pattern.
