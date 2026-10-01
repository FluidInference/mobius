# Trials

## T1 — LM graphs (2026-09-13)
- Pass-through KV following the NeuTTS layout `[L,1,H,M,D]`; one-hot blend write at `cur_len`.
- Fused frame graph unrolls 17 local-transformer passes (no local KV — prefix ≤ 17 tokens, recompute is
  cheaper than a cache round trip) and samples in-graph. `torch.clamp(max=)` lowers to `clip` with a
  tensor `beta` that Core ML rejects → `torch.minimum`.
- Sampling: exact inverse-CDF in sorted space (top-k threshold → top-p keep on exclusive cumsum →
  softmax → count(cdf < u)). Matches upstream in distribution, greedy path bit-matches (argmax).
- fp16 greedy replay: 370/375 exact over 30 s; mismatches are near-tie argmax flips (11/16, 10/16, …).

## T2 — Codec (2026-09-13)
- HF remote code (3.3k lines) ≠ GitHub repo code: attention masks by `input_lengths`, so wrappers must
  pass shape-derived lengths (`ones_like(x[:,0]).sum()`), not scalar 1 → first attempt gave −10 dB.
- Upstream `apply_rope` computes `2 / D` from a traced size → CoreML `inverse` on int32. Patched the
  module-level function with a static-D version for export.
- Full decoder builds T×T masks per stage (last stage 32 tokens/frame): keep RangeDim ≤ 125 frames.
- Encoder: fp16 → 1044/1584 codes exact, degrading from 0.96 (codebook 0) to 0.5 (codebook 15); fp32 exact.
- Streaming step: shift-append caches sized to each stage's context (500/800/1200/1600), unrotated K
  cached, RoPE applied with constant *relative* tables (q at cap−n+j, k at i) — identical dot products
  to absolute positions, zero runtime trig, fp16-safe. Torch step vs full decode 78.9 dB; CoreML GPU 56.6 dB.
- Compute units: `ALL` splits the 26-input step graph across units and costs 69 ms/frame in the
  per-frame loop vs 5–8 ms on a single unit — pin CodecStep to GPU (or ANE).
