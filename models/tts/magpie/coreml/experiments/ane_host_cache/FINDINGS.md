# decoder_step ANE unblock — §6.3 host-owned cache

Follow-up to the top open item in `SWIFT_PORT_FINDINGS.md` ("try rank-3 K/V + int32
position to unblock ANE"). Applies the *Surgical Inference* §6.3 boundary to the
autoregressive `decoder_step`: the graph does **no in-graph cache mutation** and **no
`position`-driven mask compares** — it reads the past cache read-only, concatenates the
current-step K/V, attends against a **host-supplied additive mask**, and outputs **only the
current-step K/V slice** `[1,1,H,D]`. The host owns the cache append and the mask.

This is the exact move that unlocked the streaming transformer in the paper's falsification
ladder: cache reads and current-token update *outputs* are ANE-admissible; the in-graph
*state write* is the cliff.

## Method

ANE admission is structural (op/shape/state pattern), not weight-dependent, so both graphs
use random weights at Magpie's real decoder dims (12 layers, d_model 768, H12×D64,
max_seq 512, cross-attn to T_enc 256). Both share dims, so the comparison is clean. No NeMo
model / gated download required.

- `exp_convert.py` — builds OLD (current production `TraceableDecoderStep`: in-graph blend
  write + `positions_range == position` compares) and NEW (`HostCacheDecoderStep`, §6.3).
- `exp_probe.py` — drives a real incrementing-`position` decode loop (the condition that
  triggered the documented failure) on CPU_AND_NE / CPU_AND_GPU / CPU_ONLY, plus an
  `MLComputePlan` per-op device breakdown.
- `exp_parity.py` — copies one random weight set into both formulations and compares logits
  across 8 decode steps (torch, float32).

Run (any env with torch + coremltools; NeMo not needed):

```bash
python exp_convert.py && python exp_probe.py && python exp_parity.py
```

## Results (M5 Pro, 24 GB, macOS 26 / coremltools 9.0, 2026-07-16)

**Equivalence (torch, fp32):** max |Δlogits| across 8 steps = **9.5e-7** → the rewrite is a
pure restructuring, not a new model.

**ANE admission + per-step latency (real incrementing position, 40 steps):**

| Graph | CPU_AND_NE p50 | CPU_AND_GPU | CPU_ONLY | Placement (device-assigned ops) |
|---|---|---|---|---|
| OLD (in-graph blend + pos compares) | 10.2 ms | 10.6 ms | 11.4 ms | **100% ANE** (752/752) |
| **NEW (§6.3 host-owned cache)** | **6.0 ms** | 7.7 ms | 8.7 ms | **100% ANE** (555/555) |

Two findings:

1. **The documented M2 `ANECCompile()` failure does not reproduce on M5 Pro.** The current
   graph already compiles and runs 100% on the ANE here (no `-14`), for the full
   incrementing-position loop. The block was older-ANE-compiler-specific.
2. **The §6.3 rewrite is ~40% faster per step regardless** (10.2 → 6.0 ms on ANE), still
   100% ANE, with fewer ops (555 vs 752) and far smaller I/O: current-slice outputs
   `[1,1,12,64]` vs the full-cache `[1,512,12,64]`×24 the old graph re-emits every step
   (the ~19 MB cache-out `PERF.md` flags). decode runs 50–200×/utterance, so this is a
   direct RTFx lever on top of being the portable ANE-safe formulation.

## Can §6.3 cut AR *iterations* (unroll)? No — the sampler is the blocker (`exp_unroll.py`)

The 40% per-step win doesn't change that Magpie is autoregressive (a frame at a time). The
only lever for a bigger gain is an N-frame in-graph unroll (mobius Trial 4a,
`convert_decoder_step_n2.py`, which "ANEF compile fails"). Reconverting that unroll with the
§6.3 decoder isolates the real blocker:

| §6.3 unroll | ANE placement | CPU_AND_NE | CPU-only /frame |
|---|---|---|---|
| N=1 (decoder + LT sampler fused) | **74% ANE** — 200 ops → CPU | fails to bind | 7.6 ms |
| N=2 (2 frames/call) | 71% ANE — 436 ops → CPU | fails to bind | 5.3 ms |

The ~200 CPU ops are the local-transformer **sampling tail** (`topk` / `cumsum` /
`cumsum==1` argmax / one-hot). The §6.3 decoder stays ANE-clean; **fusing the sampler is what
drops the graph off the ANE** and makes `CPU_AND_NE` fail to bind — the same reason Trial 4a
failed, and a reason §6.3 does not address.

Conclusion: the right architecture is exactly what the per-step §6.3 rewrite gives — **§6.3
decoder on ANE (6.0 ms) + sampler on the host (1.6 ms), NOT fused.** The ~2.4×→3.0× RTFx from
§6.3 is the achievable ANE ceiling for Magpie's AR loop. Beating it requires an ANE-friendly
sampler (the int32 topk/cumsum tail is the hard part) or frame-stacking (retraining) — neither
is a reconversion.

## Status / caveats

- **Proof-of-concept, not a production converter.** Random weights → this proves ANE
  admission, placement, latency, and (in torch) numerical equivalence — **not** CoreML fp16
  end-to-end parity with real weights, nor the Swift host-side cache/mask plumbing.
- **To productionize:** port the §6.3 boundary into `traceable_decoder_step.py` +
  `convert_decoder_step.py` with real weights (`from_magpie`), move the cache append + mask
  build into `MagpieModelStore.swift`, and validate audio parity vs the current pipeline.
- `MLComputePlan` worked here (Python 3.11 + coremltools 9.0), unlike the SIGBUS noted in
  `SWIFT_PORT_FINDINGS.md` for compiled `.mlmodelc` on some setups.
