# Magpie ANE overhaul — status

Hardware: Apple M2, 16 GB, macOS 26.5. fp16, iOS17 deployment target.

Cross-phase status of the Magpie TTS ANE residency overhaul. The detailed
findings + tables live in `PHASE_A.md`; this is the single-page index.

## Phase 0 — Baseline (done)

Production CoreML pipeline benchmarked via `fluidaudiocli magpie bench`.

| Component | Static ANE | Runtime | Notes |
|---|---|---|---|
| `text_encoder` | 94–99 % | clean | — |
| `decoder_prefill` | 94–99 % | clean | — |
| `decoder_step` | **99 %** | falls back | per-step ANECCompile recompile under varying `position` |
| `nanocodec_decoder` | **0 %** (1149 ops) | all CPU | `ANECCompile() FAILED` whole-graph |

Numbers: 47.4 ms/decoder step, 0.44× RTFx end-to-end, 23.49 s nanocodec
per sentence.

## Phase A — Per-module ANE diagnostics (done)

`per_module/analyze.py` converts ~14 isolated nn.Modules and reports static
ANE residency.

Confirmed:
- **Snake (sin) is op-level rejected by ANE** — `Conv1d → Snake → Conv1d`
  lands 0 % ANE; identical block with polynomial Snake lands 100 %.
- **weight_norm is NOT a blocker** — both unfolded and folded land 100 %
  identical 2-op graphs.
- **Rank-4 one-hot KV cache write is NOT a blocker** — full causal
  self-attn with KV write at decoder shapes lands 100 % ANE statically.

→ Production `decoder_step` runtime fallback is therefore not op-level. It
   is a runtime recompile churn. Investigated separately in Phase D.

Full table + raw data: `PHASE_A.md`, `ledger.json`, `raw/*.json`.

## Phase B — Snake replacement in convert_nanocodec.py (done)

`convert_nanocodec.py` updated: `_snake_plain` now uses a clamped 5th-order
Taylor expansion of `sin²(α·x)` (`SnakeTaylor5Clipped`):

```python
ax = clamp(α · x, -π/2, π/2)
sin² ≈ ax² - ax⁴/3 + 2·ax⁶/45
return x + sin² / α
```

This was the minimum needed to remove `ios17.sin` from the converted graph.
Audio parity is **insufficient** at this approximation: **11.56 dB SNR**
vs the reference sin² Snake. Replacement candidates for Phase C+ v2:
LUT-via-conv (preferred), Padé approximant, or `mod π` + Taylor5.

## Phase C — Re-convert + analyze (done; result: still 0 % ANE)

Patched nanocodec (1821 ops, no sin / no pow) re-profiled with
`coreml-cli --fallback`:

| Build | Total ops | ANE % | Failure |
|---|---|---|---|
| baseline (sin²) | 1149 | 0.0 | `ANECCompile() FAILED` |
| Taylor5Clipped patch | 1821 | 0.0 | `ANECCompile() FAILED` |

Snake fix was necessary but not sufficient. Whole-graph compile failed.

## Phase C+ — Subgraph probe / threshold finding (done; root cause)

`per_module/nano_subgraph_probe.py` builds synthetic HiFi-GAN-style
decoders progressively from a single ResBlock up through the full 5-stage
decoder, holding topology constant while varying the input time dim.

| Spec | T_in | T_out | total ops | ANE % | failure mode |
|---|---|---|---|---|---|
| `body_5stage` | 8 | 4096 | 1006 | 99.0 | 10 dilated convs CPU |
| `body_5stage_T16` | 16 | 8192 | 1006 | **98.8** | 10 dilated + 2 ops `W=16386 ∉ [1, 16384]` |
| `body_5stage_T20` | 20 | 10240 | 1006 | **0.0** | **ANECCompile() FAILED** |
| `body_5stage_T24` | 24 | 12288 | 1006 | 0.0 | failed |
| `body_5stage_T32` | 32 | 16384 | 1006 | 0.0 | failed |
| `body_5stage_T64` | 64 | 32768 | 1006 | 0.0 | failed |
| `body_5stage_T128` | 128 | 65536 | 1006 | 0.0 | failed |
| `full_decoder_T8` | 8 | 2048 | 1019 | **99.0** | 10 dilated convs CPU |
| `full_decoder` (T=256) | 256 | 262144 | 1019 | 0.0 | ANECCompile() FAILED |

### Root cause

ANE compiler imposes a hard **W ≤ 16384** dimension limit on the
space-to-batch lowering of dilated convs (HiFi-GAN dilations 1/3/5). At
T_out=8192 the W-after-lowering is 16386, just over the limit, and the
two affected ops fall back individually. At T_out ≥ 10240 the ANE
compiler refuses to plan the entire graph and the whole codec falls to
CPU under the catch-all `ANECCompile() FAILED`.

Topology, Snake replacement, weight_norm folding, kernel-7 pre_conv,
post-act, post-conv, and tanh are **NOT** the trigger. Activation tensor
size is.

### Fix path (Phase C v2 — pending)

Convert `nanocodec_decoder.mlmodelc` with **fixed input shape
`(1, 432, 16)`** (T_in ≤ 16 codec tokens, T_out = 8192 audio samples per
call). Update Swift `MagpieSynthesizer` to slide a 16-token window over
the codec output sequence and stitch the 8192-sample output chunks. For
a typical 100-token sentence: ~7 sequential codec calls instead of one
batched call. Each call should be ~99 % ANE-resident (the 10 dilated
convs out of 1006 fall back to CPU regardless of chunk size).

## Phase C+ — Audio parity (done)

`per_module/audio_parity.py` and `snake_parity.py`. Random codec tokens,
seed=42, T=256:

| Metric | Value |
|---|---|
| samples | 262 144 |
| max_abs error | 3.86e-1 |
| mean_abs error | 7.06e-3 |
| RMS ref | 5.73e-2 |
| RMS err | 1.51e-2 |
| **SNR** | **11.56 dB** |

Insufficient. The clamp at α·x = ±π/2 plus a 5th-order Taylor diverges
from sin² across the codec's full operating range (codec α can train up
to ~5; codec activations up to ~3 → α·x ≳ 10, well beyond the clamp).

Replacement plan (deferred until chunked nanocodec lands ANE):
1. **LUT-backed sin via Conv1d** (preferred) — encode sin(α·x) as a 1-D
   lookup table evaluated by a depthwise conv against a sentinel basis.
   Numerically exact to LUT resolution. Adds ~96 small convs.
2. Padé approximant of sin² in [-π/2, π/2].
3. Range reduction with `sin²(y + π) = sin²(y)` then Taylor5 over a
   smaller domain (ANE compatibility of `mod` is unverified).

## Pending

### Phase C v2 — chunked nanocodec
1. Rewrite `convert_nanocodec.py` for fixed input shape `(1, 432, 16)`.
2. Re-convert + verify ≥99 % ANE residency via `coreml-cli --fallback`.
3. Update Swift `MagpieSynthesizer` to slide a T=16 window with
   appropriate overlap (HiFi-GAN receptive field ≈ k·dilation per layer).
4. Re-run `magpie bench`, compare to Phase 0 baseline.

### Phase C+ — better Snake replacement
1. Implement LUT-via-conv Snake.
2. Verify SNR > 30 dB against chunked sin² reference.
3. Re-convert nanocodec with the better Snake; verify ANE residency
   unchanged.

### Phase D — decoder_step runtime recompile
Static analysis says 99 % ANE, runtime says CPU. Hypotheses:
- Per-step recompile churn under varying `position` (ANE caches on
  tensor values, not just shapes).
- Float dtype of `position` triggering recompile.
- Stateful program replacement bug.
1. Reproduce in isolation by running `decoder_step.mlmodelc` 100× with
   varying `position` and instrumenting `MLComputePlan` per call.
2. Try int32 `position` (despite user's earlier rejection of int32, this
   is a measurement only, no integration).
3. Try fixed `position=0` to confirm shape-vs-value cache hypothesis.

### Phase D — fused AR step
Merge `decoder_step + final_proj + local_transformer + 8 heads` into one
mlmodelc to eliminate per-call dispatch overhead. Update Swift port.

## Files

```
per_module/
├── __init__.py
├── modules.py                # diagnostic nn.Module wrappers (Snake variants, KV cache, weight_norm)
├── analyze.py                # Phase A driver: per-module conversion + ANE coverage
├── snake_parity.py           # Snake polynomial accuracy vs sin² reference
├── audio_parity.py           # codec output SNR vs PyTorch reference
├── nano_subgraph_probe.py    # Phase C+ progressive HiFi-GAN subgraph probe
└── results/
    ├── PHASE_A.md            # detailed Phase A + C+ findings (the report)
    ├── STATUS.md             # this file (cross-phase index)
    ├── ledger.json           # Phase A summary table
    ├── raw/                  # Phase A per-spec coreml-cli output
    ├── subgraph_ledger.json  # Phase C+ summary table
    └── subgraph_raw/         # Phase C+ per-spec coreml-cli output
```

## Reproducing

```bash
cd mobius/models/tts/magpie/coreml
uv sync

# Phase A (per-module ANE diagnostics)
uv run python per_module/analyze.py
cat per_module/results/ledger.json

# Phase C+ (subgraph threshold probe)
uv run python per_module/nano_subgraph_probe.py
cat per_module/results/subgraph_ledger.json
```
