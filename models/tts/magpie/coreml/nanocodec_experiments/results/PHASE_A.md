# Phase A — Per-module ANE diagnostics

Goal: identify the actual ANE blockers in Magpie TTS by converting individual
nn.Modules in isolation and measuring static ANE residency via
`coreml-cli --fallback`.

Method: `torch.jit.trace` → `ct.convert` (mlprogram, fp16, iOS17) →
`xcrun coremlcompiler compile` → `coreml-cli --fallback --json`.

Hardware: Apple M2, 16 GB, macOS 26.5.

## Summary table

| Spec | ANE % | Ops (CPU/total) | Notes |
|---|---|---|---|
| `snake_learned` (standalone) | 0.0 | 4/4 | sin op rejected; graph too small |
| `snake_poly_taylor` (standalone) | 0.0 | 6/6 | scheduler chose CPU (graph too small) |
| `snake_no_sin_pow` (standalone) | 0.0 | 6/6 | scheduler chose CPU (graph too small) |
| `kv_write_rank4_onehot` (standalone) | 0.0 | 15/15 | scheduler chose CPU (graph too small) |
| `kv_write_rank3_onehot` (standalone) | 0.0 | 15/15 | scheduler chose CPU (graph too small) |
| `kv_write_host_concat` (standalone) | 0.0 | 5/5 | scheduler chose CPU (graph too small) |
| `wn_conv1d_unfolded` | **100.0** | 0/2 | weight_norm parametrization is not a blocker |
| `wn_conv1d_folded` | **100.0** | 0/2 | identical to unfolded |
| `snake_learned_block` (Conv→Snake→Conv) | **0.0** | 6/6 | **sin op explicitly rejected** ← real blocker |
| `snake_poly_taylor_block` | **100.0** | 0/8 | ✓ ANE-clean replacement |
| `snake_no_sin_pow_block` | **100.0** | 0/8 | ✓ ANE-clean replacement |
| `causal_self_attn_rank4_cache` | **100.0** | 0/36 | rank-4 onehot KV write is 100% ANE statically |
| `kv_attn_rank4_block` | **100.0** | 0/35 | same |

## Findings

### 1. Snake activation is the real nanocodec ANE blocker

`Conv1d → Snake(α) → Conv1d` lands **0% ANE**. The cause is unambiguous:

```
1 x This platform doesn't support Neuron ZinIrNeuronType::kSin
    (aka ZinIrNonLinearSin)
4 x ANE supported but scheduler chose CPU
1 x ANE not available for this op  [ios17.conv:1]
```

`ios17.sin` is op-level rejected by the Neural Engine. The single sin pulls
the surrounding Conv1d ops to CPU because the scheduler refuses to ship a
fragmented sub-graph.

This explains the **0% ANE / 1149 ops on CPU** result observed for the
production `nanocodec_decoder.mlmodelc` in Phase 0. The codec contains 96
Snake instances, each containing a sin op.

### 2. Polynomial replacements are 100% ANE

Two drop-in replacements both land 100% ANE inside the same Conv1d ResBlock:

* **SnakePolyTaylor**: `x + α·x² − (α³/3)·x⁴` (3rd-order Taylor expansion of
  the original `x + (1/α)·sin²(α·x)`). Valid when `|α·x| ≲ 1.5`; codec
  activations are bounded by upstream LayerNorm/Conv1d so the operating
  range fits.
* **SnakeNoSinPow**: same polynomial, written entirely with `mul/add/sub` —
  no `pow`. Slightly more permissive in case `pow` is also a future ANE
  blocker.

Either is suitable for Phase B. SnakeNoSinPow is preferred for portability.

### 3. weight_norm parametrization is **not** a blocker

Hypothesis from `SWIFT_PORT_FINDINGS.md` was that PyTorch's `weight_norm`
parametrization survives `torch.jit.trace` and emits per-conv normalize
arithmetic that ANE rejects.

Actual measurement: `WeightNormConv1dUnfolded` (parametrization left in
place) and `WeightNormConv1dFolded` (`remove_weight_norm()` called) **both
land 100% ANE with identical 2-op graphs**. The tracer / coremltools
constant-folds the weight reconstruction at trace time.

→ Drop weight_norm folding from Phase B. It's a no-op.

### 4. Rank-4 onehot KV cache write is 100% ANE statically

The full causal self-attention with rank-4 onehot KV write at decoder_step
shape (`d_model=768, H=12, max_seq=512`) compiles **100% ANE** statically.
Rank-3 vs rank-4 KV layout produced identical fallback profiles.

This means the `decoder_step.mlmodelc` per-step ANECCompile() failures
observed during real synth in Phase 0 are **not** caused by op-level ANE
rejection. The static graph is ANE-clean. The runtime issue must be:

* per-step recompile churn under varying `position` (ANE caches on tensor
  shapes/values, not just the graph), or
* float dtype of `position` triggering recompile, or
* the decoder being replaced under a stateful program

Phase B will not touch the KV layout. The runtime recompile issue is
investigated separately (Phase D).

## Phase B plan (revised)

1. **Replace Snake in `convert_nanocodec.py`** with `SnakeNoSinPow`. Verify
   nanocodec_decoder lands 100% ANE.
2. **Skip** weight_norm folding (no-op).
3. **Skip** KV cache layout rewrite — static graph is fine.

## Phase C plan (revised)

1. Re-convert nanocodec only (text_encoder, decoder_prefill, decoder_step
   were already at 94–99% ANE statically).
2. Re-run `magpie bench` and compare against the Phase 0 baseline:
   * 0.44× RTFx
   * 47.4 ms/step decoder
   * 23.49s nanocodec time

## Phase B → C addendum: Snake replacement is necessary but not sufficient

After patching `convert_nanocodec.py` to use `SnakeTaylor5Clipped` (clamped 5th-order
Taylor expansion, see `_snake_plain` in that file), the converted nanocodec was
re-profiled with `coreml-cli --fallback`:

| Build | Total ops | ANE % | Notes |
|---|---|---|---|
| Baseline (sin²) | 1149 | 0.0% | `ANECCompile() FAILED` (whole-graph) |
| Patched (Taylor5Clipped) | 1821 | 0.0% | `ANECCompile() FAILED` (whole-graph) |

Both report all ops with the catch-all reason `"ANE not available for this op"`,
which is the analyzer's fallback when MLComputePlan can't produce an ANE plan
for the graph at all. The trailing stderr message `E5RT … MILCompilerForANE
error: failed to compile ANE model using ANEF. Error=_ANECompiler :
ANECCompile() FAILED.` confirms this.

Implications:

1. **The 0% ANE on baseline nanocodec was never purely about sin/pow** — even
   when the per-op rejections in Phase A unambiguously identify sin as ANE-
   incompatible, the *whole-graph* compile failure suggests a separate
   architectural issue with the 1149+ op codec graph (96 Snake instances,
   92 conv_transpose, 97 concat, many residual paths).

2. **The Snake replacement is still required** before any deeper investigation
   — without it, sin would block the smaller subgraphs we'd want to probe.

3. **Next investigation**: convert a single HiFi-GAN ResBlock with Taylor5Clipped
   in isolation (and progressively grow it) to find the graph-size or pattern
   threshold that triggers `ANECCompile() FAILED` for the full codec.

Trace verification at PyTorch fp32 was clean (`max diff = 0.0`); CoreML CPU
output range was sane (`[-0.54, 0.71]`).

## Phase C+ subgraph probe — ANE compile threshold pinned

`nanocodec_experiments/nano_subgraph_probe.py` builds synthetic HiFi-GAN-style decoders
(Taylor5Clipped Snake, weight_norm-free Conv1d, dilations 1/3/5, kernels
3/7/11) progressively from a single ResBlock up through the full 5-stage
decoder, holding topology constant while varying the input time dimension.

Result table (M2, macOS 26.5, fp16, iOS17 deployment target):

| Spec | T_in | T_out | total ops | ANE % | Failure mode |
|---|---|---|---|---|---|
| `res_block_27`        | 1024 | 1024 | 24 | 0.0  | scheduler chose CPU (graph too small) |
| `hifigan_resblock_27` | 1024 | 1024 | 70 | **100.0** | clean |
| `hifigan_reslayer_27` | 1024 | 1024 | 191 | 91.1 | 2 dilated + 14 scheduler-CPU |
| `stage_27`            | 1024 | 2048 | 202 | 99.0 | 2 dilated convs CPU |
| `body_2stage`         | 16   | 1024 | 403 | 99.0 | 4 dilated convs CPU |
| `body_3stage`         | 16   | 1024 | 604 | 99.0 | 6 dilated convs CPU |
| `body_4stage`         | 16   | 1024 | 805 | 99.0 | 8 dilated convs CPU |
| `body_5stage`         | 8    | 4096 | 1006 | 99.0 | 10 dilated convs CPU |
| `body_5stage_T16`     | 16   | 8192 | 1006 | **98.8** | 10 dilated + **2 ops "W=16386 ∉ [1, 16384]"** |
| `body_5stage_T20`     | 20   | 10240 | 1006 | **0.0** | **ANECCompile() FAILED** (whole graph) |
| `body_5stage_T24`     | 24   | 12288 | 1006 | 0.0 | ANECCompile() FAILED |
| `body_5stage_T32`     | 32   | 16384 | 1006 | 0.0 | ANECCompile() FAILED |
| `body_5stage_T64`     | 64   | 32768 | 1006 | 0.0 | ANECCompile() FAILED |
| `body_5stage_T128`    | 128  | 65536 | 1006 | 0.0 | ANECCompile() FAILED |
| `full_decoder_T8`     | 8    | 2048 | 1019 | **99.0** | 10 dilated convs CPU |
| `full_decoder` (T=256)| 256  | 262144 | 1019 | 0.0 | ANECCompile() FAILED |

### Findings

1. **The 0% ANE failure on production nanocodec is NOT a graph topology
   issue.** Identical 5-stage body graph compiles to 99% ANE at T_out=4096
   and 98.8% at T_out=8192, but to 0% at T_out=10240 and above. The Snake
   replacement, weight_norm folding, kernel-7 pre_conv, and tanh post-act
   are all unrelated.

2. **The trigger is the ANE's 16384-cell W-dimension limit on dilated-conv
   space-to-batch lowering.** At T_in=16 the rejection report cleanly
   names the constraint:

   ```
   2 x Tensor dimensions N1D1C27H1W16386 are not within supported range,
       N[1-65536]D[1-16384]C[1-65536]H[1-16384]W[1-16384].
       [ios17.conv:2]
   ```

   The W=16386 is `2·T_out + 2 = 16386` for T_out=8192 (space-to-batch
   doubles the spatial extent for dilation=2/3/5 paths). Above T_out=8192,
   even a single dilated conv lowering blows the budget, and the ANE
   compiler refuses to plan the entire graph rather than partition it.

3. **Threshold is `T_out ≤ 8192` per inference call.** With the magpie
   nanocodec stride product `8·8·4·2·2 = 512`, this translates to
   **`T_in ≤ 16 codec tokens per nanocodec call`**. At ~21.5 codec fps,
   T_in=16 corresponds to ≈743 ms of audio per call.

4. **Snake polynomial replacement is still required**, but only as a
   prerequisite — without it, the per-op sin rejection would block the
   smaller subgraphs we just used to find the threshold.

### Implication for Phase C v2

The fix is not "make the graph smaller" or "rewrite Snake better" — it is
**chunk the nanocodec call at T_in≤16 tokens and concatenate outputs**:

* Build `nanocodec_decoder.mlmodelc` with fixed input shape `(1, 432, 16)`.
* In Swift, slide a 16-token window over the codec sequence (with whatever
  overlap is needed for the receptive field) and stitch the 8192-sample
  output chunks.
* For a typical 100-token sentence: ~7 sequential nanocodec calls instead
  of one batched call.

The dilated-conv CPU fallback (10 convs out of 1006) is unfixable at the
ANE level — those 10 convs sit on CPU regardless of chunk size. Total ANE
residency per chunk is 99.0%.

### Implication for Phase C+ Snake replacement quality

With the size threshold understood, the Taylor5Clipped Snake replacement
can now be revisited in a chunked context. Audio parity (11.56 dB SNR) is
still insufficient for production, so the LUT-via-conv approach remains
the planned next step — but it should be measured against a chunked
T_in=16 reference, not the full-sequence reference.

### Audio parity vs original sin² Snake (random tokens, seed=42, T=256)

| Metric | Value |
|---|---|
| samples compared | 262 144 |
| PyTorch range | [-0.473, 0.474] |
| CoreML range  | [-0.470, 0.445] |
| max_abs error | 3.86e-1 |
| mean_abs error | 7.06e-3 |
| RMS ref | 5.73e-2 |
| RMS err | 1.51e-2 |
| **SNR** | **11.56 dB** |

11.56 dB is **not** acceptable for synthesis quality — the clamp at α·x = ±π/2
plus 5th-order Taylor diverges enough from sin² across the codec's full
operating range to produce audible distortion (~25% relative RMS error). The
output is in the right range and not NaN, but the codec activations have
enough power above α·x = π/2 (where the clamp kicks in and the polynomial
freezes) that the residual reconstruction error is significant.

**Implication for Phase B:** Taylor5Clipped is sufficient as a *conversion
probe* (it lets us continue investigating the whole-graph ANECCompile failure
without sin/pow blockers), but it is **not** the final replacement. Options
for Phase B v2:

1. **LUT-backed sin via Conv1d**: encode sin(α·x) as a 1-D lookup table,
   evaluated by a depthwise conv against a sentinel basis. Numerically exact
   to LUT resolution, no transcendental ops. Adds ~96 small convs.
2. **CORDIC or Padé approximant** of sin² in [-π/2, π/2]: rational
   approximation with much higher accuracy than degree-5 Taylor, still
   composed of mul/add/div which are ANE-clean.
3. **Range reduction with sin² periodicity**: `sin²(y + π) = sin²(y)`, so we
   can fold any α·x into [0, π] via `mod π`, then approximate over that
   smaller domain where Taylor5 is more accurate. ANE compatibility of `mod`
   is unverified.
4. **Train α down**: re-tune the codec with a regularizer that penalizes
   |α| > 1 so the clamp never bites. Out of scope for inference-time
   conversion.

Option 1 (LUT-via-conv) is the most likely to pass both audio parity AND
ANE — defer until after the whole-graph compile failure is understood, since
adding 96 more convs is wasted effort if the graph still won't compile for
ANE.

## Reproducing

```bash
cd mobius/models/tts/magpie/coreml
uv run python nanocodec_experiments/analyze.py
# Or selectively:
uv run python nanocodec_experiments/analyze.py --only snake_learned_block snake_poly_taylor_block
cat nanocodec_experiments/results/ledger.json
```

Raw fallback JSON for each spec is in `nanocodec_experiments/results/raw/<name>.json`.
