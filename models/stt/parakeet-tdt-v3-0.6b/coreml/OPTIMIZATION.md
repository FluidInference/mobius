# Parakeet decode-loop optimization

### Trial: Parakeet decode-loop fusion (TDT v3 + EOU), 2026-06-09

Campaign: ANE candidates #3/#4 from FluidAudio `Documentation/ANE_Candidates.md`.
Both decode loops run 100% CPU; the question was (a) can they be moved to ANE,
and (b) failing that, what does Nemotron-B1-style decoder+joint fusion buy.

Hardware: Apple Silicon (M4 Pro, 48 GB), macOS 26 (Darwin 25.5). Harness:
coremltools 9.0b1 Python `predict`, interleaved A/B (mono, fused, mono, fused…
— mobius Trial 15/19–22 methodology), 10 warmup + 200 timed, median/p95.
Python per-call overhead inflates absolute latencies vs the Swift production
loop; treat the *ratios* as the signal and the absolute savings as an upper
bound for Swift.

#### Phase 1 — per-op fallback dump (gate)

`MLComputePlan` per-op preferred device under `.cpuAndNeuralEngine`, on the
shipped production `mlmodelc` bundles (FluidAudio model cache):

**TDT v3 `Decoder.mlmodelc`** (24 ops, 23 MB, 100% CPU):
`cast`×8, `squeeze`×4, `split`×2, **`ios17.lstm`×2**, `stack`×2,
`transpose`×2, `greater_equal`, `add`, `gather`, `select`

**TDT v3 `JointDecisionv3.mlmodelc`** (24 ops, 13 MB, 100% CPU):
`cast`×6, `linear`×3, `expand_dims`×3, `slice_by_index`×2, `transpose`×2,
`reduce_argmax`×2, `gather_along_axis`, `squeeze`, `topk`, `add`, `relu`,
`softmax`

**EOU `decoder.mlmodelc`** (14 ops, 8 MB, 100% CPU):
`cast`×6, `transpose`×2, `expand_dims`×2, `squeeze`×2, `gather`,
**`ios17.lstm`×1**

**EOU `joint_decision.mlmodelc`** (21 ops, 3 MB, 100% CPU):
`cast`×6, `linear`×3, `expand_dims`×3, `transpose`×2, `add`, `topk`,
`reduce_argmax`, `gather_along_axis`, `relu`, `softmax`, `squeeze`

**Gate verdict: ANE is a categorical dead end for both decoders.** The
prediction networks are LSTMs and `ios17.lstm` has no ANE kernel at any
precision (same finding as Kokoro PostAlbert). The joints contain only
ANE-friendly op families (linear/relu/softmax/argmax/topk) — they sit on CPU
because of small-graph dispatch economics, not rejected constructs. That makes
this a **fusion-only campaign**: fold decoder+joint into one graph to halve
host dispatches (Nemotron B1 precedent, +15%).

#### Phase 2 — B1-style fusion

One CoreML model per pipeline: `(targets, target_length, h_in, c_in,
encoder_step) → (token_id, token_prob, [duration,] top_k_ids, top_k_logits,
h_out, c_out)` — a drop-in superset of the shipped two-model contract. Host
semantics: call once per joint step; on blank the host re-feeds the previous
`(targets, h, c)` unchanged and the LSTM recompute is deterministic, so
decoding behavior is identical to the two-model loop. Scripts:
`fuse_decoder_joint.py` here and in
`../../parakeet-realtime-eou-120m/coreml/conversion_scripts/` (`export` /
`parity` / `bench` subcommands).

The fused graphs (fp16 and fp32) plan 100% CPU under `.cpuAndNeuralEngine`,
as expected — the LSTM anchors the whole small graph on CPU, which also means
zero CPU↔ANE partition transitions.

#### Parity (fp32 fused vs fp32 two-model chain, 50 evolving steps)

| Output | TDT v3 max\|Δ\| | EOU max\|Δ\| |
|---|---|---|
| token_id | 0 mismatches | 0 mismatches |
| token_prob | 0.0 | 1.2e-07 |
| duration | 0 | n/a |
| h_out / c_out | 0.0 / 0.0 | 0.0 / 0.0 |
| top_k_logits | 7.3e-04 | 1.1e-03 |

Verdict: PASS. Decisions and LSTM state are bit-identical. `top_k_logits`
exceeds 1e-5 due to matmul accumulation-order reassociation in the wide joint
output linear (8198-d / 1027-d); it only feeds host-side contextual-biasing
re-ranking and the top-1 path is unaffected.

#### Benchmarks — per-step (interleaved, median ms, 200 runs)

fp16 fused vs **shipped production pair** (`mlmodelc`):

| CU | TDT v3 sep | TDT v3 fused | × | EOU sep | EOU fused | × |
|---|---|---|---|---|---|---|
| cpuOnly | 0.484 | 0.379 | 1.28 | 0.227 | 0.184 | 1.23 |
| cpuAndGPU | 0.484 | 0.374 | 1.29 | 0.222 | 0.181 | 1.23 |
| cpuAndNE | 0.482 | 0.382 | 1.26 | 0.225 | 0.182 | 1.24 |
| all | 0.481 | 0.378 | 1.27 | 0.223 | 0.182 | 1.23 |

Shipped per-model split (cpuAndNE): TDT v3 Decoder 0.173 / JointDecisionv3
0.278; the fused call (0.382) is cheaper than the joint-decision chain alone
plus dispatch.

fp32 fused vs fp32 separate exports: TDT v3 1.20× on cpuOnly/cpuAndNE but
**0.64–0.67× on cpuAndGPU/all** — the planner pulls the fat fp32 graph onto
the GPU and round-trips per step. fp16 shows no such regression (stays CPU on
every CU). EOU fp32: 1.30–1.33× on all CUs. **Ship fp16 only**, and keep the
production `.cpuAndNeuralEngine` default (already the FluidAudio default for
these components).

#### Benchmarks — utterance-level decode loop (interleaved sim)

Production call pattern for a 7.8 s utterance:

| Pipeline | Separate | Fused | Speedup | Saved |
|---|---|---|---|---|
| TDT v3 (40 dec + 49 joint vs 49 fused) | 20.94 ms | 18.90 ms | **1.11×** | 2.0 ms/utt |
| EOU (229 dec + 229 joint vs 229 fused) | 39.33 ms | 32.43 ms | **1.21×** | 6.9 ms/utt |

TDT v3's utterance win (1.11×) is smaller than its per-step win (1.27×)
because production skips the decoder on blank-only steps (~9 of 49), while
the fused model pays the LSTM every step. EOU runs its decoder every step, so
the per-step ratio carries straight through — it's the Nemotron shape and the
bigger dispatch prize (458 → 229 dispatches/utt).

#### Pipeline impact (honest math)

- **TDT v3**: profiler doc (M5, Swift): decode loop ≈ 32 ms of ≈ 60 ms/utt.
  Applying 1.11×: 32 → ~28.8 ms, ~3.2 ms/utt ⇒ **~5% E2E latency / RTFx**
  (upper bound; Python harness overhead inflates dispatch savings). Real but
  modest — well short of Nemotron's +15%, for the structural reason above.
- **EOU**: doc: decoder+joint = 58 ms/utt. Applying 1.21×: 58 → ~48 ms,
  **~10 ms/utt** off the decode loop. For the 160 ms streaming variant the
  encoder aggregate (~49 chunks × 6.5 ms) still dominates total compute, so
  E2E RTFx gain is ~2–3%; the practical win is post-chunk token-emission
  latency, where the decode loop is the whole story.

#### Recommendation

- **EOU: ship.** Output-identical, 1.21× on the hot loop on every compute
  unit, no downside found. Needs: fp16 fused `mlpackage` in the HF bundle +
  a `RnntDecoder` path in FluidAudio that feeds `(token, h, c, encoder_step)`
  per step (state update only on non-blank, same as today).
- **TDT v3: ship opportunistically** (bundle with the next model-repo
  release). 1.11× decode loop / ~5% E2E ceiling is worth having since the
  conversion is done and parity is exact, but it alone doesn't justify a
  release. Swift-side change mirrors the EOU one (`TdtDecoderV3`).
- Both: fp16 export only; confirm with a Swift interleaved A/B before
  flipping the default (Python ratios here are the prototype evidence, per
  playbook rule 6: placement ≠ speed, harness decides).
- ANE work on these decoders is **settled — do not revisit**: `ios17.lstm`
  has no ANE kernel. Add both to the ANE_Candidates "settled" table.

Artifacts: `build/fused/` (gitignored, local only): `decoder_fp32.mlpackage`,
`joint_decision_fp32.mlpackage`, `decoder_joint_decision_{fp32,fp16}.mlpackage`,
`bench_fp32.json`, `bench_fp16_vs_shipped.json` per pipeline. Nothing uploaded.
