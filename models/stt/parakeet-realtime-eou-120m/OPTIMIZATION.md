# Parakeet EOU decode loop: ANE gate + decoder/joint fusion (2026-06-09)

Campaign target: the worst encoder/decoder imbalance in FluidAudio —
streaming_encoder runs 6.5 ms/chunk at 97% ANE while decoder + joint_decision
burn **58 ms/utt 100% CPU** across ~229 RNNT steps (7.8 s audio), two
`MLModel.prediction` dispatches per step (458/utt). Numbers from
`Documentation/ANE_Profiler.md` (FluidAudio), M5, 160ms variant.

## TL;DR

| Verdict | Detail |
|---|---|
| ANE | **Dead end, settled.** Decoder contains `ios17.lstm` (no ANE kernel at any precision). joint_decision alone places 0% ANE under ALL and CPU_AND_NE (3 MB, under the small-graph floor). |
| Fusion (MIL, 1 graph) | **−42% per decode step** (0.215 → 0.125 ms median). But fp16 logits move up to ~3.5 absolute → argmax flips on low-margin frames → transcripts change on ~5/20 LibriSpeech files. WER-neutral (34.85% → 35.04% on the harness, +2 errors/1036 words). |
| Pipeline (2 specs, 1 dispatch) | **Bit-exact, 0% faster** on M5 (0.218 vs 0.215 ms). Host dispatch is not the bottleneck; per-stage execution overhead is. |
| Boundary-replication experiment | **Failed.** Reproducing the inter-model transpose/cast chain inside the fused graph (optimizer passes disabled) does NOT recover bit-exactness — the divergence is E5RT kernel selection below MIL control. |
| End-to-end (if fused ships) | decode 58 → ~34 ms/utt; pipeline total 65 → ~41 ms/utt ≈ **1.6× e2e** for the decode-dominated EOU pipeline. |

## 1. Gate: per-op fallback dump

`ane_ops` (compute-plan per-op dump) on the shipped 160ms mlmodelc, macOS, M5:

decoder (14 ops, 8 MB) — **LSTM-blocked, categorical ANE dead end**:

| op | CPU | GPU | ANE |
|---|---:|---:|---:|
| ios17.cast | 6 | 0 | 0 |
| ios17.squeeze | 2 | 0 | 0 |
| ios17.transpose | 2 | 0 | 0 |
| ios17.expand_dims | 2 | 0 | 0 |
| **ios17.lstm** | **1** | 0 | **0** |
| ios17.gather | 1 | 0 | 0 |

joint_decision (21 ops, 3 MB) — linear-ish but **100% CPU anyway**:

| op | CPU | GPU | ANE |
|---|---:|---:|---:|
| ios17.cast | 6 | 0 | 0 |
| ios17.expand_dims | 3 | 0 | 0 |
| ios17.linear | 3 | 0 | 0 |
| ios17.transpose | 2 | 0 | 0 |
| ios16.softmax / ios16.relu / ios17.add | 3 | 0 | 0 |
| ios17.reduce_argmax / topk / gather_along_axis / squeeze | 4 | 0 | 0 |

The compiler produces **no ANE segment at all** for joint_decision under
either ALL or CPU_AND_NE — there is no ANE plan to even benchmark. This
matches the ANE_Candidates.md small-graph floor (~50 MB transfer-overhead
threshold); the model is 3 MB. The fused graph (below) is likewise 100% CPU
and its latency is identical across CPU_ONLY / CPU_AND_NE / ALL.

Consequence: per playbook dead-end #4, the EOU decode loop can never be
ANE-resident with this prediction-network architecture. The only available
win is dispatch/overhead reduction → the Nemotron-B1-style fusion branch.

## 2. Fusion approaches

All scripts in `coreml/conversion_scripts/`. No NeMo install needed: the
fused graph is rebuilt with the MIL builder using the exact fp16 weight
blobs read out of the shipped mlmodelc (`fuse_decoder_joint_decision.py`
parses `model.mil` for blob offsets, so it works for 160/320/1280ms).

### 2a. MIL fusion — `fuse_decoder_joint_decision.py`

One iOS17 mlprogram: `targets/h_in/c_in/encoder_step` →
`token_id/token_prob/h_out/c_out` (lean; `--with-topk` adds the top-64
outputs the Swift host never reads). Replicates both source op sequences 1:1;
the decoder→joint boundary transposes cancel and the fp16→fp32→fp16 boundary
casts are identity, so the math is the same fp16 graph.

### 2b. Pipeline — `pipeline_decoder_joint_decision.py`

`ct.utils.make_pipeline(decoder, joint_decision)` with the decoder's
`decoder` output renamed to `decoder_step`. The two original compiled
programs run back-to-back inside one `MLModel` → bit-exact by construction,
one host dispatch per step. Needs the full mlpackages from HF
(`FluidInference/parakeet-realtime-eou-120m-coreml`); the FluidAudio cache
strips mlpackages to weights-only.

### 2c. Boundary replication (failed) — `--replicate-boundary`

Hypothesis: the reference's `transpose → linear` lowers to a transposed-GEMM
kernel whose fp16 accumulation differs from the fused `linear(lstm_out)`.
Keeping the exact inter-model `transpose → cast fp32 → cast fp16 → transpose`
chain (with `cast_optimization` / `reduce_transposes` / `noop_elimination`
passes removed so it survives into the artifact) reproduced the **same**
divergence — identical 197 mismatched steps over 20 files. The kernel-level
difference is introduced by E5RT below MIL, and is not controllable.

## 3. Parity (`parity_fused_decode.py`)

Real pipeline: numpy port of Swift `AudioMelSpectrogram` → shipped
`streaming_encoder.mlmodelc` with caches → greedy RNNT loop mirroring
`RnntDecoder.swift` (maxSymbolsPerStep=2, blank=1026, eou=1024). Both sides
run independent autoregressive state machines. LibriSpeech test-clean.

| metric (20 files, 2433 decode steps) | pipeline | MIL-fused |
|---|---|---|
| token sequences identical | **20/20** | 15/20 |
| h_out / c_out max abs diff | **0.0** | 0.0 on matched trajectories¹ |
| token_prob max abs diff | **0.0** | 2.1e-2 (matched steps) |
| top-1 logit max abs diff | **0.0** | 3.5 (≈4e-3 relative; fp16 GEMM) |
| STRICT (<1e-5 incl. probs) | **PASS** (675-step 5-file run) | FAIL |

¹ On step-matched comparisons h/c are bit-exact (the LSTM lowers
identically); the large h/c diffs in the 20-file fused run are downstream of
token-path divergence, not numeric error.

Transcript-level effect of the fused drift (examples):
`"...takes up an unfinisher as a berlin takes a tune"` →
`"...as a barrel organ takes a tune"`; `"a common sound judgment"` →
`"a calm and sound judgment"`. These are low-margin frames where a ~4e-3
relative logit shift flips argmax — neither side is privileged, both are
fp16 roundings of the same fp32 math.

### WER gate (`wer_ref_vs_fused.py`, 50 files, 1036 ref words)

| decode path | WER |
|---|---|
| two-model reference | 34.85% |
| MIL-fused | 35.04% (+2 errors) |

Quality-neutral within noise. (Absolute WER is high because the harness uses
simplified non-overlapping 160 ms chunking, not the production 50%-overlap
schedule; both decoders consume identical encoder frames, so the comparison
is valid.)

## 4. Benchmark (`bench_fused_decode.swift`)

Interleaved A/B/C, 10 warmup + 200 timed per variant, M5, macOS 26.
One RNNT decode step (the unit that runs ~229×/utt):

| variant | CU | median ms | p95 ms |
|---|---|---:|---:|
| ref (decoder + joint_decision, 2 dispatches) | CPU_ONLY | 0.2146 | 0.2314 |
| pipeline (bit-exact) | CPU_ONLY | 0.2177 | 0.2320 |
| **fused (MIL, lean)** | CPU_ONLY | **0.1253** | 0.1350 |
| fused (MIL, with top-k) | CPU_ONLY | 0.1513 | 0.1720 |
| ref | CPU_AND_NE | 0.2156 | 0.2291 |
| pipeline | CPU_AND_NE | 0.2205 | 0.2331 |
| fused (MIL, lean) | CPU_AND_NE | 0.1246 | 0.1336 |
| ref | ALL | 0.2151 | 0.2257 |
| pipeline | ALL | 0.2180 | 0.2323 |
| fused (MIL, lean) | ALL | 0.1246 | 0.1310 |

- Flat across CU configs ⇒ fully CPU-resident everywhere (confirms the gate).
- Pipeline ≈ ref: collapsing 2 host dispatches to 1 saves nothing on M5 —
  the cost is per-stage execution setup, which the pipeline still pays twice.
  (May differ on A-series where per-call overhead is higher; unmeasured.)
- Fused lean −41.6%; ~12 pp of that is dropping the unused 1027-way top-k
  sort (lean vs with-top-k), the rest is true single-graph execution.

### Honest pipeline math (7.8 s utterance, 160ms variant, profiler baseline)

- decode = 58 ms of ~65 ms/utt total.
- fused: 58 × (0.1253/0.2146) ≈ 33.9 ms → total ≈ 40.9 ms ≈ **1.59× e2e**.
- pipeline: no change.

## 5. Ship / no-ship

- **ANE placement: no-ship, settled.** Add EOU decoder to the
  ANE_Candidates "do not revisit" table (`ios17.lstm`, same as Kokoro
  PostAlbert). joint_decision-only ANE: nothing to ship — the compiler
  declines to produce any ANE segment for it.
- **Pipeline variant: no-ship.** Bit-exact but 0% on M5.
- **MIL-fused (lean): conditional ship — recommend, with eyes open.**
  −42%/step, −37% end-to-end, WER-neutral on the harness evidence
  (+0.19 pp over 1036 words), but it *changes transcripts* on ~25% of
  utterances at the fp16 tie-break level and cannot be made bit-exact
  (Section 2c). Before flipping the FluidAudio default: run the production
  `asr-benchmark`-style WER eval through `StreamingEouAsrManager` with the
  fused model, and take a maintainer decision on output-stability vs speed.
  Swift integration is small: `RnntDecoder` drops the inner `joint` call,
  feeds `encoder_step` into the fused model (no `target_length` input), and
  reads the same `token_id`/`h_out`/`c_out`.
  **Update: the full-scale gate + head-to-head vs the traced fusion is in
  §6** (2,620-file WER, +0.043 pp; traced fusion output-identical and
  strictly dominated).

## 6. Head-to-head vs the traced fusion + full-scale WER gate (2026-06-09/10)

Deciding run for which fused artifact (if any) ships. Two candidate fusions
existed, measured in different harnesses on different machines:

- **lean** (this branch, `fuse_decoder_joint_decision.py`): MIL-builder
  rebuild from the shipped fp16 blobs, drops the unused top-k outputs.
  Claimed −41.6%/step (Swift harness, M5).
- **traced** (`feat/parakeet-decode-fusion`, `fuse_decoder_joint.py`):
  torch.jit re-export from the NeMo checkpoint, full I/O superset
  (keeps top_k_ids/top_k_logits), fp16. Claimed 1.23–1.24×/step (Python
  harness, M4 Pro) with fp32-vs-fp32 parity ≤1.2e-7.

Both candidates were run through ONE harness (`wer_three_way.py` /
`bench_three_way.py`, this dir) against the shipped two-model reference.
Encoder outputs are computed once per file and cached (`.npy`), so all three
decode paths consume bit-identical encoder frames. M5 Pro, macOS 26.5,
coremltools 9.0b1.

### 6a. Speed, same process, interleaved (10 warmup + 200 timed)

One RNNT decode step, real encoder frame, zero state, blank token:

| CU | variant | median ms | p95 ms | × vs ref |
|---|---|---:|---:|---:|
| CPU_ONLY | ref (2 dispatches) | 0.2431 | 0.2854 | 1.00 |
| CPU_ONLY | traced fused | 0.2002 | 0.2297 | 1.21 |
| CPU_ONLY | lean + topk | 0.1742 | 0.1962 | 1.40 |
| CPU_ONLY | **lean** | **0.1448** | 0.1637 | **1.68** |
| CPU_AND_NE | ref | 0.2358 | 0.2673 | 1.00 |
| CPU_AND_NE | traced fused | 0.1906 | 0.2173 | 1.24 |
| CPU_AND_NE | lean + topk | 0.1661 | 0.1862 | 1.42 |
| CPU_AND_NE | **lean** | **0.1378** | 0.1551 | **1.71** |

This settles the 1.21× vs 1.59× discrepancy: both prior numbers reproduce in
one harness. The traced fusion really is only ~1.2×; the lean build is
~1.7×/step. The gap splits roughly evenly between the tighter MIL graph
(traced 1.24 → lean+topk 1.42 at identical I/O) and dropping the 1027-way
top-k sort the Swift host never reads (1.42 → 1.71).

### 6b. WER gate — full LibriSpeech test-clean (2,620 files, 52,576 words)

All 2,620 files, full-length audio, identical cached encoder frames for all
three variants (`wer_three_way.py`, ~0.46 s/file, ~20 min wall):

| decode path | WER | errors | token-seq diff vs ref |
|---|---:|---:|---:|
| ref (shipped pair) | 35.646% | 18,741 | — |
| traced fused fp16 | 35.689% | 18,764 | 278/2,620 files |
| lean fused | 35.689% | 18,764 | 278/2,620 files |

(Absolute WER is high for the harness reason in §3: simplified
non-overlapping 160 ms chunking, not the production overlap schedule. All
variants consume identical frames, so deltas are valid.)

**Key finding: traced ≡ lean.** Direct re-decode of all 278 divergent files
shows the traced and lean artifacts emit **identical token sequences on all
2,620 files** — same diff set, same error counts, identical hypotheses.
The traced fusion's "bit-exact" parity claim was fp32-fused vs fp32-separate
in its own harness; at fp16 against the *shipped* two-model pair it drifts
exactly like the lean build. This independently confirms §2c: the divergence
is introduced by E5RT kernel selection for the single-graph layout, below
MIL, and is a property of *any* fp16 decoder+joint fusion — not of the lean
rebuild.

Per-file tail (identical for both fused variants): 4/2,620 files where fused
WER exceeds ref by >20 pp — 6930-75918-0020 (+31.6 pp, +12 err/38 w),
908-157963-0019 (+25.6, +11/43), 1221-135766-0015 (+25.0, +2/8 — tiny-file
artifact), 1188-133604-0009 (+21.4, +15/70). All four are utterances where
the harness reference is already degenerate (ref WER 50–68% under the
non-overlap chunking); the failure mode is early truncation (a tie-level
blank/EOU flip ends emission early), not hallucination. These 4 files carry
essentially the whole +0.043 pp aggregate delta. Worth re-checking under
production overlap chunking before relying on it either way.

### 6c. Verdict (decision rule, applied)

Rule set before the run: *lean ships if (1) WER delta vs ref ≤ +0.10 pp
absolute on the gated set AND (2) no file where lean exceeds ref by >20 pp;
otherwise the traced fusion is the recommendation.*

- Clause 1: **PASS** — +0.043 pp (35.689 vs 35.646) on 2,620 files.
- Clause 2: **FAIL** — 4 files above +20 pp (max +31.6 pp).
- Rule output as written: traced fusion becomes the recommendation. **But
  the run invalidated the rule's premise**: the traced artifact emits
  token-for-token identical output to lean on every gated file, so it fails
  the identical blowup clause and offers zero quality protection while being
  1.38–1.41× slower than lean per step.

**Final recommendation:** the traced fusion is strictly dominated — never
ship it (same outputs, slower, and its export needs a multi-GB NeMo+torch
toolchain vs blob-rebuild). The real decision is fusion-vs-reference, and it
is the maintainer call already flagged in §5, now with full-scale evidence:
fusion buys 1.7×/step (≈1.6× e2e for this decode-dominated pipeline) at
+0.043 pp aggregate WER, with a 4/2,620 (0.15%) early-truncation tail on
already-degenerate utterances. If that tail is acceptable — or production
overlap chunking makes it moot — ship **lean**; if bit-exactness is a hard
requirement, ship nothing (the bit-exact pipeline variant is 0% faster, §4).

## 7. Repro

```bash
VENVPY=<python with coremltools==8.3.0 + numpy + soundfile>
MODELS="$HOME/Library/Application Support/FluidAudio/Models/parakeet-eou-streaming/160ms"

# Build (lean / topk / failed boundary experiment)
$VENVPY coreml/conversion_scripts/fuse_decoder_joint_decision.py \
    --model-dir "$MODELS" --output-dir /tmp/eou_fused [--with-topk|--replicate-boundary]

# Bit-exact pipeline (needs full mlpackages from HF)
$VENVPY coreml/conversion_scripts/pipeline_decoder_joint_decision.py \
    --decoder <hf>/160ms/decoder.mlpackage --joint <hf>/160ms/joint_decision.mlpackage \
    --output-dir /tmp/eou_fused

# Parity / WER / bench
$VENVPY coreml/conversion_scripts/parity_fused_decode.py --model-dir "$MODELS" \
    --fused /tmp/eou_fused/decoder_joint_decision_fused.mlpackage --audio <flacs...>
$VENVPY coreml/conversion_scripts/wer_ref_vs_fused.py --model-dir "$MODELS" \
    --fused /tmp/eou_fused/decoder_joint_decision_fused.mlpackage \
    --librispeech-root "$HOME/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean"
swiftc -O coreml/conversion_scripts/bench_fused_decode.swift -o /tmp/bench_eou
/tmp/bench_eou "$MODELS" /tmp/eou_fused/decoder_joint_decision_pipeline.mlpackage \
    /tmp/eou_fused/decoder_joint_decision_fused.mlpackage

# §6 head-to-head (traced artifact from feat/parakeet-decode-fusion,
# build/fused/decoder_joint_decision_fp16.mlpackage via fuse_decoder_joint.py export)
$VENVPY coreml/conversion_scripts/wer_three_way.py --model-dir "$MODELS" \
    --traced <traced>/decoder_joint_decision_fp16.mlpackage \
    --lean /tmp/eou_fused/decoder_joint_decision_fused.mlpackage \
    --librispeech-root "$HOME/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean" \
    --cache-dir /tmp/eou_enc_cache --results /tmp/eou_three_way_full.jsonl --num-files 0
$VENVPY coreml/conversion_scripts/bench_three_way.py --model-dir "$MODELS" \
    --traced <traced>/decoder_joint_decision_fp16.mlpackage \
    --lean /tmp/eou_fused/decoder_joint_decision_fused.mlpackage \
    --enc-frame /tmp/eou_enc_cache/<any>.npy
```
