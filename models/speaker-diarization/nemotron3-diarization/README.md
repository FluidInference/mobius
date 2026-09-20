# Nemotron 3 Diarization preview — CoreML

CoreML conversion of [nvidia/Nemotron-3-Diarization-preview](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview)
(early-access streaming Sortformer successor: 8 speakers, 100M params, 31-layer RoPE
Transformer encoder, 10 ms output resolution).

> **License.** The preview checkpoint is under the NVIDIA Software and Model Evaluation
> License. Converted presets are published at
> [`FluidInference/nemotron-3-diarization-coreml`](https://huggingface.co/FluidInference/nemotron-3-diarization-coreml)
> with NVIDIA's permission (gated until the public release). `build/` is gitignored.

> **PRELIMINARY — NOT FINAL, NOT NVIDIA'S OFFICIAL RESULTS.** Every number below was
> measured by Fluid Inference on the *early-access preview* checkpoint, on one M5 Pro
> MacBook, with our own harness. They are working notes from the conversion, not a
> release benchmark: the public checkpoint may differ, protocols are described per
> table, and figures will be re-run and finalized after NVIDIA's public release.
> Treat them as outdated the day the public weights land. NVIDIA's model card is the
> official source for the model's accuracy.


## Architecture vs Streaming Sortformer v2

| | v2 (`diar_streaming_sortformer_4spk-v2.1`) | Nemotron 3 preview |
|---|---|---|
| Speakers | 4 | 8 |
| Encoder | Fast-Conformer (NEST) + 18-layer Transformer | single 31-layer Transformer, RoPE, FlexAttention |
| Pre-encode | Conv subsampling | FeatureStacking (8× stack + Linear) |
| Output resolution | 80 ms | 10 ms (subpixel Conv1d upsampler ×8) |
| Speaker cache | 188 | 264 |
| NeMo class | `SortformerEncLabelModel` | same (NVIDIA-NeMo/Speech main required) |

## Variants (`conversion/build/`)

All shapes fixed; latency = (chunk_len + right_context) × 80 ms. Inputs:
`chunk (1, mel, 128)`, `chunk_lengths (1,)`, `spkcache (1, 264, 512)`,
`spkcache_lengths (1,)`, `fifo (1, F, 512)`, `fifo_lengths (1,)`.
Outputs: `speaker_preds (1, packed, 8)` @80 ms, `chunk_pre_encode_embs (1, enc, 512)`,
`chunk_pre_encode_lengths (1,)`, `speaker_preds_10ms (1, packed*8, 8)` (extra output
NVIDIA's ONNX export does not have; enables 10 ms host output).

| Variant | Latency | chunk/rc/fifo/update | mel in | packed | M5 Pro GPU | M5 Pro ANE |
|---|---|---|---|---|---|---|
| ultra | 0.32 s | 3/1/264/222 | 32 | 532 | 9.2 ms | 29 ms |
| verylow | 0.64 s | 6/2/264/222 | 64 | 536 | 9.3 ms | 30 ms |
| low | 1.04 s | 9/4/264/222 | 104 | 541 | 9.6 ms | 28 ms |
| offline | 30.4 s | 340/40/40/300 | 3040 | 684 | 11.2 ms (~2400× RT) | ANE compile fails → GPU |

spkcache_len = 264 for all variants; left_context = 0.

## Verification (all on M5 Pro, macOS 26.7)

_Preliminary, preview checkpoint — see the note at the top._

- Patched attention (export) vs FlexAttention reference: bit-exact in torch (0.0 diff).
- Single-chunk CoreML vs torch across cold-start/warm/full/partial states:
  preds ≤ 1.7e-4, hires ≤ 5e-4 (fp16).
- Closed-loop 120 s real audio (earnings22), CoreML forward inside NeMo's
  `streaming_update_async` loop vs pure torch: **99.98–99.995% frame agreement**
  (0.5 threshold), mean abs diff ~3e-4, identical active rates — fp16 state feedback
  does not diverge. (`conversion/e2e_streaming_test.py`)

## Conversion gotchas (hard-won, keep)

1. **NeMo Speech main required** — `self_attention_model='rope'` and the new
   `TransformerEncoder` are absent from all PyPI nemo-toolkit releases (≤3.0.0).
2. **FlexAttention is untraceable** — `export_patches.py` swaps in explicit
   matmul-softmax with an additive `(B,1,1,T)` key-padding bias (−30000 fp16-safe).
   RoPE/full-attention has no score_mod, so this is exact (verified bit-identical).
   Valid frames are start-packed, so RoPE positions are padding-invariant.
3. **numpy ≥2 breaks coremltools 9.0** torch frontend: every `.shape` unpack dies with
   "only 0-dimensional arrays can be converted to Python scalars". Pin numpy 1.26.4
   (and scipy 1.13.1 for numpy-1.x ABI).
4. **FeatureStacking.forward doesn't trace** (its `t_new` becomes a traced tensor →
   same coremltools `int()` failure). Inlined as static reshape + `proj` Linear in
   `wrappers.py`; export mel frames are always a multiple of 8 so the pad branch is dead.
5. **`concat_and_pad` uses `index_copy_`** (scatter) — replaced with the v2 Sortformer
   gather-based `fixed_concat_and_pad` (ANE-safe, arithmetic indices).
6. **Integer `//` traces to float divide** and the GPU fp16 lowering misrounds inputs
   > 2048 (3047→"3048"/8 = 381). Use `torch.floor_divide` (lowers to true int
   `floor_div`). Residual caveat: on the offline variant, if ANE compile fails and the
   graph lands on GPU, `chunk_pre_encode_lengths` can still read +1 — hosts should
   compute `ceil(len/8)` themselves and treat the output as advisory.
7. **Offline variant fails ANECCompile** (`E5RT ... ANECCompile() FAILED`) on
   macOS 26.7 / M5 Pro; GPU at ~2400× RT makes this a non-issue for batch use.
   Streaming variants run 99.1% ANE-resident.
8. Mel frontend is the same 128-mel / 10 ms / `normalize: NA` family as Nemotron ASR —
   FluidAudio's native `NemotronMelExtractor` should be reusable on the Swift side.

## Speed push (2026-08-28, mirrors the Nemotron ASR int8/ANE campaign)

_Preliminary, preview checkpoint — see the note at the top._

The compute profile is unlike the ASR case: 96.9% of ops are already ANE-resident and the
CPU stragglers cost <0.2 ms, so placement is solved — per-chunk cost is pure compute over
the packed sequence (spkcache 264 + fifo + chunk). Levers tried:

| Trial | ANE ms | GPU ms | Verdict |
|---|---|---|---|
| low fp16 (baseline, T=541) | 28.0 | 9.6 | — |
| weight-only int8 (ASR recipe) | 27.9 | 9.4 | no meaningful speedup; **weight footprint** 199→100 MB (peak runtime RAM not measured); quality-neutral: AMI DER 26.26 vs 26.22 fp16, worst per-file +0.33 |
| `fast` = fifo 264→40 (T=317) | **10.9** | **6.1** | **2.6x ANE win**, same 1.04 s latency; +0.54 DER on AMI (26.76 vs 26.22); advances 0.72 s audio/call → ~66x model-only RTFx on ANE |
| `efficient` = chunk 48 (T=580) | 32.4 | 10.0 | advances 3.84 s audio/call → ~119x model-only RTFx on ANE, ~384x on GPU; AMI DER 26.03% (better than low), wall RTFx 251x |
| W8A8 activation quant (calibrated, op-scoped) | 60.0 (99.8% CPU) | — | **DEAD END**: quantize/dequant pattern evicts the graph from ANE entirely |

Model-only RTFx = audio advanced per call / predict time (chunk_len × 80 ms per call, not
the latency window). Wall RTFx from the AMI harness includes mel + host state updates.

### Round 2: sweep campaign (chunk / cache / FIFO), full-AMI gated

One-lever screening (4-meeting subset) vs the full 16-meeting gate exposed interaction
effects — **do not stack state-shrink levers**:

- Right context: rc→0 costs only ~+0.1 DER alone. Cache: 160 fine, **128 is a cliff**
  (+2.2). FIFO: 16 fine, **0 is a cliff** (+3.1). All measured one lever at a time.
- Stacked combos FAIL: turbo1 (c12/rc1/cache192/fifo24) 27.69% conf 1.86; turbo2
  (c13/rc0/cache160/fifo16) 30.58% **conf 4.07** — attribution collapse. Screening
  subsets flatter combos; gate everything on full AMI.
- ANE latency is non-monotonic in T (tiling): fifo 32 (T=309) profiled *slower* than
  fifo 40 (T=317). Profile, don't extrapolate.

**Chunk ladder (the winning axis)** — fifo 40 "s" family vs fifo 264 "q" family:

| Variant | Audio/call | Latency | AMI DER | ANE/call | Wall RTFx (GPU route) |
|---|---|---|---|---|---|
| fast (chunk 9) | 0.72 s | 1.04 s | 26.76% | 10.9 ms | 85x |
| **fast24** (s24) | 1.92 s | 2.24 s | 26.28% | 11.7 ms | 227x |
| **fast32** (s32) | 2.56 s | 2.88 s | **26.21%** | 12.5 ms | 292x |
| q16/q24/q32 | 1.28–2.56 s | 1.6–2.9 s | 26.10–26.16% | 29–40 ms | 106–188x |

Bigger chunks IMPROVE quality (consistent with `efficient` at 26.03%) and recover the
small-FIFO penalty; per-call ANE cost stays nearly flat. The q family's big FIFO buys
only ~0.1 DER at 2.5–3x the ANE cost — not promoted. `fast24`/`fast32` are Swift
presets; sweep artifacts (c*, sc*, f*, s16, q*) remain in build/ for reference.

Compute-route walls for `fast` on ES2004a: ANE 59.4x (host ≈10% of wall), GPU 84.6x
(host ≈28%) — host optimization only pays on the GPU route.

### Round 3: extended chunk tier (c64–c192), full-AMI gated

| Variant | Audio/call | Latency | AMI DER | conf | ANE/call | Wall RTFx |
|---|---|---|---|---|---|---|
| c64 | 5.12 s | 5.44 s | 26.21% | 0.96 | 15.8 ms | 516x |
| c96 | 7.68 s | 8.00 s | 25.93% | 0.81 | 18.2 ms | 656x |
| **c128 = fast128** | 10.24 s | 10.56 s | **25.91%** | **0.64** | 20.6 ms | 790x |
| c160 | 12.80 s | 13.12 s | 25.94% | 0.80 | 22.8 ms | 890x |
| c192 | 15.36 s | 15.68 s | 25.83% | 0.61 | **ANECCompile FAILS** | 983x (GPU) |

- Quality improves monotonically with chunk size through c192 — no regression found
  (an earlier compacted-session summary claimed one at c160; it does not reproduce).
- **ANE compile boundary: chunk mel input between 1312 (c160, 97.7% resident) and
  1568 (c192, ANECCompile fails)** — same failure class as offline's 3040-frame input.
- c128 promoted as `.fast128` (best confusion, ~497x model-only ANE, VoxConverse
  5–10-spk avg 7.25% — best of the lineup). c160 works but sits 1 step from the
  cliff; c192 not promoted (GPU-only, dominated by `.offline` 25.70% @ 1080x for batch).

**Multi-speaker retention gate (VoxConverse test, 14 files stratified 5–10 speakers,
proper RTTM refs):** `fast32` 7.92% avg DER vs `low` 8.94% — the fifo-40 + big-chunk
preset is *better* at high speaker counts, not just equal. On the three 8-speaker
files, fast32 wins clearly (kpjud 14.10 vs 20.80, with 8/8 speakers found vs 6/8).
No cache-retention regression; the 264-frame cache + larger chunk context improves
attribution. VoxConverse absolute DERs (2–22%) are far below AMI's because the refs
are real RTTMs, not pause-padded word alignments.

Structure beats precision here: the sequence length IS the cost. `fast` and `efficient`
pass NeMo `_check_streaming_parameters` and verify at the same fp16 parity as card
profiles. W8A8 note: `linear_quantize_activations` also crashes on int32 index-arithmetic
ops unless scoped via `op_type_configs` to linear/matmul/conv — and then still loses ANE.

### Round 4: pipeline campaign (profiling → readback → VAD → ANE cliff → multi-stream)

- **Stage profile (fast32, full meeting)**: predict 78–85% of wall; output readback was
  the only real host cost (1.6 ms/chunk, 16% of GPU wall); mel ~4%, all else ~1%.
- **Readback fix**: outputs are fp16 with padded rows ([1,T,8], row stride 16);
  `shapedArrayValue` pays conversion, naive dataPointer falls to NSNumber reads.
  Bulk `vDSP.convertElements` + one `vDSP_mmov` compaction: 1.57 → 0.039 ms (40x).
  fast32 wall: GPU 260→345x, ANE 177→206x. Bit-identical output. Pipeline now
  model-bound (predict 92–96%).
- **Pipeline overlap: closed** — mel is ~4% and the only overlappable stage; <5% ceiling
  does not justify the concurrency machinery.
- **VAD gating (opt-in `speechMask` / `--vad`)**: sparse audio (55% speech) 2.0x wall;
  AMI far-field costs +0.9 DER for +8% speed at any threshold (Silero misses quiet SDM
  speech) — keep OFF by default. Long-file VAD must run in 300 s segments or it
  exhausts IOSurfaces (#752 class).
- **ANE cliff bisected**: chunk mel 1376 (c168) compiles 97.7% resident / 22.8 ms;
  1440 (c176) fails ANECCompile. rc and packed length exonerated (c192r0 fails at
  mel 1536; low compiles at T=541). The trigger is in the chunk branch
  (FeatureStacking input length) — split-graph (host pre-encode) would bypass it.
  c168 = max ANE point: 13.44 s audio per 22.8 ms call (~589x model-only).
- **Multi-stream (fast32)**: GPU aggregate 304/436/441x at 1/2/4 processes — one extra
  stream buys +43%, then saturation. ANE: 193/212/225x — jobs mostly serialize;
  don't design for ANE concurrency.

### Round 5: split graph — the cliff killer (convert_split.py / verify_split.py)

Host does feature stacking (free reshape), the 1024->512 projection (`cblas_sgemm`,
~0.06 ms), state packing, and mask construction; the model is the pure-fp
transformer+head (`packed`/`attn_bias`/`output_mask` inputs). Requires
`pre_encode_proj_t.bin` next to the models.

| | Monolithic | Split |
|---|---|---|
| CoreML-subgraph ANE residency | 99.1% (int32 on CPU) | **100.0%** (host still does proj/pack/masks on CPU, ~0.06 ms) |
| fast32 ANE | 12.5 ms | **11.2 ms** |
| c192/c256 | ANECCompile FAILS | **compile + run** |
| c256 throughput | — | 29.1 ms / 20.48 s = **~704x model-only ANE**, 590x wall |
| W8A8 | evicted to CPU (60 ms) | **works**: 9.7 ms ANE, 95 MB weights; single-chunk 0.5-threshold decisions match fp16 (max prob diff 5e-4); DER +0.12 on 4-meeting spot (PRELIMINARY) |
| DER (4-meeting spot, PRELIMINARY) | 25.41 (GPU route) | 24.93 (ANE route) — confounds split execution, fp32 host embs, and compute route; full gates + controls pending |

Swift closed-loop parity on the 120 s fixture: 99.995% frame agreement vs NeMo torch
(mean abs prob diff 5e-4 — slightly above the monolithic path's 1e-4 because host
fp32 embeddings evolve state marginally differently; binary decisions equal).
Gotcha: `ANEMemoryUtils.calculateOptimalStrides` pads innermost dims to tile
boundaries — `output_mask [1,T,1]` gets row stride 16, so linear `dataPointer`
writes scramble it (surfaced as ~90% agreement). Use plain contiguous
`MLMultiArray` for small mask inputs, or write stride-aware.

Batch farms: `nemotron3-batch --workers 2` (GPU) measured +48% aggregate over one
worker (328x -> 484x on 4 AMI meetings), identical DER; the M5 Pro GPU saturates
at two streams, ANE jobs serialize.

### Round 6: final split matrix (FULL gates — 16 AMI + frozen VoxConverse, ANE route)

| Candidate | Latency | AMI DER | Vox 5–10spk | Wall | Weights |
|---|---|---|---|---|---|
| fast32-split | 2.88 s | 26.40 | 8.35 | 219x | 189 MB |
| fast32-split-w8a8 | 2.88 s | 26.26 | 8.43 | 230x | 95 MB |
| **c128-split-w8a8** | 10.56 s | 26.13 | 7.59 | 555x | 95 MB |
| c192-split | 15.68 s | **25.84** | **6.73** | 568x | 189 MB |
| c256-split | 20.80 s | 26.18 | 6.49 | 613x | 189 MB |

- Verdicts: **c128-split-w8a8 = recommended ANE batch preset** (best latency/wall/size
  balance); c192-split = quality pick; c256-split = reference only (EN2002c shows a
  4.3% confusion outlier — very long chunks can wobble, as with c448 previously).
- Streaming default: **stays monolithic `.fast32`** — split-w8a8 is quality-equivalent
  on AMI (26.26 vs 26.21, within noise) but slightly worse on the Vox subset
  (8.43 vs 7.92; route also differs), and adds the proj-file dependency. Use
  `fast32-split-w8a8` when 100% CoreML-subgraph ANE residency or the 95 MB weight
  footprint matters (iOS).
- The earlier 4-meeting "split beats monolithic" DER edge did not survive the full
  gate — treat subset DER deltas as noise until full-gated (second time this
  campaign a subset flattered a result).
- IOSurface exhaustion, third sighting: long ANE-route runs die at ~4000+ predictions
  because output IOSurfaces never drain in the sync loop — fixed with a per-chunk
  autoreleasepool in `processComplete` (bit-identical output).


### Round 7: model-card reproduction (the capstone)

Reproduced NVIDIA's card evaluation protocol on-device: forced-alignment refs
(nttcslab-sp/diar-forced-alignment), collar 0, card configs, four card tables
(AMI MHM, AMI SDM/Array1-01, AliMeeting Far/ch0, AliMeeting Near/headset-mix).
**7 of 9 rows at parity or better, mean DER delta -0.20** — incl. exact matches
(verylow 10.13, offline SCA/MAE) and wins on both AliMeeting tables. One gap:
AMI SDM low +1.45 (far-field confusion; fp16-vs-bf16 suspected). Our presets under
card protocol all beat card-low 10.35; fast128 = 9.59 at 1004x wall. Full tables in
the private HF repo's BENCHMARKS.md.

**LABEL CORRECTION:** all earlier rounds' "AMI SDM" results used Mix-Headset audio
= the **MHM** condition (the local folder was misnamed). Relative comparisons are
unaffected. True SDM (Array1-01) results exist only in round 7.

Cross-system fidelity on NVIDIA's release demo clip (8 TTS voices): 0.134% DER vs
the NeMo reference, zero confusion frames, 8/8 speakers.


### Round 8: layer-drop sensitivity (distillation feasibility probe)

Zero-shot removal of transformer blocks (no retraining), fast32 shape, 4-meeting
FA-ref screen (baseline 9.72):

| Drop set | Layers left | DER | Δ |
|---|---|---|---|
| **{15,16} mid-stack** | 29 | **10.36** | **+0.64** |
| {14–17} contiguous mid | 27 | 13.98 | +4.3 |
| {13,15,17,19} alternating mid | 27 | 15.22 | +5.5 |
| {6,27} interior edges | 29 | 15.41 | +5.7 |
| {6,13,20,27} spread | 27 | 13.40 | +3.7 |
| {13–18} contiguous mid | 25 | 53.76 | collapse |

Findings: redundancy is concentrated in ~2 mid-stack layers (~6.5% compute for
+0.6 DER); anything deeper fails zero-shot regardless of pattern (alternating is
no better than contiguous). Layers 6 and 27 are disproportionately load-bearing.
The 31-layer stack does distinct sequential work — this checkpoint is tightly fit,
not over-parameterized (unlike Sortformer v2). Practical ceiling without training:
mid2, which is dominated by W8A8-split (-13% compute, ~0 DER). Real compression
for CPU-class targets requires distillation with training compute, or a smaller
official checkpoint (raised in the NVIDIA feedback doc).

Next-phase caveat: unstructured sparsity will NOT accelerate the ANE — width
pruning must physically shrink projection/FFN dimensions (keeping tile-friendly
alignment, e.g. multiples of 16/64) and then be fine-tuned or distilled. Order of
operations: compare Gradient Descent's independent result -> await NVIDIA's answer
on a smaller sibling -> only then structured head/FFN pruning + distillation.
Zero-shot whole-layer combinations are exhausted; do not resweep.


### Round 9: ASR-playbook transfer probes (fusion / pipelining / batching)

Tested the Nemotron 3.5 ASR port's optimization playbook against the diarizer:

- **Call fusion / dispatch overhead**: no token loop here — one predict per chunk,
  dispatch is <1% of call cost. `outputBackings` adopted anyway: no latency change
  (the ~1 ms predict-vs-benchtool gap is internal CoreML dispatch), but it removes
  per-call output IOSurface allocation — the pool-exhaustion root cause.
- **Pipelining**: structurally blocked — chunk N+1's input state IS chunk N's
  output; only mel (~4% of wall) is overlappable. Closed.
- **Vocab pruning**: no analog (8-wide head is negligible).
- **Chunk sizing**: already exhausted (rounds 2-3), and our curve differs — fixed
  state dominates attention, so bigger chunks keep improving accuracy.
- **Batch-dimension streams (new)**: b=4 split graph — **ANE rejects batch>1**
  (0% residency); GPU works at 4.5 ms/stream vs 6.2 ms b1 (−28%, ~569x model-only
  aggregate vs 484x wall for the 2-process mode). Gain over process-parallelism is
  marginal vs the ragged-stream host integration cost — measured, not integrated.

## Workflow

```bash
cd conversion
uv sync
uv run python convert.py                      # all variants -> build/
uv run python verify.py                       # single-chunk parity
uv run python e2e_streaming_test.py --variant low --wav build/test_120s.wav
```

## Host (Swift) integration — PORTED

`Sources/FluidAudio/Diarizer/Nemotron3/` (branch `feat/nemotron3-diarization`):
`Nemotron3Diarizer` + `Nemotron3StateUpdater` port `streaming_update_async` at batch 1
(fixed-capacity states, learned silence embedding from `learnable_sil_emb.bin`,
score-based compression). CLI: `nemotron3-diarize` / `nemotron3-benchmark` / `nemotron3-batch`; models load from
Hugging Face (`Nemotron3Models.loadFromHuggingFace`) or a local directory.

Swift-vs-NeMo parity on the 120 s fixture: 99.998% frame agreement, 3/12001 frames
differ (2 at the file tail from mel padding alignment, 1 ambiguous mid-file frame).

**AMI SDM test benchmark (16 meetings, word-aligned refs, collar 0, M5 Pro):**

| Model | DER | miss/fa/conf | RTFx |
|---|---|---|---|
| Nemotron 3 offline (30.4 s) | **25.70%** | 23.8/1.3/0.6 | 1080x |
| Nemotron 3 low (1.04 s) | **26.22%** | 24.2/1.2/0.8 | 60x |
| v2 Sortformer fast (1.04 s) | 29.24% | 25.2/1.4/2.6 | 52x |

Nemotron 3 low beats v2 on **all 16 meetings** (avg −3.0 DER; biggest −11.2 on TS3003b
where v2 collapses to 10.6% speaker confusion). ES2004a slope across variants:
offline 27.0 / low 27.4 / verylow 27.5 / ultra 27.9. Absolute DER runs ~2x NVIDIA's
card numbers because this harness's word-aligned ground truth counts pauses/backchannels
(miss-dominated for every model); card numbers use forced-alignment refs.

Key checkpoint facts the port relies on:
- `use_learnable_sil_emb: true` -> the running silence profile (`mean_sil_emb` /
  `n_sil_frames`) is dead code; compression uses the fixed learned embedding.
- All preview profiles have `chunk_left_context = 0`.
- First compression uses FRESH cache-region predictions; later compressions use the
  stored (frozen) predictions — `spkcache_compressed` gates this.
- 10 ms output: gather `speaker_preds_10ms[(spkcache_len+fifo_len+lc)*8 : +chunk*8]`
  per chunk, before the state mutates.
