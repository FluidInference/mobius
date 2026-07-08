# Conversion trials

## Trial 1 — TextEncoder + FmDecoder, fixed shapes, fp16 (2026-07-07)

Result: both convert and pass parity. GPU is the fast path; ANE resident but slow.

Gotchas hit (in order):

1. **`CompactRelPositionalEncoding` slices with in-graph shape math** →
   `slice_by_index` with fp32 begin, rejected by coremltools. The module is
   jit-scripted by `convert_scaled_to_non_scaled`, so instance `forward`
   monkey-patching never applies. Fix: probe each `Zipformer2Encoder`'s input
   eagerly, precompute `pos_emb`, swap `encoder_pos` for a constant-buffer
   module (`FrozenPosEmb`).
2. **`aten::Int` on 1-element arrays** (`seq_len2 = 2*seq_len - 1` shape math)
   → stock handler only takes 0-d. Fix: `patch_coremltools_int()` folds
   1-element constant arrays.
3. **`SimpleDownsample` zero-size pad** — `expand(0, ...)` + `cat` for pad=0
   materializes a spurious row under coremltools, breaking the downstream
   reshape ([1025,1,512] → [512,2,1,512]). Fix: skip the pad branch when
   `seq_len % ds == 0` (MAX_FRAMES=1024 divisible by all ds factors [1,2,4]).
4. **Token bucket**: phoneme-level tokenization is denser than expected —
   5 s prompt + one sentence = 194 tokens. Bucket at 256.
   Upstream `pad_labels` appends one pad slot whose embedding fills
   remainder frames in duration expansion; keep S+1 rows, mask from S.

Parity (CPU_ONLY CoreML vs torch, oracle: 5 s YC prompt, seed 42):

```
[text_encoder]        cos=0.999997  max_abs=1.0e-03
[fm_decoder step 0-3] cos=0.9990-0.9991
[final mel]           cos=0.998802
[wav]  waveform cos=0.715 (phase-only)  log-mel cos=0.99925  RMS 0.1479 vs 0.1463
```

Profile (M5 Pro): FmDecoder GPU 14.1 ms/step vs ANE 286 ms (98.2% resident),
CPU 155 ms. TextEncoder 1.4–4.8 ms everywhere. `all` → 100% GPU.

### Pipeline benchmark (coreml/benchmark.py, M5 Pro, oracle utterance)

3.0 s generated audio, 469 prompt + 282 gen frames padded into the fixed
1024-frame bucket — latency is bucket-constant, so RTFx scales up with
utterance length (~2.4× higher near bucket capacity).

| compute units | text_enc | fm_dec/step | core (te+4 steps) | +torch vocoder | RTFx e2e |
|---|---|---|---|---|---|
| ALL (→GPU)  | 4.7 ms | 14.2 ms | 62 ms  | 139 ms  | **21.5×** (core 48×) |
| CPU_ONLY    | 5.5 ms | 153 ms  | 621 ms | 705 ms  | 4.2× |
| CPU_AND_NE  | 4.9 ms | 286 ms  | 1152 ms| 1233 ms | 2.4× |

torch-cpu fm_decoder baseline: 405 ms/step → CoreML GPU is 28× faster,
CoreML CPU 2.6× faster. Model load: 8.7 s (ALL) / 17.7 s (ANE) cold.
Vocoder (torch cpu, 77 ms) is now the e2e bottleneck — CoreML port next.

### Audio similarity vs PyTorch wav (coreml/audio_similarity.py)

Level/energy: RMS −16.69 vs −16.60 dBFS (**Δ −0.094 dB**), peak Δ −0.34 dB,
total energy Δ −0.094 dB. Short-time envelope (25 ms/10 ms): corr **0.9954**,
per-frame diff median +0.01 dB / std 1.14 dB, envelope SNR 24.8 dB. Per-octave
band energy deltas all within **±0.27 dB** (20 Hz–24 kHz). Spectral
convergence 0.154; LSD 2.77 dB (mean). Waveform-domain cos 0.715 is phase
divergence only — energy and spectrum are match-grade.

## Open items

- **Vocoder**: dual ISTFT heads (24k + 48k upsampled) + Linkwitz-Riley
  crossover (IIR biquads). Plan: DFT-matmul ISTFT + conv_transpose overlap-add
  in CoreML; crossover host-side (Accelerate biquads) or FIR-approximated.
- **ANE latency**: 286 ms/step despite 98% residency. Zipformer runs seq-first
  (T,B,C) with heavy transposes + rel-pos gather; would need (B,C,1,S)
  restructuring per the ANE playbook. GPU path is fast enough to ship macOS;
  revisit for iPhone (no coreml-cli --fallback run yet).
- **Shape buckets**: single 1024-frame bucket (~10.9 s). Need enumerated
  buckets (256/512/1024/2048) and re-profile; attention is O(T²).
- **fp16 vs guidance**: `guidance_scale` embedding path untested at other
  scales; parity run used 3.0 (LuxTTS default).
- **ZipVoice upstream variants**: same script should convert zipvoice /
  zipvoice_distill / dialog checkpoints (24k vocoder, different tokens.txt).

### Transcription sanity check (whisper-base on generated wavs)

All CoreML samples intelligible; CoreML and PyTorch oracle transcribe
identically. Onset clipping ("The quick" -> "Brown Fox...") reproduces in
the PYTORCH oracle too - not a conversion artifact.

### Onset-clipping root cause (debugged)

Hypotheses tested, in order:
1. Whisper missing the onset - NO: +0.5s silence pad changes nothing;
   onset RMS shows speech at full level ~50ms in. Words acoustically absent.
2. Uniform avg-duration alignment mapping leading text tokens into the
   prompt region (at speed 1.3 the first 47 text tokens land inside the
   469 prompt frames) - NO: a two-segment expansion (prompt tokens pinned
   to prompt frames) produces byte-identical transcripts. The flow model
   self-aligns; text_condition alignment is a soft hint.
3. PROMPT BOUNDARY CONTINUATION - YES: the model continues speech
   seamlessly from the prompt. A prompt hard-cut mid-phrase (fixed 5.000s
   slice) makes the model elide sentence-initial function words. Same
   text/seed/speed with a prompt ending at a natural sentence boundary
   synthesizes the complete sentence including "The".

Amplifier: generate() silently multiplies speed by 1.3, squeezing the
ratio-based duration estimate; at 1.3 the elision grows from "The" to
"The quick". speed<=1.0 + clean prompt boundary = full sentence.

Swift integration guidance: trim reference clips at a VAD pause boundary
(Silero VAD already in FluidAudio) instead of a fixed duration; expose
speed, default 1.0.

### Long-sentence test (2048-frame / 512-token variant)

Converter now takes --max-tokens/--max-frames; long variant at
build/coreml-long (FmDecoder same 238MB weights). Three texts, 10.2-14.4s
generated audio, sentence-end prompt, speed 1.0: CoreML and PyTorch
transcribe word-for-word identically, sentence onsets intact, full
multi-sentence content preserved. Decoder step scaling (GPU): 14.0ms @1024
-> 35.4ms @2048 frames (2.5x for 2x frames - subquadratic thanks to the
[1,2,4,2,1] downsampled stacks). 14.4s utterance e2e wall ~0.3-0.4s with
torch vocoder (~40x realtime). Enumerated buckets remain the plan for a
Swift integration; per-bucket compiled variants work today.

### RSS measurements (python host, minus 348MB harness baseline)

Steady-state: 1024/GPU ~480MB (mlpackage load) / ~1.1GB (CompiledMLModel);
2048/GPU ~800MB / ~3.6GB; 1024/ANE ~220MB (weights in ANE-managed memory).
Load/compile peak (transient): 1.5GB @1024, 4.3GB @2048, 1.2GB @1024-ANE.
CompiledMLModel python loads hold 2-3x more resident than mlpackage loads
- treat all as macOS upper bounds; measure in Swift on device.
iPhone guidance: 1024 bucket max (chunk long text at sentence boundaries),
consider 512 bucket + 6-bit palettization; 2048 compile peak (4.3GB) will
not fit older-device jetsam limits.

### Swift harness (swift/RssBench.swift, CoreML.framework, phys_footprint)

| config | load | steady footprint | fm step | core RTFx |
|---|---|---|---|---|
| 1024 all(GPU) | 8.5s | 996 MB | 14.7 ms | 92x (5.9s gen) |
| 1024 ane | 16.5s | 658 MB | 286 ms | 5.2x |
| 2048 all(GPU) | 22.4s | 3059 MB | 34.3 ms | 114x (16.8s gen) |

Latency matches the python host exactly. Memory is the real story:
phys_footprint (jetsam metric) is ~1.0GB at the 1024 bucket steady -
activation arenas are preallocated at load for the fixed max shape
(652MB after load, +340MB after first predict). 2048 bucket = 3.0GB
steady: not iPhone-viable. Python mlpackage-path RSS (~480MB) undercounted;
the CompiledMLModel path (~1.1GB) was the honest one.
iPhone plan: 1024 bucket max + sentence chunking, add a 512 bucket
(~2.9s gen) for short utterances, 6-bit palettization to cut the weight
share, and revisit ANE layouts (658MB + lowest power, but 20x slower today).

### Quantization matrix (FmDecoder, 1024 bucket; TextEncoder stays fp16)

| variant | weights | GPU step | steady footprint | log-mel cos | RMS delta | transcript |
|---|---|---|---|---|---|---|
| fp16 | 249 MB | 14.7 ms | 996 MB | 0.99925 | -0.09 dB | identical |
| int8 linear | 131 MB | CRASHES on GPU | 674 MB (ane) / 48 MB (cpu) | 0.99870 | -0.05 dB | identical |
| 6-bit palettize | 101 MB | 15.0 ms | 419 MB | 0.99829 | -0.45 dB | identical |
| int4 palettize (per-tensor) | 72 MB | 14.1 ms | 420 MB | 0.93412 | -2.07 dB | intelligible, degraded |

- int8 linear_symmetric hits an OS bug on GPU: MPSGraph "MLIR pass manager
  failed" assertion at first predict (macOS 26.5, M5 Pro). Works on
  ane (286ms) and cpu (152ms). Do not ship int8 for the GPU path.
- 6-bit per-tensor palettization is the ship candidate: transparent
  transcript, -0.45dB, full GPU speed, footprint 996 -> 419 MB (2.4x),
  weights 249 -> 101 MB. Same recipe class as Supertonic-3 int4.
- int4 per-tensor is a bridge too far without grouped channels; retry
  with per_grouped_channel group_size=16 under an iOS18+ target.
- Palettized variants land LuxTTS below every existing FluidAudio TTS
  backend's peak RSS except Supertonic-3 (197 MB).

### ANE deep-dive

Fallback is NOT the problem: 2285/2297 ops on ANE (99.5%); the 12 CPU ops
(depthwise convs, kernel 31/15 "grouped conv with large kernel size")
cost 0.6ms. Quantization does NOT help: fp16/int8/6bit/int4 all 286ms
@1024 - ANE is not weight-bandwidth-bound here.

Bucket scaling shows the real issue - ANE time grows superlinearly
(doubling frames: x2.3, x3.2, x3.5) while GPU stays near-linear:
256f: ANE 40ms vs GPU 5.9ms (6.8x) | 512f: 91 vs 7.9 (11.4x)
1024f: 286 vs 14.1 (20.3x) | 2048f: 1009 vs 34.3 (29.4x)
=> the O(T^2) rel-pos attention (softmax + gather at [heads,B,T,T],
seq-first permutes) tiles catastrophically on ANE at T>=512. Classic
ane-transformers territory: needs (B,C,1,S) QKV restructuring, split
softmax, and replacing the rel-pos gather - a dedicated trial per the
ANE playbook, not a conversion flag.

Interim ANE options: 256-frame buckets are only 6.8x off GPU - ANE-only
sentence-chunked synthesis at ~17x realtime core (4x40ms for ~2.7s) is
already usable where power matters, and the ANE/GPU gap will narrow on
iPhone (weaker GPU, comparable ANE). System view: FluidAudio's other
models occupy ANE; TTS-on-GPU diversifies the load.

### ANE rewrite feasibility (module benchmarks @T=1024, stack0 dim512)

| module | ANE (T,B,C) | GPU | ANE (B,C,1,S) recast |
|---|---|---|---|
| attention (weights+apply) | 5.08 ms | 2.39 ms | - |
| feedforward | 6.85 ms | 1.37 ms | **0.50 ms (13.7x)** |
| conv_module | 2.52 ms | 1.41 ms | - |

Summed over 16 layers x per-stack T, the seq-first module costs fully
account for the 286ms whole-model step - it is not partition overhead;
EVERY module pays the (T,B,C) layout tax uniformly (ff worst at 5x vs
GPU). The 1x1-conv2d recast of feedforward (same weights, channels-first)
runs 13.7x faster on ANE, confirming the ane-transformers thesis.

Projected full rewrite: ff-class modules ~0.5ms, attention needs the
(B,C,1,S) QKV + last-axis softmax + constant-matrix rel-pos treatment
=> plausible 286 -> ~40-60ms/step ANE @1024 (4 steps ~200ms core,
~28x RT, 658MB, low power) - iPhone-viable ANE-first TTS.

Trial plan (next session):
1. ANE-canonical TTSZipformer forward (weights reused, no retrain):
   linear->conv2d(1x1), norms on channel dim, bypass/downsample
   channel-first, SwooshL/R kept elementwise.
2. Attention: per-head qkv via conv, softmax over last (S) axis,
   rel->abs positional via constant banded matrix (fixed shapes).
3. Per-module parity gates vs torch (the quick ff recast above skipped
   submodule details - do it exactly), then whole-decoder parity + wav.
4. coreml-cli --fallback loop per playbook; target zero CPU ops and
   subquadratic-ish scaling to 1024f.

### ANE layer rewrite (trial 2) — one full Zipformer2EncoderLayer, (1,C,1,S)

coreml/ane/: AneZipformerLayer imports weights from the post
convert_scaled_to_non_scaled layer (enc0.layers[0], dim 512, H=4, qhd=32,
phd=4, vhd=12) into ANE-canonical form: every Linear -> 1x1 conv2d,
BiasNorm/Bypass on the channel axis, per-head attention with S on the
last softmax axis, SwooshL/R elementwise (logaddexp_onnx form).

Parity (fp32 eager, S=1024, real pos_emb, coreml/ane/parity.py) —
numerically exact, every submodule and the whole layer:

| submodule | max_abs_diff | cos |
|---|---|---|
| self_attn_weights | 2.4e-07 | 1.0 |
| feed_forward1/2/3 | <2e-06 | 1.0 |
| nonlin_attention, self_attn1/2 | 0.0 | 1.0 |
| conv_module1/2 | 1.9e-06 | 1.0 |
| norm, bypass, bypass_mid | <2e-06 | 1.0 |
| WHOLE LAYER | 2.4e-06 | 1.00000000 |

CoreML (iOS17 fp16 mlprogram, S=1024, M5 Pro, coreml/ane/bench.py):

| layer | ANE | GPU | ANE placement |
|---|---|---|---|
| original (T,B,C) | 35.9 ms | 2.2 ms | 136/138 ops (2 dw convs CPU) |
| ANE-canonical | **4.14 ms (8.7x)** | 2.1 ms | **166/166 ops, zero fallback** |

fp16 CoreML (ANE) vs torch fp32: cos = 1.000000, max_abs 0.015
(orig layer converted the same way: cos 0.980 — the gather rel->abs
path costs accuracy too).

What it took (each step verified by --fallback + latency):
1. Baseline recast (skew-trick rel->abs via pad+reshape+slice): 13.6 ms,
   11 CPU ops. The 2*S*S flatten exceeds ANE dim limits -> reshape/slice
   on CPU; BiasNorm ** -0.5 pow also CPU.
2. pow -> rsqrt, and rel->abs baked into a constant buffer
   pos_abs[h,c,q,j] = linear_pos(pos_emb)[h,c,S-1-q+j] consumed as
   broadcast-mul + channel reduce (no gather/as_strided/flatten): 5.35 ms.
3. Depthwise k=31 conv: ANE limit is kernel width <= 15 (16 falls back,
   15 places). Split exactly into k=15 + k=15 + k=1 with symmetric conv
   padding + output slices (standalone pad ops are not ANE-placeable and
   drag neighbors to CPU): 4.14 ms, 100% ANE.

Caveats for the whole-decoder rewrite:
- pos_abs is (H, phd, S, S) per layer: 33.5 MB fp16 at S=1024. Fine per
  layer; naive replication over all layers/stacks adds ~100-300 MB
  (smaller S per downsampled stack shrinks it 4x/16x...). If footprint
  matters, revisit a chunked skew or share pos_abs where linear_pos
  weights allow folding.
- Projection: dominant stack layer 8.7x faster => 286 ms/step scales to
  ~35-60 ms/step ANE @1024 (smaller stacks saw less ANE penalty, so gain
  there is smaller); 4 steps ~150-250 ms core, iPhone-viable ANE-first.

Template for the full rewrite: coreml/ane/layer.py.

### ANE full-decoder rewrite (trial 3) — AneFmDecoder, all 16 layers, (1,C,1,S)

coreml/ane/decoder.py: full TTSZipformer fm_decoder in ANE-canonical form —
in/out proj as 1x1 convs, t + guidance_scale sinusoidal embedding ported to
(1,C,1,1) (cos/sin cat on the channel axis, MLPs as 1x1 convs), per-stack
time projections, SimpleDownsample as a strided depthwise conv with
softmax(bias) taps, SimpleUpsample as nearest upsample, stack out-combiners
as channel-axis Bypass, per-stack cnn kernels 31/15/7/15/31 (31 split
15+15+1; 15 and 7 place directly). Same I/O contract as the original
FmDecoder (t, x, text_condition, speech_condition, guidance_scale,
padding_mask -> v), so coreml/parity.py-style feeds and swift/RssBench work
unchanged. Mask ported as float bias: -1000 added pre-softmax on keys +
zeroing before the depthwise convs + [::ds] subsampling — exactly upstream's
masked_fill semantics.

**pos_abs sharing**: linear_pos weights are NOT shared across layers, so the
folded per-layer (H,phd,S,S) buffer of trial 2 would cost ~260 MB
decoder-wide. Every layer's posproj = PE @ W_l^T lies in col(PE) (pos_dim
48), and the SVD of the per-seq-len concatenated posprojs has numerical rank
~27: an R=32 orthonormal basis reconstructs all of them to <=3.7e-8
relative. One pos_basis[r,q,j] = U_R[S-1-q+j, r] constant per distinct S
(1024: 67 MB, 512: 16.8 MB, 256: 4.2 MB fp16 = **88 MB total, 3 buffers**),
with the per-layer basis coefficients folded into that layer's attention
in_proj (p block 16 -> H*R=128 channels). mlpackage: 313 MB vs 238 MB orig.

**Eager fp32 parity** (oracle inputs, 4-step loop): vs torch @1024+mask
max_abs <= 3.8e-4, cos = 1.00000000 every step, final mel cos 1.00000000.
vs the 751-exact oracle path: cos 0.9990-0.9997 — that is the pure-torch
padding-leak floor, NOT the rewrite: upstream SimpleDownsample folds frame
751 (computed garbage in the padded region) into downsampled frame 375 and
mask[::2] keeps it, so torch@1024+mask itself differs from torch@751 by
max_abs 1-2 / cos 0.9991-0.9997. The shipped FmDecoder bucket pays the same
floor (its trial-1 step cos 0.999 was mostly this, not fp16).

**Consolidated results** (M5 Pro, S=1024 bucket, oracle utterance 469
prompt + 282 gen frames = 3.008 s, whisper-base, 3 warmup + 10 timed):

| metric | AneFmDecoder ANE | AneFmDecoder GPU | AneFmDecoder CPU | orig ANE | orig GPU |
|---|---|---|---|---|---|
| fm step (python) | **54.6 ms** | 23.5 ms | 145 ms | 286 ms | 14.2 ms |
| fm step (Swift) | 54.5 ms | - | - | 286 ms | 14.7 ms |
| core te+4 steps | 223 ms | 96 ms | 586 ms | 1149 ms | 62 ms |
| core RTFx (3.0 s oracle) | **13.5x** | 31.3x | 5.1x | 2.6x | 48x |
| core RTFx (5.9 s full bucket) | **26.6x** | 61.7x | - | 5.2x | 92x |
| per-step cos vs torch | 0.978-0.985 | - | 0.9986-0.9990 | 0.904-0.930 | 0.999 |
| final mel cos | 0.9750 | - | 0.99868 | **0.6814** | 0.9988 |
| log-mel cos (wav) | 0.96426 | - | 0.99889 | (wav cos -0.04) | 0.99925 |
| RMS delta | -0.54 dB | - | -0.10 dB | - | -0.09 dB |
| transcript (whisper-base) | "brown fox jumps over the lazy dog and honestly it felt great." | - | identical | GARBLED: "they pack or burn fast, jump through the lazy guards. Usley." | identical |
| ANE placement | **2591/2591 (100%), 0 CPU** | - | - | 2285/2297 (12 CPU) | - |
| Swift phys_footprint steady | **25.5 MB** | - | - | 658 MB | 996 MB |
| model load | 13.7 s py / 11.9 s Swift | 1.4 s | - | 16.5 s | 8.5 s |

Transcript matches the oracle modulo casing ("Brown Fox jumps...") — the
known prompt-boundary elision, not a regression.

Takeaways:

- ANE step 286 -> **54.6 ms (5.2x)**, 100% ANE placement, zero fallback —
  inside the 40-60 ms projection from trial 2.
- The rewrite is not just faster on ANE, it is the difference between
  broken and shippable there: the ORIGINAL graph on ANE produces garbled
  audio (mel cos 0.68, unintelligible whisper transcript). The seq-first
  gather/as_strided rel-pos path loses precision on ANE (trial 2 saw the
  same per-layer: orig 0.980 vs ane 1.000000).
- Remaining ANE-vs-CPU quality gap (log-mel 0.964 vs 0.999, -0.54 dB,
  transcript intact) is fp16 accumulation compounded over 16 layers x 4
  solver steps on the ANE path — the same package at CPU_ONLY matches the
  shipped conversion exactly, and eager fp32 parity is exact.
- phys_footprint 25.5 MB steady (vs 658 MB orig-ANE / 996 MB GPU): with
  100% ANE placement, weights + activations live in ANE-managed memory
  outside the jetsam-counted footprint. Combined with 54.6 ms steps this is
  the iPhone path: ANE-first, low power, no jetsam pressure.
- GPU step of the ANE-canonical graph is 23.5 ms vs 14.2 ms original (the
  R=32 broadcast reduce costs more on GPU) — keep the original conversion
  for the macOS GPU path, ship AneFmDecoder where ANE/power/memory matter.
- Harnesses: coreml/ane/decoder_parity.py (eager gate),
  coreml/ane/convert_decoder.py (build/coreml-ane, compiles
  FmDecoder.mlmodelc for rss_bench), coreml/ane/pipeline.py (quality +
  whisper + latency).

### int4 grouped-channel (iOS18)

Retry of int4 with per_grouped_channel kmeans palettization (group_size=16,
Supertonic-3's recipe class) — needs an iOS18-target source package, so the
original-graph FmDecoder was reconverted with iOS18 (convert_coreml.py grew a
--deployment-target flag; default stays iOS17, existing packages untouched)
to build/coreml-ios18, then coreml/quantize_int4_grouped.py ->
build/coreml-int4g. Quality gauntlet identical to the quantization matrix
(coreml/parity.py CPU_ONLY + 128-mel log-mel cos + whisper-base).

| variant | weights | GPU step | steady footprint | final mel cos | log-mel cos | RMS delta | transcript |
|---|---|---|---|---|---|---|---|
| 6-bit palettize (iOS17) | 101 MB | 15.0 ms | 419 MB | 0.98501 | 0.99829 | -0.45 dB | identical |
| int4 per-tensor (iOS17) | 72 MB | 14.1 ms | 420 MB | - | 0.93412 | -2.07 dB | intelligible, degraded |
| **int4 grouped g=16 (iOS18)** | 72.7 MB | 16.1 ms | **243.8 MB** | 0.92336 | 0.98589 (80-mel) / 0.98437 (128-mel) | -0.90 dB | **identical** |

(6-bit final-mel cos measured this session for comparison: its per-step cos
is 0.977-0.988 vs int4g's 0.939-0.964.)

- Grouped channels recover most of per-tensor's loss (-2.07 -> -0.90 dB,
  log-mel 0.934 -> 0.984, transcript garble -> verbatim "Brown Fox jumps
  over the lazy dog and honestly it felt great.") at the same 72 MB — but
  it is NOT transparent: still 2x the 6-bit level error and audibly softer.
- Footprint surprise: 243.8 MB steady on GPU vs 419 MB for 6-bit — the
  iOS18 runtime keeps the grouped-LUT weights compressed in the arena
  (6-bit/iOS17 decompresses to fp16 at load). ANE run: 287 ms step
  (unchanged, not weight-bound), 121 MB.
- Verdict: 6-bit stays the quality ship candidate on iOS17; int4 grouped is
  the memory-floor option (72 MB weights / 244 MB steady, -0.45 extra dB)
  if we accept iOS18 minimum.

**Stacked on the ANE-canonical graph** (AneFmDecoder reconverted iOS18 ->
build/coreml-ane-ios18, palettized -> build/coreml-ane-int4g; pipeline.py
CPU_AND_NE + rss_bench ane): weights 328 -> 149.9 MB (pos_basis constants
resist palettization), step 55.5 ms / footprint 34.8 MB (both ~unchanged vs
fp16's 54.6 ms / 25.5 MB). But quantization error stacks on the fp16-ANE
accumulation floor: final mel cos 0.9038, log-mel 0.93490, RMS -1.77 dB
(fp16 ANE was 0.96426 / -0.54 dB), and whisper drops a word: "...over the
lays dog...". Roughly additive in dB — do NOT ship int4g on the ANE path;
if ANE memory matters it's already 25 MB at fp16.

### Decision: int4 scrapped (2026-07-07)

All int4 variants dropped from the lineup and build artifacts deleted.
Rationale: per-tensor loses 2.1dB; grouped-channel (iOS18) is verbatim-
transcript but still ~2x the 6-bit error and forces an iOS18 floor; on
the ANE graph it degrades quality (word error) with zero latency/memory
win (fp16 ANE graph is already 25.5MB). Ship set: fp16 ANE graph
(iPhone), fp16 or 6-bit original graph (macOS GPU). int8 parked on the
MPSGraph GPU bug. Numbers retained above for the record;
quantize_int4_grouped.py removed (recoverable from git history).
