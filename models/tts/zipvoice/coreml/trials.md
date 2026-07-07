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
