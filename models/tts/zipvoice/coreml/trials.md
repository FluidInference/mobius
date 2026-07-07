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
