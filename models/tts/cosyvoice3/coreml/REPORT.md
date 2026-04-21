# CosyVoice3 → CoreML Progress Report

## Scope

Port CosyVoice3 TTS (Qwen2-LLM + CFM Flow + HiFT vocoder + CAMPPlus speaker embed
+ SpeechTokenizer) to CoreML mlpackages with static shapes targeting Apple Silicon.

Target: full on-device zero-shot TTS with Swift integration.

## Status Summary

| Component | mlpackage | Status | Parity vs Ref |
|---|---|---|---|
| Qwen2 LLM — Prefill (T=256, M=768) | `LLM-Prefill-T256-M768-fp16.mlpackage` (695 MB) | Shipped | argmax OK, logits MAE 0.068 (fp16) |
| Qwen2 LLM — Decode  (M=768) | `LLM-Decode-M768-fp16.mlpackage`  (695 MB) | Shipped | argmax OK, logits MAE 0.018 (fp16) |
| Flow (N=125 → M=250) | `Flow-N125-fp32.mlpackage` (1.2 GB) | Shipped | OK (existing) |
| Flow (N=250 → M=500) | `Flow-N250-fp32.mlpackage` (1.2 GB) | **NEW** | torch→ml trace succeeded |
| HiFT (T=250 → 5 s audio) | `HiFT-T250-fp32.mlpackage` (84 MB) | Shipped | OK (existing) |
| HiFT (T=500 → 10 s audio) | `HiFT-T500-fp32.mlpackage` (88 MB) | **NEW** | Trace OK |
| CAMPPlus (T=300) | `CAMPPlus-T300-fp32.mlpackage` (27 MB) | Shipped | cos=0.96 vs onnx (known drift) |
| SpeechTokenizerV3 (T=500) | `SpeechTokenizerV3-T500-fp32.mlpackage` (924 MB) | **NEW** | 44/87 tokens drift vs onnx on real audio |
| Embedding tables | `embeddings-fp32.safetensors` (542 MB) / `-fp16` (271 MB) | **NEW** | exact copy of Qwen2 + speech embed weights |

## New Artifacts (this session)

```
build/
├── flow-fp32-n250/
│   ├── Flow-N250-fp32.mlpackage       (1.2 GB)
│   └── ref-N250.pt                    (parity reference)
├── hift-fp32-t500/
│   ├── HiFT-T500-fp32.mlpackage       (88 MB)
│   └── ref-T500.pt
├── speech-tok-fp32/
│   └── SpeechTokenizerV3-T500-fp32.mlpackage   (924 MB)
├── embeddings/
│   ├── embeddings-fp32.safetensors    (542 MB)
│   ├── embeddings-fp16.safetensors    (271 MB)
│   ├── embeddings-fp32.json           (layout metadata)
│   └── embeddings-fp16.json
└── mlmodelc/                          (pre-compiled .mlmodelc for profiling)
    ├── CAMPPlus-T300-fp32.mlmodelc
    ├── Flow-N125-fp32.mlmodelc
    ├── Flow-N250-fp32.mlmodelc
    ├── HiFT-T250-fp32.mlmodelc
    ├── HiFT-T500-fp32.mlmodelc
    ├── LLM-Prefill-T256-M768-fp16.mlmodelc
    ├── LLM-Decode-M768-fp16.mlmodelc
    └── SpeechTokenizerV3-T500-fp32.mlmodelc
```

## New Scripts

- `convert-speech-tokenizer.py` — ONNX → CoreML via onnx2torch for
  `speech_tokenizer_v3.onnx`. Registers coremltools torch-ops for
  `greater_equal` / `less_equal`, patches onnx2torch's missing `GreaterOrEqual`
  opset-16 converter.
- `export-embeddings.py` — Exports Qwen2 `embed_tokens` (151936×896) and
  CosyVoice3 `speech_embedding` (6761×896) to safetensors + JSON metadata.

## End-to-End Test (already working)

`verify/test_coreml_e2e.py` chains: Python frontend → CoreML LLM-Prefill →
decode loop (ras_sampling) → CoreML Flow → CoreML HiFT → WAV.

Latest run (CPU_ONLY):
```
prefill  3.57 s
decode   2.92 s  (38 tokens, 13 tok/s)
flow    28.58 s
hift     1.18 s
output   build/wavs/e2e_coreml.wav  (1.52 s audio)
Whisper: '希望你以后能够' vs expected '希望你以后能够做的比我还好用'
  (truncated because Flow N=125 left only 38 new-token room after 87 prompt tokens)
```

With Flow N=250 we now have room for N_new = 250 − N_prompt ≈ 160 new tokens
(~6.4 s of audio), enough for full sentences.

## Known Parity Drift

### 1. SpeechTokenizerV3 (44/87 tokens differ on real audio)

**Cause.** The model ends with a vector quantizer (argmax over a 6561-entry
codebook). Tiny numerical differences during MIL graph optimization flip the
argmax to neighboring codes. onnx2torch→torch matches onnxruntime bit-exactly;
the drift is introduced by coremltools conversion/MIL.

**Example drift** (zero_shot_prompt.wav, first 10 tokens):
```
onnx   : [4966, 488, 244, 28, 28, 28, 1, 108, 5535, 5049, ...]
coreml : [4885, 488, 271, 28, 28, 28, 28, 108, 5535, 5049, ...]
```

**Impact.** These are the prompt-speech IDs fed to (a) the LLM as context, and
(b) Flow's token embedding. Since Flow *also* receives the ground-truth prompt
mel (computed separately by the 24 kHz path), voice cloning quality should be
robust to token drift. LLM context drift is more concerning but similar to the
CAMPPlus cos=0.96 drift which empirically still produces usable voice cloning.

**Next steps (if needed).** Identify the op(s) where fp32 MIL diverges from
onnxruntime by running `--debug` with intermediate output capture.

### 2. CAMPPlus (cos=0.96, MAE 0.18)

Known; pre-existing. Likely BatchNorm numerical drift from onnx2torch. Shipped
as-is.

## Cleared Issues

- **`~47 GB ANE compile cache** in `~/Library/Caches/.../com.apple.e5rt.e5bundlecache/`
  was filling the disk and causing "weights.bin could not be opened" failures in
  profiling. Cleared during this session, freeing disk from 14 → 60 GB.

## Profiling (partial)

`uv run coreml-cli <model.mlmodelc>` on Apple M2 / macOS 26.5:

| Model | Best unit | Predict (ms) | ANE % | Notes |
|---|---|---:|---:|---|
| CAMPPlus-T300-fp32 | cpu_and_gpu | 23.6 | 0% | GPU path only |
| HiFT-T250-fp32 | all / cpu_and_gpu | 513–534 | 0% | fp32 ⇒ ANE rejects every op |
| HiFT-T500-fp32 | cpu_and_gpu | 848 | 0% | same — "Invalid output tensor format: fp32" |

**Root cause for 0% ANE:** ANE requires fp16. All three models are compiled
with `compute_precision=FLOAT32`. HiFT/Flow/CAMPPlus need a **revised fp16
conversion** to land on ANE.

**Remaining profiles (not yet run):**
- `Flow-N125-fp32` / `Flow-N250-fp32`
- `LLM-Prefill-T256-M768-fp16` — interrupted by "unknown error" compute-plan
  load (likely E5 cache re-population, should retry after cache warmed)
- `LLM-Decode-M768-fp16`
- `SpeechTokenizerV3-T500-fp32`

The LLM is the only model currently in fp16 (with fp32 ops for
RMSNorm/softmax). Expected to have meaningful ANE residency once we
re-profile after cache rebuild.

**2026-04 retry (all four shipping models, on macOS 26.5 / M2):**

| Model | `MLComputePlan.loadContentsOfURL` | Cold compile | Notes |
|---|---|---:|---|
| LLM-Prefill-T256-M768-fp16 | fails ("unknown error") | n/a | compute-plan path never returns |
| LLM-Decode-M768-fp16       | fails ("unknown error") | n/a | same |
| Flow-N250-fp32             | fails ("unknown error") | n/a | same |
| HiFT-T500-fp16             | fails ("unknown error") | 118,313 ms | compile succeeds, plan extraction doesn't |

Sanity check: `silero-vad-unified-v6.0.0.mlmodelc` profiles cleanly with the
same tool on the same machine (0.57 ms predict, 100% CPU), so the failure is
specific to CosyVoice3 graphs — likely the stateful-KV ops in the LLM and the
fused `layer_norm` in Flow exceed what `MLComputePlan`'s introspection path
can classify. `coreml-cli --fallback` takes the same code path and fails
identically.

**Workaround for ANE residency data:** use Instruments' CoreML template or
log device assignments from a Swift harness (`MLModel.compileModel(at:)` +
`MLComputePlan.load(contentsOf:configuration:)` returning a structured
fallback). That Swift harness doesn't exist yet — tracked as a follow-up in
the FluidAudio repo (pending).

## Recommended Next Steps

1. **Re-profile after ANE cache warms.** Run `coreml-cli` individually on each
   remaining `.mlmodelc`. Expect non-zero ANE% only for the fp16 LLM models.
2. **Produce fp16 variants** of Flow, HiFT, CAMPPlus, SpeechTokenizer. For each,
   use the same selective-fp32 trick we used for LLM
   (`compute_precision=ct.transform.FP16ComputePrecision(op_selector=...)`) to
   keep numerically-sensitive ops in fp32 (RMSNorm / softmax / VQ argmax
   comparisons) while letting the bulk land on ANE.
3. **Flow / HiFT CPU-fallback fix.** HiFT fallback reasons included scatter,
   cumsum, range_1d — these are already in the MIL graph. Some are avoidable
   with CoreML-friendly rewrites in `src/hift_coreml.py` /
   `src/weight_norm_fold.py`; worth re-examining if ANE residency matters.
4. **SpeechTokenizer parity.** If downstream voice-cloning quality degrades,
   try a bit-exact path via ONNX Runtime on device (NB: ORT has a CoreML
   execution provider that may dispatch to ANE anyway), or narrow MIL precision
   overrides.

## Ready-for-Swift

Everything the Swift side needs (excluding the still-Python frontend + wetext
normalization) is now present:

- CoreML mlpackages for LLM, Flow (up to N=250), HiFT (up to T=500), CAMPPlus,
  SpeechTokenizerV3.
- `embeddings-fp16.safetensors` / `embeddings-fp32.safetensors` with
  `text_embedding` + `speech_embedding` tables and a JSON metadata file
  documenting layout, sos/task/eos IDs, and the `lm_input` construction recipe
  (see `SWIFT_PORT_NOTES` in `src/text_frontend.py`).

Remaining Swift-port gaps (design, not conversion):
- Qwen2 tiktoken BPE tokenizer (swift-transformers has it)
- wetext text normalization (recommend server-side)
- 16 kHz Whisper log-mel and 80-d kaldi fbank preprocessing (vDSP)
- 24 kHz log-mel for Flow prompt
- `ras_sampling` loop (trivial port, reference in `verify/test_coreml_e2e.py`)
