# NVIDIA NemotronLabs VoiceChat 11B — FluidAudio Prep

Feasibility trial for running [nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
(released 2026-08-03) on Apple Silicon. End-to-end full-duplex speech-to-speech
model: streaming speech understanding + speech generation + tool calling in one
unified architecture (~450 ms turn-taking latency, #2 open FD on VoiceBench,
first open FD model with tool calling). License: OpenMDW-1.1 — permissive ("deal in the Model Materials without restriction"; retain the license text and notices in distributions).

Reference implementation: [NVIDIA-NeMo/Speech @ `nemotron-labs-voicechat`](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat),
model class `nemo/collections/speechlm2/models/nemotron_voicechat.py`.

## Component map (from safetensors header, all F32 = 44.4 GB)

| Component | Keys | Params | Apple target |
|---|---|---:|---|
| Fast Conformer encoder | `stt_model.perception.encoder.*` | 609 M | CoreML/ANE — same family as shipped `nemotron-speech-streaming-en-0.6b` (24L, d1024, 128 mel, 8x subsampling, cache-aware `chunked_limited` att_context **[70, 0]** = fully causal, 80 ms frames) |
| Perception proj | `stt_model.perception.proj.*` | 4.6 M | CoreML (1024 → 4480 into LLM embed space) |
| LLM backbone | `stt_model.llm.layers.*` | 7 714 M | **CoreML int8** (lossless; 4 stateful shards, 43.5 ms/step measured — Phase 3) — Nemotron Nano v2 9B (NemotronH, hidden 4480; verified from keys: 56 layers = 27 Mamba2 + 25 MLP + 4 attention at indices 14/21/30/39 → tiny KV cache, O(1) Mamba state). MLX remap = fallback only |
| Token embeddings | `stt_model.embed_tokens.weight` | 587 M | host-side fp16 lookup table (131072 × 4480; shards consume `inputs_embeds` directly) |
| Text head | `stt_model.lm_head.weight` | 587 M | CoreML (heads shard) |
| Function head | `stt_model.function_head.weight` | 587 M | CoreML, heads shard (second lm_head on same hidden state → tool-call channel) |
| RNNT decoder+joint | `stt_model.rnnt_decoder.*`, `stt_model.rnnt_joint.*` | ~13 M | CoreML — verified from keys: 2-layer LSTM prednet (hidden 640, embed 1025×640), joint enc 1024→640 / pred 640→640 → 1025 logits; same shapes the 0.6b conversion scripts already handle (tokenizer in `rnnt_tokenizer/`) |
| TTS backbone | `tts_model.tts_model.backbone.*` | 595 M | CoreML/ANE — `gemma3_text` 28L, hidden 1152, per-step decode with KV cache |
| TTS MoG head | `tts_model.tts_model.mog_head.*` | 159 M | CoreML (mixture-of-gaussians over latent 512, 1024 predictions, low-rank 64) |
| TTS misc | `embed_subword`, `rvq_embs`, `gated_fusion_audio_text`, `audio_prompt_projection_W` | ~43 M | CoreML |
| Audio codec decoder | `tts_model.audio_codec.decoder.*` + `prvq` | 108 M | CoreML/ANE — 31 quantizers, codebook 1024, 12.5 Hz → 22.05 kHz (wav_to_token_ratio 1764) |
| Audio codec encoder | `tts_model.audio_codec.encoder.*` | 92 M | Only needed to encode speaker-prompt audio; pre-baked speaker latents ("Aria") ship in the checkpoint, so optional |

## Runtime loop (per 80 ms frame `t`)

From `NemotronVoiceChat.offline_inference` + `DuplexSTTModel._step_inference`:

1. **Perception**: mel (128-dim, 10 ms hop) → conformer chunk step → `audio_embeds[t]`
   (4480-d) and `asr_emb[t]` (1024-d encoder tap, pre-proj).
2. **Fusion**: `fused = fusion(agent_text_emb[t-1], audio_embeds[t], function_emb[t-1])`
   — embeds of the *previous* generated text/function tokens plus current audio frame
   (config: `use_gated_fusion_for_text_audio: true` on TTS side; STT fusion is add/gated per cfg).
3. **LLM step**: 9B NemotronH step on `fused` with cache
   (note `use_cache_for_nemotron: false` in config — Mamba state carries; attention
   layers use sliding window) → `text_logits` (sampled) + `function_logits` (greedy).
4. **TTS step**: `infer_codes_one_step(current_subword, prev_subword, prev_codes, kv_cache)`
   → gemma3 backbone step → MoG head → 31-codebook frame. Delay: `num_delay_speech_tokens: 2`,
   classifier-free guidance scale 0.2 (i.e. 2 backbone evals/step unless distilled out).
5. **RNNT branch** (optional user transcript): one `asr_emb` frame → prednet/joint greedy step.
6. **Codec decode**: codes → 22.05 kHz waveform (incremental every step, or batched at end).

Tool calling: function channel emits `<TOOLCALL>[...]</TOOLCALL>` on its own channel;
host executes and injects `<TOOL_RESPONSE>` tokens back into the function channel.

## Feasibility

Per-frame budget is **80 ms**. Measured on M5 Pro / 24 GB, macOS 26.6 (2026-08-03):

| Step | Measured | Notes |
|---|---|---|
| Conformer encoder chunk (ANE) | **9.5 / 8.6 / 10.3 ms** for 560/1120/2240 ms tiers | cached `nemotron-multilingual` encoders, CPU_AND_NE (71–89% ANE). Latency is **flat across chunk sizes** → fixed-overhead-dominated, so per-80ms-frame stepping also lands ~8–10 ms. Chunked encoding amortizes to <1 ms/frame (1120 ms tier = 8.6 ms/14 frames) at the cost of added response latency; `[70,0]` causal allows either. English 0.6b int8 encoder (560 ms tier, HF): 9.7 ms GPU / 11.9 ms ANE — int8 does not reduce latency. |
| 9B LLM decode (MLX 4-bit) | **21.2 ms/tok** (47.2 tok/s) | mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits via mlx-lm 0.31.3; peak 5.24 GB. Confirms CoreML non-viability call (riva-translate-4b GEMV floor ⇒ 9B ≈ 80–90 ms/tok). |
| 9B LLM prefill (MLX 4-bit) | **546 tok/s** | 1442-token system prompt ingests in 2.6 s at session start (one-time). |
| RNNT decoder+joint step | **0.5 ms** (CPU; 0.23 + 0.27 ms split) | cached `decoder_joint.mlmodelc` / `decoder` / `joint`; negligible. |
| TTS backbone + MoG step | **22 ms measured** (backbone 14.2 GPU incl. CFG batch-2 + 8 MoG iterations × 0.96) | Phase 2b actuals — the ~6 ms magpie proxy under-counted the 8-iteration MaskGIT RVQ refinement and CFG. Levers: guidance off → batch-1 backbone; fewer refinement iterations. |
| Codec decoder | **3.2 ms measured** (per-frame T=1, GPU) | Phase 2c actuals; 1 s batches decode at 67× RT. |

**Frame total (all components now measured): encoder 12 + LLM int8 43.5 +
TTS 22 + RNNT 0.5 + codec 3.2 ≈ 81 ms vs the 80 ms budget — serial
worst-case is right at the line.** Paths back under budget: disable CFG
(batch-1 backbone, saves ~5–7 ms; quality impact is an open question below),
fewer MoG refinement iterations, amortized chunked encoding (~11 ms → ~1
ms/frame at +latency), or pipelining the LLM (GPU) against TTS/codec across
frames. The earlier 45–55 ms MLX-4-bit projection is superseded (4-bit RTN
fails the quality gate); MLX rows remain as the fallback-path baseline.
~13 GB resident with the int8 LLM.

### CoreML 9B floor benchmark (`bench_llm_coreml_floor.py`) — MLX-vs-CoreML revisited

Exact-geometry single-step stacks (random weights, real shapes), int4 per-block
(linear_symmetric, block 32), measured on M5 Pro:

| Stack | GPU median | ANE |
|---|---|---|
| Mamba2 ×9 (of 27) | 3.85 ms | ANEF compile fails → CPU fallback |
| MLP ×9 (of 25) | 3.43 ms | rejected |
| Attention ×4 (1024-frame KV window) | 4.01 ms | rejected |
| lm_head + function_head (2× 4480→131072) | 2.75 ms | rejected |
| **Full 9B step, sum-of-parts (as-if-fused)** | **≈ 28 ms** | |
| **Full 9B step, chained 8-model calls (measured)** | **50.6 ms** | ~2.8 ms/dispatch overhead |

Optimized v2 (`bench_llm_coreml_floor2.py`): iOS18 **stateful** shards — all
mamba conv/ssm states and the KV window live on-device via `ct.StateType`
(in-place slice-assignment in the torch source; whole-buffer `.copy_()` is NOT
matched by the converter pass), layers interleaved into 4 realistic shards
(7 Mamba2 + 6 MLP + 1 attn each, ~1.0 GB int4/shard) + heads:

| Variant | 9B step |
|---|---|
| v1 stateless, 8 calls, ~280 MB/step state I/O | 50.6 ms |
| v1 sum of isolated stacks (as-if-fused) | ~28 ms |
| **v2 stateful, 4 shards + heads (5 calls)** | **26.6 ms** |
| MLX 4-bit reference | 21.2 ms |

Conclusion: **CoreML int4 on the M5 GPU reaches near-parity with MLX** —
26.6 ms vs 21.2 ms, well inside the 80 ms budget, ~4.7 GB int4 resident.
The riva-translate-4b 40 ms/tok floor does not reproduce at this geometry on
M5. The sharded stateful pipeline also sidesteps the >24 GB fused-trace RAM
problem entirely. The LLM runs on GPU (ANE rejects int4 stacks);
encoder/TTS on ANE — no compute-unit contention.

### Quantization quality on the REAL fine-tuned weights (`measure_int4_quality.py`)

Dual-track forward (fp32 baseline vs quantize-dequantize, same code both
tracks), all 56 layers streamed from `components/llm.safetensors`, 3 real
prompts, coremltools-equivalent per-block-32 linear_symmetric RTN.
**SUPERSEDED**: these numbers were measured before the gated-norm fix (see the
calibrated-gate section below); the corrected frontier there replaces them.
Directionally the conclusions hold, but int8 is ~99% top-1 on the corrected
forward, not exactly 100%:

| Scheme | top-1 agree | top-5 | KL | fn-head top-1 | step latency | LLM weights |
|---|---|---|---|---|---|---|
| fp16 cast (noise floor) | 100% | 100% | ~0 | 100% | — (19 GB, doesn't fit) | — |
| **int8 pb32** | **100%** | **100%** | **0.006** | **100%** | **45.4 ms** (measured, stateful shards) | ~9.4 GB |
| int4 body + int8 heads | 80.6% | 99.1% | 0.20 | 85.8% | ~28 ms est. | ~5.6 GB |
| int4 pb16 | 76.3% | 99.1% | 0.19 | 89.4% | ~27 ms est. | ~5.2 GB |
| int4 pb32 | 74.3% | 99.1% | 0.22 | 85.8% | 26.6 ms (measured) | ~4.7 GB |

Findings: naive int4 RTN is NOT acceptable; the function-head degradation
comes from accumulated body drift, not the head matrix. **int8 is
effectively lossless** (corrected forward: 98.9% top-1, KL 0.0002; the
shipped conversion separately validates 100% prefill argmax vs fp32 torch on
its test prompt) and still fits the budget: frame total
with int8 = encoder 12 + LLM 45.4 + TTS 6–12 + RNNT 0.5 + codec ≈ 65–72 ms
of 80 ms (thin but real-time on M5 Pro; needs ≥16–24 GB RAM at ~13 GB
resident). NOTE: naive 4-bit RTN is also what mlx-community quants use — the
quality problem applies to MLX equally; an MLX 8-bit run (~2× 21.2 ms) would
land near CoreML int8. Path to recover 4–6-bit quality if headroom is needed:
GPTQ/AWQ-calibrated int4 (coremltools layerwise compression), 6-bit
palettization (validate — pal4 was pathological in riva trial), or
sensitivity-based mixed int8/int4 per layer.

### Calibrated sub-8-bit gate (`measure_calibrated_quant.py`)

Extends the dual-track harness with AWQ-style activation-aware per-input-channel
scales (grid-searched per linear on a conversational-text calibration set; at
deployment 1/s folds into the preceding norm weight, so the scheme stays
coremltools-exact), a per-linear sensitivity proxy recorded during calibration
that drives mixed int8/int6/int4 promotion under an effective-bits budget, and
exact effective-bits accounting (weight bits + fp16 scale overhead).
`calib_scales.npz` is the calibration output (scales + sensitivities).
Gate: **≥98% top-1 at ≤5.0 effective bits** ⇒ sub-8-bit ships (and a
half-duplex browser LLM becomes plausible); fail ⇒ lossless int8 stands.

A first pass of this gate ran with a WRONG forward — full-width gated norm at
eps 1e-6 instead of the model's per-group (8 × 1280) RMSNorm at eps 1e-5
(caught in review; the shipped `convert_llm_real.py` always had the correct
semantics). The forward was fixed, `calib_scales.npz` regenerated, and the
entire frontier re-measured (`rerun_quant_gate.sh`); everything below is from
the corrected forward. Sub-5-bit numbers improved by 4–5 points but the
verdict did not change.

**Result: GATE FAILED.** Frontier (294 eval positions, 16 prompts, dual-track
fp32 reference; fp16 floor = 100%/100%, KL 0):

| Scheme | top-1 agree | top-5 | KL | fn-head top-1 | effective bits |
|---|---|---|---|---|---|
| RTN int4 pb32 | 80.0% | 98.9% | 0.174 | 75.3% | 4.5 |
| AWQ int4 | 87.4% | 100% | 0.063 | 83.9% | 4.5 |
| **GPTQ + AWQ int4 (best top-1 in budget)** | **92.2%** | **100%** | **0.032** | 89.6% | **4.5** |
| GPTQ + AWQ int4 + int8/chan heads | 90.8% | 100% | 0.028 | **90.7%** | 4.963 |
| GPTQ + AWQ int5 | 94.6% | 100% | 0.007 | 95.5% | 5.5 |
| AWQ int6 (~7.1 GB — pragmatic ship candidate) | 95.5% | 100% | 0.004 | 97.4% | 6.5 |
| int8 pb32 | 98.9% | 100% | 0.0002 | 99.6% | 8.5 |

Best in-budget top-1 is 92.2% at 4.5 effective bits vs the ≥98% bar (the
int8-heads variant trades ~1.4 pts of top-1 for +1.1 pts of function-head
accuracy — the drift is in the body either way). The frontier climbs slowly
and smoothly through 94.6% @ 5.5 and 95.5% @ 6.5 — no small set of promotable
outlier layers exists, so mixed precision cannot reach the gate under the
bits budget. Note **even int8 is 98.9%, not exactly 100%**, on this
294-position eval (the shipped conversion's own validation still shows 100%
prefill argmax on its test prompt). Consequence: the 9B stays **int8
(~9.4 GB) for native**; a browser-resident LLM is shelved (int8 ≈ 13 GB
resident is not browser-deliverable) — browser stays STT (+TTS/codec later),
full-duplex stays native. Next tiers if ever revisited: palettized K-means
LUT, QuIP#/SpinQuant-style rotations, QAT on fused embeddings. Caveat: the
calibration set is plain text, a proxy for the deployed fused
audio-embedding regime. Harness cost: full GPTQ pass ≈ 14 min / 15 GB RAM on
M5 Pro (batched layer-major streaming).

**Conclusion: CoreML-only execution.** CoreML/ANE for encoder + TTS + codec +
RNNT (≈ 1.6 B params total, ~all reusable patterns from prior trials) and
**CoreML int8 stateful shards for the 9B backbone + heads** (98.9% top-1 /
KL 0.0002 at int8 — effectively lossless; int4/int5/int6 fail the quality
gate — see above). Memory: int8 LLM ≈ 9.4 GB +
fp16 rest ≈ 3.5 GB → **~13 GB resident**. **macOS-only target** (Apple Silicon
≥ 16 GB, realistically 24 GB); iPhone is out of reach for v1. **MLX is
explicitly fallback-only** (kept as the escape hatch if CoreML shard dispatch
regresses on some OS); it would add an `mlx-swift` dependency FluidAudio does
not otherwise carry, which is another reason it is not the primary path.

## Plan

- [x] Checkpoint downloaded to `~/Documents/models/voicechat-11b` (44 GB, fp32)
- [x] Speech repo cloned at `~/Documents/models/voicechat-11b/Speech`
- [ ] **Phase 0 — slice**: `uv run python slice_checkpoint.py` → per-component
      safetensors (encoder/rnnt/tts/codec in fp32, LLM in bf16). Verify key coverage = 100%.
- [ ] **Phase 1 — torch parity harness**: run `offline_voicechat_infer.py` on CPU/MPS
      (mamba-ssm/causal-conv1d are CUDA-only → needs the pure-torch Mamba2 fallback path,
      or capture per-component I/O on a Linux GPU box) to produce golden tensors for
      each component on `turn_taking.wav`.
- [x] **Phase 2a — encoder + RNNT CoreML** (`convert_encoder.py`, `convert_rnnt.py`):
  - encoder: **native per-frame streaming confirmed** — `setup_streaming_params()`
    yields `chunk_size=[1,8]`, 70-frame channel cache, 9-mel pre-encode cache,
    `valid_out_len=1`. Exported with proj + asr_emb tap. Chained-cache parity
    torch↔CoreML: fp32 ≤8e-07, fp16 ≤1.05e-02 (25 steps). Per-frame step measured
    **11.4 ms GPU / 12.3 ms ANE (68% resident)** on M5 Pro.
  - rnnt: fp32 decoder + joint (fp16 rejected: LSTM state drift → 0.16/9.9 max|Δ|
    over 20 chained steps; fp32 is 1.4e-05/5.5e-04). Measured **0.37 + 0.13 ms/step**.
  - env gotchas: `numpy==2.2.6` (2.4.x makes coremltools' `aten::Int` handler throw
    TypeError on NeMo's size-1 `max_audio_length` array), `torch==2.12.1`.
  - **e2e STT validated** (`test_e2e_stt.py`): `sample_general.wav` → mel →
    CoreML encoder (fp16, per-frame streaming steps) → CoreML RNNT greedy →
    "Hello, do you know what color the sky is" — token-identical to the torch
    reference, asserted with nonzero exit on mismatch. Windowing streams the
    real mel history (left-padded only for missing frames) and processes the
    zero-padded final partial chunk. Real-audio behavioral parity for the
    full user-transcription chain.
- [x] **Phase 2b — TTS backbone + MoG head** (`convert_tts.py`): two CoreML models
      + host glue. `backbone_step_fp16`: manual gemma3_text single-frame step
      (28L h1152, 16 heads hd72, q/k RMSNorm, sandwich norms, 5:1
      sliding/full layer pattern with RoPE theta 10k/1M, scale 256^-0.5),
      batch 2 for CFG, rolling 1024-slot KV `ct.StateType` with pos masking —
      parity vs the HF backbone 2.96e-05 (12-step prefill), fp16 chained
      7.7e-02. `mog_dense`: mlp_stack + CFG combine → mixture logits / logs /
      mu_res / guided hidden; sampling + the low-rank mu gathers
      (proj_mus[idx] 64×1152, low_mat[idx] 512×64) and the RVQ
      depthsum encode/decode run host-side (npy exports). Deterministic
      e2e (argmax pick, noise 0): torch vs CoreML codes **identical 4/4
      frames**; full fp16 chain 124/124 code agreement. **Measured M5 Pro:
      backbone step 14.2 ms GPU (ANE rejects the stateful graph), MoG dense
      0.96 ms GPU / 0.82 ms ANE × 8 iterations → TTS ≈ 22 ms/frame.**
      Notes: `disable_eos_prediction` (EOS comes from the LLM text channel);
      per-frame generation is 8-iteration MaskGIT-style RVQ refinement, not
      one decoder call — the old ~6 ms proxy under-counted it.
- [x] **Phase 2c — codec decoder** (`convert_codec.py`): conv stack in CoreML
      (flexible T), PRVQ code→latent (31 embedding sums; codebooks ==
      tts rvq_embs, exported npy) + 16-point iSTFT tail host-side. Parity on
      real audio through the codec's own encoder: fp32 wav 7.5e-07 corr
      1.000000, fp16 wav 7.5e-03 corr 0.99995; round-trip corr vs source
      0.941. **Measured: 3.2 ms GPU per 80 ms frame (T=1); 15.6 ms for 1 s
      batches (67× RT).**
- [x] **Phase 3 — LLM on CoreML (real weights)** (`convert_llm_real.py`): the
      fine-tuned 9B converted to 4 stateful int8 shards (~1.9 GB each) + heads
      (1.2 GB) + fp16 embedding table for host-side lookup (1.1 GB, consumes
      `inputs_embeds` directly — fusion output plugs straight in). Semantics
      verified against mlx-lm nemotron_h (no RoPE; per-group gated RMSNorm
      g=1280; eps 1e-5; dt softplus). KV window 1024 with pos-state masking
      (exact < 1024 ctx, then 82 s sliding window; Mamba carries long-range).
      **Validated: 100% prefill argmax agreement vs fp32 torch reference;
      coherent greedy generation ("...Nemotron, created by NVIDIA. You are a
      helpful, respectful, and honest assistant..."). Step: 43.5 ms.**
      MLX remap kept as fallback only (not needed).
- [ ] **Phase 4 — Swift host**: streaming mel (NemotronMelExtractor reusable) →
      encoder step → fusion (tiny, host-side) → CoreML int8 LLM step (4 shards +
      heads, 5 dispatches, states on-device) → TTS step → codec chunks;
      barge-in = feed user audio continuously, agent yields when text channel emits EOS.
      Tool-call channel surfaced as an API callback.
- [ ] **Phase 5 — publish**: HF `FluidInference/nemotron-voicechat-11b-coreml`
      (CoreML bundles: encoder + RNNT + int8 LLM shards + heads + fp16 embed
      table + TTS + codec) after confirming repo with Alex. No MLX artifacts
      unless the fallback is ever needed.

## WebGPU/WASM port knowledge (browser STT)

The user-transcription chain (encoder + RNNT) is ported to the browser as the
`asr-voicechat` engine in fluidaudio-web
([PR #48](https://github.com/FluidInference/fluidaudio-web/pull/48)), running on
the shared FastConformer WGSL runtime (geometry is manifest-compatible; RNNT is
nemotron-shaped with vocab 1025 / blank 1024, no prompt kernel). Parity-gated
**byte-identical** to the torch/CoreML reference transcript on
`sample_general.wav` ("Hello, do you know what color the sky is").
Load-bearing facts for porting this encoder to any non-NeMo runtime:

- **Attention chunk = 1, not 8.** NeMo `chunked_limited` with att_context
  `[70, 0]` puts the chunk grid at `right + 1 = 1`. Running the runtime's usual
  chunk 8 leaks intra-chunk future frames through attention — output degrades
  subtly instead of failing.
- **The `batch_norm`-named conv tensors are a LayerNorm.** This config sets
  `conv_norm_type=layer_norm`, so the conv module is the EOU-style
  dw-conv → LayerNorm → SiLU path despite the `batch_norm` key names.
- **LSTM gate order.** RNNT prednet weights are PyTorch `ifgo`; ONNX-style
  kernels expect `iofc` — reorder at extraction time.
- **Mel log guard is 2^-24** (NeMo default), not 1e-10; the wrong guard shifts
  the mel floor in silence and breaks parity.

Extractor: fluidaudio-web `scripts/extract-voicechat-stt.py` (fp16 encoder
1.22 GB + fp32 decoder 36 MB); smoke test `scripts/ci-smoke-voicechat.mjs`.
WASM runs ≈ 1.0× real-time; the WebGPU path has not been exercised in-browser yet.

## Published benchmarks (evaluation targets for the port)

NVIDIA model card + [Artificial Analysis](https://artificialanalysis.ai/articles/nemotron-3-voicechat-leader-speech-pareto):

| Benchmark | VoiceChat score | Context |
|---|---|---|
| VoiceBench | #2 open full-duplex | no absolute score published |
| FDB 1.0 smooth turn-taking | TOR 0.82, **448 ms latency** | our per-frame budget must preserve this |
| FDB 1.0 user interruption | TOR 1.0, 480 ms, GPT-4o 4.33/5 | barge-in quality |
| FDB pause handling | TOR 0.153 synth / 0.255 Candor | lower = better |
| Conversational Dynamics (FDB, AA) | 77.8% | PersonaPlex 91.0 > **VC 77.8** > FLM-Audio 62 > Moshi 61 > Freeze-Omni 58.7 |
| Big Bench Audio (speech reasoning, AA) | 29.2% | Freeze-Omni 33.9; proprietary 87–96; VC = only open model top-3 on BOTH axes |
| AU-Harness BFCL-v3 (spoken tool calling) | 56.1% avg | Simple 58.5 / Multiple 62.5 / Parallel 42.5 / Irrelevance 89.6 |
| FDB-v3 (multi-step tool use) | Tool selection 82.5%, args 44.2%, Pass@1 33% | |

Runnable harnesses for the CoreML port (once TTS+codec+loop exist):
[VoiceBench](https://github.com/MatthewCYM/VoiceBench) (LLM-judged QA subsets),
[Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) 1.0/v3 (turn-taking,
interruption, tool use — needs the interactive loop),
[AU-Harness](https://github.com/ServiceNow/AU-Harness) (spoken BFCL),
Big Bench Audio (HF dataset, speech-reasoning QA). Port-acceptance criteria:
on-device outputs ≈ H100 reference on VoiceBench subset + turn-taking latency
≤ 448 ms + int8 text-channel parity (already 100% at prefill).

## Open questions

- CFG (guidance_enabled, scale 0.2): 2× TTS backbone evals per step. Check if
  `inference_guidance_enabled: false` degrades quality acceptably, else batch the
  uncond/cond pair through one ANE call.
- MoG head sampling (`inference_top_p_or_k 0.95`, noise 0.001) — host-side sampling
  from CoreML-emitted mixture params, same pattern as neutts sampling head.
- (fallback path only) Nemotron Mamba2 state handling in MLX single-frame stepping: `use_cache_for_nemotron: false`
  in the NeMo config — read `duplex_stt_model.forward` cache path carefully; full-duplex
  sessions run minutes long, sliding-window attn (if any) + Mamba state should be O(1) memory.
