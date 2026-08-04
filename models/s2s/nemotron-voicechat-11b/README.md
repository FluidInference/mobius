# NVIDIA NemotronLabs VoiceChat 11B — FluidAudio Prep

Feasibility trial for running [nvidia/NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
(released 2026-08-03) on Apple Silicon. End-to-end full-duplex speech-to-speech
model: streaming speech understanding + speech generation + tool calling in one
unified architecture (~450 ms turn-taking latency, #2 open FD on VoiceBench,
first open FD model with tool calling). License: OpenMDW-1.1, research only.

Reference implementation: [NVIDIA-NeMo/Speech @ `nemotron-labs-voicechat`](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat),
model class `nemo/collections/speechlm2/models/nemotron_voicechat.py`.

## Component map (from safetensors header, all F32 = 44.4 GB)

| Component | Keys | Params | Apple target |
|---|---|---:|---|
| Fast Conformer encoder | `stt_model.perception.encoder.*` | 609 M | CoreML/ANE — same family as shipped `nemotron-speech-streaming-en-0.6b` (24L, d1024, 128 mel, 8x subsampling, cache-aware `chunked_limited` att_context **[70, 0]** = fully causal, 80 ms frames) |
| Perception proj | `stt_model.perception.proj.*` | 4.6 M | CoreML (1024 → 4480 into LLM embed space) |
| LLM backbone | `stt_model.llm.layers.*` | 7 714 M | **MLX** — Nemotron Nano v2 9B (NemotronH, hidden 4480; verified from keys: 56 layers = 27 Mamba2 + 25 MLP + 4 attention at indices 14/21/30/39 → tiny KV cache, O(1) Mamba state); mlx-lm supports the arch (mlx-community ships 4/6-bit quants of the base model) |
| Token embeddings | `stt_model.embed_tokens.weight` | 587 M | MLX (131072 × 4480) |
| Text head | `stt_model.lm_head.weight` | 587 M | MLX |
| Function head | `stt_model.function_head.weight` | 587 M | MLX (second lm_head on same hidden state → tool-call channel) |
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
| TTS backbone + MoG step | ~6 ms est. (×2 with CFG ≈ 12 ms) | proxy: magpie decoder_step host-cache rewrite measured 6.0 ms/step 100% ANE on this machine (similar-class decoder). Measure after Phase 2. |
| Codec decoder | unmeasured | 108 M convnet, expect low single-digit ms per 80 ms frame. (NeuCodec-fp16 batch decode was profiled as a candidate proxy — 366 ms ANE-off — but it is a far heavier vocoder doing whole-utterance decode; not representative.) |

**Frame total ≈ 45–55 ms of an 80 ms budget → real-time viable on M5 Pro (~1.5–1.8× headroom), ~9 GB resident.**
Base-model (not fine-tuned) MLX weights were benchmarked; fine-tuned weights are the same
architecture so speed carries over. 6-bit quant (~7 GB, est. ~28 ms/tok) still fits if 4-bit
hurts quality after fine-tune conversion.

**Conclusion: hybrid execution.** CoreML/ANE for encoder + TTS + codec + RNNT
(≈ 1.6 B params total, ~all reusable patterns from prior trials), MLX for the
9B backbone + 3 heads. Memory: 4-bit LLM ≈ 4.7 GB + fp16 rest ≈ 3.5 GB → ~8–9 GB.
**macOS-only target** (Apple Silicon ≥ 16 GB); iPhone is out of reach for v1.
FluidAudio currently has no MLX dependency — the LLM runner either becomes an
optional SPM product (`FluidAudioVoiceChat`) depending on `mlx-swift`, or the
whole thing ships as a separate example app. Decide before Swift work starts.

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
    CoreML encoder (fp16, 195 per-frame steps) → CoreML RNNT greedy →
    "Hello, do you know what color the sky is" — token-identical to the torch
    reference. Real-audio behavioral parity for the full user-transcription chain.
- [ ] **Phase 2b — TTS backbone + MoG head**: gemma3 28L×1152 stateful KV single-step,
  magpie/neutts decoder-step playbook applies (host-cache if needed)
- [ ] **Phase 2c — codec decoder (+ prvq dequant)**: conv stack, straightforward
- [ ] **Phase 3 — LLM to MLX**: remap `stt_model.llm.*`/`embed_tokens`/`lm_head` onto
      NemotronH HF layout → `mlx_lm.convert` (4-bit + 6-bit) → custom step runner that
      accepts `inputs_embeds` (fusion output) instead of token ids and returns hidden
      state for both heads. Function head = extra 4480×131072 matmul.
- [ ] **Phase 4 — Swift host**: streaming mel (NemotronMelExtractor reusable) →
      encoder step → fusion (tiny, host-side) → MLX step → TTS step → codec chunks;
      barge-in = feed user audio continuously, agent yields when text channel emits EOS.
      Tool-call channel surfaced as an API callback.
- [ ] **Phase 5 — publish**: HF `FluidInference/nemotron-voicechat-11b-coreml`
      (CoreML bundles + MLX quant) after confirming repo with Alex.

## Open questions

- CFG (guidance_enabled, scale 0.2): 2× TTS backbone evals per step. Check if
  `inference_guidance_enabled: false` degrades quality acceptably, else batch the
  uncond/cond pair through one ANE call.
- MoG head sampling (`inference_top_p_or_k 0.95`, noise 0.001) — host-side sampling
  from CoreML-emitted mixture params, same pattern as neutts sampling head.
- Nemotron Mamba2 state handling in MLX single-frame stepping: `use_cache_for_nemotron: false`
  in the NeMo config — read `duplex_stt_model.forward` cache path carefully; full-duplex
  sessions run minutes long, sliding-window attn (if any) + Mamba state should be O(1) memory.
