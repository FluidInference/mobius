# Riva-Translate-4B-Instruct-v2 → CoreML feasibility probe

Probe conversion of [nvidia/Riva-Translate-4B-Instruct-v2](https://huggingface.co/nvidia/Riva-Translate-4B-Instruct-v2)
(text-to-text NMT, 37 languages) to CoreML, to answer: does a 4B Mistral-class
translation LLM run acceptably on-device via CoreML?

## Architecture

Standard `MistralForCausalLM` (pruned/distilled from Mistral-NeMo-12B):
34 layers, hidden 3072, 32 Q / 8 KV heads (GQA×4), head_dim 128,
intermediate 8640, vocab 131072 (Tekken), tied embeddings, 8k context,
~4.2B params.

## Conversion recipe

Same stateful-KV-cache pattern as `models/stt/qwen3-asr-0.6b/coreml/convert_stateful_decoder.py`:

- 34-layer decoder stack as ONE stateful model, 68 fp16 KV state buffers
  (`max_seq_len` 1024), `RangeDim` query length → single model serves both
  prefill and decode
- traced end-to-end in **fp16** (no fp32 master copy — converts on a 24GB host,
  peak well under RAM; conversion takes ~70s)
- `embed_tokens` kept host-side (`embed_tokens_fp16.npy`, 768MB, mmap lookup)
- lm_head = final RMSNorm + tied 3072×131072 projection, separate model
- RoPE cos/sin computed host-side and passed as inputs; additive causal mask input

Scripts: `convert_stateful_decoder.py` (convert), `quantize_int4.py`
(post-training linear int4, per-block 32), `run_reference.py` /
`run_coreml.py` (parity + latency harness).

## Results (M-series, 24GB, macOS 26)

| variant | decoder size | load | prefill (36 tok) | decode mean | greedy parity vs fp16 torch |
|---|---|---|---|---|---|
| fp16, CPU_AND_GPU | 7.0GB | 24.3s | 1.61s (22 tok/s) | 86.9ms/tok (11.5 tok/s, p50 63ms) | 14/14 tokens, logits corr 0.999999 |
| int4 pb32, CPU_AND_GPU | 2.0GB | 3.8s | 0.55s (65 tok/s) | 81.9ms/tok (12.2 tok/s, p50 70ms) | 0/14 (corr 0.959) — but output is a correct alternative translation |
| int4 pb32, CPU_AND_NE | 2.0GB | 56.7s | 7.06s (5.1 tok/s) | 209ms/tok (4.8 tok/s) | 0/14 (corr 0.785) — same correct text as int4 GPU |

- fp16 en→de output: "In San Francisco ist es für diese Jahreszeit ungewöhnlich warm."
- int4 en→de output: "Das Wetter in San Francisco ist für diese Jahreszeit ungewöhnlich warm."
  (greedy path flips to an equally valid phrasing; no degradation visible on this sample —
  proper eval would need COMET on FLORES)

## Findings

1. **Conversion itself is a non-issue.** fp16 tracing + stateful states convert
   cleanly in ~70s with coremltools 9.0; smoke + parity pass. The
   vocab-131072 lm_head converts fine as a plain matmul (no topk in-graph, so
   the known CoreML topk mod-131072 bug is not triggered).
2. **Decode is overhead-bound in this harness, not bandwidth-bound.**
   int4 shrank weights 3.5× and sped prefill 3×, but decode stayed ~80ms/tok.
   ~63ms p50 for a 2GB weight read ≈ 32GB/s effective — far below the chip's
   bandwidth. Per-step cost is dominated by predict dispatch (2 model calls
   per token from Python + RangeDim shape handling). A Swift host with
   preallocated buffers would likely land in the 20–40ms/tok range, but that
   is still 25–50 tok/s at best — MLX gets similar or better with far less
   machinery.
3. **Quality survives int4 per-block-32** on the sampled prompt; keep lm_head
   fp16 if logit fidelity matters (int4 head dropped first-step corr to 0.959).
4. **Memory**: int4 stack = 2.0GB decoder + 216MB head + 768MB embedding
   (embedding could be int8/int4'd too → ~200–400MB). Total ≈ 2.5GB + KV cache
   (68 × 1×8×1024×128 fp16 = 272MB at seq 1024). Fits Mac/iPad easily;
   iPhone Pro-class only with the extended-memory entitlement and little else
   resident — pairing it with an ASR stack on-device would be tight.
5. **ANE is strictly worse** (int4, CPU_AND_NE): 56.7s load (ANE compile),
   prefill 7.1s (4× slower than GPU), decode 209ms/tok (2.5× slower), and
   first-step logits corr drops to 0.785 — consistent with partial placement
   plus CPU-fallback sync overhead. Same outcome class as the Qwen3-0.6B
   LLM-on-ANE trial: autoregressive decode of this shape does not benefit
   from the ANE. Output text was still the correct translation.

## Verdict

Converting works; running is "fine on Mac, marginal on iOS". For FluidAudio
purposes a 4B translation LLM remains better served by MLX on macOS; the
CoreML route offers no advantage for GPU-bound autoregressive decode and the
ANE does not change the picture for this shape of model.
