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

## Optimization pass (profiling, quant variants, fused model, MLX baseline)

Follow-up pass to find where decode time goes and whether CoreML can approach
bandwidth-optimal decode. Scripts: `profile_decode.py`, `quantize_variants.py`,
`convert_stateful_decoder.py --fused`.

Decoder-only latency (Q=1, short context, steady loop, CPU_AND_GPU):

| weights | size | ms/tok | effective BW |
|---|---|---|---|
| fp16 | 7.0GB | 58.8 | ~142GB/s |
| int8 per-channel | 3.5GB | 47.5 | 74GB/s |
| int4 per-block-32 | 2.0GB | 41.3 | 49GB/s |
| int4 per-channel | 1.8GB | 39.1 | 46GB/s |
| 4-bit palettized LUT | 1.8GB | 95.5 | (avoid — slowest) |

Findings, in causal order:

1. **RangeDim shape churn is a non-issue** — growing `end_step` per decode
   step costs nothing vs constant shape. No need for static-shape decode
   models or position-input scatter designs.
2. **CoreML quantized GEMV kernels hit a ~40ms floor** on this GPU: int4
   moves 4× less data than fp16 but only decodes 1.5× faster. Palettized LUT
   dequant is pathological (2.3× slower than fp16). fp16 runs at ~52% of
   chip bandwidth; int4 at ~18%.
3. **Alternating between two MLModels costs ~24ms/step** (decoder 41.1ms +
   head 1.5ms alone, but 66.8ms interleaved). Fixed by the `--fused` variant
   (34 layers + final norm + tied lm_head in one graph, last-position logits
   output): prefill 228 tok/s, one predict per token.
4. **Host-side gaps between predicts are nearly free** (busy-wait probe:
   +0.5→10ms gaps add only 0–3ms net), so a Swift host loop wouldn't beat
   the Python harness by much — the floor is in the kernels, not the host.
5. **The machine is bimodal under sustained GPU load**: identical benchmarks
   oscillate between ~41ms and ~68ms regimes (GPU clock management /
   thermals). Steady-state best ≈ 24 tok/s; observed sustained ≈ 15 tok/s.
6. **MLX 4-bit baseline on the same machine: 106.7 tok/s decode**
   (`out/mlx-4bit`, 4.5 bits/weight, 3.5GB peak memory, same correct
   translation output). MLX's quantized GEMV runs near bandwidth-optimal —
   ~4.4× faster than CoreML's best steady-state. CoreML wins only prefill
   (228 vs ~39 tok/s on this 36-token prompt, where MLX is setup-dominated).

Not pursued (and why): speculative decoding needs a draft model and host
machinery disproportionate to a probe; chunked multi-model pipelines add
handoffs (the thing that costs 24ms) without reducing sequential work; ANE
already shown strictly worse.

## Verdict

Converting works; running is "fine on Mac, marginal on iOS". The optimization
pass makes the conclusion quantitative: CoreML's quantized GEMV kernels cap
decode at ~24 tok/s while MLX hits 106.7 tok/s on the same silicon — a 4.4×
gap that no amount of model restructuring closes, because it lives inside the
kernels. For FluidAudio purposes a 4B translation LLM is an MLX workload on
macOS. CoreML remains the right tool for the encoder-heavy sub-1B models the
framework is built around, and its strong prefill suggests the hybrid worth
remembering: CoreML/ANE encoders + MLX decoder.
