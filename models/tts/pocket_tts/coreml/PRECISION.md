# PocketTTS Precision and Quantization

How the four PocketTTS CoreML artifacts are quantized for the
`FluidInference/pocket-tts-coreml` HF builds, why specific layers are
deliberately left at full precision, and why this mixed-precision recipe
preserves speech quality.

References:
- Upstream implementation: [kyutai-labs/pocket-tts#147](https://github.com/kyutai-labs/pocket-tts/pull/147)
  ("Add int8 quantization for FlowLM transformer linears")
- Motivating discussion: [kyutai-labs/pocket-tts#7](https://github.com/kyutai-labs/pocket-tts/issues/7)
  ("Lower-precision deployment of PocketTTS")

---

## TL;DR

Per HF language pack we ship five `.mlmodelc` artifacts:

| Artifact                | Precision        | Notes                                       |
|-------------------------|------------------|---------------------------------------------|
| `cond_step.mlmodelc`    | fp16             | KV-cache prefill, runs ~141 times per chunk |
| `flow_decoder.mlmodelc` | fp16             | Flow-matching velocity field, 8 Euler steps |
| `flowlm_step.mlmodelc`  | fp16             | Default autoregressive step (legacy)        |
| `flowlm_stepv2.mlmodelc`| **selective int8** | Drop-in replacement for `flowlm_step`     |
| `mimi_decoder.mlmodelc` | fp16             | Audio synthesis (Mimi VAE decoder, conv-heavy) |

> **v2.1 note:** the optimized v2.1 packs add `flow_decoder_fused` (fp16, the
> 8-step LSD loop unrolled — the only model that reaches the ANE) and
> `cond_prefill` (fp16, one-shot conditioning). Same weights, re-converted for
> speed. The fastest *flowlm* on-device is the existing int8 `flowlm_stepv2`
> (3.16 ms @ cpuAndGpu); 4-bit palettization shrinks size further but is slower
> (dequant overhead > bandwidth saved) — int4 is a size lever, not a speed one.

Only `flowlm_stepv2.mlmodelc` is int8-quantized. `cond_step`, `flow_decoder`,
`mimi_decoder`, and the legacy `flowlm_step` are pure fp16. The Swift loader
defaults to `flowlm_step.mlmodelc`; callers opt into the int8 variant by
selecting `flowlm_stepv2`.

---

## Why selective and not blanket

PocketTTS is a flow-matching language model. Per upstream issue #7, the
authors flagged the architecture as "sensitive to numerical precision —
the iterative LSD denoiser amplifies quantization error step over step,"
and an early naive dynamic-int8 attempt produced SNR < 0 dB output. The
recipe that landed in PR #147 quantizes only the **transformer body**
linears and leaves everything else alone:

| Layer category                            | Decision   | Why                                                                 |
|-------------------------------------------|------------|---------------------------------------------------------------------|
| `attn{i}_in_proj` (3072×1024)             | **int8**   | Large GEMMs dominate compute and weight memory                      |
| `attn{i}_out_proj` (1024×1024)            | **int8**   | Same                                                                |
| `linear{i}_1` (FFN expand, 4096×1024)     | **int8**   | Largest tensor in the model                                         |
| `linear{i}_2` (FFN contract, 1024×4096)   | **int8**   | Same                                                                |
| `input_linear` (32×1024)                  | fp32 → fp16 | 32-D latent → 1024-D embedding; tiny, but it gates every step       |
| `out_eos` (1024×1, sigmoid head)          | fp32 → fp16 | Termination decision; collapse here breaks EOS detection (see below)|
| LayerNorms (`norm{i}_1`, `norm{i}_2`, `out_norm`) | fp16 | Norm scale/bias are tiny; quantizing destroys distribution shape    |
| `cond_step` linears                       | fp16       | Voice/text prefill; runs once per chunk, not on the hot path        |
| `flow_decoder` MLP-AdaLN                  | fp16       | Iterative ODE solver — quantization error compounds over 8 steps    |
| `mimi_decoder` (conv VAE)                 | fp16       | Conv kernels, not matmuls; less benefit, more sensitivity           |

This is the same selection upstream's
`pocket_tts.quantization.apply_dynamic_int8(flow_lm, {"attention", "ffn"})`
makes. We only diverge in the **backend** (see next section).

---

## Method: weight-only PTQ via `coremltools.optimize.torch`

Upstream uses `torch.ao.quantization.quantize_dynamic` (true dynamic int8:
int8 weights, int8-quantized activations at runtime, int8 GEMM via FBGEMM
or torchao). `ct.convert` cannot ingest the resulting
`quantized::linear_dynamic` ops, so we substitute a CoreML-native
**weight-only PTQ** with the same module selection:

```python
# convert_models/convert/convert_flowlm_step.py
from coremltools.optimize.torch.quantization import (
    PostTrainingQuantizer,
    PostTrainingQuantizerConfig,
    ModulePostTrainingQuantizerConfig,
)

body_linear_names = []
for i in range(num_layers):
    body_linear_names.extend([
        f"attn{i}_in_proj",
        f"attn{i}_out_proj",
        f"linear{i}_1",
        f"linear{i}_2",
    ])

body_cfg = ModulePostTrainingQuantizerConfig(
    weight_dtype="int8",
    granularity="per_channel",
    quantization_scheme="symmetric",
)
ptq_config = PostTrainingQuantizerConfig(
    global_config=None,                  # ← do not quantize anything by default
    module_name_configs={name: body_cfg for name in body_linear_names},
)
quantizer = PostTrainingQuantizer(step_model, ptq_config)
quantized_model = quantizer.compress(inplace=False)
```

What this produces in the CoreML graph:

- Each named linear's `weight` tensor is stored as int8 with a per-output-
  channel fp16 scale.
- At inference, the Apple Silicon runtime dequantizes the weight to fp16
  on the fly (compute-time dequant), runs the existing fp16 matmul kernel,
  and discards the int8 buffer. **Activations stay fp16 throughout.**
- `global_config=None` is load-bearing: it means "do not quantize anything
  unless explicitly named." Drop it and the converter will silently
  quantize `out_eos`, `input_linear`, and the LayerNorm parameters too,
  reproducing the SNR < 0 dB failure that issue #7 documents.

This is **not** the same scheme as upstream's `quantize_dynamic`. The two
have the same memory footprint (8-bit weights) but different compute
paths:

|                       | Upstream `torch.ao` dynamic | Ours (CoreML weight-only PTQ) |
|-----------------------|------------------------------|-------------------------------|
| Weight storage        | int8                         | int8                          |
| Activation precision  | int8 (quantized at runtime)  | fp16                          |
| GEMM kernel           | int8 × int8 → int32          | fp16 × fp16 → fp16            |
| Backend               | FBGEMM (x86) / torchao (ARM) | Apple Silicon ANE/GPU fp16    |
| Speedup source        | int8 GEMM throughput         | weight bandwidth (DRAM→SRAM)  |

We pay the dequant cost on every step but never lose precision in the
activations or accumulator. On Apple Silicon the relevant win is **memory
bandwidth**, not GEMM throughput — the ANE has no exposed int8 GEMM kernel,
its MAC array is fp16-native.

---

## Why the excluded layers must stay fp16

### `out_eos` (1024 → 1 sigmoid head)

This is the EOS termination logit. It has output dimension 1, so
"per-channel symmetric int8" degenerates to **per-tensor** int8: a single
scale spanning all 1024 input weights. EOS is a small-margin scalar
decision (the synthesizer triggers stop when `eos_logit > -4.0`), and
collapsing per-channel granularity wipes out the decision margin.
Empirically, quantizing `out_eos` produces a model that either never
emits EOS (audio runs to `max_frames`, giving truncated cut-offs at
chunk boundaries) or emits it constantly (single-frame outputs). This
matches the early-int8 failure mode in issue #7.

### `input_linear` (32 → 1024)

The 32-dim flow latent is fed back through this linear on every step
of the autoregressive loop. Errors here compound over the entire
generation. The matrix is also tiny (32 KB), so quantizing it saves no
meaningful memory.

### LayerNorms

LayerNorm's `weight` and `bias` are per-feature scaling vectors with
small dynamic range. Per-channel int8 scaling them re-quantizes
activations that have already been carefully normalized to unit
variance, which is precisely what LayerNorm is meant to prevent.

### `flow_decoder` and `mimi_decoder`

The flow decoder runs 8 Euler integration steps per generated audio
frame; quantization noise added per step turns into a divergent integral.
The Mimi decoder is a convolutional VAE — conv kernels have less weight
mass than the transformer FFN linears (lower bandwidth payoff) and
upsampling layers are noticeably more sensitive to quantization than
matmul layers.

### `cond_step`

`cond_step` handles voice + text prefill. It runs ~141 times per chunk
but only **once per chunk**, not per audio frame. The bandwidth payoff is
swamped by the per-frame cost of `flowlm_step` running 20–50× per chunk.
Not worth the complexity or the risk of corrupting voice conditioning.

---

## Why this works

Three independent reasons:

1. **The hot tensors really are the FFN/attention linears.** A 24-layer
   pack with 8-bit FFN/attn weights drops from 1.1 GB to 291 MB
   (~74% reduction) without touching anything sensitive. 6-layer goes
   from 145 MB to 74 MB (~49%). The weights we leave at fp16 are a
   rounding error on disk.

2. **Per-channel granularity matches the geometry.** The body linears
   have output dimension ≥ 1024, so per-channel scales preserve enough
   dynamic range for each row to keep its independent magnitude. The
   layers we exclude either have output dim 1 (`out_eos`) — where
   per-channel collapses to per-tensor — or are tiny enough that
   quantization adds noise without saving memory.

3. **The body computation is robust to fp16 dequant.** Softmax in the
   attention layer is shift-invariant: a small additive bias from
   weight noise becomes a multiplicative constant after `exp`, which
   the softmax denominator divides out. Residual connections shorten
   the effective depth that error has to traverse. LayerNorm at every
   block re-centers the signal. The flow-decoder and EOS heads — the
   parts of the pipeline that *are* sensitive — are exactly the parts
   we leave at fp16.

Upstream measures a WER delta of **−0.022 ± 0.032** versus the fp16
baseline on their internal eval (PR #147 description), i.e.
indistinguishable from noise. Our CoreML port uses the same module
selection, so we expect the same quality envelope; informal A/B
listening on each of the 10 language packs confirms no audible
degradation.

---

## Build and ship

`convert_flowlm_step.py` accepts `--int8`. With the flag, the script
applies the wrapper above pre-trace, traces the quantized module, and
saves the result as `flowlm_stepv2.mlpackage` (so it sits next to the
fp16 `flowlm_step.mlpackage` rather than overwriting it). The orchestrator
in `convert_all_languages.sh` runs both passes per language; both
artifacts end up under `build/<language>/` and get uploaded to
`FluidInference/pocket-tts-coreml/v2/<language>/`.

Swift consumers select the variant by name via the model store. Defaulting
to `flowlm_step` keeps existing English deployments byte-identical to the
pre-quantization shipment.
