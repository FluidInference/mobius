# LocalVQE → Core ML

Streaming Core ML export of [LocalVQE](https://github.com/localai-org/LocalVQE)
(Apache-2.0): neural acoustic echo cancellation + noise suppression +
dereverberation for 16 kHz speech, a CPU-tuned derivative of DeepVQE
(Indenbom et al., Interspeech 2023). Requested in
[FluidAudio#49](https://github.com/FluidInference/FluidAudio/issues/49#issuecomment-5719663475).

Published weights: `FluidInference/localvqe-coreml` (see [Outputs](#outputs)).
Swift consumer: `LocalVqeManager` / `LocalVqeStream` in FluidAudio
(`Sources/FluidAudio/Enhancement/LocalVQE`).

## What was converted

| Upstream checkpoint | Params | Exports |
|---|---:|---|
| `localvqe-v1.3-4.8M.pt` (current) | 4.8 M | `localvqe-v1.3-4.8M-16ms.mlmodelc`, `localvqe-v1.3-4.8M-256ms.mlmodelc` |
| `localvqe-v1.2-1.3M.pt` | 1.3 M | `localvqe-v1.2-1.3M-16ms.mlmodelc`, `localvqe-v1.2-1.3M-256ms.mlmodelc` |

Not converted: `v1.4-AEC` (GGUF-only, a C++ adaptive-filter front-end + tiny
mask with no PyTorch reference) and the GTCRN "pi" line (same DAF front-end
dependency). Both would need the DSP front-end ported by hand.

## Streaming formulation

The upstream module is fully causal but written as a whole-clip graph (causal
zero padding on every time-axis conv, a Python-loop S4D recurrence, and a
`fold`-based overlap-add). `src/localvqe_coreml/streaming.py` re-expresses it
as a pure function

    (mic_hop, ref_hop, *states) -> (enhanced_hop, *new_states)

that processes `frames` consecutive 256-sample hops per call and carries all
causal context as 33 explicit `in_*` / `out_*` tensors:

- 3 past frames per time-axis conv (7 encoder blocks × 2 convs, 5 decoder
  blocks × 2 convs, using the post-norm input since `CausalGroupNorm` is
  per-frame);
- AlignBlock: 63 past frames of `K` and of `x_ref` (the `dmax=64` unfold) and
  4 past frames of the similarity map for the (5,3) smoothing conv;
- S4D bottleneck: complex hidden state (`h_real`, `h_imag`);
- CCM: 2 past frames of the input spectrum;
- DCT analysis: 256 samples of PCM history per input; DCT synthesis: the
  256-sample overlap-add tail.

Feeding outputs back as the next call's inputs reproduces the whole-clip
forward to 4e-7 (fp32), for `frames=1` and `frames>1` alike.

Two conventions had to be pinned against the shipped GGML engine, not the
PyTorch reference:

- **Output level.** `DCTDecoder` divides the overlap-add by the overlap count
  (2). The GGML engine does not (`scale = 1.0` for `localvqe.version >= 2`),
  so its output is exactly 2× the PyTorch reference. The Core ML export uses
  the GGML level (`ola_scale=1.0`) — that is what the OBS plugin, the HF demo
  and the user who filed the request have been listening to.
- **Timing.** Frame `k` covers samples `[256k-256, 256k+256)`, so the hop
  emitted after consuming hop `k` is input hop `k-1`: output lags input by
  256 samples (16 ms). The GGML streaming API emits that delayed stream as-is
  (its first hop is the `t<0` region); FluidAudio's `LocalVqeStream` drops the
  first hop and flushes one hop of zeros at the end so whole-clip output is
  sample-aligned with the input.

`fold_temperature()` (the trained AlignBlock softmax temperature, 0.145 for
v1.3 / 0.10 for v1.2) is baked into the smoothing conv at load time, as the
upstream README requires.

## Precision

fp32 only. fp16 breaks parity: the 1e-12 epsilons in the power-law front-end
underflow and the S4D recurrence (|A| up to 0.99) accumulates error.

| Export | Compute units | SNR vs PyTorch fp32 |
|---|---|---:|
| fp32 | CPU / GPU / CPU+ANE | 101.8 dB (max abs diff 1e-5) |
| fp16 | CPU_ONLY | 5.0 dB |
| fp16 | CPU_AND_NE | 32.6 dB |

ANE is not a win for this graph even ignoring precision: the fp16 256 ms
export ran 3.2 ms/call on CPU+NE vs 7.3 ms fp32 on CPU in Python, but at
32.6 dB that is not the same audio. Left as a follow-up (selective fp16 with
the front-end / bottleneck kept fp32).

## Parity

Upstream double-talk demo clip (`examples/dt_mic.wav` / `dt_ref.wav` from
`LocalAI-io/LocalVQE-demo`, 10 s):

| Comparison | max abs diff | SNR |
|---|---:|---:|
| Core ML fp32 (Python) vs PyTorch ×2 | 1.0e-5 | 101.8 dB |
| Swift `LocalVqeStream` vs PyTorch ×2 (both as 16-bit WAV) | 3.8e-5 | 74.0 dB |
| Swift `LocalVqeStream` vs upstream GGML CLI, shifted 256 samples | 2.8e-5 | 79.7 dB |

Upstream regression fixture (`ggml/tests/fixtures/regression_input.f32` →
`localvqe-v1.3-4.8M-f32.out.f32`): streaming PyTorch × ola_scale 1.0 matches
the GGML fixture to 2.3e-6 after the 256-sample shift.

## Speed

Apple M5 Pro, FluidAudio release build, `fluidaudiocli enhance --streaming`
on the 10 s demo clip, per-call latency of the Core ML prediction incl. state
hand-off. RTFx is audio-per-call ÷ p50 latency.

| Model | Chunk | Compute | p50 | p99 | RTFx |
|---|---|---|---:|---:|---:|
| v1.3 4.8M | 256 ms | CPU | 7.12 ms | 14.1 ms | 36× |
| v1.3 4.8M | 256 ms | GPU | 6.16 ms | 118 ms (first-call JIT) | 42× |
| v1.3 4.8M | 16 ms | CPU | 1.16 ms | 2.30 ms | 14× |
| v1.2 1.3M | 256 ms | CPU | 4.23 ms | 8.09 ms | 60× |
| v1.2 1.3M | 16 ms | CPU | 0.67 ms | 0.91 ms | 24× |

Whole-clip wall RTFx (v1.3, 256 ms, CPU): 35×. The 16 ms export is bound by
per-prediction overhead (35 tensors in, 34 out), not compute — CPU is the
right default; GPU only helps the 256 ms chunk and pays a ~110 ms first-call
compile.

## Usage

```bash
uv sync
# checkpoints: https://huggingface.co/LocalAI-io/LocalVQE (.pt files)
uv run python convert-coreml.py --ckpt localvqe-v1.3-4.8M.pt --frames 1 16 --output-dir build
uv run python convert-coreml.py --ckpt localvqe-v1.2-1.3M.pt --frames 1 16 --output-dir build

# parity + timing vs the PyTorch reference
uv run python verify-coreml.py --ckpt localvqe-v1.3-4.8M.pt \
    --model build/localvqe-v1.3-4.8M-256ms.mlpackage --mic dt_mic.wav --ref dt_ref.wav

# PyTorch-only checks (OLA scale, streaming == whole-clip, GGML fixture)
uv run python verify_torch.py --ckpt localvqe-v1.3-4.8M.pt --upstream /path/to/LocalVQE
```

Model I/O (all Float32): `mic`, `ref` `[1, 256·frames]`; `enhanced`
`[1, 256·frames]`; `in_<state>` / `out_<state>` pairs (33). Metadata keys:
`frames_per_call`, `samples_per_call`, `output_delay_samples`, `ola_scale`,
`state_names`.

## Layout

```
convert-coreml.py            checkpoint -> streaming .mlpackage + .mlmodelc
verify-coreml.py             Core ML vs PyTorch parity + per-call timing
verify_torch.py              PyTorch-only sanity (OLA scale, streaming, GGML fixture)
src/localvqe_coreml/
  streaming.py               explicit-state wrapper (the conversion)
  common.py                  checkpoint loader (model_config + fold_temperature)
  upstream/                  vendored pytorch/localvqe from upstream (Apache-2.0)
LICENSE-upstream, CITATION.cff
```
