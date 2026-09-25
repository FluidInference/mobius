---
license: apache-2.0
base_model: LocalAI-io/LocalVQE
tags:
  - coreml
  - audio
  - speech-enhancement
  - acoustic-echo-cancellation
  - noise-suppression
  - fluidaudio
library_name: fluidaudio
---

# LocalVQE Core ML

Streaming Core ML exports of [LocalVQE](https://github.com/localai-org/LocalVQE)
(weights: [LocalAI-io/LocalVQE](https://huggingface.co/LocalAI-io/LocalVQE),
Apache-2.0): neural acoustic echo cancellation + noise suppression +
dereverberation for 16 kHz speech, a CPU-tuned derivative of DeepVQE
(Indenbom et al., Interspeech 2023).

Consumed by [FluidAudio](https://github.com/FluidInference/FluidAudio)
(`LocalVqeManager` / `LocalVqeStream`); conversion code in
[FluidInference/mobius](https://github.com/FluidInference/mobius)
`models/enhancement/localvqe/coreml`.

## Files

| File | Checkpoint | Params | Samples per call |
|---|---|---:|---:|
| `localvqe-v1.3-4.8M-256ms.mlmodelc` | `localvqe-v1.3-4.8M.pt` | 4.8 M | 4096 (16 hops) |
| `localvqe-v1.3-4.8M-16ms.mlmodelc` | `localvqe-v1.3-4.8M.pt` | 4.8 M | 256 (1 hop) |
| `localvqe-v1.2-1.3M-256ms.mlmodelc` | `localvqe-v1.2-1.3M.pt` | 1.3 M | 4096 (16 hops) |
| `localvqe-v1.2-1.3M-16ms.mlmodelc` | `localvqe-v1.2-1.3M.pt` | 1.3 M | 256 (1 hop) |

All fp32, iOS 17 / macOS 14 minimum deployment target. The two chunk sizes
produce identical audio; they trade per-call overhead against latency.

## Model I/O

Inputs (Float32): `mic` `[1, N]`, `ref` `[1, N]` (far-end reference — what the
loudspeaker played), and 33 `in_<state>` tensors. Outputs: `enhanced` `[1, N]`
and the matching `out_<state>` tensors. Start with all states zero and feed
each call's `out_*` back as the next call's `in_*`.

The enhanced hop lags the input by 256 samples (16 ms): after consuming input
hop *k* the model emits input hop *k−1*. The first emitted hop covers t < 0
and can be dropped; feed one hop of zeros at the end to drain. Output level
matches the upstream GGML engine (2× the upstream PyTorch reference's
overlap-add convention).

## Parity / speed

Swift output vs the upstream GGML CLI on the upstream double-talk demo clip:
2.8e-5 max abs diff, 80 dB SNR. Apple M5 Pro, per-call p50 on CPU: v1.3 256 ms
7.1 ms (36× RT), v1.3 16 ms 1.2 ms (14× RT), v1.2 256 ms 4.2 ms (60× RT),
v1.2 16 ms 0.7 ms (24× RT).

## Citation

Cite the upstream repository (`CITATION.cff` in
[localai-org/LocalVQE](https://github.com/localai-org/LocalVQE)) and the
DeepVQE paper it derives from (Indenbom et al., Interspeech 2023,
[arXiv:2306.03177](https://arxiv.org/abs/2306.03177)).
