# LocalVQE → Core ML

Streaming Core ML export of [LocalVQE](https://github.com/localai-org/LocalVQE)
(Apache-2.0): neural acoustic echo cancellation + noise suppression +
dereverberation for 16 kHz speech, a CPU-tuned derivative of DeepVQE
(Indenbom et al., Interspeech 2023). Requested in
[FluidAudio#49](https://github.com/FluidInference/FluidAudio/issues/49#issuecomment-5719663475).

Published weights: `FluidInference/localvqe-coreml` (see [Outputs](#outputs)).
Swift consumer: `LocalVqeManager` / `LocalVqeStream` in
[FluidAudio #930](https://github.com/FluidInference/FluidAudio/pull/930).
The consumer serializes each stream's push/flush/reset operations and handles
queued and active cancellation. It remains beta pending live call-pipeline
validation on target devices; offline model parity does not certify that integration.

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

## Quality: ICASSP 2022 AEC-Challenge blind test set

**Summary: port fidelity validated; published benchmark substantially
reproduced, with unresolved v1.2 far-end differences.**

The upstream README / HF model-card table is AECMOS on the 800-clip ICASSP
2022 blind set (real recordings; mirror `richiejp/aec-challenge-16k`,
`blind_test_set_icassp2022/`). `render-blind.sh` renders every mic/lpb pair
through either engine; `score_blind.py` scores AECMOS echo / degradation MOS,
blind ERLE (plain energy ratio and the technical-report gated definition) and
DNSMOS OVRL; `compare-renders.py` does an aligned A/B of two render dirs.
Two evaluation protocols are kept separately labelled because they answer
different questions:

- **challenge** (our reference): Microsoft's current scenario-aware AECMOS
  model (`Run_1663915512_Stage_0`) with the AECMOS README trimming rules
  (far-end single talk → last half, double talk → last (len−15)/2 s, near-end
  → whole clip), i.e. the convergence portions are excluded as the challenge
  instructs. DNSMOS on the same rated segment.
- **upstream** (HF table reproduction check): the legacy AECMOS model
  (`Run_1663829550_Stage_0`, no scenario marker) over the whole clip, which
  its 20 s cap makes the first 20 s. This reproduces the card's unprocessed
  baseline to the displayed digit (2.67 / 2.56 / 1.90 / 2.13 / 5.00), and the
  same first-20-s implementation is in the author's public predecessor
  evaluation code (deepvqe-ggml `train/src/metrics.py`). The card's OVRL is
  DNSMOS on the rated segment and its ERLE is the technical-report gated
  definition; both definitions are compared below.

Saved per-recording results, metric availability and recomputed aggregates are
published in the [validation index](validation/benchmark-index.json), alongside
the [800-recording manifest](validation/blind-manifest.txt). The stored upstream
runs skipped DNSMOS. Rated DNSMOS measurements are available in the separate
challenge files. The fixed scorer computes that same rated segment independently
of AECMOS protocol, but these saved upstream files must not be represented as
full DNSMOS runs.

### Challenge protocol (reference)

Core ML (Swift `fluidaudiocli enhance`, float32 WAV output). ERLE is the
gated definition; OVRL is DNSMOS on the rated segment.

| Scenario | n | unproc. echo | v1.3 echo / deg / ERLE / OVRL | v1.2 echo / deg / ERLE / OVRL |
|---|--:|--:|---|---|
| doubletalk | 115 | 2.17 | 4.35 / 3.93 / 6.3 dB / 2.89 | 4.20 / 3.63 / 6.2 dB / 2.77 |
| doubletalk-with-movement | 185 | 2.21 | 4.35 / 3.86 / 6.1 dB / 2.84 | 4.13 / 3.57 / 6.0 dB / 2.73 |
| farend-singletalk | 107 | 1.95 | 2.49 / 5.00 / 54.2 dB / 1.95 | 3.92 / 5.00 / 53.2 dB / 1.89 |
| farend-singletalk-with-movement | 193 | 2.23 | 3.08 / 5.00 / 55.9 dB / 1.96 | 4.13 / 5.00 / 47.3 dB / 1.80 |
| nearend-singletalk | 200 | 5.00 | 4.99 / 4.14 / 2.3 dB / 3.17 | 4.99 / 4.09 / 2.1 dB / 3.17 |

### Upstream protocol (HF table reproduction)

AECMOS echo / deg over the first 20 s; ERLE is the gated definition over the
whole clip; OVRL is DNSMOS on the challenge-rated segment (the segment the
card's OVRL matches — `--dnsmos-region rated` selects it under either
protocol). Core ML and GGML rows are kept separate — GGML is the upstream CLI's raw output (256-sample
delay, 16-bit PCM), Core ML is aligned float32.

| Scenario | HF card v1.3 (echo / deg / ERLE / OVRL) | Core ML v1.3 | GGML v1.3 |
|---|---|---|---|
| doubletalk | 4.73 / 2.62 / 8.5 / 2.89 | 4.73 / 2.62 / 8.5 / 2.89 | 4.73 / 2.47 / 8.2 / — |
| doubletalk-with-movement | 4.67 / 2.43 / 8.3 / 2.85 | 4.66 / 2.44 / 8.2 / 2.84 | 4.68 / 2.34 / 7.8 / — |
| farend-singletalk | 3.69 / 4.83 / 50.9 / 1.94 | 3.54 / 4.82 / 50.1 / 1.95 | 3.66 / 4.82 / 50.9 / — |
| farend-singletalk-with-movement | 3.88 / 4.98 / 49.9 / 1.96 | 3.75 / 4.96 / 49.6 / 1.96 | 3.84 / 4.95 / 49.4 / — |
| nearend-singletalk | 5.00 / 4.18 / 2.4 / 3.17 | 5.00 / 4.18 / 2.3 / 3.17 | 5.00 / 4.12 / 1.0 / — |

| Scenario | HF card v1.2 (echo / deg / ERLE / OVRL) | Core ML v1.2 | GGML v1.2 |
|---|---|---|---|
| doubletalk | 4.72 / 2.37 / 8.4 / 2.83 | 4.72 / 2.39 / 8.5 / 2.77 | 4.71 / 2.24 / 8.1 / — |
| doubletalk-with-movement | 4.65 / 2.30 / 8.1 / 2.79 | 4.64 / 2.31 / 8.1 / 2.73 | 4.66 / 2.19 / 7.6 / — |
| farend-singletalk | 3.78 / 4.91 / 45.7 / 1.80 | 4.07 / 4.93 / 47.6 / 1.89 | 4.14 / 4.92 / 48.0 / — |
| farend-singletalk-with-movement | 4.12 / 4.96 / 40.6 / 1.75 | 4.27 / 4.96 / 41.3 / 1.80 | 4.32 / 4.97 / 41.1 / — |
| nearend-singletalk | 5.00 / 4.16 / 2.1 / 3.17 | 5.00 / 4.17 / 2.1 / 3.17 | 5.00 / 4.09 / 1.1 / — |

Unprocessed baseline under this protocol: 2.67 / 2.56 / 1.90 / 2.13 / 5.00
echo MOS, identical to the card.

**What is and is not reproduced.** Unprocessed baseline: exact.
v1.3 (Core ML vs card): echo MOS within 0.01 on double-talk and near-end and
0.15 low on far-end (3.54 / 3.75 vs 3.69 / 3.88; the raw GGML output, with
its delay and 16-bit quantisation of the near-silent residual, scores within
0.04); deg within 0.02; gated ERLE within 0.8 dB (50.1 / 49.6 vs
50.9 / 49.9); OVRL within 0.01. v1.2 (Core ML vs card): double-talk and
near-end within 0.02 echo, 0.02 deg, 0.1 dB ERLE and 0.06 OVRL; **far-end is
not reproduced on any metric** — echo +0.29 / +0.15 (4.07 / 4.27 vs
3.78 / 4.12), gated ERLE +1.9 / +0.7 dB (47.6 / 41.3 vs 45.7 / 40.6; GGML
+2.3 / +0.5 dB), OVRL +0.09 / +0.05 (1.89 / 1.80 vs 1.80 / 1.75). Our values
are above the published ones from either runtime, so this is not an
output-format effect; it is not evidence that the port outperforms upstream,
and +0.29 echo MOS is not rounding noise. The
earlier statement in this README that the v1.3 far-end row "was not produced
from the published v1.3 weights" is retracted: it was a scorer / segment
protocol mismatch. The v1.2 far-end gap is unexplained; the private LocalVQE
scoring script is not public, so exact reproduction of every cell is not
established.

**v1.2 far-end investigation.** Hypotheses tested to explain the v1.2
far-end cells, all rendered through the PyTorch reference over the 800 clips
(`render-torch.py`) and scored under the upstream protocol:

| Render | FE-ST echo / deg / gERLE | FE-ST-mov echo / deg / gERLE | DT echo / deg |
|---|---|---|---|
| HF card v1.2 | 3.78 / 4.91 / 45.7 | 4.12 / 4.96 / 40.6 | 4.72 / 2.37 |
| published weights as shipped (dmax 64) | 4.07 / 4.93 / 47.6 | 4.27 / 4.96 / 41.3 | 4.72 / 2.39 |
| **dmax 32** (pre-v1.2 delay window) | 4.29 / 4.88 / **44.9** | 4.41 / 4.96 / **40.5** | 4.70 / 2.40 |
| dmax 32 + GGML CLI output emulated (delay, int16) | 4.32 / 4.87 / 43.8 | 4.43 / 4.96 / 39.8 | 4.70 / 2.24 |
| softmax temperature 1.0 (unfolded) | 2.79 / 4.77 / 13.4 | 3.02 / 4.91 / 12.9 | 3.81 / 2.60 |
| ReLU6 reference (arch_version 2), dmax 32 or 64 | 2.17 / 4.88 / −7 | 2.39 / 4.94 / −7 | 2.78 / 3.30 |

Scorer-side variations (both AECMOS models, with / without scenario marker,
whole / first-half / last-half / middle-20 s segments, on Core ML and raw
GGML renders) never reach 3.78 while keeping deg near 4.91. The delay window
is not stored in the checkpoint, and the reference config switched from
dmax 32 to 64 on the day the row was published (2026-05-14). Rendering at
dmax 32 brings far-end ERLE (44.9 / 40.5 vs 45.7 / 40.6 dB) and degradation
(4.88 / 4.96 vs 4.91 / 4.96) close to the card. This supports a historical
configuration hypothesis, but does not confirm which configuration upstream
used. Far-end echo MOS moves farther from the card (4.29 / 4.41 vs
3.78 / 4.12), and none of the tested settings reproduces the whole table.
Those echo cells remain unexplained. Upstream's evaluation configuration,
rendered audio or scoring script would help resolve the discrepancy.

A separate [historical-runtime and precision investigation](REPRODUCTION.md)
compares the published PT/GGUF tensors and isolates an upstream state-copy
defect. Applying only the later two-phase copy fix makes the historical
engine match current GGML exactly on four real-recording controls. The
original engine was also rendered and scored on all 300 far-end recordings;
its echo MOS (4.12 / 4.28) still does not reproduce the card. The linked report
includes per-recording results, manifests, failed precision trials and the
limits of these diagnostics. These are separate from the main 800-clip
benchmark and do not replace its quality reference.

**Port fidelity.** The upstream GGML CLI was run on the same 800 clips on the
same machine and both render sets scored with `compare-renders.py --shift-b
256 --quantize-a` (whole hops only, challenge protocol): per-scenario echo /
deg / ERLE means identical to two decimals; per-clip deltas — echo mean
+0.0002 (p95 0.017, max 0.09), deg p95 0.0004, ERLE p95 0.005 dB; aligned
waveform SNR median 84 dB (numerically equivalent within 16-bit quantisation,
not bit-identical). Normalised first, all artefacts of the upstream CLI: its
output is one hop (256 samples) late, its whole-clip mode zero-fills the
trailing partial hop and truncates some outputs, and its 16-bit writer
quantises the ~1e-4-RMS far-end residual and wraps samples above full scale
(1.02 → −0.98) on clips whose mic is already clipped — the Swift CLI writes
float32 and keeps them. Those wrapped clips are the only ones whose aligned
waveform SNR stays low (min 2.7 dB); their AECMOS scores still agree.

```bash
./render-blind.sh coreml blind renders/coreml-v1.3 /path/to/fluidaudiocli v1.3 4
./render-blind.sh ggml   blind renders/ggml-v1.3 /path/to/localvqe localvqe-v1.3-4.8M-f32.gguf 4
PY="uv run --no-project --python 3.12 --with librosa --with onnxruntime --with soundfile --with scipy python"
$PY score_blind.py --blind-dir blind --enh-dir unprocessed --aecmos-dir aecmos --protocol upstream --no-dnsmos
$PY score_blind.py --blind-dir blind --enh-dir renders/coreml-v1.3 --aecmos-dir aecmos --output scores.json
$PY compare-renders.py --blind-dir blind --a renders/coreml-v1.3 --b renders/ggml-v1.3 --shift-b 256 --quantize-a --aecmos-dir aecmos
```

(`aecmos/` holds the AECMOS and DNSMOS ONNX files from microsoft/AEC-Challenge
and microsoft/DNS-Challenge. The scorer runs under Python 3.12: scipy wheels
for the 3.10 project env fail to dlopen on macOS 27.)

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
render-blind.sh              render an AEC-Challenge blind set through the Swift or GGML CLI
score_blind.py               AECMOS / ERLE / DNSMOS scoring of blind-set renders
compare-renders.py           aligned, quantisation-matched A/B of two render dirs
render-torch.py              PyTorch-reference render of a blind set (diagnostics: --dmax, --no-fold, --arch-version)
src/localvqe_coreml/
  streaming.py               explicit-state wrapper (the conversion)
  common.py                  checkpoint loader (model_config + fold_temperature)
  upstream/                  vendored pytorch/localvqe from upstream (Apache-2.0)
LICENSE-upstream, CITATION.cff
```
