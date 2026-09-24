# parakeet-ultra → Core ML

Conversion of [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) (full-precision
post-training of `nvidia/parakeet-tdt-0.6b-v3`, CC-BY-4.0) into the v3 component contract: `Preprocessor` /
`Encoder` / `Decoder` / `JointDecisionv3`, 15 s window. Published as `FluidInference/parakeet-ultra-coreml`; loaded in
Swift via `AsrModelVersion.ultra`.

## What the checkpoint is

* Same architecture, HF-transformers parameter names and tokenizer as parakeet-redux, but no ternary layers: 356
  fp16 + 349 fp32 tensors (+24 int64 BN counters). 651 of 723 non-preprocessor tensors differ from nvidia v3
  (encoder layers up to 15.7 % relative, joint up to 6 %), so every component is re-exported.
* Same extra `vad_head` as redux; not exported.

## Pipeline

Reuses the redux scripts (`../../parakeet-redux/coreml`), which load either checkpoint (no `ternary.json` → plain
load). Run with the v3 `uv` env.

```bash
PY=../../parakeet-tdt-v3-0.6b/coreml/.venv/bin/python
R=../../parakeet-redux/coreml
W=~/Documents/parakeet-ultra-work
$PY $R/redux_weights.py --hf-dir $W/hf                                  # strict load + diff vs nvidia v3
$PY $R/export_encoder_fp16.py --hf-dir $W/hf --work $W --name ultra     # iOS 18 fp16 reference (parity/latency A/B)
$PY $R/export_encoder_fp16.py --hf-dir $W/hf --work $W --name ultra --target ios17   # shipped: iOS 17 / macOS 14
$PY quantize_encoder_int8.py $W/encoder_fp16_ultra_ios17.mlpackage --out $W/encoder_int8_ultra_ios17.mlpackage
$PY $R/export_decoder_joint.py --hf-dir $W/hf --out $W/components --name ultra
$PY $R/validate_encoder.py $W/encoder_int8_ultra_ios17.mlmodelc --ref $W/encoder_ref_ultra_ios17.npz --units ane,gpu
```

`Preprocessor.mlmodelc` and the vocab files are the v3 copies.

## Encoder encoding

int8 linear, symmetric, per output channel (data-free) — the same encoding as the v3 `Encoder_v2.mlmodelc`
(FluidAudio #760), not the v3 6-bit LUT, which flips tokens on some windows.

| Encoder | Size | Placement (ANE) | cos vs torch (GPU / ANE) |
|---|---:|---|---|
| fp16 | 1187 MB | 1381 ANE, 4 CPU | 0.999999 / 0.99933 |
| int8 per-channel, iOS 17 (ships) | 595 MB | 1381 ANE, 4 CPU | 0.99594 / 0.99064 |

The int8 tensor cosine is lower than for stock v3 int8 (0.99957) even though the per-tensor weight quantization
error is identical to stock (median ratio 1.00 over 264 tensors): the post-trained activations are more sensitive.
It does not reach the transcripts — int8 and fp16 score the same WER (below).

## Results (M-series Mac, macOS 27, full corpora, corpus-level WER)

| | v3 | redux | ultra fp16 | **ultra int8** |
|---|---:|---:|---:|---:|
| LibriSpeech test-clean (2620) | 2.27 % | 2.67 % | 2.13 % | **2.12 %** |
| LibriSpeech test-other (2939) | 4.12 % | 5.15 % | 3.79 % | **3.79 %** |
| FLEURS mean, 24 langs × 100 | 14.81 % | 13.06 % | — | **11.67 %** |

Ultra int8 beats v3 on all 24 FLEURS languages (largest: lv −8.4, lt −7.4, sl −7.4, mt −6.9, fi −5.3) and the
direction agrees with the upstream card in 24/24. The ANE/GPU choice is WER-neutral (2.12 / 2.13 %).

Speed (RTFx = total audio / processing time), v3 / redux / ultra run back to back per row, full sets:

| Set | Encoder | v3 | redux | ultra |
|---|---|---:|---:|---:|
| test-clean | ANE | 88.3× | 67.8× | **89.5×** |
| test-clean | GPU | 93.1× | 86.6× | **94.2×** |
| test-other | ANE | 92.2× | 68.0× | **96.5×** |
| test-other | GPU | 93.4× | 89.2× | **100.7×** |
| FLEURS 24 × 100 | ANE | 136.9× | — | 135.3× |
| FLEURS 24 × 100 | GPU | 120.0× | — | **129.2×** |

The shipped encoder is the iOS 17 export (same int8 encoding, spec version 8). Paired against the iOS 18 export on
test-other it matches in WER (3.79 / 3.80 %) and speed (ANE 96.4× vs 93.3×, GPU 96.7× vs 96.5×), so one file serves
iOS 17+.

Benchmark hygiene: an AC Low Power Mode (`pmset -g custom`, `powermode 1`) and other ANE workloads cost 25–40 %
RTFx on this machine; every speed number above is a same-time pair with Low Power Mode off.
