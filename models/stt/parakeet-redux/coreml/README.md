# parakeet-redux → Core ML

Conversion of [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) (ternary re-training of
`nvidia/parakeet-tdt-0.6b-v3`, CC-BY-4.0) into the same component contract as
`FluidInference/parakeet-tdt-0.6b-v3-coreml`: `Preprocessor` / `Encoder` / `Decoder` / `JointDecisionv3`, 15 s window.
Published as `FluidInference/parakeet-redux-coreml`; loaded in Swift via `AsrModelVersion.redux`.

## What the checkpoint is

* Same architecture and tokenizer as v3 (HF-transformers parameter names). Only the 24 encoder layers' linears are
  ternary (`feed_forward{1,2}.linear{1,2}`, `q/k/v/o_proj`, `relative_k_proj`, `pointwise_conv{1,2}`): group size
  128 along the input dim, fp16 scale per (row, block), `w = scale[r, c // 128] * (code - 1)`, codes packed 5 per
  byte in base 3 (`ternary.json`, format `thrush-ternary-v2`).
* Everything else was fine-tuned too: layer norms, depthwise conv, subsampling, decoder (fp16-rounding level) and the
  joint (up to 8.5 % relative change) — so the decoder and joint are re-exported, not reused from v3.
* Extra `vad_head` on the subsampler (213 k params) used by Photon for long-form segmentation. Not exported;
  FluidAudio has its own windowing.

## Pipeline (uses the v3 `uv` env in `../../parakeet-tdt-v3-0.6b/coreml`)

```bash
PY=../../parakeet-tdt-v3-0.6b/coreml/.venv/bin/python
$PY redux_weights.py                 # unpack ternary, map HF→NeMo names, strict load, diff vs nvidia v3
$PY export_encoder_fp16.py           # fp16 iOS18 encoder mlpackage + torch reference outputs
$PY ternarize_encoder.py --mode joint  # exact 2-bit re-encoding of the 240 ternary consts → Encoder.mlmodelc
$PY export_decoder_joint.py          # Decoder.mlpackage + JointDecisionv3.mlpackage (iOS17, same I/O as v3)
$PY validate_encoder.py <Encoder.mlmodelc> --units ane,gpu   # parity vs torch, latency, compute placement
$PY nemo_reference_wer.py <asr-benchmark.json>               # NeMo fp32 decode of the same files, same scorer
```

`Preprocessor.mlmodelc` and `parakeet_vocab.json` are copied from the v3 repo (redux does not ship the mel
front-end; NeMo's own is used, unchanged).

### Encoder weight encoding

`ct.convert` names the traced consts `module_<nemo path>_weight_to_fp16`, so the ternary tensors are matched by
name and verified value-exact against the dequantized fp16 before replacement. `linear_pos` (24 tensors) is
constant-folded by the converter into the fixed-window positional projection (`[1, 8, 128, 375]` fp16, 18 MB total)
and is left as is.

| `--mode`        | MIL form                                                                        | Exact | Encoder |
|-----------------|---------------------------------------------------------------------------------|-------|---------|
| `joint` (ships) | `constexpr_lut_to_dense` uint2 idx, int8 LUT {-1,0,1} → `constexpr_blockwise_shift_scale` fp16 [out, in/128] | yes | 183 MB |
| `blockwise-lut` | N-D blockwise `constexpr_lut_to_dense`, fp16 LUT [-s,0,s,0] per (row, block)     | yes   | 210 MB  |
| `int4-block`    | `constexpr_blockwise_shift_scale` int4 data, fp16 [out, in/128] scale             | yes   | 330 MB  |
| `joint-perchannel` | as `joint` but one scale per row (diagnostic, **inexact**)                    | no    | 183 MB  |
| `lut-perrow`    | grouped-channel fp16 LUT, group 1 on axis 0 (diagnostic, **inexact**)            | no    | 177 MB  |

`coremltools.optimize.coreml.palettize_weights` cannot express a per-(row, block) LUT ("general block-wise
palettization is not supported"), hence the custom graph pass.

## Results (M-series Mac, macOS 27, coremltools 9.0b1)

Encoder, one 15 s window, vs the PyTorch fp32 redux encoder on the same mel:

| Encoder                    | Size   | GPU load / latency | ANE load / latency | ANE placement          | cos vs torch |
|----------------------------|--------|--------------------|--------------------|------------------------|--------------|
| v3 shipped 6-bit LUT       | 445 MB | 3.8 s / 18.3 ms    | 11.5 s / 23.5 ms   | 1381 ANE, 4 CPU        | (other weights) |
| redux fp16 (uncompressed)  | 1.19 GB| —                  | 8.7 s / 20.8 ms    | 1381 ANE, 4 CPU        | 0.99985      |
| redux `joint` 2-bit        | 183 MB | 0.6 s / 21.3 ms    | 368 s / 52.2 ms    | 1271 ANE, 108 CPU      | 0.99984      |
| redux `blockwise-lut` 2-bit| 210 MB | 0.6 s / 21.1 ms    | 320 s / 44.8 ms    | 1271 ANE, 108 CPU      | 0.99984      |

(GPU parity is measured at fp16 accumulation: `cos 0.999997`, `max|d| 0.0017`. The ANE column's parity is the same
encoding measured through the ANE's own fp16 pipeline.)

### The ANE does not want these weights

Full sweep, ANE (`CPU_AND_NE`), same graph, only the weight encoding changed:

| Encoding                    | First load | Latency | Placement          |
|-----------------------------|-----------:|--------:|--------------------|
| fp16, iOS17                 | 8.3 s      | 21.2 ms | 1381 ANE, 4 CPU    |
| fp16, iOS18                 | 8.7 s      | 20.8 ms | 1381 ANE, 4 CPU    |
| 2-bit, per-(row, block) scale (`joint`)   | 368 s | 52.2 ms | 1271 ANE, 108 CPU |
| 2-bit, per-(row, block) LUT (`blockwise-lut`) | 320 s | 44.8 ms | 1271 ANE, 108 CPU |
| int4, per-(row, block) scale (`int4-block`)   | 1585 s | 44.9 ms | 1271 ANE, 108 CPU |
| 2-bit, per-row scale (`joint-perchannel`) | 44.9 s | 70.4 ms | 1381 ANE, 4 CPU |
| 2-bit, per-row LUT (`lut-perrow`)         | 45.2 s | 70.6 ms | 1381 ANE, 4 CPU |

Two separate problems, and no encoding avoids both:

* **Blockwise scales break the compiler.** Any per-(row, 128-column) scale — 2-bit or int4, LUT or shift-scale — makes
  `ANECompilerService` run 5–26 minutes at 100 % CPU on first load and pushes 72 `select` + 24 `softmax` (the
  attention mask path, which has nothing to do with the weights) onto the CPU, costing ~2× latency.
* **Per-row scales compile fine and are slower anyway.** Dropping to one scale per row restores the fp16 placement
  exactly (1381 ANE, 4 CPU) and a 45 s compile, but lands at 70 ms — 3× the fp16 encoder. Both per-row variants are
  deliberately inexact (`cos 0.25`); they exist only to isolate the compiler behaviour, and latency does not depend
  on the values.

The same graph with plain fp16 weights compiles in 9 s and runs at 20.8 ms, so this is the weight encoding, not the
iOS18 target and not the graph. **On this OS/hardware every 2-bit encoding is slower on the ANE than the shipped
6-bit v3 encoder (23.5 ms).** The GPU, by contrast, decompresses in-kernel: 21 ms and a 0.6 s load, matching fp16.
Consequence for the host: FluidAudio keeps the ANE default (iOS background execution needs it) and documents
`.cpuAndGPU` as the fast-load option. Re-test when the ANE compiler gains real
blockwise-palettization support.

End-to-end, FluidAudio `asr-benchmark`, **full** LibriSpeech. Corpus WER = total edit distance / total reference
words; RTFx = total audio / total processing time.

| Set | Encoder | v3 WER | redux WER | v3 RTFx | redux RTFx |
|-----|---------|-------:|----------:|--------:|-----------:|
| test-clean (2620) | GPU | 2.30 % | 2.67 % | 117.9× | 104.6× |
| test-other (2939) | GPU | 4.10 % | 5.15 % | 106.2× | 99.0×  |
| test-clean (2620) | ANE | 2.27 % | 2.68 % | 101.4× | 80.4×  |
| test-other (2939) | ANE | 4.12 % | —      | 104.0× | —      |

v3 wins on English by +0.37 (clean) and +1.05 (other), reproducing the model card's own deltas (+0.44, +1.21).
Absolute values are higher than the card because FluidAudio decodes in 15 s windows and this scorer is simpler than
the Open ASR Leaderboard's (lowercase, strip punctuation, keep apostrophes) — both models pay it equally. Compute
placement is WER-neutral for both (redux 2.67 GPU / 2.68 ANE).

**Do not trust small subsets here.** The first-100-file slice of test-clean gave the *opposite* ranking (redux 2.00 %
vs v3 2.43 %); the full 2620 files reverse it. Per-file mean WER is also unusable on this corpus — one 2-word
utterance (`STEPHANOS DEDALOS` → 150 %) moves it by more than a point.

Conversion fidelity is separately confirmed: NeMo fp32 full-context decode of 100 of these files scores redux
3.62 % / stock 2.82 % (same scorer), and the CoreML redux transcripts differ from the PyTorch redux transcripts by
0.19 % WER. The English regression is the checkpoint's, not the conversion's.

One caveat on ANE timing: the first `test-clean` + ANE run for v3 recorded 25× overall because two short files
stalled for 299 s and 225 s (both transcribed correctly). Re-running the identical command gave 101.4× with a
bit-identical 2.27 % WER. Treat isolated multi-hundred-second files on ANE as a compile/eviction artifact — re-run
rather than reporting the total.

### FLEURS (the reason to ship this model)

`fleurs-benchmark --samples 100`, 24 of 25 languages (`es_es` is missing from FluidAudio's FLEURS map), GPU encoder:

| | v3 | redux |
|---|---:|---:|
| mean WER, 24 languages | 14.81 % | **13.06 %** |
| duration-weighted      | 14.65 % | **12.89 %** |
| languages won          | 11      | **13**      |
| RTFx                   | **148.8×** | 134.1×   |

Redux takes the low-resource end by large margins — Latvian −11.14, Maltese −7.72, Slovene −7.17, Estonian −7.02,
Greek −5.19, Lithuanian −5.10 — and pays for it on the high-resource end: French +3.66, Russian +2.57, Dutch +1.92,
English +1.86, Polish +1.68, Ukrainian +1.43.

Per-language direction matches the upstream card in **23 of 24** languages (Bulgarian is the lone flip: we see
+0.97, the card −0.67, both small). Our absolute WERs and our deltas both run larger than the card's — 15 s windowed
decoding plus a simpler normalizer — but the ordering reproduces. Bulgarian v3 landed at 11.91 % against the card's
11.90 %, so the protocols are close.

**Verdict:** v3 for English, redux for multilingual and for download size. The conversion adds nothing measurable to
either (0.19 % WER vs PyTorch); every difference above belongs to the checkpoint.

## iOS 17 / macOS 14 int8 build (evaluated, not shipped)

The 2-bit encoding needs iOS 18 ops, so iOS 17 gets data-free int8 per-channel quantization of the iOS 17 fp16 export
(`coremltools.optimize.coreml.linear_quantize_weights`, symmetric per-channel; the ternary weights round cleanly
into int8). It was not in
the sweep above, and it runs well on the ANE: 1381 ANE ops, ~14 s first compile, 27.5 ms/window ANE, 16.3 ms GPU,
cos 0.9996 vs torch. 595 MB. Full-set WER on ANE matches the 2-bit build:

| | 2-bit (iOS 18) | int8 (iOS 17) |
|---|---:|---:|
| test-clean | 2.68 % | 2.69 % |
| test-other | 5.14 % | 5.14 % |
| FLEURS 24 × 100 | 13.1 % | 13.1 % |

Not published: at 595 MB it is larger than the v3 encoder, so Redux stays iOS 18+ only and FluidAudio refuses to
load it on iOS 17 / macOS 14.
