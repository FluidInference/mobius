# Phonon-2 → Core ML

Conversion of [FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) (Fermion Research's
quantization-aware re-training of `nvidia/parakeet-tdt-0.6b-v3`, English, CC-BY-4.0) into the v3 component contract:
`Preprocessor` / `Encoder` / `Decoder` / `JointDecisionv3`, 15 s window. Published as `FluidInference/phonon-2-coreml`;
loaded in Swift via `AsrModelVersion.phonon2`.

## What the checkpoint is

* One 164 MB download, `phonon-2.bps.tar.zst` (zstd tar with a byte-plane transform; `package_release_bps.py unpack` in
  [fermionresearch/phonon](https://github.com/fermionresearch/phonon) restores `model.fermion`, sha-checked). The
  container format is `fermion-five-value-parakeet-v1`; `fermion_container.py` (vendored here, Apache-2.0) reads it into
  HF-transformers `ParakeetForTDT` names — the same 723-key set as moondream/parakeet-redux, so the redux HF→NeMo name
  map applies unchanged.
* **264 five-value modules** = the 11 linears of every encoder layer (`feed_forward{1,2}.linear{1,2}`, `q/k/v/o_proj`,
  `relative_k_proj`, `pointwise_conv{1,2}`; 604 M weights): `w[r, c] = sign[r, c] · (hi[r] if is_hi[r, c] else lo[r])`,
  so each **output row** takes at most five values `{0, ±lo, ±hi}` with two fp16 magnitudes per row. 51 % of the weights
  are zero, 31 % `±lo`, 18 % `±hi`. `hi/lo` is 2.1 at the median and up to 100×, so no affine (shift-scale) grid holds
  all five exactly — a palette does.
* **35 int6 tables** (one fp16 scale per output row): subsampling 1×1 convs + linear, depthwise convs, decoder embedding
  and LSTM, decoder/encoder projectors, joint head. **424 fp16** tensors: norms, biases, BN stats.
* No mel front-end is shipped; NeMo's v3 preprocessor and the v3 tokenizer are used as is (the card says tokenizer and
  output conventions are the original's).

## Pipeline (uses the v3 `uv` env in `../../parakeet-tdt-v3-0.6b/coreml`)

```bash
PY=../../parakeet-tdt-v3-0.6b/coreml/.venv/bin/python
W=~/Documents/phonon2-work
python package_release_bps.py unpack phonon-2.bps.tar.zst $W/container   # from fermionresearch/phonon
$PY phonon2_weights.py                        # container → NeMo names, strict load, diff vs nvidia v3
$PY export_encoder_fp16.py                    # fp16 iOS18 encoder mlpackage + torch reference outputs
$PY fivevalue_encoder.py --mode lut6          # exact palette re-encoding → components/Encoder_lut6.mlmodelc
$PY fivevalue_encoder.py --mode lut3          #   … and the small variant
$PY export_decoder_joint.py                   # Decoder.mlpackage + JointDecisionv3.mlpackage (iOS17, v3 I/O)
$PY ../../parakeet-redux/coreml/validate_encoder.py $W/components/Encoder_lut6.mlmodelc --ref $W/encoder_ref_phonon2.npz --units gpu,ane
$PY score.py $W/bench/*.json                  # corpus WER of FluidAudio asr-benchmark outputs
$PY nemo_reference_wer.py $W/bench/phonon2_test-clean_ane.json   # NeMo fp32 decode of the same files
```

## Encoder encoding

The fp16 export's 240 five-value consts (`module_<nemo path>_weight_to_fp16`; the 24 `linear_pos` tensors are
constant-folded into the fixed-window positional projection) are matched by name, checked value-exact against the
container, and replaced by iOS 18 `constexpr_lut_to_dense` ops with a **grouped-channel palette**: the LUT of a group
of G consecutive output rows is the sorted multiset of their `{0, ±lo, ±hi}` (5G entries, zero-padded to 2^nbits), and
each weight stores the index of its value. Reconstruction is bit-exact in fp16 — nothing is re-quantized on our side.

| `--mode` | Rows per LUT | Index bits | Encoder | GPU latency / load | ANE latency / first load |
|---|---:|---:|---:|---|---|
| `lut3` | 1 | 3 | **253 MB** | 16.2 ms / 0.7 s | 72.4 ms / 72 s |
| `lut4` | 2 | 4 | 325 MB | 20.7 ms / 0.8 s | 40.0 ms / 24 s |
| `lut6` | 8 | 6 | 470 MB | 16.2 ms / 0.6 s | **18.6 ms** / 14 s |
| v3 shipped 6-bit LUT (reference) | — | 6 | 445 MB | 18.3 ms | 23.5 ms |
| fp16 (reference) | — | 16 | 1187 MB | — | 20.8 ms |

One 15 s window on an M5 Pro (macOS 27, coremltools 9.0b1); parity vs the PyTorch fp32 encoder on the same mel is
identical for all three (GPU cos 0.999998, ANE cos 0.99959 through the ANE's fp16 pipeline; 1381 ANE / 4 CPU ops).

**The ANE's palette cost scales with the number of LUT groups, not with the index width.** Per-row palettes (lut3) run at
3× the fp16 encoder, exactly like the per-row diagnostic in the redux sweep; 8 rows per palette (lut6) is *faster* than
both the fp16 encoder and the shipped v3 6-bit encoder, and compiles in 14 s. The GPU decompresses in-kernel and does
not care. The published repo therefore carries both: `Encoder.mlmodelc` = lut6 (the FluidAudio default, ANE) and
`Encoder_lut3.mlmodelc` for GPU-only / size-constrained use (same transcripts).

## Results

End-to-end, FluidAudio `asr-benchmark`, **full** LibriSpeech, M5 Pro, models run back to back. Corpus WER = total edit
distance / total reference words; RTFx = total audio / total processing time.

| Set | Encoder units | v3 WER | phonon2 WER | v3 RTFx | phonon2 lut6 RTFx | phonon2 lut3 RTFx |
|-----|---------------|-------:|------------:|--------:|------------------:|------------------:|
| test-clean (2620) | ANE | 2.27 % | 2.47 % | 148.7× | **155.1×** | 70.0× |
| test-other (2939) | ANE | 4.12 % | 4.62 % | 138.1× | **143.0×** | 64.6× |
| test-clean (2620) | GPU | 2.30 % | 2.46 % | 171.9× | 150.6× | 154.1× |

lut3 and lut6 give identical transcripts (2620 / 2620); the ANE WER / GPU WER difference is the usual fp16 pipeline
noise. The +0.20 / +0.50 gap to v3 reproduces the card's own deltas to its teacher (+0.20 / +0.79, leaderboard
protocol). NeMo fp32 full-context decode of the first 100 test-clean files: Phonon-2 1.79 % vs Core ML 1.83 %,
Core ML-vs-NeMo hypothesis WER 0.34 % (stock v3 for comparison: 1.88 %, 1.36 % against its own Core ML build) — the
conversion is lossless to within fp16.

### Sparse encoding (the 164 MB question)

The upstream 164 MB is a zstd archive of a ~2.1-bit custom packing (trits 5 per byte + a bitmask over the non-zeros);
Core ML has no entropy-coded weight format, and a five-value palette needs 3-bit indices. The Core ML route that gets
close is iOS 18 sparsity: 51 % of the weights are zero and the non-zeros take four values per row, so
`constexpr_lut_to_sparse` (palette over the non-zeros only) + `constexpr_sparse_to_dense` (1-bit mask) is exact at
1 + 0.49 × nbits bits per weight. Grouping rows per palette costs bits but, as with the dense LUTs, buys ANE speed:

| `--mode` | Rows per LUT | Non-zero index bits | Encoder | ANE latency | ANE first load | GPU |
|---|---:|---:|---:|---:|---:|---|
| `sparse-g1` | 1 | 2 | **176 MB** | 70.0 ms | 90 s | 16.1 ms, but ~150 s load **every** time |
| `sparse-g2` | 2 | 3 | 211 MB | — | — | 17.2 ms, 164 s load |
| `sparse-g4` | 4 | 4 | **246 MB** | **24.3 ms** (v3: 23.5) | 56 s | 15.9 ms, 158 s load |
| `sparse-g8` | 8 | 6 | 321 MB | **18.6 ms** (= lut6) | 52 s | — |

All exact (same parity as the dense builds). The ANE compiles the sparse weights on first load; Core ML's cache made a
second `sparse-g1` load take 0.1 s, but `sparse-g4`/`g8` recompiled (52–55 s) on their second load after several other
large encoders had been compiled in between, so treat the compile as "usually cached" rather than guaranteed. The GPU
path instead materializes the sparse weights at every load (~2.5 min of CPU, never cached), so a sparse file is wrong
for `.cpuAndGPU` users — keep `lut3` for them.

End-to-end on the ANE (full test-clean, same session, v3 control 148.7–151.5×): `sparse-g8` **159.0×**, `lut6` 155.1×,
`sparse-g4` 140.0×, `lut3` 70.0×; all four produce the same 2620 transcripts (2.47 %).

**Shipping candidates**, all exact and WER-identical:

| File | Size | ANE RTFx | GPU | Role |
|---|---:|---:|---|---|
| `lut6` | 470 MB | 155× | 16 ms, 0.6 s load | fastest everywhere, biggest |
| `sparse-g8` | 321 MB | 159× | 150 s load | ANE-only default candidate: lut6 speed at 68 % of the size |
| `sparse-g4` | 246 MB | 140× | 150 s load | ANE-only, v3 speed, half of lut6 |
| `lut3` | 253 MB | 70× | 16 ms, 0.7 s load | the GPU / Mac file |
| `sparse-g1` | 176 MB | ~70× (70 ms) | 150 s load | size floor, 7 % above the upstream download |

### 60-minute long-form file (Earnings-22, four concatenated calls)

`transcribe` on the 3600 s `earnings22_top4_1h.wav` (M5 Pro, default ANE encoder, best of 2–3 runs, processing time
excludes model load). Reference = the concatenated Earnings-22 chunk transcripts, same normalizer as above.

| Model | Processing time | RTFx | WER |
|---|---:|---:|---:|
| v3 | 10.9 s | 331× | 16.5 % |
| Ultra | 7.7 s | 469× | **13.5 %** |
| Redux | 14.2 s | 254× | 14.8 % |
| Phonon-2 default (sparse, 321 MB) | 7.5 s | **478×** | 17.2 % |
| Phonon-2 `Encoder_lut6` (470 MB) | 7.4 s | 486× | 17.2 % |
| Phonon-2 `Encoder_sparse-g4` (246 MB) | 9.0 s | 399× | 17.2 % |
| Phonon-2 `Encoder_sparse-g1` (176 MB) | 23.4 s | 154× | 17.2 % |
| Phonon-2 `Encoder_lut3` (253 MB) | 23.6 s | 152× | 17.2 % |

On conversational long-form audio Phonon-2 is the fastest model we ship (1.45× v3's throughput, on par with Ultra) but
the least accurate of the four: Ultra and Redux both beat v3 here while Phonon-2 trails it by 0.8 points, consistent
with the upstream card's Earnings-22 row (6.96 % vs its teacher's 5.85 %). All five Phonon-2 encoders produce the same
transcript.
