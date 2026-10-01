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

WER_TABLE_PLACEHOLDER
