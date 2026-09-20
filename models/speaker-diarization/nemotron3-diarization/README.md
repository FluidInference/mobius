# Nemotron 3 Diarization — CoreML

CoreML conversion of [nvidia/Nemotron-3-Diarization-preview](https://huggingface.co/nvidia/Nemotron-3-Diarization-preview):
the streaming Sortformer successor — 8 speakers, ~100M params, 31-layer RoPE Transformer
encoder, 10 ms output resolution, one checkpoint for every latency profile.

Converted presets are published at
[`FluidInference/nemotron-3-diarization-coreml`](https://huggingface.co/FluidInference/nemotron-3-diarization-coreml)
with NVIDIA's permission (gated until the public release). The Swift host lives in
FluidAudio (`Sources/FluidAudio/Diarizer/Nemotron3/`).

> **License.** The preview checkpoint is under the NVIDIA Software and Model Evaluation
> License. Evaluation results may not be disclosed without NVIDIA's consent, so this
> README documents the conversion and the methodology only; accuracy and throughput
> tables are published with the public release. `build/` is gitignored.

## Architecture vs Streaming Sortformer v2

| | v2 (`diar_streaming_sortformer_4spk-v2.1`) | Nemotron 3 |
|---|---|---|
| Speakers | 4 | 8 |
| Encoder | Fast-Conformer (NEST) + 18-layer Transformer | single 31-layer Transformer, RoPE, FlexAttention |
| Pre-encode | Conv subsampling | FeatureStacking (8× stack + Linear) |
| Output resolution | 80 ms | 10 ms (subpixel Conv1d upsampler ×8) |
| Speaker cache | 188 | 264 |
| NeMo class | `SortformerEncLabelModel` | same (NVIDIA-NeMo/Speech main required) |

## Exported graph

All shapes fixed; latency = (chunk_len + right_context) × 80 ms. Inputs:
`chunk (1, mel, 128)`, `chunk_lengths (1,)`, `spkcache (1, 264, 512)`,
`spkcache_lengths (1,)`, `fifo (1, F, 512)`, `fifo_lengths (1,)`.
Outputs: `speaker_preds (1, packed, 8)` @80 ms, `chunk_pre_encode_embs (1, enc, 512)`,
`chunk_pre_encode_lengths (1,)`, and `speaker_preds_10ms (1, packed*8, 8)` — an extra
output NVIDIA's ONNX export does not have; it enables 10 ms host output.

State (speaker cache, FIFO) lives host-side: the Swift port of `streaming_update_async`
runs cache compression with the checkpoint's learned silence embedding, first-vs-later
compression prediction freezing, and NeMo-exact tail chunking.

### Variants

| Variant | Latency | chunk / rc / fifo / update | Notes |
|---|---|---|---|
| `ultra` | 0.32 s | 3 / 1 / 264 / 222 | model-card profile |
| `verylow` | 0.64 s | 6 / 2 / 264 / 222 | model-card profile |
| `low` | 1.04 s | 9 / 4 / 264 / 222 | model-card profile |
| `offline` | 30.4 s | 340 / 40 / 40 / 300 | model-card profile; fails ANECCompile, runs on GPU |
| `fast` | 1.04 s | 9 / 4 / 40 / 40 | small FIFO: much cheaper per call than `low` |
| `fast32` | 2.88 s | 32 / 4 / 40 / 40 | bigger chunk amortizes the fixed state |
| `fast128` | 10.56 s | 128 / 4 / 40 / 40 | largest monolithic chunk that compiles for the ANE |
| `s32-split`, `c128-split` (+`-w8a8`) | 2.88 s / 10.56 s | as above | split graph, 100% ANE-resident, optional W8A8 |

spkcache_len = 264 and left_context = 0 for every variant. Bigger chunks improve
accuracy on this model — the fixed cache+FIFO state dominates attention, so the chunk
is nearly free — which is why the ladder exists.

**Split graph** (`convert_split.py` / `verify_split.py`): the host does feature
stacking (a reshape) and the 1024→512 pre-encode projection (one `cblas_sgemm` with
`pre_encode_proj_t.bin`), packing and masks; the CoreML model is the pure
floating-point transformer + head. This bypasses an ANE compiler limit on long chunk
inputs (monolithic chunk mel ≤ ~1376 frames compiles, above that `ANECCompile` fails)
and gives a graph that quantizes cleanly to W8A8 (`w8a8_split.py`, calibrated with
real streaming states, scoped to linear/matmul/conv).

## Conversion gotchas (hard-won, keep)

1. **NeMo Speech main required** — `self_attention_model='rope'` and the new
   `TransformerEncoder` are absent from all PyPI nemo-toolkit releases (≤3.0.0).
2. **FlexAttention is untraceable** — `export_patches.py` swaps in explicit
   matmul-softmax with an additive `(B,1,1,T)` key-padding bias (−30000 fp16-safe).
   RoPE/full-attention has no score_mod, so this is exact (verified bit-identical).
   Valid frames are start-packed, so RoPE positions are padding-invariant.
3. **numpy ≥2 breaks coremltools 9.0** torch frontend: every `.shape` unpack dies with
   "only 0-dimensional arrays can be converted to Python scalars". Pin numpy 1.26.4
   (and scipy 1.13.1 for numpy-1.x ABI).
4. **FeatureStacking.forward doesn't trace** (its `t_new` becomes a traced tensor →
   same coremltools `int()` failure). Inlined as static reshape + `proj` Linear in
   `wrappers.py`; export mel frames are always a multiple of 8 so the pad branch is dead.
5. **`concat_and_pad` uses `index_copy_`** (scatter) — replaced with the v2 Sortformer
   gather-based `fixed_concat_and_pad` (ANE-safe, arithmetic indices).
6. **Integer `//` traces to float divide** and the GPU fp16 lowering misrounds inputs
   > 2048. Use `torch.floor_divide` (lowers to true int `floor_div`). Residual caveat:
   on the offline variant, if the graph lands on GPU, `chunk_pre_encode_lengths` can
   still read +1 — hosts should compute `ceil(len/8)` themselves.
7. **Offline variant fails ANECCompile** (`E5RT ... ANECCompile() FAILED`) on
   macOS 26 / M5 Pro and runs on the GPU. Streaming variants are ANE-resident apart
   from a few index/gather ops around state packing; the split graph is fully resident.
8. **Model outputs are fp16 with padded rows** ([1,T,8] at row stride 16): read them
   with a stride-aware compaction, never a linear `dataPointer` copy.
9. **ANE rejects batch > 1** for this graph (0% residency); GPU batching works but
   gains little over two processes. Do not design for ANE concurrency.
10. **Long ANE runs exhaust the IOSurface pool** after a few thousand predictions
    unless the host drains an autoreleasepool per chunk (same failure class as
    FluidAudio #752). Output backings remove the per-call allocation entirely.
11. Mel frontend is the same 128-mel / 10 ms / `normalize: NA` family as Nemotron ASR.

## What was tried (methodology; numbers deferred)

- **Weight-only int8** (`quantize_int8.py`): halves the weight footprint, speed-neutral,
  quality-neutral.
- **W8A8 on the monolithic graph**: dead end — the experimental pass evicts the graph
  to CPU. On the split surface it works (see above).
- **Speaker-cache / FIFO shrinking** (`speed_campaign.py`): individual levers look free
  on small subsets and fail the full-AMI gate when stacked (speaker confusion). Do not
  stack state-shrink levers; gate every claim on the full set.
- **Chunk ladder**: accuracy improves monotonically with chunk length at near-flat
  per-call ANE cost up to the compiler cliff; the split graph continues the ladder.
- **VAD gating** (`speechMask` in the host): large wall-clock win on sparse audio,
  but Silero misses quiet far-field speech, so it costs accuracy on meeting audio —
  opt-in only.
- **Zero-shot layer/sub-layer drops** (`drop_layers.py --explicit/--sub/--combo`):
  the checkpoint is tightly fit; removing two mid-stack blocks is the only tolerable
  cut and it is small. CPU-class targets need real distillation or a smaller sibling.
- **Batch dimension**: measured, not integrated (see gotcha 9).

## Workflow

```bash
cd conversion
uv sync
uv run python convert.py                       # card + ladder variants -> build/
uv run python convert_split.py [--batch N]     # split-graph variants (+ pre_encode_proj_t.bin)
uv run python verify.py                        # single-chunk parity vs torch
uv run python verify_split.py                  # split-graph parity
uv run python e2e_streaming_test.py --variant low --wav build/test_120s.wav
uv run python quantize_int8.py                 # weight-only int8 variants
uv run python w8a8_split.py                    # W8A8 on the split graph
uv run python speed_campaign.py --help         # config sweeps, full-AMI gated
uv run python stage_hf.py                      # assemble the HF upload set
```

## Host (Swift) integration

`Sources/FluidAudio/Diarizer/Nemotron3/` in FluidAudio: `Nemotron3Diarizer` +
`Nemotron3StateUpdater` port `streaming_update_async` at batch 1; `Nemotron3Models`
loads from Hugging Face or a local directory; an audio-in streaming API
(`appendAudio` / `processBufferedAudio` / `finishStream`) is frame-exact with the batch
path. CLI: `nemotron3-diarize`, `nemotron3-benchmark`, `nemotron3-batch`.

Key checkpoint facts the port relies on:
- `use_learnable_sil_emb: true` → the running silence profile (`mean_sil_emb` /
  `n_sil_frames`) is dead code; compression uses the fixed learned embedding.
- All profiles have `chunk_left_context = 0`.
- First compression uses FRESH cache-region predictions; later compressions use the
  stored (frozen) predictions — `spkcache_compressed` gates this.
- 10 ms output: gather `speaker_preds_10ms[(spkcache_len+fifo_len+lc)*8 : +chunk*8]`
  per chunk, before the state mutates.
