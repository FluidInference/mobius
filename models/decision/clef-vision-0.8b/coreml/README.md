`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)# clef-vision-0.8b → Core ML
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)Core ML export of [FluidInference/clef-vision-0.8b-coreml](https://huggingface.co/FluidInference/clef-vision-0.8b-coreml):
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)a 0.92B image decision model distilled from Cloudflare's [Clef-flash](https://huggingface.co/Cloudflare/clef-flash)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)(Qwen3.5-9B) into a [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) backbone with a copy of Clef's joint
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)schema head. Same contract as Clef / Jev / SystemOne: `state` + images + typed questions (`noul`, `choice`, `score`)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)→ one probability per allowed option, one prefill pass, nothing generated. Swift host: `ClefVisionManager` in
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)[FluidUse](https://github.com/FluidInference/FluidUse). Training and evaluation live in Fluid Inference's model-lab.
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)## Packages
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| Package | Trace | Size | Precision | M5 Pro GPU |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)|---|---|---:|---|---:|
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| `Vision_P784/VisionTower_fp32` | Qwen3.5 ViT, one image right-padded to 784 patches (448² cap), host `pos_embeds` / 2-D rope / pad mask in, 196 merged tokens out | 377 MB | fp32 | 35–51 ms |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| `LM_L{512,1024,2048}/LMRows_fp16` | the Kev / Cua-S1 prefill-only Qwen3.5 decoder (`qwen35_export.py`) + final norm; host embedding gather with vision tokens spliced, host M-RoPE tables | ~960 MB each | fp16 | 190 ms (L1024) |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| `Head_L{512,1024,2048}_Q16_O64/Head_fp32` | mask-aware fixed-shape rewrite of Clef's `JointSchemaHead`: span-mean matrices, option→question map, lexical rows, per-question option softmax | 264 MB | fp32 | 14 ms |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)Plus `embeddings.f16` (tied token table, also the head's lexical rows) and `Vision_P784/pos_embed.f32` (the learned
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)48×48 position grid the host resamples per image). `config.json` on the Hub lists every shape and constant.
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)## Run
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)```bash
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)uv sync
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)uv run python convert-coreml.py --merged <student>/merged/backbone --student <student>/best --out build/coreml
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)```
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)`<student>/merged/backbone` is the LoRA-folded HF checkpoint (`model.safetensors`, `config.json`, processor files);
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)`<student>/best` holds `joint_head.safetensors` + `joint_head_config.json`. Conversion uses torch 2.7.0 + coremltools
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)9.0 (coremltools 9 fails on traces from newer torch). The per-module scripts also run alone (`vision_export.py`,
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)`lm_export.py`, `head_export.py`).
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| Check | Result |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)|---|---|
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| vision torch module vs HF `Qwen3_5VisionModel` (real images, incl. padded 432/748-patch cases) | max 3.0e-4 |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| vision Core ML fp32 vs torch | 7e-4 · **fp16 0.27, mixed (fp32 norms + softmax) 0.33** → this ViT's activation range overflows fp16; ship fp32 (`reports/vision-fp16-vs-fp32.json`) |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| LM rows torch module vs HF text model (records with 1–2 images) | 4.0e-5 relative |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| head torch module vs Clef's head | 1.2e-6 (after switching the option softmax to a per-question max: a global max underflowed weaker questions' exponentials) |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| end to end, 9 held-out records, `CPU_AND_GPU` | max logit diff 6.3e-3, probability drift 2.3e-4, argmax 100%, ≈ 260 ms / record vs 604 ms torch MPS (`reports/coreml-e2e-m5pro.json`) |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)| Swift host (FluidUse `ClefVisionCheck parity`) vs Python reference on 4 fixtures (1–4 images) | token ids / spans / M-RoPE positions exact; max logit diff 2.2e-2, max probability drift 8.2e-3, argmax 8/8; warm 136–157 ms per single-image record (vision 39, LM 95–113, head 5 ms), 640 ms for a 4-image / 1,111-token grid |
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)## Host contract (what FluidUse implements)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)1. Tokenize Clef's record (`<|im_start|>system … STATE: <images> <state json> SCHEMA FIELDS … JOINT SCHEMA DECISIONS:`);
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)   the question and option spans are the head's averaging windows.
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)2. Per image: smart-resize to multiples of 32 within 16,384–200,704 pixels, torchvision-style antialiased bicubic,
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)   rescale + normalize (mean/std 0.5), patchify in spatial-merge-block order with the frame repeated twice
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)   (temporal patch), bilinear align-corners resample of `pos_embed.f32`, axial 2-D rope, additive pad mask.
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)3. Embedding gather, vision tokens into the `<|image_pad|>` slots, 3-D M-RoPE positions (text advances all axes;
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)   an image spans `(t, t+row, t+col)` over its merged grid and advances the cursor by `max(h, w)/2`), interleaved
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)   rope tables with sections `[11, 11, 10]`.
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)4. Head inputs: `q_mean [16,L]`, `o_mean [64,L]`, `o2q [64,16]`, `lexical [64,1024]` (mean embedding rows of each
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)   option's tokens), `type_oh [16,3]`, masks; softmax per question over the returned logits.
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)## Notes
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)- The teacher labels came from Clef-flash run locally in MLX (8-bit backbone + an exact MLX port of the head,
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)  `reports/teacher-mlx-head-parity.json`); the student itself was trained in PyTorch on an M5 Pro.
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)- The Qwen3.5 decoder does not run on the Neural Engine (see the Kev / Cua-S1 notes); with the vision tower in fp32
`vision_parity.py` (image files → trace module and Core ML package vs HF; needs `uv sync --extra check`) is the
runnable sanity check; the LM / head parity and the 9-record end-to-end check ran from the training repo.

## Fidelity (M5 Pro)  this is a GPU model.
