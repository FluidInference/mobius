---
license: apache-2.0
base_model:
  - jaredpalmer/kev-0.6b
  - Qwen/Qwen3-0.6B-Base
tags:
  - coreml
  - decision-model
  - apple-silicon
---

# Kev 0.6B Core ML

This is a fixed-shape Core ML conversion of the **trained** [Kev 0.6B](https://huggingface.co/jaredpalmer/kev-0.6b) LoRA adapter and pointer head merged with [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base). The `source/` files in this repository contain the exact conversion scripts and reproducibility notes.

## Artifact

`kev_0_6b_fp16_L128_options32.mlpackage/` is the FP16 Core ML package; `kev_0_6b_w8_L128_options32.mlpackage/` is the 8-bit weight compressed variant. Both target iOS 17/macOS 14 or later and accept one Kev typed question per call, at most 128 tokens and 32 option slots. Each package contains the merged Qwen3 backbone and trained pointer head. `source/runtime.py` uses the included tokenizer, pinned serving configuration, and Core ML package without loading the upstream adapter or Qwen weights. The state may be shortened to fit L128, and requests with an overlong question branch fail explicitly. These variants do not implement Kev's native packed multiquestion forward in one call.

Download the repository with `snapshot_download("FluidInference/kev-0.6b-coreml")`. From its root, install the pinned Python dependencies and run an **unlabelled** one-question request:

```bash
uv sync --project source
uv run --project source python source/runtime.py --model-dir . --request-json request.json
```

For example, `request.json` can be `{"state":"The customer was charged twice.","questions":{"team":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Charges and refunds","support":"Technical issues"}}}}`. Pass `--precision w8` for the compressed package. The runtime returns the upstream typed System One answer. `source/verify.py` is a separate **conversion parity** script that intentionally loads the upstream PyTorch model.

The Core ML inputs are `input_ids` `[1,128]`, `attention_mask` `[1,128]`, `decide_map` `[1,1,128]`, and `option_map` `[1,32,128]`; outputs are `logits` and `probabilities` over 32 slots. Slots absent from the request carry a large negative logit.

## Validation and limits

The explicit PyTorch exporter matched the trained Kev model within `9.54e-7` logits on the export fixture. Both Core ML packages selected the same answer on 4/4 local requests covering Choice, Noul, and Score. FP16 had maximum probability difference `0.002031` and size `1,194 MB`. W8 had maximum probability difference `0.008554` and size `599 MB`. The standalone host passed offline, unlabelled Choice, Noul, and Score requests without native weights; see `source/reports/standalone-runtime.json`. Short `CPU_AND_NE` model-call medians on one Apple Silicon Mac ranged `13.10–15.35 ms` for FP16 and `12.27–23.15 ms` for W8 across two tiny runs; they reversed the speed ordering, so no variant-level speed claim is made. The compute setting permits the Neural Engine but does not establish full ANE residency. These checks do not constitute the Jev Decision Index or a 2048 benchmark, and the tracker’s 31.30 score was measured on a separate serving stack.

The adapter/head and Qwen base are Apache-2.0. The base license text is included as `LICENSE`. Upstream author: Jared Palmer; Qwen base: Qwen team; Core ML conversion: Fluid Inference. Exact source revisions and trained-weight SHA256 hashes are in `source/assets.lock.json`.
