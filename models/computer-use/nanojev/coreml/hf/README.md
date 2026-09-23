---
base_model: C-Tianyu/NanoJev
tags:
  - coreml
  - decision-model
  - conversion-recipe
---

# NanoJev Core ML conversion source — no weights

This repository contains a validated conversion recipe for the [current NanoJev unified-games checkpoint](https://huggingface.co/C-Tianyu/NanoJev), pinned to revision `047b927b30882a1138fc504821b82ac145a4b81a`. **It does not contain a converted model or trained weights.** The upstream model card labels its source code MIT but does not state a redistribution license for the trained weights. Converted Core ML weights will be published only if that permission is clarified.

The source preserves NanoJev's Qwen3 candidate encoder, trained scalar head, and Choice-specific attention over the full candidate set. It also implements the Boolean and Score output paths. `source/assets.lock.json` records the selected checkpoint SHA256 `f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b`, which contains 596,250,498 trained parameters including heads. The exporter creates separate candidate-encoder and decision-head Core ML packages so one backbone serves all three question types.

The local FP16 L128/K4 export passed native parity on Choice, Boolean, and Score (maximum logit error `1.67e-6`). Core ML selected the same answer as native PyTorch on all three requests; maximum probability errors were `0.003851`, `0.000847`, and `0.005882`, respectively. The encoder package is about 1.1 GiB and the shared head about 420 KiB. The packages remain local because trained-weight redistribution is unresolved. These are parity checks, not the Decision Index or 2048 benchmark.

The current checkpoint is newer than the one likely used for the tracker entry, whose exact revision has not been verified. A local Core ML package from this recipe should not be described as reproducing the tracker's 26.19 score without that identity check and the official suite run.

See `source/README.md` for commands and `source/RESULTS.md` for the local results. This repository's conversion scripts are original Fluid Inference code. NanoJev's source code and checkpoint remain at the upstream repository; the base Qwen model keeps its own license.

`LICENSE-CODE` covers this repository's conversion scripts only. It does not license the upstream trained NanoJev weights.
