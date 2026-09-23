---
tags:
- coreml
- conversion-toolkit
- gated-model
---

# System One Gemma Core ML conversion toolkit (source only)

**Blocked, no model weights or Core ML package here.** This repo hosts pinned conversion source for [`akash-kamat/system-one-gemma`](https://github.com/akash-kamat/system-one-gemma) at `cc75aa8042dec965003210fdba033fee8759735e` with [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) at `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. The current authenticated account receives HTTP 403 for the gated base. The source's trained adapter contains the scalar score head, but neither the adapter nor any base or derivative weights are redistributed here.

Read [README-toolkit.md](README-toolkit.md) for reproducible commands, exact native semantics, validation gates, and license/access notes. The scripts are prepared and **untested against the complete model** until access is granted. No latency, ANE, size, parity, or Decision Index claim is made.

The upstream scorer README states that the pretrained weights inherit a noncommercial restriction. [Gemma Terms](https://ai.google.dev/gemma/terms) govern the base and converted derivatives. Any future artifact requires both legitimate access and a separate redistribution-rights review.

`LICENSE-CODE` covers this repository's conversion scripts only; it does not change the base, adapter, or derivative model terms.
