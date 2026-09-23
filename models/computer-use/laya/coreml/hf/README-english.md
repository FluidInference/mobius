---
license: apache-2.0
library_name: coreml
pipeline_tag: text-classification
base_model: convaiinnovations/laya
tags:
  - coreml
  - apple-silicon
  - decision-model
  - laya
---

# Laya English root checkpoint for Core ML

This repository hosts the **English root checkpoint** of [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), pinned at `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. It is a 421,293,827-parameter ModernBERT-large model with trained choice, score, noul and action heads. This is separate from [`FluidInference/laya-coreml`](https://huggingface.co/FluidInference/laya-coreml), which hosts the roughly 322M multilingual checkpoint.

The L128 and L512 packages are fixed-shape FP16 Core ML programs with 32 option slots, targeting iOS 17/macOS 14 or newer. Use L128 when the full rendered input fits 128 tokens and L512 for 129–512 tokens. Inputs: `input_ids` `[1,L]` int32, `attention_mask` `[1,L]` int32, `marker_map` `[1,32,L]` float32, `question_type` `[1,3]` float32. Outputs: raw `logits`, uncalibrated `probabilities`, and `action_probabilities`. The host must use `rl_agent_config.json` to apply the English model's type- and option-count-specific temperatures to logits. The included standalone `calibration.py` contains the temperature and softmax helper; `preprocessing.py` needs the upstream `laya` Python package for rendering. Requests that exceed 32 options or the installed sequence bucket must be rejected or explicitly handled under the upstream API's truncation policy.

The original Laya root config has `max_len=512` and `head_max_len=192`. Native PyTorch parity for both packages is recorded in `verification-english-L128.json` and `verification-english-L512.json`. On Apple M5 Pro, the validated **automatic/GPU** route matched all 16 fixture choices at both lengths. Worst calibrated probability errors were 0.00225 (L128) and 0.00425 (L512); median model call times were 7.79 and 20.79 ms. Forced CPU+ANE also matched every choice, but its worst calibrated probability error was **0.0355**, failing the predeclared 0.02 gate on a 20-option case. The English checkpoint uses a temperature of 0.10058 there, which amplifies small logit differences. Use `.all` for these packages; CPU+ANE is not validated for calibrated outputs. The source lock and conversion reports record SHA-256 hashes and parameter count.

The Decision Index tracker reports Laya at 16.39, but its historical adapter's selected checkpoint has not yet been authenticated. This English artifact must not be presented as a verified reproduction of that score. See [FluidInference/mobius](https://github.com/FluidInference/mobius) for conversion source and [upstream Laya](https://huggingface.co/convaiinnovations/laya) for the Apache-2.0 model and its original evaluation.
