---
license: apache-2.0
library_name: coreml
pipeline_tag: text-classification
base_model: heman10x/rlcd-modernbert-151m
tags:
  - coreml
  - apple-silicon
  - decision-model
  - verdict
---

# Verdict Core ML (L128 and L512 FP16)

This is a Core ML conversion of the **trained Verdict checkpoint** [`heman10x/rlcd-modernbert-151m`](https://huggingface.co/heman10x/rlcd-modernbert-151m), pinned at `8af2496eb63c7fa66d7d234e1f62629380030eb4`. It has 151,378,177 parameters and retains the full encoder plus trained decision head. The L128 package is 303,210,832 bytes; L512 is 304,390,482 bytes. Both target iOS 17 / macOS 14 or newer. Choose L128 when the full rendered request fits 128 tokens and L512 for 129–512 tokens. Do not truncate an overlength request into an installed bucket.

Inputs are `input_ids` and `attention_mask` of shape `[1,L]`, and `class_marker_map` of shape `[1,25,L]`, where `L` is 128 or 512. The graph returns raw logits and an uncalibrated softmax. For a native Verdict decision, render the request with `native_reference.py`, append the `__insufficient_evidence__` candidate, and apply `calibrator.json` to raw logits. This means at most **24 substantive candidates**. The released calibrator has a separate temperature for several candidate counts. Calling the graph with arbitrary labels or treating `probabilities` as calibrated changes the model's behavior.

Verified on Apple M5 Pro/macOS 27.0 against the pinned PyTorch model: L128 passed 4/4 smoke decisions (including two native abstentions), worst calibrated probability difference 0.00341. L512 passed 5/5 (including one 283-token public Decision Index row), worst difference 0.00070. The 20-iteration L128 `coreml-cli` median was 3.627 ms CPU+GPU, 3.710 ms CPU+ANE, 3.915 ms automatic, and 11.721 ms CPU only. L512 median automatic model call time across its five parity requests was 8.15 ms. Rendering and tokenization add time. The verification and profile JSON files include the details.

The public Decision Index tracker reports Verdict at 13.38. Its historical checkpoint and renderer have not been authenticated against this pinned release, so this artifact does **not** claim to reproduce that score. The two FP16 buckets cover its 512-token public limit.

An experimental L128 8-bit k-means LUT cut package size to 151,878,624 bytes, but **failed** the predeclared 100-request FP16 comparison: 96/100 calibrated selection and abstention agreement (four abstention flips), versus a 99/100 gate. P95 calibrated probability error was 0.0197; worst was 0.0297. Median measured model-call time was 3.48 ms LUT8 versus 3.34 ms FP16. The LUT8 package is **not included** here as a validated model. See `lut8-L128-suite-parity.json` and `verification-verdict_lut8_kmeans_per_tensor_L128_candidates25.json` for the reports.

The Hub repo includes the runtime renderer, calibration helper, tokenizer, verification report and SHA-256 asset lock. Conversion source is maintained in [FluidInference/mobius](https://github.com/FluidInference/mobius). Upstream model and code: [Verdict model](https://huggingface.co/heman10x/rlcd-modernbert-151m), [Verdict source](https://github.com/Heman10x-NGU/Verdict-open-jev), Apache-2.0. Base architecture: [`knowledgator/gliclass-modern-base-v2.0`](https://huggingface.co/knowledgator/gliclass-modern-base-v2.0).
