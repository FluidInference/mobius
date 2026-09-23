# LFM2.5-350M-RLCD Core ML conversion

This toolkit targets the pinned [`notnotsamuel/LFM2.5-350M-RLCD`](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/tree/deb589d803d141cabd158ef55f6617b128529f36) checkpoint, whose weights are a byte-for-byte copy of `LiquidAI/LFM2.5-350M` according to its release manifest. The trained behavior here is the RLCD engine's constrained inference protocol, not a fine-tuned classifier. The source weights use the [LFM Open License v1.0](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/blob/deb589d803d141cabd158ef55f6617b128529f36/LICENSE); the RLCD code has its own license.

The upstream engine prefills the prompt once, forks its hybrid attention/convolution cache, and scores the **full token sequence** of each candidate value. The Core ML graph in this toolkit recomputes each prompt and candidate as an independent sequence, then sums the same value-token log probabilities. This avoids mutable caches but costs more work per candidate. It does not use first-token approximations. The fixed-shape boundary rejects requests that exceed its sequence or value-token buckets; it never silently truncates them.

```bash
uv sync --group dev
uv run pytest -q
uv run python verify-native.py
uv run python convert-coreml.py --trace-only
uv run python convert-coreml.py
```

`verify-native.py` compares the same pinned weights and tokenizer under the upstream cached engine and this toolkit's uncached scoring path. Conversion and Core ML parity must pass before uploading an artifact. No Decision Index or game score is claimed by the conversion alone.

The validated release uses the FP16 L256/B8/V16 package with Core ML `.all`. Nine upstream task cases preserved their selected values, with a maximum candidate likelihood difference of 0.0478. Forced CPU+ANE failed numerical parity, so the publisher stages only the `.all`-validated FP16 package. `publish.py` stages tokenizer, licenses, conversion/runtime source and reports; `--upload` publishes to its separate Hub repo.

An exploratory W8 quantization of the tied token embedding/output matrix reduced the package from 709,563,326 to 642,651,900 bytes. Asymmetric scaling kept the same selected values in 9/9 upstream cases, but its maximum log-likelihood difference was 0.2913, exceeding the existing 0.1 release gate. Symmetric scaling changed one selected value; a 32-column block variant could not be applied to the iOS 17 package because Core ML requires an iOS 18 deployment target for per-block quantization. Forced CPU+ANE still produced an invalid 45,402.8 score error. These variants remain local and are **not published**. See `reports/embedding-w8-linear.json`.
