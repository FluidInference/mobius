---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/blob/deb589d803d141cabd158ef55f6617b128529f36/LICENSE
library_name: coreml
tags:
  - coreml
  - structured-prediction
  - apple-silicon
---

# LFM2.5-350M-RLCD Core ML

This is a fixed-shape Core ML export of the [LFM2.5-350M-RLCD](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD) constrained-decision path. It uses the unchanged 354,483,968-parameter LFM2.5-350M checkpoint and the upstream RLCD prompt/schema renderer. For each allowed value, the model sums the log probabilities of **all** value tokens. The Core ML graph recomputes the prompt for each candidate instead of sharing the upstream hybrid attention/convolution cache; it is therefore a faithful scoring method but not the same compute schedule or speed as the upstream cached implementation.

## Validated artifact

| File | Shape | Bytes | Validated compute units |
| --- | --- | ---: | --- |
| `lfm350_rlcd_fp16_L256_B8_V16.mlpackage` | 256 tokens, 8 candidates per call, 16 value tokens | 709,563,326 | `.all` on the tested Apple Silicon Mac |

On nine selected [upstream RLCD task cases](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD/blob/deb589d803d141cabd158ef55f6617b128529f36/rlcd/tasks.py), Core ML `.all` chose the same values as the native full-value scorer in **9/9** cases; the largest individual candidate log-likelihood difference was **0.0478**. Ten model calls across those cases had a **65.1 ms median after the first call** at B8/L256. This measures model calls only. It is not an official Decision Index score, accuracy estimate, or end-to-end request latency. The 9-candidate routing case needed two model calls.

The FP16 graph **must not be forced to CPU+ANE**: on a selected fixture, that route produced a large likelihood error despite choosing the same values. CPU-only also missed the predeclared 0.5 likelihood-error gate. `.all` passed. A local FP32 reference export also matched the native scorer, but is not included here. See `reports/` for exact inputs, errors, and timings.

The runtime accepts only a closed, flat JSON schema with required string-enum or boolean fields, matching the upstream engine. Inputs over 256 tokens or values over 16 tokens raise an error. More than eight candidates are processed in multiple calls. No input is silently truncated. The conversion does not include unrestricted text generation.

```bash
python -m pip install -r runtime-requirements.txt
python runtime.py lfm350_rlcd_fp16_L256_B8_V16.mlpackage \
  --tokenizer-dir . \
  --context "The invoice has a duplicate charge." \
  --schema schema.json
```

`schema.json` must contain a closed flat schema, for example:

```json
{"type":"object","properties":{"route":{"type":"string","enum":["billing","support"]}},"required":["route"],"additionalProperties":false}
```

## Provenance and license

- Source checkpoint: `notnotsamuel/LFM2.5-350M-RLCD` at `deb589d803d141cabd158ef55f6617b128529f36` (`model.safetensors` SHA256 `1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e`). Its release manifest states these are byte-for-byte copies of `LiquidAI/LFM2.5-350M` weights.
- The model weights are distributed under **LFM Open License v1.0**, included as `LICENSE-LFM`. Its commercial-use grant has a revenue threshold; read the included license before use. The conversion is a modified artifact and the upstream weights are credited here. Upstream RLCD code license is included as `LICENSE-CODE`.
- Conversion source, runtime source, pinned dependencies, and validation reports are included. No claim is made that this Core ML artifact reproduces the tracker’s reported Decision Index value.
