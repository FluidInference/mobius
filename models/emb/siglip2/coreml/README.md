# SigLIP 2 Core ML

Core ML conversion of Google's [SigLIP 2](https://arxiv.org/abs/2502.14786)
image and text encoders, starting with
[`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256)
(375M parameters: 93M image encoder, 282M text encoder of which 197M is the
256k-token embedding table). Apache-2.0, same as upstream.

Both towers are exported, so labels can be written as text at runtime
(zero-shot classification) instead of being fixed at conversion time.

## Packages

| Package | Input | Output |
| --- | --- | --- |
| `siglip2-base-patch16-256-image-fp16.mlpackage` | `pixel_values` float32 `[1, 3, 256, 256]` | `image_embeds` float32 `[1, 768]`, L2-normalized |
| `siglip2-base-patch16-256-text-fp16.mlpackage` | `input_ids` int32 `[1, 64]` | `text_embeds` float32 `[1, 768]`, L2-normalized |

`config.json` records the preprocessing and the scoring constants.

- **Image:** resize to 256×256 (bilinear), scale to [0, 1], then
  `(x - 0.5) / 0.5` per channel.
- **Text:** lowercase, Gemma tokenizer, pad to 64 tokens (`padding="max_length"`).
  SigLIP was trained without an attention mask, so padding must match.
- **Score:** `sigmoid(logit_scale * cos + logit_bias)` gives an independent
  probability per label; `argmax(cos)` gives a single choice. Text embeddings
  depend only on the labels, so compute them once and reuse them for every
  image.

## Usage

```bash
uv sync
uv run python convert-coreml.py                      # build/siglip2-base-patch16-256/
uv run python compare-models.py --limit 1000         # Core ML vs PyTorch on ImageNet-1k
```

`compare-models.py` downloads the ImageNet-1k test split from
[`clip-benchmark/wds_imagenet1k`](https://huggingface.co/datasets/clip-benchmark/wds_imagenet1k)
(about 6.4 GB) and scores zero-shot classification with the prompt
`this is a photo of {class}.`, lowercased.

## Validation

All numbers from an Apple M5 Pro (macOS 27).

**ImageNet-1k zero-shot, all 50,000 test images**, fp16 packages on
CPU + Neural Engine, against the fp32 PyTorch model under the same protocol
([report](reports/imagenet1k-zeroshot-base-256-fp16-ane.json)):

| | PyTorch fp32 | Core ML fp16 |
| --- | ---: | ---: |
| Top-1 | 76.79% | 76.76% |
| Same prediction as PyTorch | — | 99.32% |
| Image embedding cosine, mean / min | — | 0.99990 / 0.975 |
| Text embedding cosine (1,000 class prompts), mean / min | — | 0.99998 / 0.9989 |

Google reports 79.1% for this checkpoint with its own class names and prompts.
The single-prompt protocol here scores lower for both backends; the table
compares Core ML with PyTorch, not with Google's number.

**Latency per call** (batch 1, median):

| Backend | Image encoder | Text encoder |
| --- | ---: | ---: |
| Core ML, CPU + Neural Engine (100% of ops on ANE for image) | 5.2 ms | 1.3 ms |
| Core ML, CPU + GPU | 3.5 ms | 3.0 ms |
| Core ML, CPU only | 17.4 ms | 4.1 ms |
| PyTorch fp32, MPS | 19.1 ms | — |
| PyTorch fp32, CPU | 91.4 ms | — |

## Notes

- The converter warns about fp16 overflow while folding constants; the
  comparison above shows it does not affect outputs.
- `coreml-cli` cannot build a compute plan for the text package (likely the
  256k-row embedding gather), but the package loads and runs on every compute
  unit.
- NaFlex (variable-resolution) checkpoints are not supported: they need dynamic
  shapes, and the SigLIP 2 authors report the fixed-resolution checkpoints are
  better on dense tasks.
