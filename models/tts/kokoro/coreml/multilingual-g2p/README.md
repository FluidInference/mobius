# CharsiuG2P ByT5 — Multilingual G2P CoreML Conversion

Converts [charsiu/g2p_multilingual_byT5_tiny_16_layers_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100) to CoreML for on-device grapheme-to-phoneme conversion in 100+ languages.

Pre-converted models: [FluidInference/charsiu-g2p-byt5-coreml](https://huggingface.co/FluidInference/charsiu-g2p-byt5-coreml)

## Quick start

```bash
uv sync
uv run python convert-to-coreml.py
uv run python compile-modelc.py
```

## Architecture

ByT5 encoder-decoder with byte-level tokenization (no separate tokenizer vocab — raw UTF-8 bytes + 3 offset).

- **Encoder**: input_ids (1, seq_len) + attention_mask (1, seq_len) → hidden states (1, seq_len, 768)
- **Decoder**: autoregressive, takes decoder_input_ids + encoder output → logits over byte vocab

Conversion path: PyTorch → ONNX (`torch.onnx.export`) → CoreML (`coremltools 7.x`).

## Outputs

| File | Description | Size |
|------|-------------|------|
| `G2PEncoder.mlpackage` | ByT5 encoder | ~55 MB |
| `G2PDecoder.mlpackage` | ByT5 decoder + lm_head | ~25 MB |
| `g2p_config.json` | Token IDs, sequence length limits | <1 KB |

## Kokoro TTS languages

The model supports 100+ languages. These 9 are used by Kokoro TTS voices:

| Code | Language | Kokoro voices |
|------|----------|---------------|
| `eng-us` | American English | af, am |
| `eng-uk` | British English | bf, bm |
| `spa` | Spanish | ef, em |
| `fra` | French | ff, fm |
| `ita` | Italian | if, im |
| `por-bz` | Brazilian Portuguese | pf, pm |
| `hin` | Hindi | hf, hm |
| `jpn` | Japanese | jf, jm |
| `cmn` | Mandarin Chinese | zf, zm |

## Input format

Words are prefixed with a language tag:

```
<eng-us>: hello    → h ə l oʊ
<fra>: bonjour     → b ɔ̃ ʒ u ʁ
<jpn>: 東京        → t o ː kʲ o ː
```

## Performance

~30 ms/word on Apple Silicon (CPU compute unit). CPU-only is fastest for this model due to GPU/ANE dispatch overhead on small autoregressive decoder steps.

## License

Source model: [Apache 2.0](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100)
