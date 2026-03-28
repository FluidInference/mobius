# KittenTTS CoreML Conversion

## Status: Complete

KittenTTS Nano is a distilled Kokoro/StyleTTS2 model (15M params, 24kHz) that ships as ONNX-only (INT8 quantized).
Successfully converted to CoreML by reconstructing the PyTorch model from the ONNX graph, dequantizing weights, and tracing.

## Architecture

| Component | KittenTTS Nano | Kokoro-82M |
|-----------|---------------|------------|
| BERT embedding dim | 128 | 768 |
| BERT hidden (ALBERT) | 768 | 768 |
| bert_encoder output | 128 | 768 |
| Style dim | 128 | 128 |
| Generator channels | 256->128->64 | 512->256->128 |
| Total params | 15M | 82M |
| Source format | ONNX INT8 | PyTorch |

## Usage

```bash
cd coreml
python convert_kittentts.py --seconds 5 --output kittentts_5s.mlpackage
```

See [coreml/README.md](coreml/README.md) for full documentation.
