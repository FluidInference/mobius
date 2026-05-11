# PocketTTS Voice Cloning for CoreML

Export custom voices for use with FluidAudio's PocketTTS Swift implementation.

> **Two equivalent paths** as of 2026-05-11:
>
> - `export_voice_v2.py` — PyTorch pipeline, bakes a v2 KV-cache
>   `.safetensors` (drop-in for `build/<lang>/constants_bin/`). Requires
>   `pip install pocket-tts==2.0.0`.
> - `export_voice_coreml.py` — pure CoreML pipeline, writes a v1
>   `<voice>_audio_prompt.bin` consumed by `generate_coreml_v4.py`'s v1
>   `cond_step` prefill path. Requires the retraced
>   `mimi_encoder.mlpackage` produced by
>   `convert_models/convert/convert_mimi_encoder.py`.
>
> Both produce intelligible speech against the deployed Apr 27 cond_step /
> flowlm_step / flow_decoder mlpackages. Pick `export_voice_v2.py` when you
> want PyTorch parity (e.g. matching deployed voice caches bit-for-bit) and
> `export_voice_coreml.py` when you want a Python toolchain without
> PyTorch + pocket-tts installed.
>
> Historical note (FluidAudio #592): the originally shipped
> `mimi_encoder.mlmodelc` was traced against a pre-2.0.0 pocket-tts and
> produced conditioning in the wrong latent space, causing immediate EOS.
> The new mlpackage built from `convert_mimi_encoder.py` fixes that, and
> the v1 `generate_coreml_v4.py` cond_step path now also prepends the
> required `bos_before_voice` token.

## Quick Start

```bash
cd mobius/models/tts/pocket_tts

# Install pocket-tts 2.0.0 (must match the deployed mlpackage trace)
pip install "pocket-tts==2.0.0" safetensors torch

# Bake a v2 KV-cache voice file (drop-in for build/<lang>/constants_bin/)
python coreml/voice_cloning/export_voice_v2.py \
    your_voice.wav \
    -o build/english/constants_bin/your_voice.safetensors
```

## Full Workflow: Export → Test → Evaluate

```bash
# 1. Bake voice as v2 KV-cache.
python coreml/voice_cloning/export_voice_v2.py speaker.wav \
    -o build/english/constants_bin/speaker.safetensors

# 2. Run the CoreML pipeline against the freshly baked voice.
python coreml/generate_coreml_v4.py \
    --language english --voice speaker \
    --text "Hello, this is a voice cloning test." \
    --output test_output.wav

# 3. (Optional) Speaker similarity vs reference.
python coreml/voice_cloning/evaluate_voice.py speaker.wav test_output.wav
```

## Testing Exported Voices

```bash
# Test with pre-exported .bin file
python coreml/test_voice_coreml.py \
    --voice custom_audio_prompt.bin \
    --text "Testing my custom voice"

# Test with .safetensors file
python coreml/test_voice_coreml.py \
    --voice alba.safetensors \
    --text "Testing alba voice"
```

## Requirements

1. **PocketTTS model with voice cloning** - Accept terms at https://huggingface.co/kyutai/pocket-tts then:
   ```bash
   huggingface-cli login
   ```

2. **Python dependencies**:
   ```bash
   pip install "pocket-tts==2.0.0" safetensors torch
   ```

   `pocket-tts` 2.0.0 (released 2026-04-21) is the exact library that traced
   the deployed `cond_step.mlpackage` / `flowlm_step.mlpackage` /
   `flow_decoder_fused8.mlpackage`. Newer 2.x versions should also work; the
   1.x line is incompatible (different HF revision + repo path layout).

## Pure-CoreML export with `export_voice_coreml.py`

The mlpackage shipped originally as `mimi_encoder.mlmodelc` was traced from a
pre-2.0.0 pocket-tts revision and produced conditioning in a latent space
incompatible with the deployed cond_step (FluidAudio #592 — immediate EOS).
Re-trace the encoder against pocket-tts 2.0.0 first:

```bash
# Bake a fresh mlpackage from pocket-tts 2.0.0 weights into
# coreml/build/<lang>/mimi_encoder.mlpackage and symlink it into voice_cloning/.
python coreml/convert_models/convert/convert_mimi_encoder.py --language english
ln -sfn ../build/english/mimi_encoder.mlpackage \
    coreml/voice_cloning/mimi_encoder.mlpackage
```

After that, `export_voice_coreml.py` writes a v1
`<voice>_audio_prompt.bin` that pairs with the patched
`generate_coreml_v4.py` v1 prefill path (which now prepends the required
`bos_before_voice` token).

## Usage

### Single Voice Export

```bash
# Export with auto-naming (uses input filename)
python coreml/export_voice_coreml.py voice.wav --output-dir ./constants_bin/
# Creates: constants_bin/voice_audio_prompt.bin

# Export with custom name
python coreml/export_voice_coreml.py recording.mp3 --name speaker1 --output-dir ./constants_bin/
# Creates: constants_bin/speaker1_audio_prompt.bin

# Export to specific file
python coreml/export_voice_coreml.py voice.wav -o my_voice_audio_prompt.bin
```

### Batch Export

```bash
# Export all audio files in a directory
python coreml/export_voice_coreml.py ./voices/ --output-dir ./constants_bin/
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output` | Output .bin file path | - |
| `--output-dir` | Output directory | Current dir |
| `--name` | Voice name for filename | Input filename |
| `--config` | Model config variant | `610b0b2c` |
| `--device` | cpu or cuda | `cpu` |
| `--frames` | Conditioning length | 125 |
| `--safetensors` | Also save .safetensors | False |

## Audio Guidelines

For best results:
- **Duration**: 5-30 seconds of speech
- **Quality**: Clean audio, minimal background noise
- **Content**: Clear speech, natural prosody
- **Format**: WAV, MP3, FLAC, M4A, OGG supported

The script automatically:
- Resamples to 24kHz mono
- Truncates to 30 seconds (configurable)
- Pads/truncates to 125 frames

## Using with FluidAudio (Swift)

1. Copy the exported `{voice}_audio_prompt.bin` to your app's model directory
2. Update the voice list in your app or use the custom voice API:

```swift
// Load custom voice
let voiceData = try store.voiceData(for: "custom")

// Use in synthesis
let result = try await PocketTtsSynthesizer.synthesize(
    text: "Hello world",
    voice: "custom"
)
```

## Output Format

The `.bin` file contains:
- Raw Float32 values (little-endian)
- Shape: `[125, 1024]` flattened to `[128000]` floats
- Size: ~500 KB per voice

This matches FluidAudio's `PocketTtsConstantsLoader` expectations.

## Troubleshooting

### "Voice cloning unsupported" error
Accept the model terms at https://huggingface.co/kyutai/pocket-tts and login:
```bash
huggingface-cli login
```

### Out of memory
Use CPU device (default) or reduce audio length:
```bash
python coreml/export_voice_coreml.py voice.wav --device cpu
```

### Poor voice quality
- Ensure clean audio without background noise
- Use at least 5 seconds of speech
- Avoid audio with multiple speakers

## Evaluating Voice Quality

Use `evaluate_voice.py` to measure speaker similarity using neural embeddings:

```bash
pip install resemblyzer  # Required

python coreml/evaluate_voice.py reference_speaker.wav tts_output.wav
```

**Output:**
```
Reference:   reference_speaker.wav
Synthesized: tts_output.wav

Reference duration:   5.23s
Synthesized duration: 3.45s

Computing speaker similarity...

  Speaker Similarity: 0.8234
  Quality:            Good
```

### Why Speaker Embeddings?

Neural speaker embeddings (Resemblyzer) measure "is this the same person?" by:
- Extracting voice characteristics independent of content
- Using models trained on millions of speaker pairs
- Working even when saying completely different words

Unlike spectral similarity which is affected by what words are spoken.

### Quality Thresholds

| Score | Quality | Meaning |
|-------|---------|---------|
| 0.85+ | Excellent | Very close voice match |
| 0.75+ | Good | Clearly same speaker |
| 0.65+ | Fair | Some similarity |
| <0.65 | Poor | Different speaker characteristics |

### Visual Comparison

```bash
python coreml/evaluate_voice.py reference.wav synthesized.wav --plot
```

Generates `speaker_comparison.png` showing embedding comparison
