# PocketTTS Voice Cloning for CoreML

Export custom voices for use with FluidAudio's PocketTTS Swift implementation.

## Quick Start

```bash
cd mobius/models/tts/pocket_tts

# Install dependencies
pip install torch safetensors torchaudio

# Export a voice from audio file
python coreml/export_voice_coreml.py your_voice.wav -o constants_bin/custom_audio_prompt.bin

# The output file can be used with FluidAudio
```

## Full Workflow: Export → Test → Evaluate

```bash
# One command does it all: export voice, run CoreML inference, evaluate
python coreml/test_voice_coreml.py \
    --reference speaker.wav \
    --text "Hello, this is a voice cloning test." \
    --output test_output.wav \
    --evaluate
```

This will:
1. Export the voice from `speaker.wav` → `speaker_audio_prompt.bin`
2. Run CoreML TTS with the exported voice
3. Evaluate spectral similarity against reference

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
   pip install torch safetensors torchaudio
   ```

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
