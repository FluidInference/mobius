# ZipVoice / LuxTTS → CoreML

CoreML conversion for [LuxTTS](https://huggingface.co/YatharthS/LuxTTS) (48 kHz
zero-shot voice cloning), a distilled [ZipVoice](https://github.com/k2-fsa/ZipVoice)
(k2-fsa) with a 4-step flow-matching solver and a dual-head 48 kHz Vocos vocoder.
The same pipeline applies to upstream ZipVoice checkpoints (en+zh, dialog) —
identical graphs, different weights/vocoder config.

Requested in FluidAudio issue #49 ([comment](https://github.com/FluidInference/FluidAudio/issues/49#issuecomment-4762131530)).

## Status

| Component | CoreML | Parity (vs torch) | Notes |
|---|---|---|---|
| TextEncoder (Zipformer, 4M) | ✅ fp16, 8.3 MB | cos 0.999997 | fixed 256-token bucket + mask |
| FmDecoder (Zipformer, 119M) | ✅ fp16, 238 MB | cos ≥0.9987/step | fixed 1024-frame bucket + mask |
| Duration expansion + solver | host (Python/Swift) | exact | anchor-Euler, 4 steps, t_shift 0.5 |
| Vocoder (dual-head Vocos 48k) | ⬜ torch for now | — | ISTFT×2 + Linkwitz-Riley crossover; needs DFT-matmul ISTFT |
| Prompt transcription | out of scope | — | upstream uses Whisper; FluidAudio would use Parakeet |
| End-to-end audio | — | log-mel cos 0.99925, RMS match | phase-only waveform divergence |

## Performance (M5 Pro, macOS 26.5)

| Model | CPU | GPU | ANE (98–99% resident) |
|---|---|---|---|
| FmDecoder (per step) | 155 ms | **14.1 ms** | 286 ms |
| TextEncoder | 5.5 ms | 1.4 ms | 4.8 ms |

`all` compute units picks GPU. ~8 s utterance ≈ 5 + 4×14 ≈ **~60 ms** for the
core (≈130× realtime) before vocoder. ANE residency is high but latency is
poor (seq-first layouts / rel-pos attention transposes) — see trials.md.

## Layout

- `coreml/convert_coreml.py` — export TextEncoder + FmDecoder (component split
  mirrors upstream `zipvoice/bin/onnx_export.py`; duration expansion stays host-side)
- `coreml/parity.py` — per-component + end-to-end parity vs torch oracle
- `scripts/reference_infer.py` — pure-torch reference (parity oracle)
- `upstream/` — clones of ysharma3501/LuxTTS and k2-fsa/ZipVoice (gitignored)

## Usage

```bash
uv venv --python 3.11 .venv
# install deps (see pyproject; universal uv sync fights the cu126 pins upstream)
.venv/bin/python scripts/reference_infer.py --prompt-audio <ref.wav>
.venv/bin/python coreml/convert_coreml.py
.venv/bin/python -m coreml.parity
```

## Licenses

LuxTTS & ZipVoice: Apache-2.0. Frontend uses piper-phonemize (espeak-ng
data — GPL-3.0); a FluidAudio integration should reuse the in-house G2P
frontend instead.
