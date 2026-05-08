# StyleTTS 2 — CoreML conversion

This directory holds the fresh CoreML conversion effort for the
[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) LibriTTS
multi-speaker model.

The PyTorch ground-truth inference (`run_inference.py`) is the parity
reference every conversion stage will be validated against.

## Layout

```
models/tts/styletts2/
├── README.md
├── pyproject.toml          # uv project for ground-truth inference
├── run_inference.py        # PyTorch end-to-end ground truth (CPU, fp32)
├── scripts/
│   └── bootstrap.py        # one-shot fetch: vendor + checkpoint + ref audio
└── (gitignored)
    ├── vendor/StyleTTS2/                 # upstream repo clone
    ├── checkpoints/LibriTTS/             # yl4579/StyleTTS2-LibriTTS weights
    │   ├── config.yml
    │   └── epochs_2nd_00020.pth          # ~771 MB
    ├── reference_audio/                  # voice references for cloning
    └── *.wav                             # generated outputs
```

## Setup

```bash
brew install espeak-ng                    # macOS phonemizer backend
cd models/tts/styletts2
uv sync                                   # install ground-truth deps
uv run python scripts/bootstrap.py        # clone vendor + download model
```

`bootstrap.py` is idempotent and skips any asset already present.

## Run ground-truth inference

```bash
uv run python run_inference.py \
    --text "StyleTTS 2 is a text to speech model." \
    --reference reference_audio/696_92939_000016_000006.wav \
    --output out.wav \
    --seed 0
```

Defaults: 5-step ADPM2 diffusion, alpha=0.3, beta=0.7, embedding_scale=1.0,
seed=0. Output is 24 kHz mono float64 WAV. CPU-only (the diffusion sampler
hits known issues on MPS).

Reference timing on M-series CPU, cold: ~13–19 s for ~8 s of audio
(RTF ≈ 1.6–2.3).

## Ground-truth contract

For parity testing, fix:
- `--seed 0`
- text from a small fixed set (committed in `parity_texts.txt` once we
  start writing parity tests)
- reference audio from `reference_audio/` (e.g. `696_92939_000016_000006.wav`)
- diffusion steps = 5

Outputs of the CoreML pipeline are compared to `run_inference.py` on:
- log-mel cosine similarity
- ECAPA-TDNN speaker-embedding cosine vs reference

## License

The yl4579 LibriTTS weights ship with non-permissive consent / disclosure
terms in the upstream README. Treat any converted artifacts as
**local-only** — do not redistribute without reading upstream's license.
