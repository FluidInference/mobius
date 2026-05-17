# Supertonic-3 — CoreML conversion

Hand-port of [Supertone Supertonic-3 v1.7.3](https://huggingface.co/Supertone/supertonic-3)
from ONNX to PyTorch to CoreML. 31 languages, 44.1 kHz, flow-matching
diffusion (8 denoising steps, classifier-free guidance baked into the
ONNX graph via batch-2 duplication).

End-to-end pipeline:

```
text → UnicodeProcessor → token_ids, text_mask
   ├── duration_predictor → duration_sec
   └── text_encoder       → text_emb [B, 256, T]
                          ↓
        sample_noisy_latent(duration_sec) → noisy [B, 144, L], latent_mask
                          ↓
        for 8 steps: vector_estimator(noisy, text_emb, style, masks, step, total)
                          ↓
                       vocoder(denoised_latent) → wav [B, 512*6*L]
```

Audio chunk granularity:
- AE / vocoder frame: 512 / 44100 ≈ **11.6 ms**
- TTL latent slot (model "tick"): 512 × 6 / 44100 ≈ **69.7 ms**

## Layout

```
models/tts/supertonic-3/
├── README.md
├── pyproject.toml          # uv project (Python 3.11, torch + coremltools 8)
└── coreml/
    ├── trials.md           # numerical-parity bug log (4 vector_estimator gotchas)
    ├── __init__.py
    ├── common.py           # ONNX-graph loader utilities (assign_param, etc.)
    ├── text_encoder.py     # PyTorch port: build_text_encoder_from_onnx
    ├── duration_predictor.py
    ├── vector_estimator.py
    ├── vocoder.py
    ├── convert_coreml.py   # PyTorch trace -> .mlpackage for all 4 modules
    ├── validate.py         # ONNX vs PyTorch parity check
    ├── verify_coreml.py    # CoreML vs PyTorch parity check
    ├── infer.py            # end-to-end PyTorch TTS driver (text -> wav)
    └── infer_coreml.py     # end-to-end CoreML TTS driver (text -> wav)
```

## Setup

```bash
cd models/tts/supertonic-3/
uv sync

# Fetch upstream ONNX + style + tokenizer assets
mkdir -p build/_onnx build/voice_styles
HF=https://huggingface.co/Supertone/supertonic-3/resolve/main
for f in text_encoder duration_predictor vector_estimator vocoder; do
    curl -L $HF/_onnx/${f}.onnx -o build/_onnx/${f}.onnx
done
curl -L $HF/_onnx/tts.json -o build/_onnx/tts.json
curl -L $HF/_onnx/unicode_indexer.json -o build/_onnx/unicode_indexer.json
curl -L $HF/voice_styles/M1.json -o build/voice_styles/M1.json
```

## Convert

```bash
# FP32 (numerical reference; ALL modules fall back to CPU on ANE)
uv run python -m coreml.convert_coreml build/_onnx --out-dir build/_mlpackage

# FP16 (required for ANE residency; 3/4 modules land on ANE — see Profile below)
uv run python -m coreml.convert_coreml build/_onnx --fp16 --out-dir build/_mlpackage_fp16

# Fixed-shape VectorEstimator variant for ANE profiling (RangeDim/Enum hit
# ANE shape limits — see trials.md "Dynamic shapes vs ANE"):
uv run python -m coreml.convert_ve_fixed \
    --onnx build/_onnx/vector_estimator.onnx \
    --out  build/_mlpackage_fp16_fixed/VectorEstimator_L128.mlpackage \
    --L 128 --T 128
```

Produces four `.mlpackage` bundles (FP32 ~380 MB, FP16 ~190 MB; mlprogram,
iOS 18+):

| Module             | FP32  | FP16   | Variable axes                             |
| ------------------ | ----- | ------ | ----------------------------------------- |
| vocoder            | 97 MB | 48 MB  | `latent.L_ttl` = RangeDim(4..512)         |
| text_encoder       | 35 MB | 17 MB  | fixed `text.T = 128`                      |
| duration_predictor | 3.5 MB| 1.8 MB | fixed `text.T = 128`                      |
| vector_estimator   | 244 MB| 122 MB | `latent.L` & `text.T` = RangeDim(17..512) |

## Validate

```bash
# ONNX vs PyTorch port (per module)
uv run python -m coreml.validate

# CoreML vs PyTorch port (per module)
uv run python -m coreml.verify_coreml

# End-to-end PyTorch (writes WAV)
uv run python -m coreml.infer \
    --onnx-dir build/_onnx \
    --voice-style build/voice_styles/M1.json \
    --text "Hello world."

# End-to-end CoreML (writes WAV)
uv run python -m coreml.infer_coreml \
    --mlpackage-dir build/_mlpackage \
    --tts-json build/_onnx/tts.json \
    --unicode-indexer build/_onnx/unicode_indexer.json \
    --voice-style build/voice_styles/M1.json \
    --text "Hello world."
```

Final parity vs ONNX-Runtime CPU:

| Module             | PyTorch vs ONNX max_abs | CoreML vs PyTorch max_abs |
| ------------------ | ----------------------- | ------------------------- |
| vocoder            | 2.53e-4                 | 1.41e-6                   |
| text_encoder       | 9.77e-2 (relaxed tol)   | 2.33e-4                   |
| duration_predictor | 3.04e-6                 | 3.82e-6                   |
| vector_estimator   | 1.21e-3                 | 2.96e-5                   |

End-to-end CoreML on M-series CPU+ANE: **~0.74 s** to synthesize
6.32 s of audio for a single English sentence (RTFx ≈ 8.5x), 8
denoising steps. ASR-verified against FluidAudio Parakeet TDT.

## Profile (FP16, Apple M2, macOS 26.5, `cpu_and_neural_engine`)

| Module                              | CPU% | GPU% | ANE% | Predict | Notes |
| ----------------------------------- | ---- | ---- | ---- | ------- | ----- |
| duration_predictor                  | 100  | 0    | 0    | 0.82 ms | tiny, CPU-bound |
| text_encoder (T=128)                | 38   | 0    | 62   | 2.15 ms | partial ANE |
| vocoder (RangeDim L 4..512)         | 0    | 0    | 100  | 1.17 ms | full ANE, 4× vs FP32 |
| vector_estimator (RangeDim 17..512) | —    | —    | —    | —       | ANE compile fails (dynamic shapes disabled) |
| vector_estimator (fixed L=128 T=128)| 100  | 0    | 0    | 8.32 ms | 89.6% ops *eligible*, blocked by 1 bool `tile` op |

See `coreml/trials.md` → "ANE residency profiling" for the full breakdown,
the bool-tile blocker, and the EnumeratedShapes runtime stride gotcha.

## Critical gotchas

See `coreml/trials.md` for the full log. Highlights:

1. **CFG via batch-2 duplication** — the ONNX vector_estimator tiles
   inputs to batch=2, runs cond + uncond in parallel, then combines
   with `(noisy + (1/total)*(4*cond - 3*uncond)) * mask`. The cond
   style key is **not** the user `style_ttl` — it is a learned
   initializer at `/vector_estimator/Expand_output_0`.
2. **Rotary is length-normalized** — `angles = (pos / sum(mask)) * theta`,
   divisor differs for Q (latent_mask) and K (text_mask).
3. **Attention divisor is 16.0**, not `sqrt(dk)=8`. Off-by-2x in scoring.
4. **Style attention applies `tanh(K)`** before the score matmul; text
   attention does not.
5. **Replicate-pad lower bound** — ConvNeXt depthwise pads scale with
   dilation: `pad = (K-1)*D/2`. CoreML enforces `pad ≤ dim-1` at load
   time, hence `RangeDim.lower_bound = 17` for vector_estimator and
   `4` for vocoder.
6. **int32 vs int64 tokens** — CoreML wants int32, PyTorch indexes int64.
   Wrap modules with a tiny `_Int32Wrapper` that casts inside the
   traced graph so the external input stays int32.
7. **Python 3.14 has no BlobWriter** — pin `requires-python = ">=3.11,<3.13"`.

## Upstream + downstream

- Upstream: <https://huggingface.co/Supertone/supertonic-3>
- Reference Python driver: <https://github.com/supertone-inc/supertonic/blob/main/py/helper.py>
- Republished CoreML: `FluidInference/supertonic-3-coreml` (HuggingFace)
- FluidAudio Swift integration: `Sources/FluidAudio/TTS/Supertonic3/`
