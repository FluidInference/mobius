# Conversion Guide

This note documents the current, working path for exporting the `pyannote/speaker-diarization-community-1` pipeline to Core ML. The guidance reflects what the checked-in tooling (`convert-coreml.py`, `coreml_wrappers.py`, `compare-models.py`) actually does today.

## Pipeline Components & Shapes

- **Segmentation** – `pyannote.audio.models.segmentation.PyanNet` with a 10 s receptive field (160 000 waveform samples at 16 kHz). The traced Torch module emits `log_probs` with shape `(batch, 589, 7)`: 589 frames per chunk and seven powerset classes that encode up to two concurrent speakers.
- **Speaker embedding** – `pyannote.audio.models.embedding.wespeaker.WeSpeakerResNet34`. We trace it with the same 10 s window so the Core ML bundle expects `(1, 1, 160 000)` audio samples plus a `(1, 589)` weight vector aligned with the segmentation frames. The wrapper interpolates weights internally to match the pooling layer and applies L2 normalisation before returning the 256‑dim embedding.
- **PLDA transforms** – `plda-community-1.mlpackage` (x‑vector transform + PLDA projection) and `plda_rho-community-1.mlpackage` (same transform plus `sqrt(phi)` scaling for VBx). Both accept `(batch, 256)` embeddings and export `(batch, 128)` tensors.
- **Auxiliary resources** – `convert-coreml.py` also emits JSON payloads in `coreml_models/resources/` that mirror the PLDA tensors for runtimes that still consume NumPy-style arrays.

For the diarization pipeline configuration see `pyannote-speaker-diarization-community-1/config.yaml` (`VBxClustering`, `Fa=0.07`, `Fb=0.8`, `threshold=0.6`).

## Environment Setup

The target environment is Python 3.10.12 with dependencies pinned in this directory’s `pyproject.toml`/`uv.lock` (Torch 2.4, Pyannote Audio 4.0, Core ML Tools 7.2, NumPy < 2, SciPy ≥ 1.10).

```bash
# Set local caches so the sandbox stays inside the repo tree
export UV_CACHE_DIR="$(pwd)/.uv-cache"
export MPLCONFIGDIR="$(pwd)/.matplotlib"

uv sync
```

`uv sync` creates `.venv/` and installs the exact versions required by the conversion scripts and comparison tooling.

## Running the Conversion Script

`convert-coreml.py` is the single source of truth for tracing and exporting all four Core ML bundles and the JSON resources.

```bash
uv run python convert-coreml.py \
  --model-root ./pyannote-speaker-diarization-community-1 \
  --output-dir ./build/coreml \
  --selective-fp16          # optional, see below
```

Key behaviours:

- **Segmentation**: traced with a batch of 32 windows and exported with `ct.EnumeratedShapes` so the Core ML bundle supports batches from 1 to 32. The wrapper (`CoreMLSegmentationModule`) still feeds one chunk per call because that is what mobile currently expects; switching to batched prediction requires downstream work.
- **Embedding**: exported as two bundles — `fbank` handles `(batch, 1, 160000)` audio on the CPU and produces `(batch, 1, 80, 998)` FBANK features, while `embedding` consumes those features plus `(batch, 589)` weights. Core ML still limits multi-input ML Programs to fixed leading dimensions on iOS 17, so batching happens in Python/Swift via loops.
- **PLDA / PLDA rho**: traced with batch 32 tensors and exported with enumerated shapes 1–32. Both bundles bake in the original float64 parameters (converted to float32 inside the model) and remove the need for ad‑hoc JSON decoding.
- **Resources**: `xvector-transform.json` and `plda-parameters.json` retain the original dtype of every tensor. Consumers can trust the metadata written by `convert_resource`.

The CLI prints the paths of every generated artefact so you can copy the `.mlpackage` bundles to `coreml_models/` or an external build directory.

## Selective FP16 Conversion

Pass `--selective-fp16` to request Core ML Tools’ `FP16ComputePrecision`. In that mode `convert-coreml.py` keeps numerically sensitive operations in FP32 by using the `skip_sensitive_ops_*` selectors defined near the top of the script:

- **Segmentation** keeps `log`, `softmax`, normalisation layers, reductions, and divisions in FP32 to protect the log-probabilities.
- **Embedding** keeps all arithmetic, normalisation, reductions, matrix multiplications, and logarithmic/exponential ops in FP32. The fbank frontend and stats pooling are particularly sensitive in FP16.
- **PLDA** defaults to standard FP16 compute precision (the math is mostly affine) and continues to match the NumPy reference at 1e‑6 absolute error.

The compare script (`uv run python compare-models.py --coreml-dir ./build/coreml --audio-file ./yc_first_10s.wav`) reports per-chunk metrics, so run it after every conversion toggle. Expect correlations ≥ 0.999 and cosine similarities ≥ 0.99 when the selective filters are left untouched. If a model diverges, log the offending op types by printing inside `skip_sensitive_ops_*` and expand the FP32 list.

## Wrapper Expectations

The runtime wrappers in `coreml_wrappers.py` define how mobile clients should load and execute the bundles:

- `CoreMLSegmentationModule` runs one chunk at a time, logging shapes to help diagnose batching mistakes. It keeps the Pyannote prototype metadata (`receptive_field`, `audio`, `specifications`) so the rest of the pipeline behaves identically to the Torch version.
- `CoreMLEmbeddingModule` runs the FBANK frontend on the CPU, interpolates weights with SciPy’s `ndimage.zoom` (order‑1) when they do not match the expected frame count, and applies L2 normalisation before returning embeddings. Consumers must provide the `(batch, 589)` overlap weights emitted by segmentation.
- `wrap_pipeline_with_coreml(...)` replaces both modules on a `Pipeline` instance and keeps PLDA resources on disk so Swift can re-use them.

The wrappers trace every Core ML call, which `compare-models.py` relies on to benchmark CPU/GPU/ANE execution. Keep those logs intact; they are the fastest way to debug on-device inputs.

## Verification Checklist

1. Convert models (optionally with `--selective-fp16`).
2. Run `compare-models.py` against a representative WAV (`yc_first_10s.wav` or `longconvo-30m-last5m.wav`). Inspect the JSON report for:
   - Segmentation `mae`/`rmse` < 1e‑3.
   - Embedding cosine similarities ≥ 0.99.
   - PLDA rho parity (scores differ < 1e‑6).
3. Run `test.py` to ensure the diarization pipeline still obtains DER/JER within thresholds on AMI Mix-Headset.
4. Package the `.mlpackage` bundles under `coreml_models/` (or copy them into the iOS project) along with the JSON resources if legacy code requires them.

## Troubleshooting Notes

- **Sandboxed caches**: set `UV_CACHE_DIR` and `MPLCONFIGDIR` before running `uv` commands to avoid permission errors.
- **Missing Core ML models**: `wrap_pipeline_with_coreml` looks for `segmentation-community-1.mlpackage`, `fbank-community-1.mlpackage`, and `embedding-community-1.mlpackage` next to the wrapper. Name the bundles exactly or adjust the paths before shipping.
- **Non-normalised embeddings**: the raw PyTorch model returns vectors with norms ≈0.75. The Core ML wrapper adds L2 normalisation; ensure downstream NumPy checks also normalise before PLDA when calling `load_plda_pipeline_from_npz` directly.
- **Neural Engine tuning**: if a new hardware target requires more aggressive FP16 usage, start by relaxing the segmentation selector (it is the more tolerant model). The embedding selector should stay conservative until you re-run the correlation benchmarks.

Keep this guide close when refreshing the Core ML artefacts; any deviations from these steps should be documented here so the repo remains the canonical reference for Community-1 conversions.
