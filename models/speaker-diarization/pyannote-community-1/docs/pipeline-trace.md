# Pipeline Trace (Speaker Diarization Community-1)

This note walks through the full execution path triggered by `test.py` so we can reimplement the diarization pipeline in another language/runtime. It documents where each component lives on disk, how the YAML config is expanded, and the runtime steps executed by `pyannote.audio.pipelines.SpeakerDiarization`.

## 1. Entry Point

- Script: `test.py`
- Call stack: `Pipeline.from_pretrained(PIPELINE_CONFIG)` → instantiated pipeline → `pipeline({"waveform": waveform, "sample_rate": sample_rate})`.
- Pipeline config: `pyannote-speaker-diarization-community-1/config.yaml`.

`Pipeline.__call__` (pyannote 4.x core) ensures the object is instantiated, validates the file dict, and delegates to `SpeakerDiarization.apply` with the audio payload.

## 2. Config Expansion

`Pipeline.from_pretrained` (pyannote.audio/core/pipeline.py)

1. Treats the provided path as a local checkpoint and loads YAML.
2. Recursively rewrites `$model/<subfolder>` entries into dictionaries:

   ```yaml
   segmentation: {"checkpoint": model_id, "subfolder": "segmentation", "token": token, "cache_dir": cache_dir}
   embedding:   {"checkpoint": model_id, "subfolder": "embedding",   ...}
   plda:        {"checkpoint": model_id, "subfolder": "plda",         ...}
   ```

3. Imports `pyannote.audio.pipelines.SpeakerDiarization` and instantiates with the `params` block plus the expanded descriptors.
4. Calls `pipeline.instantiate(config["params"])`, freezing defaults such as:

   ```yaml
   params:
     clustering: {threshold: 0.6, Fa: 0.07, Fb: 0.8}
     segmentation: {min_duration_off: 0.0}
   ```

## 3. Pipeline Initialization

Inside `SpeakerDiarization.__init__` (pyannote/audio/pipelines/speaker_diarization.py):

| Component | Source | Notes |
|-----------|--------|-------|
| Segmentation model | `get_model(segmentation)` | Loads 10 s powerset segmentation network, wraps it in `Inference` with configurable stride and batch size.
| Embedding extractor | `PretrainedSpeakerEmbedding(embedding)` | Loads 5 s Wespeaker backbone, sets up audio resampling + batching; respects `embedding_exclude_overlap` flag.
| PLDA parameters | `get_plda(plda)` | Loads VBx PLDA assets (means, transforms) used during clustering.
| Clustering backend | `Clustering["VBxClustering"]` | Builds VBx variational Bayes instance seeded by a centroid-linkage Agglomerative Hierarchical Clustering (AHC) pass and configured with PLDA + cosine metric from embeddings.

Key attributes produced:

- `_segmentation`: inference helper over the segmentation network.
- `_embedding`: batched embedding extractor; `_audio` handles cropping waveforms for each diarization chunk.
- `clustering`: VBx algorithm configured with thresholds (`Fa`, `Fb`, `threshold`) from `config.yaml`.

## 4. Runtime Flow (`apply`)

The `apply` method receives an `AudioFile` dict (`{"waveform": Tensor, "sample_rate": int}`) and optional speaker-count hints. The main steps are below; each can be ported independently.

1. **Instantiate hook & speaker bounds**
   - Normalizes `num_speakers`, `min_speakers`, `max_speakers` using `set_num_speakers`.
   - VBx requires `num_speakers` when the clustering backend expects a fixed count; Community-1 uses VBx with free cluster count, so the hint is optional.

2. **Segmentation Inference**
   - `_segmentation(file)` runs the 10 s sliding-window model over the waveform.
   - Pyannote batches chunks internally via `Inference`. Our Core ML wrapper mirrors the upstream behaviour but still loops over chunks and calls `mlmodel.predict` with `(1, 1, 160000)` tensors to keep memory predictable.
   - Output shape: `(num_chunks, 589, local_num_speakers)` where `local_num_speakers` is the powerset size (7) for Community-1.
   - Hook label: `"segmentation"`.

3. **Binarization**
   - If the model is not in `powerset` mode, apply `binarize` with the learned threshold (`0.6` default) to obtain frame-level speech activity per local speaker stream.

4. **Speaker Counting**
   - `speaker_count` runs on the binarized masks with the network receptive field, producing `count` (frames × 1) with the instantaneous speaker count.
   - Used to cap cluster assignments and produce exclusive diarization later.

5. **Early Exit**
   - If `max(count) == 0`, return empty `Annotation`s and zero embeddings.

6. **Embedding Extraction**
   - `get_embeddings` iterates over each `(chunk, local speaker)` pair:
     - Crops waveform segments using the embedding inference helper (`PretrainedSpeakerEmbedding`).
     - Applies speech masks; optionally removes overlapping frames based on `embedding_exclude_overlap` (true in Community-1).
     - Batches into size `embedding_batch_size` (32 from config) and runs `_embedding` to get 256‑dim vectors.
   - The Core ML wrapper currently mirrors this batching but still performs one predict call per chunk because the exported model takes fixed `(1, 1, 160000)` audio plus `(1, 589)` weights.
   - Output shape: `(num_chunks, local_num_speakers, 256)`.

7. **AHC → VBx Clustering**
   - `self.clustering(...)` first normalizes the filtered embeddings and runs a centroid-linkage AHC step to produce initial clusters before handing them to the VBx refinement loop.
   - The VBx stage (`cluster_vbx`) consumes the AHC assignments, PLDA-transformed embeddings, and Fa/Fb hyperparameters to iteratively update speaker posteriors.
   - Returns `hard_clusters` (cluster ID per `(chunk, speaker)`) and `centroids` (global speaker embeddings), optionally re-running KMeans when a fixed speaker count is requested.
   - Applies Fa/Fb penalties and oracle logic tied to the VBx reference implementation.

8. **Discrete Diarization Reconstruction**
   - Assigns inactive speakers to a throwaway cluster (`-2`).
   - `reconstruct` aggregates chunk-level masks into a continuous timeline using cluster IDs and the frame-level speaker counts.

9. **Annotation Construction**
   - `to_annotation` converts discrete masks to `Annotation` objects with `min_duration_off` from config.
   - Two passes:
     - **`speaker_diarization`**: uses the original speaker count → overlaps preserved.
     - **`exclusive_speaker_diarization`**: clamps `count` to 1 → forced one-speaker-per-frame timeline (better for ASR alignment).

10. **Label Mapping & Output**
    - If a reference annotation is packaged with `file`, performs optimal mapping; otherwise generates sequential IDs (`SPEAKER_00`, ...).
    - Packs `speaker_diarization`, `exclusive_speaker_diarization`, and `speaker_embeddings` (ordered like `speaker_diarization.labels()`) into `DiarizeOutput` dataclass.

## 5. Reimplementation Checklist

Use the table below when porting. Every row corresponds to a Python step that must be mirrored in another language.

| Stage | Python Source | Responsibilities | Porting Notes |
|-------|---------------|------------------|---------------|
| Config loader | `Pipeline.from_pretrained` | Parse YAML, resolve `$model/...`, hydrate params, set defaults. | Implement YAML parser, support relative asset paths, replicate parameter instantiation rules. |
| Segmentation | `Inference(model, duration, step, batch_size)` | Slide 10 s window, run NN, stitch logits. | Core ML/TorchScript can provide raw logits. Make sure overlap (step 0.1) matches; implement stitching if not handled by exporter. |
| Binarization | `binarize(segmentations, threshold)` | Convert soft scores to 0/1 masks. | Simple threshold + hysteresis (optional). Must run per local speaker stream. |
| Speaker count | `speaker_count` | Estimate active speakers per frame. | Uses convolution with receptive field; port the logic or precompute offline if acceptable. |
| Embeddings | `PretrainedSpeakerEmbedding.__call__` | Crop audio, mask frames, produce 256-dim embeddings. | Ensure identical window length, normalization, and overlap exclusion. Core ML embedding block + host-side batching. |
| VBx clustering | `VBxClustering.__call__` | Variational Bayes, PLDA scoring, hyperparameters Fa/Fb. | Rewrite in target language (Swift/Accelerate, C++). Needs double precision linear algebra and iterative convergence. |
| Reconstruction | `reconstruct` + `to_annotation` | Convert clustered masks to time spans. | Implement timeline merging with `min_duration_off` smoothing and label mapping. |
| Output struct | `DiarizeOutput` | Return diarization timelines, exclusive timeline, speaker embeddings. | Design equivalent struct/class; keep serialization schema for parity. |

## 6. Guidance for Other Languages

- Keep compute-heavy steps (segmentation, embedding) inside converted neural bundles (Core ML, TensorRT, etc.).
- Implement VBx in a systems language with efficient BLAS (Accelerate on Apple, Eigen/ATLAS on C++). Reuse existing VBx reference if licensing allows.
- Serialize config parameters (thresholds, Fa/Fb, window sizes) to a shared JSON so mobile/desktop ports load the exact values.
- Match torch’s float32 precision for clustering; avoid automatic FP16 downcasts in VBx logic.
- Validate every stage individually: segmentation logits vs. PyTorch, embedding cosine similarity, final diarization DER on reference clips.

## 7. Useful File References

- Test harness: `test.py`
- Pipeline config: `pyannote-speaker-diarization-community-1/config.yaml`
- Segmentation weights: `pyannote-speaker-diarization-community-1/segmentation/pytorch_model.bin`
- Embedding weights: `pyannote-speaker-diarization-community-1/embedding/pytorch_model.bin`
- PLDA assets: `pyannote-speaker-diarization-community-1/plda/*`
- Source for runtime logic: `.venv/lib/python3.10/site-packages/pyannote/audio/pipelines/speaker_diarization.py`

Use this document alongside `docs/conversion-guide.md` and `docs/batch-processing-optimization.md` when planning a port or embedding the pipeline in a new runtime.

## 8. Instrumented Python Trace

The snippet below instruments `Pipeline.from_pretrained(...).apply` with a hook so each major step in `SpeakerDiarization` reports its intermediate tensor shapes. Running it with `uv` keeps the existing virtual environment and produces deterministic output for the bundled 10 s sample clip.

```bash
UV_CACHE_DIR=$(pwd)/.uv-cache \
MPLCONFIGDIR=$(pwd)/.matplotlib \
uv run python - <<'PY'
from pathlib import Path
import numpy as np
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation

root = Path.cwd()
config_path = root / "pyannote-speaker-diarization-community-1" / "config.yaml"
audio_path = root / "yc_first_10s.wav"

pipeline = Pipeline.from_pretrained(config_path)

def summarize(artefact):
    if (
        hasattr(artefact, "data")
        and hasattr(artefact.data, "shape")
        and hasattr(artefact.data, "dtype")
    ):
        return f"shape={artefact.data.shape}, dtype={artefact.data.dtype}"
    if isinstance(artefact, Annotation):
        timeline = artefact.get_timeline()
        return f"tracks={len(list(artefact.itertracks()))}, duration={timeline.duration:.3f}s"
    arr = np.asarray(artefact)
    return f"shape={arr.shape}, dtype={arr.dtype}" if arr.ndim else f"scalar={arr.item()}"

def trace(step, artefact, *, file, total=None, completed=None, **kwargs):
    tag = f"[{step}]"
    if completed is not None and total is not None:
        tag += f" ({completed}/{total})"
    print(tag, summarize(artefact))

waveform, sample_rate = torchaudio.load(str(audio_path))
output = pipeline({
    "waveform": waveform,
    "sample_rate": sample_rate,
    "uri": "yc_first_10s",
}, hook=trace)

print("---", type(output).__qualname__, output.speaker_embeddings.shape)
PY
```

Key observations from the trace:

- `segmentation`: `(1, 589, 7)` powerset log probabilities (589 frames at ≈16.9 ms frame rate after resampling).
- `speaker_counting`: `(594, 1)` integers describing active-speaker counts per frame.
- `embeddings`: single batch `shape=(1, 3, 256)` corresponding to three local speaker streams extracted from the segmentation chunks.
- `discrete_diarization`: `(594, 2)` mask holding the VBx cluster assignments aligned to the frame grid.
- Final `DiarizeOutput`: two diarization tracks with overlaps preserved, one exclusive track, and `(2, 256)` speaker centroids emitted by VBx.

These numbers align with the expectations documented above and are useful sanity checks when porting VBx clustering and label reconstruction to another runtime.
