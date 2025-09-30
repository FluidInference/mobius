# Community-1 (v4) vs Legacy v3 Pipeline

This note contrasts the pyannote `speaker-diarization-community-1` pipeline (shipped with `pyannote.audio` 4.x) with the legacy v3 release built on the 2.x stack. Use it alongside `docs/pipeline-trace.md` when porting the Core ML models to new runtimes.

## High-level Changes

| Area | Community-1 (v4) | Legacy v3 |
| --- | --- | --- |
| Runtime stack | `pyannote.audio` 4.0 `SpeakerDiarization` pipeline | `pyannote.audio` 2.x multi-stage diarization graph |
| Segmentation network | 10 s powerset head (`7` local speakers) producing log-probabilities; same window drives masking and speaker counting | 2-class speech/non-speech and speaker-change detectors run in cascade with separate window sizes |
| Embedding backbone | WeSpeaker ResNet34, 256-dim embeddings normalised before PLDA, fed by segmentation-guided crops | x-vector style encoder (512 dim) operating on fixed 5 s windows independent of segmentation |
| Clustering | VBx with PLDA `rho` scoring + cosine fallback, configurable via `Fa`, `Fb`, `threshold` | Agglomerative hierarchical clustering (AHC) with cosine/PLDA scoring and heuristic resegmentation |
| Overlap handling | `embedding_exclude_overlap=true` and powerset segmentation encode concurrent speakers in the same chunk | Dedicated overlap detection model; embeddings rely on heuristic overlap filtering |
| Batch execution | Segmentation and embedding traced with batch support but the Core ML wrappers still loop over `(1, 1, 160000)` chunks to mirror Pyannote’s runtime expectations | Sliding-window inference loops over one chunk at a time; batching handled inside `Inference` helper |
| PLDA assets | Core ML bundles `plda-community-1.mlpackage` (features) and `plda_rho-community-1.mlpackage` (rho) plus JSON fallbacks | `.npz` parameter files loaded into NumPy, applied in Python |

## v4 Pipeline Flow (Community-1)

1. **Segmentation** – A 10 s powerset model (`Segmentation.modelc`) ingests padded waveform chunks (1 × 160000) and emits log-probs over 7 local speakers. Output is reshaped to `(num_chunks, num_frames, 7)`.
2. **Binarization & Speaker Count** – Log-probs are thresholded to binary masks; `speaker_count` integrates across the receptive field to estimate active-speaker counts per frame.
3. **Early Exit** – If no speech is detected, the pipeline immediately returns empty annotations.
4. **Embedding Extraction** – Weights derived from speech masks are resampled to match the embedding model’s expected frame count before invoking `Embedding.modelc`. Outputs are 256-dim vectors.
5. **Clustering** – VBx consumes embeddings and the PLDA `rho` scores from `plda_rho-community-1.mlpackage`, iterating until convergence with the `Fa`, `Fb`, `threshold` hyperparameters.
6. **Post-processing** – Chunk-level assignments are mapped back to the timeline, overlaps are resolved according to the speaker count signal, and `min_duration_off` removes short gaps.

See `docs/pipeline-trace.md` for a step-by-step reference with stack traces and hook points.

## Practical Porting Notes

- **Padding & Chunking** – Replicate the padding logic in `coreml_wrappers.py` when preparing Swift inputs. Each Core ML model expects batch size `1`, so the wrapper loops across chunks, padding partial windows to 160000 samples.
- **Weight Frame Alignment** – The embedding wrapper rescales soft VAD weights with `scipy.ndimage.zoom` to match the Core ML spec. A Swift port must resample masks to the embedding frame count (use `vDSP` or custom interpolation).
- **Batch Providers** – `MLBatchProvider` lets us submit chunks sequentially without rebuilding the graph; mirror the Python batching logs to keep debugging parity.

## What Stayed the Same

- Both pipelines rely on frame-wise segmentation, speaker embeddings, and PLDA-based clustering.
- Sliding-window inference with 16 kHz audio and 10 s context remains standard to balance latency and accuracy.
- Hyperparameters such as clustering thresholds continue to live in the YAML config and should be surfaced in runtime apps for experimentation.

When documenting Swift integrations, reference this note to highlight why the v4 Core ML conversion ships three model bundles (`Segmentation`, `Embedding`, `PldaRho`) and how they replace the Python counterparts.
