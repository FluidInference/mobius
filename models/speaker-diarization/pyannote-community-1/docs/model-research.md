# Community-1 Speaker Diarization Model Research

This note captures the current understanding of the `pyannote/speaker-diarization-community-1` pipeline as downloaded into `pyannote-speaker-diarization-community-1/`. It focuses on the model architectures, the high-level processing stages, and the artifacts that must be handled when porting the pipeline to Core ML with 10 s fixed windows on iOS 17+.

## 1. Pipeline At A Glance

- **Input requirements**: 16 kHz mono audio. Multi-channel sources are averaged to mono and resampled automatically by `pyannote.audio`.
- **Sliding window strategy**: Segmentation consumes 10 s (160k-sample) windows and produces 589 frames per chunk (~16.95 ms hop). The Core ML export drives the embedding model with the same 10 s window so that weights align chunk-for-chunk.
- **Primary components**: segmentation network → speech activity post-processing → embedding network → VBx clustering backed by PLDA statistics.
- **Offline-friendly**: all weights ship locally (`segmentation/`, `embedding/`, `plda/`), enabling disconnected use once the Hugging Face repository is cloned.

## 2. Segmentation Network

- **Checkpoint**: `pyannote-speaker-diarization-community-1/segmentation/pytorch_model.bin`.
- **Architecture**: `pyannote.audio.models.segmentation.PyanNet` (powerset diarization). It models speaker activity using a multi-label formulation capped at two concurrent speakers.
- **Context & stride**: 10 s receptive field, ≈16.95 ms frame resolution (589 frames per window when traced at 16 kHz).
- **Outputs**: 7 logits corresponding to all non-empty subsets of up to two speakers (powerset scheme). A softmax + marginalisation recovers per-speaker activity probabilities.
- **Loss & reference**: Uses the powerset multi-class cross-entropy loss introduced by Plaquet & Bredin (Interspeech 2023), cited in the upstream README.
- **Conversion considerations**: supports TorchScript tracing with `(batch, 1, 160000)` tensors. Expect Core ML output shape `(batch, 589, 7)` (batch 1–32 via enumerated shapes).

## 3. Speaker Embedding Network

- **Checkpoint**: `pyannote-speaker-diarization-community-1/embedding/pytorch_model.bin` (copied from `pyannote/wespeaker-voxceleb-resnet34-LM`).
- **Architecture**: WeSpeaker ResNet34 with large-margin softmax head; produces 256-dim embeddings that are L2-normalised inside the Core ML wrapper and PLDA transform.
- **Receptive field**: Architecturally 5 s waveform (80k samples at 16 kHz). The conversion wrapper feeds a full 10 s window (160k samples) so that the overlap weights emitted by segmentation map directly onto the embedding pool without re-windowing.
- **Pre-processing**: The model expects raw waveform and performs internal feature extraction (log-Mel). `WeSpeakerTraceableWrapper` in `convert-coreml.py` replaces the Kaldi frontend with a TorchScript-friendly implementation and handles frame-wise centring.
- **References**: Wang et al., "WeSpeaker: A research and production oriented speaker embedding learning toolkit," ICASSP 2023.

### 3.1 Differences vs. upstream WeSpeaker releases

- **Training recipe**: Community-1 weights have been re-trained inside Pyannote's Lightning/`Task` stack with diarization-oriented batches, augmentation, and objectives. Upstream WeSpeaker checkpoints target speaker verification (ASV) on VoxCeleb and therefore ship different margin/loss schedules.
- **Feature pipeline**: Pyannote keeps the Kaldi log-Mel extraction inside the model so a raw 16 kHz waveform goes straight into the ResNet. The canonical WeSpeaker toolkit expects you to run feature extraction externally (or via Kaldi-compatible scripts) before feeding tensors to the network.
- **Head configuration**: Community-1 fixes a single 256-dim embedding head (no classifier branches) tuned for VBx clustering. Some WeSpeaker repos export multiple heads or auxiliary classification layers geared toward verification benchmarks.
- **Packaging & metadata**: The Community-1 bundle includes `config.yaml`, VBx parameters, and PLDA/X-vector transforms calibrated to these embeddings. GitHub WeSpeaker releases distribute standalone Torch/ONNX models without diarization back-end assets.
- **Integration guarantees**: `pyannote.audio.Model.from_pretrained` loads Community-1 as a drop-in pipeline component alongside segmentation. That ensures consistent hyperparameters (frame shift, windowing, centering) for the downstream Core ML conversion flow, which is absent in the off-the-shelf WeSpeaker toolkit.

## 4. VBx Clustering & PLDA Backend

- **Configuration**: `config.yaml` declares `clustering: VBxClustering` with hyperparameters `threshold=0.6`, `Fa=0.07`, `Fb=0.8`.
- **Artifacts**:
  - `plda/plda.npz`: PLDA global mean (`mu`), transforms (`tr`), and diagonal terms (`psi`).
  - `plda/xvec_transform.npz`: whitening/centering matrices (`mean1`, `mean2`, `lda`).
- **Algorithm outline**:
  1. Center + project embeddings using `xvec_transform` (LDA & mean subtraction).
  2. Score pairs with PLDA (from `plda.npz`).
  3. Run VBx variational Bayes iterations with parameters `(threshold, Fa, Fb)` to obtain diarization tracks.
- **Implementation note**: VBx relies on dynamic loops and float64 math, which Core ML cannot express. The plan is to reimplement VBx (and PLDA scoring) in Swift using Accelerate or bridge the original NumPy logic.
- **Reference**: Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization," CSL 2022.

## 5. Pipeline Data Flow

1. **Audio ingestion**: `Pipeline.__call__` accepts file paths or waveform dictionaries. When given a path, `pyannote.audio.core.io.AudioDecoder` handles decoding (falls back to `torchaudio` if `torchcodec` is unavailable, as in our environment).
2. **Segmentation pass**: 10 s windows → frame-level activity probabilities (powerset classes) → binarization into speech regions.
3. **Region post-processing**: Minimum off-duration `min_duration_off=0` (from `config.yaml`) keeps short gaps. Overlaps remain represented in multi-speaker frames.
4. **Embedding extraction**: Pyannote crops per-speaker regions using the embedding model's own inference parameters (5 s receptive field) and stacks them into batches of up to 32. Our Core ML wrapper consumes the same 10 s waveform slices that fed segmentation and relies on the overlap weights to focus on active frames; the additional context is effectively ignored once weights are applied.
5. **Clustering**: VBx groups embeddings into speaker labels, optionally respecting hints such as `num_speakers`.
6. **Outputs**: produces two timelines. `speaker_diarization` keeps the full multi-speaker annotation (overlaps preserved). `speaker_diarization_exclusive` is derived by a downstream re-labeling pass that assigns the single most likely speaker per frame using VBx posteriors reweighted by the segmentation scores, which is what upstream Pyannote recommends when aligning diarization with ASR word-level timestamps.

### 5.1 Output Timelines & Thresholding

- **`speaker_diarization`**: direct VBx clustering result where concurrent speakers remain when the segmentation head fires multiple powerset classes in the same frame.
- **`speaker_diarization_exclusive`**: computed inside `pyannote.audio.pipelines.SpeakerDiarization.__call__` by collapsing overlaps; it iterates over the VBx posterior lattice and picks the argmax speaker for each frame, ensuring a single label per timestamp. This mirrors the “exclusive diarization” feature documented in the upstream README bundled with the model snapshot.
- **Clustering thresholds**: the YAML configuration pins `threshold=0.6` together with calibration terms `Fa=0.07` and `Fb=0.8`. These values come from Pyannote’s validation sweep on their Community-1 development set and balance DER against speaker purity. The pipeline keeps them as defaults but still accepts runtime overrides (`num_speakers`, `min_speakers`, `max_speakers`, or explicit `threshold` overrides when instantiating VBx) if an application needs different behaviour.

## 6. Interaction With iOS Conversion Goals

- **Fixed 10 s window**: matches `PyanNet` context; Core ML model can accept `(1, 1, 160000)` tensors per inference.
- **On-device audio handling**: Need 16 kHz mono capture/resampling prior to Core ML invocation.
- **Embedding aggregation**: For 10 s chunks, derive sub-windows (e.g., 1 s hops) to maintain VBx performance; experiment to balance latency vs. accuracy.
- **Storage footprint**: Segmentation (~12 MB) + embedding (~27 MB) + VBx params (few kB). Consider compression and lazy loading when shipping with an app.

## 7. Open Questions & Next Steps

- Verify exact output tensor shapes by running Torch tracing (see `docs/conversion-guide.md`).
- Decide whether VBx will run offline after full audio capture or incrementally for near-real-time feedback.
- Confirm licensing obligations (segmentation: CC-BY 4.0, embedding inherits VoxCeleb CC-BY 4.0).
- Explore replacing VBx with a differentiable clustering head if Core ML-only deployment is required.

## 8. References

- `pyannote-speaker-diarization-community-1/README.md`
- `pyannote-speaker-diarization-community-1/config.yaml`
- `docs/conversion-guide.md`
- Plaquet & Bredin, Interspeech 2023.
- Wang et al., ICASSP 2023.
- Landini et al., CSL 2022.
