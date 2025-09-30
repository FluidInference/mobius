"""Core ML-backed wrappers for the pyannote community-1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import coremltools as ct
import numpy as np
from scipy import ndimage
import torch
from pyannote.audio import Model, Pipeline

from embedding_io import EMBEDDING_SAMPLES, SEGMENTATION_FRAMES

DEFAULT_EMBEDDING_WINDOW = EMBEDDING_SAMPLES  # 10 s @ 16 kHz (matches segmentation window)


class CoreMLSegmentationModule(torch.nn.Module):
    """Expose a Core ML segmentation model with the torch module API."""

    def __init__(
        self,
        mlmodel: ct.models.MLModel,
        prototype: Model,
        output_key: str,
    ) -> None:
        super().__init__()
        self._mlmodel = mlmodel
        self._output_key = output_key
        self.specifications = getattr(prototype, "specifications", None)
        self.receptive_field = getattr(prototype, "receptive_field", None)
        self.audio = getattr(prototype, "audio", None)
        self.hparams = getattr(prototype, "hparams", None)
        self._device = torch.device("cpu")
        self._supports_batch = True

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "CoreMLSegmentationModule":
        self._device = torch.device(device)
        return self

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        audio_batch = waveforms.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        batch_size = len(audio_batch)
        print(f"[Segmentation] Input batch shape: {audio_batch.shape}, processing {batch_size} chunks")

        if batch_size == 0:
            raise ValueError("Segmentation received an empty batch")

        if self._supports_batch or batch_size == 1:
            try:
                prediction = self._mlmodel.predict({"audio": audio_batch})
                output = prediction[self._output_key]
                output_array = np.asarray(output, dtype=np.float32)
                if output_array.shape[0] != batch_size:
                    raise RuntimeError(
                        f"Batch prediction returned unexpected shape {output_array.shape}"
                    )
                print(f"[Segmentation] Batch predict succeeded: {output_array.shape}")
                return torch.from_numpy(output_array).to(waveforms.device)
            except Exception as exc:  # pragma: no cover - runtime dependent
                print(f"[Segmentation] Batch predict failed ({exc}); falling back to per-chunk loop")
                self._supports_batch = False

        outputs: list[np.ndarray] = []

        # Process one chunk at a time (fallback for runtimes without batching support)
        for idx in range(batch_size):
            audio_sample = audio_batch[idx:idx+1]  # Keep batch dimension: (1, 1, 160000)
            prediction = self._mlmodel.predict({"audio": audio_sample})
            output = prediction[self._output_key]
            output_array = np.asarray(output, dtype=np.float32)
            print(f"[Segmentation] Chunk {idx}: input {audio_sample.shape} -> output {output_array.shape}")
            outputs.append(output_array)

        stacked = np.concatenate(outputs, axis=0)
        print(f"[Segmentation] Final stacked output shape: {stacked.shape}")
        return torch.from_numpy(stacked).to(waveforms.device)


class CoreMLEmbeddingModule(torch.nn.Module):
    """Expose Core ML-backed frontend+backend for speaker embeddings."""

    def __init__(
        self,
        embedding_mlmodel: ct.models.MLModel,
        fbank_mlmodel: ct.models.MLModel,
        prototype: Model,
        output_key: str,
        fbank_output_key: str = "fbank_features",
        embedding_window: int = DEFAULT_EMBEDDING_WINDOW,
    ) -> None:
        super().__init__()
        self._mlmodel = embedding_mlmodel
        self._fbank_model = fbank_mlmodel
        self._output_key = output_key
        self._fbank_output_key = fbank_output_key
        self.dimension = getattr(prototype, "dimension", None)
        self.audio = getattr(prototype, "audio", None)
        self.specifications = getattr(prototype, "specifications", None)
        self.hparams = getattr(prototype, "hparams", None)
        self._device = torch.device("cpu")
        self._embedding_window = int(embedding_window)
        self._supports_batch = True
        self._fbank_supports_batch = True

        spec = embedding_mlmodel.get_spec()
        self._expected_weight_frames = SEGMENTATION_FRAMES
        self._expected_feature_shape: tuple[int, ...] | None = None
        for input_desc in spec.description.input:
            name = getattr(input_desc, "name", "")
            array_type = getattr(input_desc.type, "multiArrayType", None)
            if array_type is None:
                continue
            shape = tuple(int(dim) for dim in getattr(array_type, "shape", []))
            if not shape:
                continue
            if name == "fbank_features":
                self._expected_feature_shape = shape
            elif name == "weights":
                self._expected_weight_frames = int(shape[-1])

        if self._expected_feature_shape is None:
            raise RuntimeError("Embedding CoreML model is missing expected fbank_features input shape")

        if self._expected_weight_frames != SEGMENTATION_FRAMES:
            print(
                f"[Embedding] CoreML backend expects {self._expected_weight_frames} weight frames; "
                f"segmentation outputs {SEGMENTATION_FRAMES}"
            )

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "CoreMLEmbeddingModule":
        self._device = torch.device(device)
        return self

    def _run_fbank(self, audio_batch: np.ndarray) -> np.ndarray:
        batch_size = len(audio_batch)
        if self._fbank_supports_batch or batch_size == 1:
            try:
                prediction = self._fbank_model.predict({"audio": audio_batch})
                features = prediction[self._fbank_output_key]
                feature_array = np.asarray(features, dtype=np.float32)
                if feature_array.shape[0] != batch_size:
                    raise RuntimeError(
                        f"FBANK batch output mismatch: expected {batch_size}, got {feature_array.shape}"
                    )
                print(f"[FBANK] Batch predict succeeded: {feature_array.shape}")
                return feature_array
            except Exception as exc:  # pragma: no cover - runtime dependent
                print(f"[FBANK] Batch predict failed ({exc}); falling back to per-chunk loop")
                self._fbank_supports_batch = False

        outputs: list[np.ndarray] = []
        for idx in range(batch_size):
            audio_sample = audio_batch[idx:idx+1]
            prediction = self._fbank_model.predict({"audio": audio_sample})
            features = prediction[self._fbank_output_key]
            feature_array = np.asarray(features, dtype=np.float32)
            print(f"[FBANK] Chunk {idx}: input {audio_sample.shape} -> features {feature_array.shape}")
            outputs.append(feature_array)

        stacked = np.concatenate(outputs, axis=0)
        print(f"[FBANK] Final stacked feature shape: {stacked.shape}")
        return stacked

    def forward(
        self,
        waveforms: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        audio_batch = waveforms.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        weight_batch = weights.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        batch_size = len(audio_batch)
        print(
            f"[Embedding] Input batch shapes: audio {audio_batch.shape}, "
            f"weights {weight_batch.shape}, processing {batch_size} chunks"
        )

        # Resample weights to match expected frame count
        if weight_batch.shape[-1] != self._expected_weight_frames:
            zoom_factor = self._expected_weight_frames / weight_batch.shape[-1]
            weight_batch = ndimage.zoom(weight_batch, (1.0, zoom_factor), order=1)
            print(
                f"[Embedding] Resampled weights from {weights.shape[-1]} to "
                f"{self._expected_weight_frames} frames"
            )
        weight_batch = weight_batch.astype(np.float32, copy=False)

        features_batch = self._run_fbank(audio_batch)
        expected_without_batch = tuple(self._expected_feature_shape[1:])
        if features_batch.shape[1:] != expected_without_batch:
            raise RuntimeError(
                f"FBANK features shape mismatch: expected (*, {expected_without_batch}), "
                f"got {features_batch.shape}"
            )

        outputs: list[np.ndarray] = []
        if self._supports_batch and batch_size > 1:
            try:
                prediction = self._mlmodel.predict({
                    "fbank_features": features_batch,
                    "weights": weight_batch,
                })
                output = prediction[self._output_key]
                output_array = np.asarray(output, dtype=np.float32)
                if output_array.shape[0] != batch_size:
                    raise RuntimeError(
                        f"Batch embedding output mismatch: expected {batch_size}, got {output_array.shape}"
                    )
                print(f"[Embedding] Batch backend predict succeeded: {output_array.shape}")
                return torch.from_numpy(output_array).to(waveforms.device)
            except Exception as exc:  # pragma: no cover - runtime dependent
                print(f"[Embedding] Batch backend predict failed ({exc}); falling back to per-chunk loop")
                self._supports_batch = False

        for idx in range(batch_size):
            feature_sample = features_batch[idx:idx+1]
            weight_sample = weight_batch[idx:idx+1]
            prediction = self._mlmodel.predict({
                "fbank_features": feature_sample,
                "weights": weight_sample,
            })
            output = prediction[self._output_key]
            output_array = np.asarray(output, dtype=np.float32)
            print(
                f"[Embedding] Chunk {idx}: features {feature_sample.shape}, "
                f"weights {weight_sample.shape} -> embedding {output_array.shape}"
            )
            outputs.append(output_array)

        stacked = np.concatenate(outputs, axis=0)
        print(f"[Embedding] Final stacked output shape: {stacked.shape}")
        return torch.from_numpy(stacked).to(waveforms.device)


@dataclass
class WrappedPipeline:
    pipeline: Pipeline
    segmentation_path: Path
    fbank_path: Path
    embedding_path: Path


def wrap_pipeline_with_coreml(
    pipeline: Pipeline,
    coreml_dir: Path,
    compute_unit: ct.ComputeUnit,
    embedding_window: int = DEFAULT_EMBEDDING_WINDOW,
) -> WrappedPipeline:
    """Replace segmentation and embedding modules with Core ML-backed versions."""

    seg_path = coreml_dir / "segmentation-community-1.mlpackage"
    fbank_path = coreml_dir / "fbank-community-1.mlpackage"
    emb_path = coreml_dir / "embedding-community-1.mlpackage"

    print(f"Loading CoreML segmentation model from: {seg_path}")
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Segmentation model not found at {seg_path}. "
            f"Ensure CoreML models are converted and placed in {coreml_dir}"
        )

    print(f"Loading CoreML FBANK model from: {fbank_path}")
    if not fbank_path.exists():
        raise FileNotFoundError(
            f"FBANK model not found at {fbank_path}. "
            f"Ensure CoreML models are converted and placed in {coreml_dir}"
        )

    print(f"Loading CoreML embedding model from: {emb_path}")
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embedding model not found at {emb_path}. "
            f"Ensure CoreML models are converted and placed in {coreml_dir}"
        )

    try:
        segmentation_ml = ct.models.MLModel(str(seg_path), compute_units=compute_unit)
        print(f"Loaded segmentation model with compute unit: {compute_unit}")
    except Exception as e:
        print(f"Failed to load segmentation model from {seg_path}: {e}")
        raise RuntimeError(f"Failed to load CoreML segmentation model: {e}") from e

    try:
        fbank_ml = ct.models.MLModel(str(fbank_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        print("Loaded FBANK model with compute unit: CPU_ONLY")
    except Exception as e:
        print(f"Failed to load FBANK model from {fbank_path}: {e}")
        raise RuntimeError(f"Failed to load CoreML FBANK model: {e}") from e

    try:
        embedding_ml = ct.models.MLModel(str(emb_path), compute_units=compute_unit)
        print(f"Loaded embedding model with compute unit: {compute_unit}")
    except Exception as e:
        print(f"Failed to load embedding model from {emb_path}: {e}")
        raise RuntimeError(f"Failed to load CoreML embedding model: {e}") from e

    seg_prototype: Model = pipeline._segmentation.model  # type: ignore[attr-defined]
    emb_prototype: Model = pipeline._embedding.model_  # type: ignore[attr-defined]

    pipeline._segmentation.model = CoreMLSegmentationModule(  # type: ignore[attr-defined]
        segmentation_ml,
        seg_prototype,
        output_key="log_probs",
    ).eval()
    pipeline._segmentation.device = torch.device("cpu")  # type: ignore[attr-defined]
    print("Wrapped segmentation module with CoreML")

    pipeline._embedding.model_ = CoreMLEmbeddingModule(  # type: ignore[attr-defined]
        embedding_ml,
        fbank_ml,
        emb_prototype,
        output_key="embedding",
        embedding_window=embedding_window,
    ).eval()
    pipeline._embedding.device = torch.device("cpu")  # type: ignore[attr-defined]
    print("Wrapped embedding module with CoreML")

    return WrappedPipeline(
        pipeline=pipeline,
        segmentation_path=seg_path,
        fbank_path=fbank_path,
        embedding_path=emb_path,
    )


__all__ = [
    "CoreMLEmbeddingModule",
    "CoreMLSegmentationModule",
    "DEFAULT_EMBEDDING_WINDOW",
    "WrappedPipeline",
    "wrap_pipeline_with_coreml",
]
