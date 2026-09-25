"""Checkpoint loading + audio helpers shared by the conversion/verify scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .upstream.model import LocalVQE

SAMPLE_RATE = 16000


def load_model(ckpt_path: str | Path, arch_version: int = 3) -> LocalVQE:
    """Build a LocalVQE from a published ``.pt`` (uses its saved ``model_config``)
    and bake the trained AlignBlock temperature into the smoothing conv."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    cfg.pop("transform", None)
    sd = ckpt["model_state_dict"]
    model = LocalVQE(**cfg, arch_version=arch_version, dmax=_infer_dmax(sd))
    model.load_state_dict(sd)
    model.eval()
    model.align.fold_temperature()
    return model


def _infer_dmax(sd) -> int:
    # Published checkpoints: v1.1 -> 32, v1.2/v1.3 -> 64 (README + tests/manifest.yaml).
    # Neither value is stored in the state dict; key on the activation family.
    return 64


def read_wav(path: str | Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    return data[:, 0]


def write_wav(path: str | Path, pcm: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), np.asarray(pcm, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
