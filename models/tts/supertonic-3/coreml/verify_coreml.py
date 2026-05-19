"""Load each .mlpackage and compare CoreML predictions against the PyTorch port."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct


def _compare(label: str, ct_out: np.ndarray, torch_out: np.ndarray) -> bool:
    if ct_out.shape != torch_out.shape:
        print(f"  [FAIL] {label}: shape {ct_out.shape} vs {torch_out.shape}")
        return False
    diff = np.abs(ct_out - torch_out)
    print(f"  [{label}] max_abs={diff.max():.3e}  mean_abs={diff.mean():.3e}  shape={ct_out.shape}")
    return True


def verify_vocoder(mlp: Path, onnx: Path, tts_json: Path) -> None:
    from .vocoder import build_vocoder_from_onnx
    print("=== Vocoder ===")
    model = ct.models.MLModel(str(mlp))
    pt = build_vocoder_from_onnx(onnx, tts_json_path=tts_json).eval()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 144, 4)).astype(np.float32)
    ct_out = model.predict({"latent": x})["wav"]
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x)).numpy()
    _compare("vocoder.wav", ct_out, pt_out)


def verify_text_encoder(mlp: Path, onnx: Path) -> None:
    from .text_encoder import build_text_encoder_from_onnx
    print("=== TextEncoder (fixed T=128) ===")
    model = ct.models.MLModel(str(mlp))
    pt = build_text_encoder_from_onnx(onnx).eval()
    rng = np.random.default_rng(0)
    T = 128
    text_ids = rng.integers(0, 8322, size=(1, T)).astype(np.int32)
    style = rng.standard_normal((1, 50, 256)).astype(np.float32)
    mask = np.ones((1, 1, T), dtype=np.float32)
    ct_out = model.predict({"text_ids": text_ids, "style_ttl": style, "text_mask": mask})["text_emb"]
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(text_ids.astype(np.int64)), torch.from_numpy(style), torch.from_numpy(mask)).numpy()
    _compare("text_encoder.emb", ct_out, pt_out)


def verify_duration_predictor(mlp: Path, onnx: Path) -> None:
    from .duration_predictor import build_duration_predictor_from_onnx
    print("=== DurationPredictor (fixed T=128) ===")
    model = ct.models.MLModel(str(mlp))
    pt = build_duration_predictor_from_onnx(onnx).eval()
    rng = np.random.default_rng(0)
    T = 128
    text_ids = rng.integers(0, 8322, size=(1, T)).astype(np.int32)
    style_dp = rng.standard_normal((1, 8, 16)).astype(np.float32)
    mask = np.ones((1, 1, T), dtype=np.float32)
    ct_out = model.predict({"text_ids": text_ids, "style_dp": style_dp, "text_mask": mask})["duration"]
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(text_ids.astype(np.int64)), torch.from_numpy(style_dp), torch.from_numpy(mask)).numpy()
    _compare("duration_predictor.dur", ct_out, pt_out)


def verify_vector_estimator(mlp: Path, onnx: Path) -> None:
    from .vector_estimator import build_vector_estimator_from_onnx
    print("=== VectorEstimator (L=24, T_text=24) ===")
    model = ct.models.MLModel(str(mlp))
    pt = build_vector_estimator_from_onnx(onnx).eval()
    rng = np.random.default_rng(0)
    L, T_text = 24, 24
    noisy = rng.standard_normal((1, 144, L)).astype(np.float32)
    text = rng.standard_normal((1, 256, T_text)).astype(np.float32)
    style = rng.standard_normal((1, 50, 256)).astype(np.float32)
    lmask = np.ones((1, 1, L), dtype=np.float32)
    tmask = np.ones((1, 1, T_text), dtype=np.float32)
    cur = np.array([7.0], dtype=np.float32)
    tot = np.array([8.0], dtype=np.float32)
    ct_out = model.predict({
        "noisy_latent": noisy, "text_emb": text, "style_ttl": style,
        "latent_mask": lmask, "text_mask": tmask,
        "current_step": cur, "total_step": tot,
    })["denoised_latent"]
    with torch.no_grad():
        pt_out = pt(
            torch.from_numpy(noisy), torch.from_numpy(text), torch.from_numpy(style),
            torch.from_numpy(lmask), torch.from_numpy(tmask),
            torch.from_numpy(cur), torch.from_numpy(tot),
        ).numpy()
    _compare("vector_estimator.den", ct_out, pt_out)


if __name__ == "__main__":  # pragma: no cover
    onnx_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/supertonic-3-coreml/_onnx")
    mlp_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("build/supertonic-3-coreml/_mlpackage")
    tts_json = onnx_dir / "tts.json"
    verify_vocoder(mlp_dir / "Vocoder.mlpackage", onnx_dir / "vocoder.onnx", tts_json)
    verify_text_encoder(mlp_dir / "TextEncoder.mlpackage", onnx_dir / "text_encoder.onnx")
    verify_duration_predictor(mlp_dir / "DurationPredictor.mlpackage", onnx_dir / "duration_predictor.onnx")
    verify_vector_estimator(mlp_dir / "VectorEstimator.mlpackage", onnx_dir / "vector_estimator.onnx")
