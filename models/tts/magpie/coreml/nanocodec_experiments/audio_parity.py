"""Audio parity test: PyTorch sin² Snake vs CoreML Taylor5Clipped.

Loads the NeMo Magpie codec, runs random tokens through:
  1. Original nanocodec with sin² Snake (PyTorch fp32 reference)
  2. Patched CoreML nanocodec with Taylor5Clipped Snake (fp16, CPU compute)

Reports max abs error, mean abs error, and SNR. Run from magpie/coreml dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct


HERE = Path(__file__).resolve().parent
COREML_DIR = HERE.parent  # magpie/coreml


def load_pytorch_codec():
    """Load PyTorch reference (original sin² Snake)."""
    sys.path.insert(0, str(COREML_DIR))
    from nemo.collections.tts.models import MagpieTTSModel
    print("Loading nvidia/magpie_tts_multilingual_357m...", file=sys.stderr)
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()
    return model


def main() -> int:
    coreml_path = COREML_DIR / "build" / "nanocodec_decoder_taylor5_clipped.mlpackage"
    if not coreml_path.exists():
        print(f"ERROR: missing {coreml_path}", file=sys.stderr)
        print("Run: uv run python convert_nanocodec.py "
              "--output build/nanocodec_decoder_taylor5_clipped.mlpackage",
              file=sys.stderr)
        return 1

    model = load_pytorch_codec()

    # Use random tokens at the conversion's traced shape
    num_codebooks = 8
    T = 256
    codebook_size = 2016

    rng = np.random.default_rng(seed=42)
    tokens_np = rng.integers(0, codebook_size, size=(1, num_codebooks, T)).astype(np.int32)
    tokens_torch = torch.from_numpy(tokens_np.astype(np.int64))  # codec wants int64 typically

    # ---- PyTorch reference (original sin² Snake — no replacement) ----
    print("Running PyTorch reference...", file=sys.stderr)
    with torch.no_grad():
        codec = model._codec_model
        codec.eval()
        ref_audio_t = codec.decode(tokens=tokens_torch, tokens_len=torch.tensor([T], dtype=torch.long))
    if isinstance(ref_audio_t, tuple):
        ref_audio_t = ref_audio_t[0]
    ref_audio = ref_audio_t.detach().cpu().numpy()
    if ref_audio.ndim == 3:
        ref_audio = ref_audio[:, 0]
    print(f"  PyTorch audio shape: {ref_audio.shape}", file=sys.stderr)
    print(f"  PyTorch range: [{ref_audio.min():.4f}, {ref_audio.max():.4f}]", file=sys.stderr)

    # ---- CoreML Taylor5Clipped (CPU) ----
    print("Running CoreML Taylor5Clipped (CPU)...", file=sys.stderr)
    coreml_model = ct.models.MLModel(str(coreml_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = coreml_model.predict({"tokens": tokens_np})
    cm_audio = out["audio"]
    print(f"  CoreML audio shape: {cm_audio.shape}", file=sys.stderr)
    print(f"  CoreML range: [{cm_audio.min():.4f}, {cm_audio.max():.4f}]", file=sys.stderr)

    # Align lengths
    n = min(ref_audio.shape[-1], cm_audio.shape[-1])
    ref = ref_audio[..., :n].reshape(-1)
    cm = cm_audio[..., :n].reshape(-1)

    err = ref - cm
    max_abs = np.max(np.abs(err))
    mean_abs = np.mean(np.abs(err))
    rms_ref = np.sqrt(np.mean(ref ** 2)) + 1e-12
    rms_err = np.sqrt(np.mean(err ** 2)) + 1e-12
    snr_db = 20.0 * np.log10(rms_ref / rms_err)

    print()
    print(f"=== Audio parity (random tokens, seed=42) ===")
    print(f"samples compared : {n}")
    print(f"max_abs error    : {max_abs:.4e}")
    print(f"mean_abs error   : {mean_abs:.4e}")
    print(f"RMS ref          : {rms_ref:.4e}")
    print(f"RMS err          : {rms_err:.4e}")
    print(f"SNR              : {snr_db:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
