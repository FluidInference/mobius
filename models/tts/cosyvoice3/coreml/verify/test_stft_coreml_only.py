"""Convert STFT alone and check CoreML parity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import coremltools as ct
import numpy as np
import torch

from src.stft_coreml import STFT


def main():
    n_fft, hop, L = 16, 4, 120000
    stft = STFT(n_fft, hop).eval()

    torch.manual_seed(0)
    x = torch.randn(1, L) * 0.3

    with torch.no_grad():
        real_t, imag_t = stft(x)
    print(f"torch real: shape={tuple(real_t.shape)}")

    traced = torch.jit.trace(stft, x, strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="real", dtype=np.float32), ct.TensorType(name="imag", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    out_dir = Path(__file__).parent.parent / "build"
    mlp = out_dir / "stft-only.mlpackage"
    ml.save(str(mlp))

    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({"x": x.numpy()})
    real_m = out["real"]
    imag_m = out["imag"]

    d = np.abs(real_t.numpy() - real_m)
    print(f"real MAE={d.mean():.3e} max={d.max():.3e}")
    T = d.shape[-1]
    for lbl, start in [("start", 0), ("mid", T // 2), ("end", T - T // 10)]:
        chunk = d[..., start:start + T // 10]
        print(f"  {lbl}: MAE={chunk.mean():.3e} max={chunk.max():.3e}")


if __name__ == "__main__":
    main()
