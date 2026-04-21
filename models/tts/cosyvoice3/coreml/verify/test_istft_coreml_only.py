"""Convert only our ISTFT module and check CoreML parity with PyTorch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import coremltools as ct
import numpy as np
import torch

from src.stft_coreml import ISTFT


def main():
    n_fft, hop, T = 16, 4, 30001
    istft = ISTFT(n_fft, hop).eval()

    torch.manual_seed(0)
    real = torch.randn(1, n_fft // 2 + 1, T) * 2.0
    imag = torch.randn(1, n_fft // 2 + 1, T) * 2.0

    with torch.no_grad():
        audio_torch = istft(real, imag)
    print(f"torch output: shape={tuple(audio_torch.shape)} range=[{audio_torch.min().item():.4f}, {audio_torch.max().item():.4f}]")

    traced = torch.jit.trace(istft, (real, imag), strict=False)
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="real", shape=real.shape, dtype=np.float32),
            ct.TensorType(name="imag", shape=imag.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    out_dir = Path(__file__).parent.parent / "build"
    out_dir.mkdir(exist_ok=True)
    mlp = out_dir / "istft-only.mlpackage"
    ml.save(str(mlp))

    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({"real": real.numpy(), "imag": imag.numpy()})
    audio_ml = out[list(out.keys())[0]]
    print(f"coreml output: shape={audio_ml.shape} range=[{audio_ml.min():.4f}, {audio_ml.max():.4f}]")

    a_t = audio_torch.numpy().flatten()
    a_m = audio_ml.flatten()
    L = min(a_t.size, a_m.size)
    diff = np.abs(a_t[:L] - a_m[:L])
    corr = np.corrcoef(a_t[:L], a_m[:L])[0, 1]
    print(f"ISTFT-only MAE: {diff.mean():.3e}  max: {diff.max():.3e}  corr: {corr:.6f}")

    # Regional
    N = L // 10
    for start in [0, L // 2 - N // 2, L - N]:
        d = diff[start:start + N]
        print(f"  [{start:>6d}, {start+N:>6d}]  MAE={d.mean():.3e}  max={d.max():.3e}")


if __name__ == "__main__":
    main()
