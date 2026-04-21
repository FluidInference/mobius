"""Isolate the cumsum + sin pipeline in CoreML and test where precision fails."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn


class PhaseSin(nn.Module):
    """Stand-in for SineGen's phase+sin math with controllable input."""
    def __init__(self, upsample_scale=480):
        super().__init__()
        self.upsample_scale = upsample_scale

    def forward(self, x_frame):
        # x_frame: (B, N_frames, D) values in cycles per sample
        # same math as SineGen2CoreML._f02sine tail
        import numpy as np
        phase_norm = torch.cumsum(x_frame, dim=1)
        phase_norm = phase_norm - torch.floor(phase_norm)
        phase = phase_norm * (2.0 * np.pi * self.upsample_scale)
        phase = phase - torch.floor(phase / (2.0 * np.pi)) * (2.0 * np.pi)
        return torch.sin(phase), phase_norm, phase


def main():
    torch.manual_seed(0)
    # Use realistic-ish f0 scaled to cycles per sample
    # f0 ~ 100-200 Hz, upsample_scale=480, sampling rate=24000
    N_frames = 250
    D = 9
    x = (torch.rand(1, N_frames, D) * 400 + 50) / 24000  # f0 range 50-450
    # Scale by upsample_scale and back via interpolate isn't needed; mimic eval-mode input
    ps = PhaseSin().eval()

    with torch.no_grad():
        sin_t, pn_t, ph_t = ps(x)

    traced = torch.jit.trace(ps, x, strict=False)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32)],
        outputs=[
            ct.TensorType(name="sin", dtype=np.float32),
            ct.TensorType(name="pn", dtype=np.float32),
            ct.TensorType(name="ph", dtype=np.float32),
        ],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    out_dir = Path(__file__).parent.parent / "build"
    mlp = out_dir / "phase-sin.mlpackage"
    ml.save(str(mlp))
    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({"x": x.numpy()})
    sin_m = out["sin"]
    pn_m = out["pn"]
    ph_m = out["ph"]

    print(f"phase_norm (after mod 1):")
    d = np.abs(pn_t.numpy() - pn_m)
    print(f"  MAE={d.mean():.3e} max={d.max():.3e}")
    # Per frame
    per_frame_pn = d.mean(axis=(0, 2))
    print(f"  tail frames pn MAE: {per_frame_pn[-10:]}")

    print(f"phase (after *2pi*scale, mod 2pi):")
    d = np.abs(ph_t.numpy() - ph_m)
    print(f"  MAE={d.mean():.3e} max={d.max():.3e}")
    per_frame_ph = d.mean(axis=(0, 2))
    print(f"  tail frames ph MAE: {per_frame_ph[-10:]}")

    print(f"sin:")
    d = np.abs(sin_t.numpy() - sin_m)
    print(f"  MAE={d.mean():.3e} max={d.max():.3e}")
    per_frame_sin = d.mean(axis=(0, 2))
    print(f"  tail frames sin MAE: {per_frame_sin[-10:]}")

    # Print a few actual phase values at broken frames
    print(f"\nph_t tail 5 frames per-D mean: {ph_t.numpy().mean(axis=2)[0, -5:]}")
    print(f"ph_m tail 5 frames per-D mean: {ph_m.mean(axis=2)[0, -5:]}")


if __name__ == "__main__":
    main()
