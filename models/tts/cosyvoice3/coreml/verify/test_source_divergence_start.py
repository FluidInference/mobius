"""Find exactly where coreml s diverges from torch s."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from src.hift_coreml import HiFTCoreML
from hyperpyyaml import load_hyperpyyaml


mlp = Path(__file__).parent.parent / "build" / "source-only.mlpackage"
ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)

here = Path(__file__).parent.parent
with open(here / "cosyvoice3_dl" / "cosyvoice3.yaml") as f:
    cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
gen = cfg["hift"]
sd = torch.load(str(here / "cosyvoice3_dl" / "hift.pt"), map_location="cpu", weights_only=False)
gen.load_state_dict(sd, strict=False)
gen.eval()
w = HiFTCoreML(gen).eval()


class SourceOnly(nn.Module):
    def __init__(self, w): super().__init__(); self.w = w
    def forward(self, mel):
        ww = self.w
        f0 = ww.gen.f0_predictor(mel, finalize=True)
        s = ww.gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = ww.gen.m_source(s)
        return s.transpose(1, 2), f0


torch.manual_seed(42)
raw = torch.randn(1, 80, 250) * 0.3
mel = torch.cumsum(raw, dim=-1) * 0.05 + torch.linspace(-2.0, -6.0, 80).view(1, 80, 1)
src = SourceOnly(w).eval()

with torch.no_grad():
    s_t, _ = src(mel)
out = ml.predict({"mel": mel.numpy()})
s_m = out["s"]
s_t = s_t.numpy()[0, 0]
s_m = s_m[0, 0]

# Find smallest index where |diff| > 1e-4
diff = np.abs(s_t - s_m)
mask = diff > 1e-4
if mask.any():
    first_idx = np.where(mask)[0][0]
    print(f"First index where diff > 1e-4: {first_idx}")
    print(f"f0 frame at that audio index: {first_idx // 480}")
    # Show per-480-block MAE
    per_frame = diff.reshape(-1, 480).mean(axis=1)
    print(f"Per-frame MAE around frame {first_idx // 480}:")
    start_frame = max(0, first_idx // 480 - 5)
    for k in range(start_frame, min(250, start_frame + 30)):
        print(f"  frame {k}: MAE={per_frame[k]:.3e}  max={diff[k*480:(k+1)*480].max():.3e}")
