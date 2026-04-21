"""Look at the actual tail values of the source signal in torch vs coreml."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct
import numpy as np
import torch

mlp = Path(__file__).parent.parent / "build" / "source-only.mlpackage"
ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)

# Build torch reference
import torch.nn as nn
from src.hift_coreml import HiFTCoreML
from hyperpyyaml import load_hyperpyyaml

here = Path(__file__).parent.parent
with open(here / "cosyvoice3_dl" / "cosyvoice3.yaml") as f:
    cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
gen = cfg["hift"]
sd = torch.load(str(here / "cosyvoice3_dl" / "hift.pt"), map_location="cpu", weights_only=False)
gen.load_state_dict(sd, strict=False)
gen.eval()
w = HiFTCoreML(gen).eval()


class SourceOnly(nn.Module):
    def __init__(self, w): super().__init__(); self.w=w
    def forward(self, mel):
        ww=self.w
        f0=ww.gen.f0_predictor(mel, finalize=True)
        s=ww.gen.f0_upsamp(f0[:,None]).transpose(1,2)
        s,_,_=ww.gen.m_source(s)
        return s.transpose(1,2), f0


torch.manual_seed(42)
raw = torch.randn(1, 80, 250) * 0.3
mel = torch.cumsum(raw, dim=-1) * 0.05 + torch.linspace(-2.0, -6.0, 80).view(1, 80, 1)
src = SourceOnly(w).eval()

with torch.no_grad():
    s_t, f0_t = src(mel)
out = ml.predict({"mel": mel.numpy()})
s_m = out["s"]

s_t = s_t.numpy()[0, 0]
s_m = s_m[0, 0]

# f0 at mel rate: 250 frames. Audio rate: 120000. last 480 samples = last f0 frame.
print(f"f0 last 5 frames: {f0_t.numpy()[0, -5:]}")
print(f"s_t last 10: {s_t[-10:]}")
print(f"s_m last 10: {s_m[-10:]}")
print(f"s_t [-500:-490]: {s_t[-500:-490]}")
print(f"s_m [-500:-490]: {s_m[-500:-490]}")

# Is it a phase shift? Check cross-correlation at tail
tail = 2000
diff = s_t[-tail:] - s_m[-tail:]
print(f"\nTail {tail} samples stats:")
print(f"  torch: mean={s_t[-tail:].mean():.4e} std={s_t[-tail:].std():.4e} range=[{s_t[-tail:].min():.4e},{s_t[-tail:].max():.4e}]")
print(f"  coreml: mean={s_m[-tail:].mean():.4e} std={s_m[-tail:].std():.4e} range=[{s_m[-tail:].min():.4e},{s_m[-tail:].max():.4e}]")
print(f"  diff: mean={diff.mean():.4e} std={diff.std():.4e} max={np.abs(diff).max():.4e}")

# What about where the divergence starts?
L = len(s_t)
for start in range(0, L, 10000):
    end = min(start + 10000, L)
    d = np.abs(s_t[start:end] - s_m[start:end])
    print(f"  [{start:>6d}:{end:>6d}] MAE={d.mean():.3e} max={d.max():.3e}")
