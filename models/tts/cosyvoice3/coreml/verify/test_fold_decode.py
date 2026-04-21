"""Isolate: same model before vs after weight_norm fold, decoding the SAME source."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

from src.weight_norm_fold import fold_weight_norm


def build():
    yaml_path = Path(__file__).parent.parent / "cosyvoice3_dl" / "cosyvoice3.yaml"
    hift_pt = Path(__file__).parent.parent / "cosyvoice3_dl" / "hift.pt"
    with open(yaml_path, "r") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    m = cfg["hift"]
    sd = torch.load(str(hift_pt), map_location="cpu", weights_only=False)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def main():
    m_u = build()  # unfolded
    m_f = build()
    fold_weight_norm(m_f)  # folded

    torch.manual_seed(0)
    mel = torch.randn(1, 80, 250)

    with torch.no_grad():
        f0 = m_u.f0_predictor(mel, finalize=True)
        s = m_u.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = m_u.m_source(s)
        s = s.transpose(1, 2)

        a_u = m_u.decode(x=mel, s=s, finalize=True)
        a_f = m_f.decode(x=mel, s=s, finalize=True)

    L = min(a_u.shape[-1], a_f.shape[-1])
    d = (a_u[..., :L] - a_f[..., :L]).abs()
    corr = np.corrcoef(a_u[..., :L].flatten().numpy(), a_f[..., :L].flatten().numpy())[0, 1]
    print(f"unfold vs fold (same src): MAE={d.mean().item():.3e} max={d.max().item():.3e} corr={corr:.6f}")


if __name__ == "__main__":
    main()
