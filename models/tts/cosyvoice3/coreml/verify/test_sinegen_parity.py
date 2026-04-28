"""Verify SineGen2CoreML matches upstream SineGen2 in causal-eval mode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

from src.sinegen_coreml import SineGen2CoreML, patch_source_module


def main():
    yaml_path = Path(__file__).parent.parent / "cosyvoice3_dl" / "cosyvoice3.yaml"
    hift_pt = Path(__file__).parent.parent / "cosyvoice3_dl" / "hift.pt"
    sd = torch.load(str(hift_pt), map_location="cpu", weights_only=False)

    def build():
        with open(yaml_path, "r") as f:
            cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
        m = cfg["hift"]
        m.load_state_dict(sd, strict=False)
        m.eval()
        return m

    m_ref = build()
    m_patch = build()

    # Patch m_patch's source module's sinegen.
    patch_source_module(m_patch.m_source)

    # Check direct SineGen output parity on synthetic F0.
    torch.manual_seed(0)
    # f0 input: (B, L, 1), realistic range
    L = 120000  # 5s at 24kHz
    f0 = torch.zeros(1, L, 1)
    f0[:, : L // 2, 0] = 200.0  # voiced
    f0[:, L // 2 :, 0] = 0.0  # unvoiced

    with torch.no_grad():
        # Upstream SineGen2 through m_ref.m_source.l_sin_gen
        s_ref, uv_ref, n_ref = m_ref.m_source.l_sin_gen(f0)
        s_new, uv_new, n_new = m_patch.m_source.l_sin_gen(f0)

    print(f"SineGen direct: sine MAE={(s_ref - s_new).abs().mean().item():.3e} max={(s_ref - s_new).abs().max().item():.3e}")
    print(f"                uv MAE={(uv_ref - uv_new).abs().mean().item():.3e}")
    print(f"                noise MAE={(n_ref - n_new).abs().mean().item():.3e}")

    # Now end-to-end inference parity.
    torch.manual_seed(0)
    mel = torch.randn(1, 80, 250)
    with torch.no_grad():
        audio_ref, source_ref = m_ref.inference(mel, finalize=True)
        audio_new, source_new = m_patch.inference(mel, finalize=True)

    print(f"\nEnd-to-end audio MAE: {(audio_ref - audio_new).abs().mean().item():.3e}  max={(audio_ref - audio_new).abs().max().item():.3e}")
    print(f"End-to-end source MAE: {(source_ref - source_new).abs().mean().item():.3e}")


if __name__ == "__main__":
    main()
