"""Verify HiFTCoreML wrapper output matches upstream CausalHiFTGenerator.inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
from hyperpyyaml import load_hyperpyyaml

from src.hift_coreml import HiFTCoreML


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
    m_ref = build()
    m_wrap_gen = build()
    wrapper = HiFTCoreML(m_wrap_gen).eval()

    torch.manual_seed(0)
    mel = torch.randn(1, 80, 250)

    # Also test: wrapper WITHOUT the f0_predictor FP32 downgrade — run its f0 path in FP64.
    # Monkey: run upstream inference with f0 in FP32 to isolate the precision effect.
    m_ref2 = build()

    with torch.no_grad():
        audio_ref, _ = m_ref.inference(mel, finalize=True)  # FP64 f0_predictor (upstream default)

        # Upstream but force FP32 f0 path (to match wrapper):
        f0_fp32 = m_ref2.f0_predictor(mel, finalize=True)
        s_fp32 = m_ref2.f0_upsamp(f0_fp32[:, None]).transpose(1, 2)
        s_fp32, _, _ = m_ref2.m_source(s_fp32)
        s_fp32 = s_fp32.transpose(1, 2)
        audio_ref_fp32, _ = m_ref2.decode(x=mel, s=s_fp32, finalize=True), None

        audio_wrap = wrapper(mel)

    L = min(audio_ref.shape[-1], audio_wrap.shape[-1], audio_ref_fp32.shape[-1])
    a_ref = audio_ref[..., :L]
    a_ref_fp32 = audio_ref_fp32[..., :L]
    a_wrap = audio_wrap[..., :L]

    import numpy as np

    def stats(name, a, b):
        mae = (a - b).abs().mean().item()
        mx = (a - b).abs().max().item()
        corr = np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1]
        print(f"{name}: MAE={mae:.3e} max={mx:.3e} corr={corr:.6f}")

    stats("ref(fp64)  vs ref(fp32)", a_ref, a_ref_fp32)
    stats("ref(fp64)  vs wrapper  ", a_ref, a_wrap)
    stats("ref(fp32)  vs wrapper  ", a_ref_fp32, a_wrap)


if __name__ == "__main__":
    main()
