"""Isolate decode-only parity: feed same source signal to upstream decode vs wrapper._decode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import numpy as np
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

    # Build same source for both
    with torch.no_grad():
        # Use ref path: FP32 f0 predictor
        f0 = m_ref.f0_predictor(mel, finalize=True)
        s = m_ref.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = m_ref.m_source(s)
        s = s.transpose(1, 2)  # (B, 1, L)

        # Decode via upstream (still has weight_norm params)
        audio_ref = m_ref.decode(x=mel, s=s, finalize=True)
        # Decode via wrapper's _decode (folded weights + matmul ISTFT)
        audio_wrap = wrapper._decode(mel, s)

    # Also: pass the wrapper's OWN precomputed source to _decode for fuller test
    with torch.no_grad():
        f0w = wrapper.gen.f0_predictor(mel, finalize=True)
        sw = wrapper.gen.f0_upsamp(f0w[:, None]).transpose(1, 2)
        sw, _, _ = wrapper.gen.m_source(sw)
        sw = sw.transpose(1, 2)
        audio_wrap_owns = wrapper._decode(mel, sw)

    L = min(audio_ref.shape[-1], audio_wrap.shape[-1])
    audio_ref = audio_ref[..., :L]
    audio_wrap = audio_wrap[..., :L]
    audio_wrap_owns = audio_wrap_owns[..., :L]

    def stats(name, a, b):
        d = (a - b).abs()
        corr = np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1]
        print(f"{name}: MAE={d.mean().item():.3e} max={d.max().item():.3e} corr={corr:.6f}")

    # Same-source comparisons (isolate decode/istft/fold effects)
    stats("ref.decode vs wrap._decode (same src)", audio_ref, audio_wrap)
    # Source differences due to weight_norm folding effects upstream in m_source.l_linear?
    stats("wrap._decode same-src vs wrap-own-src", audio_wrap, audio_wrap_owns)

    # Compare the sources themselves
    with torch.no_grad():
        f0_ref = m_ref.f0_predictor(mel, finalize=True)
        f0_wrap = wrapper.gen.f0_predictor(mel, finalize=True)
    print(f"f0 diff: MAE={(f0_ref - f0_wrap).abs().mean().item():.3e} max={(f0_ref - f0_wrap).abs().max().item():.3e}")

    # Compare source signals
    print(f"source diff: MAE={(s - sw).abs().mean().item():.3e} max={(s - sw).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
