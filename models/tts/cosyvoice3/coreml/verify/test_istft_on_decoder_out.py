"""Test ISTFT parity on realistic decoder outputs (magnitude=exp(...), phase=sin(...))."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml
from scipy.signal import get_window

from src.stft_coreml import ISTFT


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
    m = build()
    n_fft = m.istft_params["n_fft"]
    hop = m.istft_params["hop_len"]

    torch.manual_seed(0)
    mel = torch.randn(1, 80, 250)

    # Run full decode up to conv_post to get realistic magnitude/phase
    with torch.no_grad():
        f0 = m.f0_predictor(mel, finalize=True)
        s = m.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = m.m_source(s)
        s = s.transpose(1, 2)

        # Replicate decode() steps but capture pre-istft values
        s_real, s_imag = m._stft(s.squeeze(1))
        x = m.conv_pre(mel)
        s_stft = torch.cat([s_real, s_imag], dim=1)
        import torch.nn.functional as F
        for i in range(m.num_upsamples):
            x = F.leaky_relu(x, m.lrelu_slope)
            x = m.ups[i](x)
            if i == m.num_upsamples - 1:
                x = m.reflection_pad(x)
            si = m.source_downs[i](s_stft)
            si = m.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(m.num_kernels):
                r = m.resblocks[i * m.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / m.num_kernels
        x = F.leaky_relu(x)
        x = m.conv_post(x)
        magnitude = torch.exp(x[:, :n_fft // 2 + 1, :])
        phase = torch.sin(x[:, n_fft // 2 + 1:, :])
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        # upstream ISTFT
        window = torch.from_numpy(get_window("hann", n_fft, fftbins=True).astype(np.float32))
        audio_t = torch.istft(
            torch.complex(real, imag), n_fft, hop, n_fft, window=window
        )

        # our ISTFT
        istft = ISTFT(n_fft, hop)
        audio_o = istft(real, imag)

    L = min(audio_t.shape[-1], audio_o.shape[-1])
    d = (audio_t[..., :L] - audio_o[..., :L]).abs()
    print(f"real range: [{real.min().item():.3f}, {real.max().item():.3f}]")
    print(f"imag range: [{imag.min().item():.3f}, {imag.max().item():.3f}]")
    print(f"audio torch shape: {tuple(audio_t.shape)}")
    print(f"audio ours  shape: {tuple(audio_o.shape)}")
    print(f"ISTFT diff: MAE={d.mean().item():.3e} max={d.max().item():.3e}")
    corr = np.corrcoef(audio_t[..., :L].flatten().numpy(), audio_o[..., :L].flatten().numpy())[0, 1]
    print(f"corr: {corr:.6f}")


if __name__ == "__main__":
    main()
