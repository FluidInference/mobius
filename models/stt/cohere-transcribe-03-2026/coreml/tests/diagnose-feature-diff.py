"""Diagnose remaining mel parity error between HF extractor and v2 port."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))

from cohere_features_v2 import CohereMelSpectrogram as CohereMelV2  # noqa: E402


def main():
    from transformers import AutoFeatureExtractor

    pytorch_dir = ROOT.parent / "cohere-pytorch"
    fe = AutoFeatureExtractor.from_pretrained(str(pytorch_dir), trust_remote_code=True)

    wav = ROOT / "fleurs_samples/en_us/sample_0000.wav"
    audio, sr = sf.read(str(wav), dtype="float32")
    assert sr == 16000

    # Default (dither=1e-5)
    out_default = fe(audio, sampling_rate=16000, return_tensors="pt")
    mel_default = out_default["input_features"].float().cpu().numpy()

    # Dither disabled
    fb = fe.filterbank
    saved_dither = fb.dither
    fb.dither = 0.0
    out_nodither = fe(audio, sampling_rate=16000, return_tensors="pt")
    mel_nodither = out_nodither["input_features"].float().cpu().numpy()
    fb.dither = saved_dither

    v2 = CohereMelV2()
    mel_v2, _ = v2(audio)

    def cmp(name, a, b):
        t = min(a.shape[-1], b.shape[-1])
        d = np.abs(a[..., :t] - b[..., :t])
        print(f"  {name:40s} max={d.max():.4e} mean={d.mean():.4e} p99={np.percentile(d, 99):.4e}")

    print("Ref mel shape:", mel_default.shape)
    print()
    cmp("default  vs nodither", mel_default, mel_nodither)
    cmp("nodither vs v2",       mel_nodither, mel_v2)
    cmp("default  vs v2",       mel_default, mel_v2)

    # Also show where the worst disagreement is located
    t = min(mel_nodither.shape[-1], mel_v2.shape[-1])
    diff = np.abs(mel_nodither[0, :, :t] - mel_v2[0, :, :t])
    worst_mel, worst_time = np.unravel_index(diff.argmax(), diff.shape)
    print(f"\nWorst nodither-vs-v2 at mel={worst_mel}, frame={worst_time}")
    print(f"  HF val = {mel_nodither[0, worst_mel, worst_time]:.6f}")
    print(f"  v2 val = {mel_v2[0, worst_mel, worst_time]:.6f}")


if __name__ == "__main__":
    main()
