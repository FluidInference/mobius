"""Use a real 24 kHz speech prompt, compute its mel, and compare PyTorch vs CoreML HiFT outputs.

This validates the mlpackage on a realistic mel distribution (not synthetic random).
Optionally saves WAVs for manual listening and Whisper ASR.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

from src.hift_coreml import HiFTCoreML
from hyperpyyaml import load_hyperpyyaml


def build_gen():
    here = Path(__file__).parent.parent
    with open(here / "cosyvoice3_dl" / "cosyvoice3.yaml") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    gen = cfg["hift"]
    sd = torch.load(str(here / "cosyvoice3_dl" / "hift.pt"), map_location="cpu", weights_only=False)
    gen.load_state_dict(sd, strict=False)
    gen.eval()
    return gen, cfg


def main():
    wav_path = Path(__file__).parent / "CosyVoice" / "asset" / "zero_shot_prompt.wav"
    audio, sr = sf.read(str(wav_path))
    assert sr == 24000, f"expected 24kHz, got {sr}"
    audio = torch.from_numpy(audio).float().unsqueeze(0)  # (1, L)
    print(f"Input audio: {audio.shape}  duration={audio.shape[-1]/sr:.2f}s")

    from matcha.utils.audio import mel_spectrogram
    # Use the exact config from cosyvoice3.yaml
    mel = mel_spectrogram(
        audio,
        n_fft=1920,
        num_mels=80,
        sampling_rate=sr,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=None,
        center=False,
    )  # (1, 80, T_mel)
    print(f"Mel: {mel.shape}  range=[{mel.min():.3f}, {mel.max():.3f}]")

    # HiFT expects a specific mel length; our converted mlpackage was built with T=250.
    # Pad to 250 frames; track true length so we can trim output.
    T_target = 250
    T_actual = min(mel.shape[-1], T_target)
    if mel.shape[-1] >= T_target:
        mel = mel[:, :, :T_target]
        T_actual = T_target
    else:
        mel = torch.nn.functional.pad(mel, (0, T_target - mel.shape[-1]))
    num_valid_frames = torch.tensor([T_actual], dtype=torch.int32)
    print(f"Padded mel to: {mel.shape}  (valid frames: {T_actual})")

    # PyTorch HiFT
    gen, _ = build_gen()
    wrapper = HiFTCoreML(gen).eval()
    with torch.no_grad():
        a_torch_full, alen_torch = wrapper(mel, num_valid_frames)
    a_torch_full = a_torch_full.numpy().flatten()
    alen_torch = int(alen_torch.item())
    print(f"PyTorch  : audio shape {a_torch_full.shape}  alen={alen_torch}")

    # CoreML HiFT
    mlp = Path(__file__).parent.parent / "build" / "hift-fp32" / "HiFT-T250-fp32.mlpackage"
    ml = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({"mel": mel.numpy(), "num_valid_frames": num_valid_frames.numpy()})
    a_coreml_full = np.asarray(out["audio"]).flatten()
    alen_coreml = int(np.asarray(out["audio_length_samples"]).ravel()[0])
    print(f"CoreML   : audio shape {a_coreml_full.shape}  alen={alen_coreml}")
    assert alen_torch == alen_coreml, f"length mismatch: {alen_torch} vs {alen_coreml}"

    # Trim both to the real audio length.
    a_torch = a_torch_full[:alen_torch]
    a_coreml = a_coreml_full[:alen_coreml]
    L = a_torch.size

    d = np.abs(a_torch - a_coreml)
    corr = np.corrcoef(a_torch, a_coreml)[0, 1]
    print(f"\nAudio length: {L} samples = {L/24000:.2f}s")
    print(f"torch:  range=[{a_torch.min():.4f}, {a_torch.max():.4f}]  std={a_torch.std():.4f}")
    print(f"coreml: range=[{a_coreml.min():.4f}, {a_coreml.max():.4f}]  std={a_coreml.std():.4f}")
    print(f"Overall: MAE={d.mean():.3e}  max={d.max():.3e}  corr={corr:.6f}")

    N = L // 10
    for i in range(10):
        s, e = i * N, (i + 1) * N
        c = np.corrcoef(a_torch[s:e], a_coreml[s:e])[0, 1]
        print(f"  [{s:>6d}:{e:>6d}] MAE={d[s:e].mean():.3e} max={d[s:e].max():.3e} corr={c:.5f}")

    # Save WAVs for listening / ASR
    out_dir = Path(__file__).parent.parent / "build" / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "real_mel_torch.wav"), a_torch, 24000)
    sf.write(str(out_dir / "real_mel_coreml.wav"), a_coreml, 24000)
    print(f"\nSaved WAVs to {out_dir}")


if __name__ == "__main__":
    main()
