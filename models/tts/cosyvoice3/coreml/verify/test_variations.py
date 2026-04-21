"""Generate 5 HiFT-CoreML audio variations for listening tests.

Since LLM+Flow aren't converted yet, 'variations' means different mel inputs fed
through the same HiFT mlpackage:
  1. zero_shot prompt → full 3.48s via CoreML
  2. cross_lingual prompt → full via CoreML
  3. zero_shot prompt → half length (demonstrates num_valid_frames trimming)
  4. zero_shot prompt → via PyTorch (A/B reference for CoreML)
  5. cross_lingual prompt → via PyTorch (A/B reference)
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
from matcha.utils.audio import mel_spectrogram


HERE = Path(__file__).parent
ROOT = HERE.parent
ASSET = HERE / "CosyVoice" / "asset"
OUT = ROOT / "build" / "wavs"
MLP = ROOT / "build" / "hift-fp32" / "HiFT-T250-fp32.mlpackage"

T_FIXED = 250  # mel frames the mlpackage was converted for
HOP = 480       # samples per mel frame


def build_wrapper():
    with open(ROOT / "cosyvoice3_dl" / "cosyvoice3.yaml") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    gen = cfg["hift"]
    sd = torch.load(str(ROOT / "cosyvoice3_dl" / "hift.pt"), map_location="cpu", weights_only=False)
    gen.load_state_dict(sd, strict=False)
    gen.eval()
    return HiFTCoreML(gen).eval()


def audio_to_mel(audio_np, sr=24000):
    a = torch.from_numpy(audio_np).float().unsqueeze(0)
    mel = mel_spectrogram(
        a, n_fft=1920, num_mels=80, sampling_rate=sr,
        hop_size=HOP, win_size=1920, fmin=0, fmax=None, center=False,
    )
    return mel  # (1, 80, T)


def pad_mel(mel, T_valid):
    """Pad mel (1,80,T_valid) to (1,80,T_FIXED) and return mel + num_valid_frames."""
    T_valid = min(T_valid, T_FIXED)
    mel = mel[:, :, :T_valid]
    if T_valid < T_FIXED:
        mel = torch.nn.functional.pad(mel, (0, T_FIXED - T_valid))
    return mel, torch.tensor([T_valid], dtype=torch.int32)


def run_coreml(ml, mel, nvf):
    out = ml.predict({"mel": mel.numpy(), "num_valid_frames": nvf.numpy()})
    audio = np.asarray(out["audio"]).flatten()
    alen = int(np.asarray(out["audio_length_samples"]).ravel()[0])
    return audio[:alen]


def run_torch(wrapper, mel, nvf):
    with torch.no_grad():
        audio, alen = wrapper(mel, nvf)
    return audio.numpy().flatten()[: int(alen.item())]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Loading mlpackage: {MLP.name}")
    ml = ct.models.MLModel(str(MLP), compute_units=ct.ComputeUnit.CPU_ONLY)
    print("Building PyTorch wrapper...")
    wrapper = build_wrapper()

    prompts = {
        "zeroshot": ASSET / "zero_shot_prompt.wav",
        "cross":    ASSET / "cross_lingual_prompt.wav",
    }

    import torchaudio.functional as AF
    mels = {}
    for name, p in prompts.items():
        audio, sr = sf.read(str(p))
        if sr != 24000:
            t = torch.from_numpy(audio).float()
            t = AF.resample(t, sr, 24000)
            audio = t.numpy()
            sr = 24000
        mel = audio_to_mel(audio, sr)
        print(f"  {name}: input {len(audio)} samp ({len(audio)/sr:.2f}s) -> mel {tuple(mel.shape)}")
        mels[name] = mel

    # --- 5 CoreML variations ---
    variations = []

    # 1. zero_shot full 3.48s
    m, nvf = pad_mel(mels["zeroshot"], mels["zeroshot"].shape[-1])
    variations.append(("1_zeroshot_full.wav", run_coreml(ml, m, nvf),
                       f"zero_shot full ({int(nvf):d} frames)"))

    # 2. cross_lingual first 5s (frames 0..250 of 687)
    m, nvf = pad_mel(mels["cross"][:, :, :T_FIXED], T_FIXED)
    variations.append(("2_crosslingual_seg1.wav", run_coreml(ml, m, nvf),
                       f"cross_lingual segment 1/3 ({int(nvf):d} frames)"))

    # 3. cross_lingual middle 5s (frames 218..468)
    start = (mels["cross"].shape[-1] - T_FIXED) // 2
    m, nvf = pad_mel(mels["cross"][:, :, start:start + T_FIXED], T_FIXED)
    variations.append(("3_crosslingual_seg2.wav", run_coreml(ml, m, nvf),
                       f"cross_lingual segment 2/3 (frames {start}-{start+T_FIXED})"))

    # 4. cross_lingual last 5s
    start = mels["cross"].shape[-1] - T_FIXED
    m, nvf = pad_mel(mels["cross"][:, :, start:start + T_FIXED], T_FIXED)
    variations.append(("4_crosslingual_seg3.wav", run_coreml(ml, m, nvf),
                       f"cross_lingual segment 3/3 (frames {start}-{start+T_FIXED})"))

    # 5. zero_shot half 1.74s (exercises num_valid_frames trimming at half length)
    T_half = mels["zeroshot"].shape[-1] // 2
    m, nvf = pad_mel(mels["zeroshot"], T_half)
    variations.append(("5_zeroshot_half.wav", run_coreml(ml, m, nvf),
                       f"zero_shot half ({int(nvf):d} frames → trim demo)"))

    print("\nWriting WAVs:")
    for fname, audio, desc in variations:
        path = OUT / fname
        sf.write(str(path), audio, 24000)
        print(f"  {fname}  ({len(audio)/24000:.2f}s)  — {desc}")

    print(f"\nAll saved to {OUT}")


if __name__ == "__main__":
    main()
