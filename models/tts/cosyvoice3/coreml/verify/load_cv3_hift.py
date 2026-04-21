"""Load real CosyVoice3 HiFT and run baseline inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
from hyperpyyaml import load_hyperpyyaml


def main():
    yaml_path = Path(__file__).parent.parent / "cosyvoice3_dl" / "cosyvoice3.yaml"
    hift_pt = Path(__file__).parent.parent / "cosyvoice3_dl" / "hift.pt"

    with open(yaml_path, "r") as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})

    model = configs["hift"]
    sd = torch.load(str(hift_pt), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded CausalHiFTGenerator. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    mel = torch.randn(1, 80, 250)  # 250 frames @ 50Hz = 5s @ 24kHz
    with torch.no_grad():
        audio, source = model.inference(mel, finalize=True)
    print(f"Input mel:   {tuple(mel.shape)}")
    print(f"Output audio: {tuple(audio.shape)}  (expect ~{250*480} samples)")
    print(f"Source:       {tuple(source.shape)}")
    print(f"Audio range: [{audio.min().item():.3f}, {audio.max().item():.3f}]")
    torch.save({"mel": mel, "audio": audio, "source": source},
               Path(__file__).parent / "out" / "hift_reference.pt")
    print("Saved reference to verify/out/hift_reference.pt")


if __name__ == "__main__":
    main()
