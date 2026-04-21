"""Check if two fresh model instances give identical output (isolate fold error from non-determinism)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
from hyperpyyaml import load_hyperpyyaml


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
    m1 = build()
    m2 = build()
    torch.manual_seed(0)
    mel = torch.randn(1, 80, 250)
    with torch.no_grad():
        a1, s1 = m1.inference(mel, finalize=True)
        a2, s2 = m2.inference(mel, finalize=True)
    print(f"Two fresh models audio MAE: {(a1-a2).abs().mean().item():.3e}  max={(a1-a2).abs().max().item():.3e}")
    print(f"Two fresh models source MAE: {(s1-s2).abs().mean().item():.3e}  max={(s1-s2).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
