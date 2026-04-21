"""Verify fold_weight_norm preserves outputs on real CausalHiFTGenerator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import copy
import torch
import torch.nn.utils.parametrize as P
from hyperpyyaml import load_hyperpyyaml

from src.weight_norm_fold import fold_weight_norm


def count_parametrized(m):
    return sum(1 for sm in m.modules() if P.is_parametrized(sm))


def count_legacy(m):
    return sum(1 for sm in m.modules() if hasattr(sm, "weight_g") and hasattr(sm, "weight_v"))


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

    model_a = build()  # will be folded
    model_b = build()  # kept as reference (parametrized)

    print(f"Before fold: parametrized={count_parametrized(model_a)} legacy={count_legacy(model_a)}")
    fold_weight_norm(model_a)
    print(f"After  fold: parametrized={count_parametrized(model_a)} legacy={count_legacy(model_a)}")

    # Compare outputs - do reference FIRST so parametrized f0_predictor isn't perturbed by a .to(float64) if deepcopy affected it.
    torch.manual_seed(0)
    mel = torch.randn(1, 80, 250)

    with torch.no_grad():
        audio_b, source_b = model_b.inference(mel, finalize=True)
        audio_a, source_a = model_a.inference(mel, finalize=True)

    mae_a = (audio_a - audio_b).abs().mean().item()
    max_a = (audio_a - audio_b).abs().max().item()
    print(f"Audio shape: {tuple(audio_a.shape)}")
    print(f"Audio MAE (folded vs original): {mae_a:.3e}  max={max_a:.3e}")
    print(f"Source MAE: {(source_a - source_b).abs().mean().item():.3e}")


if __name__ == "__main__":
    main()
