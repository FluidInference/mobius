"""Fetch StyleTTS2-LibriTTS weights and strip them down to the inference modules.

The upstream `.pth` is ~750 MB because it carries optimizer state, EMA copies,
and discriminators (WavLM, MultiPeriod, MultiResSpec). For inference we only
need: text_encoder, predictor, predictor_encoder, style_encoder, bert,
bert_encoder, decoder, diffusion (sampler).

Output: checkpoints/styletts2_libritts_inference.pt — a dict mapping module
name → state_dict for only the modules listed above.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REPO_ID = "yl4579/StyleTTS2-LibriTTS"
CHECKPOINT_FILE = "Models/LibriTTS/epochs_2nd_00020.pth"
INFERENCE_MODULES = (
    "text_encoder",
    "predictor",
    "predictor_encoder",
    "style_encoder",
    "bert",
    "bert_encoder",
    "decoder",
    "diffusion",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "checkpoints" / "styletts2_libritts_inference.pt",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[00] downloading {REPO_ID}/{CHECKPOINT_FILE} …")
    src = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)
    print(f"[00] loading {src}")
    raw = torch.load(src, map_location="cpu", weights_only=False)

    # Upstream stores under either "net" or directly at top level depending on
    # checkpoint vintage. Probe both.
    state = raw.get("net", raw)
    stripped: dict[str, dict] = {}
    for name in INFERENCE_MODULES:
        if name not in state:
            print(f"[00]   ! module {name!r} missing from checkpoint")
            continue
        sd = state[name]
        # If wrapped in EMA dict ({'params': ...}), unwrap.
        if isinstance(sd, dict) and "params" in sd and "shadow_params" not in sd:
            sd = sd["params"]
        # Upstream trained with DataParallel; every key has a `module.` prefix.
        # Demo notebook strips it before load_state_dict(..., strict=False).
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        stripped[name] = sd
        n_params = sum(t.numel() for t in sd.values() if torch.is_tensor(t))
        print(f"[00]   - {name}: {n_params/1e6:.2f}M params")

    print(f"[00] writing {args.out}")
    torch.save(stripped, args.out)
    total = sum(t.numel() for sd in stripped.values() for t in sd.values() if torch.is_tensor(t))
    print(f"[00] done. inference total: {total/1e6:.2f}M params")


if __name__ == "__main__":
    main()
