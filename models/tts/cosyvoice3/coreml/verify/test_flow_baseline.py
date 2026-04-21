"""Load flow.pt and run a baseline inference with dummy inputs.

Goals:
 1. Validate load path (hyperpyyaml + state_dict)
 2. Confirm input/output shapes
 3. Establish a reference output for later parity tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
from hyperpyyaml import load_hyperpyyaml


def main() -> None:
    here = Path(__file__).parent.parent
    with open(here / "cosyvoice3_dl" / "cosyvoice3.yaml") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
    flow = cfg["flow"]
    sd = torch.load(str(here / "cosyvoice3_dl" / "flow.pt"), map_location="cpu", weights_only=False)
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    print(f"flow loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")
    flow.eval()
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"flow params: {n_params:,}")

    # Dummy inference: 40 new tokens + 40 prompt tokens → 80 mel frames
    torch.manual_seed(0)
    N_new = 40
    N_prompt_tok = 40
    N_prompt_mel = N_prompt_tok * 2  # token_mel_ratio=2

    token = torch.randint(0, 6561, (1, N_new), dtype=torch.int64)
    token_len = torch.tensor([N_new], dtype=torch.int64)
    prompt_token = torch.randint(0, 6561, (1, N_prompt_tok), dtype=torch.int64)
    prompt_token_len = torch.tensor([N_prompt_tok], dtype=torch.int64)
    prompt_feat = torch.randn(1, N_prompt_mel, 80)
    prompt_feat_len = torch.tensor([N_prompt_mel], dtype=torch.int64)
    embedding = torch.randn(1, 192)

    print("\nRunning inference(finalize=True, streaming=False)...")
    with torch.no_grad():
        mel, _ = flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=False,
            finalize=True,
        )
    print(f"mel shape: {tuple(mel.shape)}")
    print(f"      expected: (1, 80, {N_new * 2}) for new tokens only")
    print(f"      range=[{mel.min().item():.3f}, {mel.max().item():.3f}]  std={mel.std().item():.3f}")


if __name__ == "__main__":
    main()
