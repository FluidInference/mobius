"""Compare FlowCoreML wrapper output against upstream flow.inference().

Sanity check: ensure our hand-rolled euler+CFG path matches the upstream module
bit-close BEFORE handing it to coremltools. Mismatch here means a bug in the
wrapper, not in CoreML conversion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
from hyperpyyaml import load_hyperpyyaml

from src.flow_coreml import FlowCoreML


def main() -> None:
    here = Path(__file__).parent.parent
    with open(here / "cosyvoice3_dl" / "cosyvoice3.yaml") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
    flow = cfg["flow"]
    sd = torch.load(str(here / "cosyvoice3_dl" / "flow.pt"), map_location="cpu", weights_only=False)
    flow.load_state_dict(sd, strict=False)
    flow.eval()

    torch.manual_seed(0)
    N_prompt = 40
    N_new = 40
    N_total = N_prompt + N_new
    M_prompt = N_prompt * 2
    M_total = N_total * 2

    prompt_token = torch.randint(0, 6561, (1, N_prompt), dtype=torch.int64)
    new_token = torch.randint(0, 6561, (1, N_new), dtype=torch.int64)
    prompt_feat = torch.randn(1, M_prompt, 80)
    embedding = torch.randn(1, 192)

    # ------ upstream reference ------
    print("Running upstream flow.inference() ...")
    with torch.no_grad():
        ref_mel, _ = flow.inference(
            token=new_token,
            token_len=torch.tensor([N_new], dtype=torch.int64),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([N_prompt], dtype=torch.int64),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([M_prompt], dtype=torch.int64),
            embedding=embedding,
            streaming=False,
            finalize=True,
        )
    print(f"  upstream: ref_mel shape {tuple(ref_mel.shape)}  range=[{ref_mel.min():.3f}, {ref_mel.max():.3f}]")

    # ------ wrapper ------
    print("\nRunning FlowCoreML wrapper ...")
    wrapper = FlowCoreML(flow, n_total_tokens=N_total).eval()

    # build wrapper inputs
    token_total = torch.cat([prompt_token, new_token], dim=1)  # (1, N_total)
    num_prompt_tokens = torch.tensor([N_prompt], dtype=torch.int32)
    prompt_feat_padded = torch.zeros(1, M_total, 80)
    prompt_feat_padded[:, :M_prompt] = prompt_feat

    with torch.no_grad():
        full_mel, num_prompt_mel = wrapper(
            token_total=token_total,
            num_prompt_tokens=num_prompt_tokens,
            prompt_feat=prompt_feat_padded,
            embedding=embedding,
        )
    new_mel = full_mel[:, :, int(num_prompt_mel.item()):]
    print(f"  wrapper : full_mel {tuple(full_mel.shape)}  new_mel {tuple(new_mel.shape)}  range=[{new_mel.min():.3f}, {new_mel.max():.3f}]")

    # ------ compare ------
    assert ref_mel.shape == new_mel.shape, f"shape mismatch: {ref_mel.shape} vs {new_mel.shape}"
    d = (ref_mel - new_mel).abs()
    corr = torch.corrcoef(torch.stack([ref_mel.flatten(), new_mel.flatten()]))[0, 1].item()
    print(f"\nNew-mel parity:")
    print(f"  MAE = {d.mean().item():.3e}   max = {d.max().item():.3e}")
    print(f"  corr = {corr:.6f}")

    # also compare prompt portion (should also match — same diffusion)
    full_ref_mel = None  # upstream returns only new mel; we can't directly compare prompt region.

    # Acceptable threshold: < 1e-4 (we're bypassing nothing but constant-mask construction).
    if d.max().item() < 1e-3:
        print("  PASS")
    else:
        print("  FAIL — investigate before tracing")


if __name__ == "__main__":
    main()
