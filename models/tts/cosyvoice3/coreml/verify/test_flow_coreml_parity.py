"""Compare CoreML Flow mlpackage output against the PyTorch wrapper reference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import coremltools as ct
import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mlpackage", required=True)
    p.add_argument("--ref", required=True, help="Path to ref-N{N}.pt saved at conversion time")
    args = p.parse_args()

    ref = torch.load(args.ref, map_location="cpu", weights_only=False)
    ref_mel = ref["mel"].numpy()            # (1, 80, M)
    ref_num = int(ref["num_prompt_mel"].item())
    print(f"Reference mel: {ref_mel.shape}  num_prompt_mel={ref_num}")

    print("Loading CoreML mlpackage (CPU_ONLY)...")
    ml = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({
        "token_total":      ref["token_total"].to(torch.int32).numpy(),
        "num_prompt_tokens": ref["num_prompt_tokens"].numpy(),
        "prompt_feat":      ref["prompt_feat"].numpy(),
        "embedding":        ref["embedding"].numpy(),
    })

    cm_mel = np.asarray(out["mel"])
    cm_num = int(np.asarray(out["num_prompt_mel"]).ravel()[0])
    print(f"CoreML   mel  : {cm_mel.shape}  num_prompt_mel={cm_num}")

    assert ref_mel.shape == cm_mel.shape, f"shape mismatch: {ref_mel.shape} vs {cm_mel.shape}"
    assert ref_num == cm_num, f"num_prompt_mel mismatch: {ref_num} vs {cm_num}"

    d = np.abs(ref_mel - cm_mel)
    # compare only the "new" portion that callers actually use
    new_ref = ref_mel[:, :, ref_num:]
    new_cm = cm_mel[:, :, ref_num:]
    d_new = np.abs(new_ref - new_cm)
    corr_new = float(np.corrcoef(new_ref.flatten(), new_cm.flatten())[0, 1])

    print(f"\nFull mel parity:")
    print(f"  MAE={d.mean():.3e}  max={d.max():.3e}")
    print(f"New-mel (post-prompt) parity:")
    print(f"  MAE={d_new.mean():.3e}  max={d_new.max():.3e}  corr={corr_new:.6f}")


if __name__ == "__main__":
    main()
