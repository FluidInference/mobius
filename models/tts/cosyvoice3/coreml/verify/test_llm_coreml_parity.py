"""Compare CoreML LLM-Prefill / LLM-Decode outputs vs their PyTorch wrappers.

Inputs:
  --build-dir  directory containing LLM-Prefill-*.mlpackage and LLM-Decode-*.mlpackage
               plus the accompanying ref-T<T>-M<M>-{fp32,fp16}.pt file.

For parity we load the .pt reference produced by convert-llm.py (it contains the
pre-captured PyTorch outputs), then run both mlpackages with CoreML and compare
logits / last_hidden / KV.  Also runs one decode step starting from the prefill's
KV and compares against the PyTorch decode reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent


def _load_ref(build_dir: Path):
    cands = sorted(build_dir.glob("ref-T*-M*-*.pt"))
    if not cands:
        raise SystemExit(f"No ref-*.pt found in {build_dir}")
    # Prefer the one that matches the mlpackage precision in the directory name.
    ref = cands[0]
    print(f"[ref] {ref.name}")
    return torch.load(str(ref), map_location="cpu", weights_only=False), ref


def _find_mlpackage(build_dir: Path, prefix: str) -> Path:
    cands = sorted(build_dir.glob(f"{prefix}*.mlpackage"))
    if not cands:
        raise SystemExit(f"No {prefix}*.mlpackage found in {build_dir}")
    return cands[0]


def _mae(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return float(d.mean().item()), float(d.max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--compute-units", default="ALL",
                    choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"])
    args = ap.parse_args()

    import coremltools as ct

    build_dir = Path(args.build_dir)
    ref, _ = _load_ref(build_dir)

    cu = {
        "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE":  ct.ComputeUnit.CPU_AND_NE,
        "ALL":         ct.ComputeUnit.ALL,
    }[args.compute_units]

    prefill_mlp = _find_mlpackage(build_dir, "LLM-Prefill")
    decode_mlp  = _find_mlpackage(build_dir, "LLM-Decode")
    print(f"[load] prefill: {prefill_mlp.name}")
    print(f"[load] decode : {decode_mlp.name}")

    pre_model = ct.models.MLModel(str(prefill_mlp), compute_units=cu)
    dec_model = ct.models.MLModel(str(decode_mlp),  compute_units=cu)

    # ---------------- Prefill parity ----------------
    lm_input  = ref["lm_input"].detach().numpy().astype(np.float32)
    input_len = ref["input_len"].detach().numpy().astype(np.int32)
    print(f"\n[prefill] feed  lm_input {lm_input.shape}  input_len {input_len}")
    pre_out = pre_model.predict({"inputs_embeds": lm_input, "input_len": input_len})
    lh  = torch.from_numpy(pre_out["last_hidden"])
    sl  = torch.from_numpy(pre_out["speech_logits"])
    kk  = torch.from_numpy(pre_out["kv_k"])
    vv  = torch.from_numpy(pre_out["kv_v"])

    mae_lh  = _mae(lh, ref["prefill_last_hidden"])
    mae_sl  = _mae(sl, ref["prefill_speech_logits"])
    mae_kk  = _mae(kk, ref["prefill_kv_k"])
    mae_vv  = _mae(vv, ref["prefill_kv_v"])
    last_idx = int(input_len[0]) - 1
    am_ref = int(ref["prefill_speech_logits"][0, last_idx].argmax().item())
    am_cm  = int(sl[0, last_idx].argmax().item())

    print(f"[prefill] last_hidden   MAE={mae_lh[0]:.3e}  max={mae_lh[1]:.3e}")
    print(f"[prefill] speech_logits MAE={mae_sl[0]:.3e}  max={mae_sl[1]:.3e}")
    print(f"[prefill] kv_k          MAE={mae_kk[0]:.3e}  max={mae_kk[1]:.3e}")
    print(f"[prefill] kv_v          MAE={mae_vv[0]:.3e}  max={mae_vv[1]:.3e}")
    print(f"[prefill] argmax ref={am_ref}  coreml={am_cm}  {'OK' if am_ref == am_cm else 'MISMATCH'}")

    # ---------------- Decode parity ----------------
    # Use the same dummy embed stored in ref by convert-llm.py so results are reproducible.
    if "decode_input_embed" not in ref:
        print("[decode] (no decode reference in ref.pt) -- skipping")
        return
    embed = ref["decode_input_embed"].detach().numpy().astype(np.float32)
    cur_len = ref["decode_cur_len"].detach().numpy().astype(np.int32)
    print(f"\n[decode] feed  embed {embed.shape}  cur_len {cur_len}")
    dec_out = dec_model.predict({
        "inputs_embeds": embed,
        "kv_k": ref["prefill_kv_k"].detach().numpy().astype(np.float32),
        "kv_v": ref["prefill_kv_v"].detach().numpy().astype(np.float32),
        "cur_len": cur_len,
    })
    sl_d  = torch.from_numpy(dec_out["speech_logits"])
    kk_d  = torch.from_numpy(dec_out["kv_k_out"])
    vv_d  = torch.from_numpy(dec_out["kv_v_out"])

    mae_sl_d = _mae(sl_d, ref["decode_speech_logits"])
    mae_kk_d = _mae(kk_d, ref["decode_kv_k_out"])
    mae_vv_d = _mae(vv_d, ref["decode_kv_v_out"])
    am_ref_d = int(ref["decode_speech_logits"][0, 0].argmax().item())
    am_cm_d  = int(sl_d[0, 0].argmax().item())

    print(f"[decode ] speech_logits MAE={mae_sl_d[0]:.3e}  max={mae_sl_d[1]:.3e}")
    print(f"[decode ] kv_k_out      MAE={mae_kk_d[0]:.3e}  max={mae_kk_d[1]:.3e}")
    print(f"[decode ] kv_v_out      MAE={mae_vv_d[0]:.3e}  max={mae_vv_d[1]:.3e}")
    print(f"[decode ] argmax ref={am_ref_d}  coreml={am_cm_d}  {'OK' if am_ref_d == am_cm_d else 'MISMATCH'}")


if __name__ == "__main__":
    main()
