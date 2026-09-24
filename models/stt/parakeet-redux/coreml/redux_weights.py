#!/usr/bin/env python3
"""Unpack moondream/parakeet-redux ternary weights and map them onto the NeMo parakeet-tdt-0.6b-v3 module tree.

Ternary packing (ternary.json, format "thrush-ternary-v2"):
  * qweight: uint8 [out, ceil(in/5)], 5 base-3 digits per byte, least-significant digit first
  * scales:  fp16  [out, in/128]
  * w[r, c] = scales[r, c // 128] * (code - 1), code in {0, 1, 2}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

GROUP = 128
ELEMENTS_PER_BYTE = 5

_POW3 = np.array([3**i for i in range(ELEMENTS_PER_BYTE)], dtype=np.int32)


def unpack_codes(qweight: np.ndarray, in_features: int) -> np.ndarray:
    """uint8 [out, ceil(in/5)] -> int8 codes [out, in] in {0, 1, 2}."""
    q = qweight.astype(np.int32)[:, :, None]  # [out, B, 1]
    digits = (q // _POW3[None, None, :]) % 3  # [out, B, 5], LSD first
    codes = digits.reshape(q.shape[0], -1)[:, :in_features]
    return codes.astype(np.int8)


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """codes int8 [out, in], scales fp16 [out, in/128] -> fp32 [out, in]."""
    out, inp = codes.shape
    assert inp % GROUP == 0 and scales.shape == (out, inp // GROUP), (codes.shape, scales.shape)
    s = np.repeat(scales.astype(np.float32), GROUP, axis=1)
    return s * (codes.astype(np.float32) - 1.0)


def hf_to_nemo_name(name: str) -> str:
    """Map moondream/HF-transformers parameter names to NeMo EncDecRNNTBPEModel names."""
    n = name
    n = re.sub(r"^encoder\.subsampling\.layers\.(\d+)\.", r"encoder.pre_encode.conv.\1.", n)
    n = n.replace("encoder.subsampling.linear.", "encoder.pre_encode.out.")
    n = n.replace(".self_attn.q_proj.", ".self_attn.linear_q.")
    n = n.replace(".self_attn.k_proj.", ".self_attn.linear_k.")
    n = n.replace(".self_attn.v_proj.", ".self_attn.linear_v.")
    n = n.replace(".self_attn.o_proj.", ".self_attn.linear_out.")
    n = n.replace(".self_attn.relative_k_proj.", ".self_attn.linear_pos.")
    n = n.replace(".self_attn.bias_u", ".self_attn.pos_bias_u")
    n = n.replace(".self_attn.bias_v", ".self_attn.pos_bias_v")
    n = n.replace(".conv.norm.", ".conv.batch_norm.")
    n = n.replace("decoder.embedding.weight", "decoder.prediction.embed.weight")
    n = n.replace("decoder.lstm.", "decoder.prediction.dec_rnn.lstm.")
    n = n.replace("decoder.decoder_projector.", "joint.pred.")
    n = n.replace("encoder_projector.", "joint.enc.")
    n = n.replace("joint.head.", "joint.joint_net.2.")
    return n


def load_redux(hf_dir: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, torch.Tensor]]:
    """Returns (nemo_state_dict fp32, ternary {nemo_name: (codes int8, scales fp16)}, vad_head tensors)."""
    from safetensors import safe_open

    meta = json.loads((hf_dir / "ternary.json").read_text())
    assert meta["format"] == "thrush-ternary-v2", meta["format"]
    assert meta["quant"]["group_size"] == GROUP
    qmods = {m["name"]: m for m in meta["quantized_modules"]}

    f = safe_open(str(hf_dir / "model.safetensors"), "np")
    raw = {k: f.get_tensor(k) for k in f.keys()}

    state: Dict[str, torch.Tensor] = {}
    ternary: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    vad: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("vad_head."):
            vad[k] = torch.from_numpy(v.astype(np.float32))
            continue
        if k.endswith(".scales"):
            continue
        if k.endswith(".qweight"):
            mod = k[: -len(".qweight")]
            info = qmods[mod]
            codes = unpack_codes(v, info["in_features"])
            scales = raw[mod + ".scales"]
            assert codes.shape == (info["out_features"], info["in_features"])
            w = dequantize(codes, scales)
            nemo_name = hf_to_nemo_name(mod + ".weight")
            if info["as_conv1d"]:
                w = w[:, :, None]  # Conv1d kernel 1 -> [out, in, 1]
            state[nemo_name] = torch.from_numpy(w)
            ternary[nemo_name] = (codes, scales)
            continue
        if v.dtype == np.int64:
            state[hf_to_nemo_name(k)] = torch.from_numpy(v)
        else:
            state[hf_to_nemo_name(k)] = torch.from_numpy(v.astype(np.float32))
    return state, ternary, vad


def load_into_nemo(asr_model, hf_dir: Path, verbose: bool = True):
    """Load redux weights into a NeMo parakeet-tdt-0.6b-v3 model in place. Returns (ternary dict, diff report)."""
    state, ternary, vad = load_redux(hf_dir)
    orig = {k: v.detach().clone() for k, v in asr_model.state_dict().items()}
    # Preprocessor buffers (mel filterbank, window) are not shipped by redux; keep NeMo's.
    for k in orig:
        if k.startswith("preprocessor."):
            state[k] = orig[k]
    missing = sorted(set(orig) - set(state))
    unexpected = sorted(set(state) - set(orig))
    if missing or unexpected:
        raise KeyError(f"state dict mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    for k in orig:
        if tuple(orig[k].shape) != tuple(state[k].shape):
            raise ValueError(f"shape mismatch {k}: nemo {tuple(orig[k].shape)} vs redux {tuple(state[k].shape)}")
    asr_model.load_state_dict(state, strict=True)

    report = {}
    for k in orig:
        if k.startswith("preprocessor."):
            continue
        a = orig[k].float()
        b = state[k].float()
        diff = (a - b).abs().max().item()
        rel = diff / (a.abs().max().item() + 1e-12)
        report[k] = (diff, rel)
    if verbose:
        changed_non_ternary = {k: v for k, v in report.items() if k not in ternary and v[0] > 0}
        print(f"ternary modules: {len(ternary)}")
        print(f"non-ternary tensors changed vs nvidia v3: {len(changed_non_ternary)} / {len(report) - len(ternary)}")
        by_group = {}
        for k, (d, r) in changed_non_ternary.items():
            g = k.split(".")[0] if not k.startswith("encoder.layers") else "encoder.layers"
            by_group.setdefault(g, []).append(r)
        for g, rs in sorted(by_group.items()):
            print(f"  {g:16s} n={len(rs):4d} max_rel_diff={max(rs):.4f} median_rel_diff={float(np.median(rs)):.4f}")
    return ternary, vad, report


if __name__ == "__main__":
    import argparse

    import nemo.collections.asr as nemo_asr

    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, default=Path("~/Documents/parakeet-redux-work/hf").expanduser())
    ap.add_argument("--out", type=Path, default=None, help="Optional .pt to save the mapped fp32 NeMo state dict")
    args = ap.parse_args()

    m = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
    m.eval()
    ternary, vad, report = load_into_nemo(m, args.hf_dir)
    # Sanity: every dequantized ternary tensor has <= 3 unique values per (row, group)
    k0 = "encoder.layers.0.feed_forward1.linear1.weight"
    codes, scales = ternary[k0]
    print("layer0 ff1.linear1 code histogram:", np.bincount(codes.flatten().astype(np.int64), minlength=3))
    print("layer0 ff1.linear1 scales range:", float(scales.min()), float(scales.max()))
    if args.out:
        torch.save(m.state_dict(), args.out)
        print("saved", args.out)
