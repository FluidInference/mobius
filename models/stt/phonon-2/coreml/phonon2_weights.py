#!/usr/bin/env python3
"""Load FermionResearch/Phonon-2 (`fermion-five-value-parakeet-v1` container) onto the NeMo parakeet-tdt-0.6b-v3 module tree.

Container records (HF-transformers ParakeetForTDT names, same key set as moondream/parakeet-redux):
  * five_value (264, the 11 linears of each encoder layer): w[r, c] = sign[r, c] * (hi[r] if is_hi[r, c] else lo[r]),
    sign in {-1, 0, +1} packed 5 per byte in base 3, is_hi a bitmask over the non-zero entries, lo/hi fp16 per output row.
    Every row therefore takes at most five values {0, +-lo, +-hi}; hi/lo is not an integer, so this is a palette, not
    an affine grid.
  * int6 (35: subsampling 1x1 convs + linear, depthwise convs, decoder embedding/LSTM/projector, joint encoder
    projector and head): q in [-32, 31] with one fp16 scale per output row.
  * fp16 (424): norms, biases, pos biases, BN stats.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "parakeet-redux" / "coreml"))

from fermion_container import read_container  # noqa: E402
from redux_weights import hf_to_nemo_name  # noqa: E402

DEFAULT_CONTAINER = Path("~/Documents/phonon2-work/container/model.fermion").expanduser()


class FiveValue:
    """One five-value record: sign int8 [out, in], is_hi bool [out, in], lo/hi fp16 [out]."""

    def __init__(self, sign: np.ndarray, is_hi: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        self.sign, self.is_hi, self.lo, self.hi = sign, is_hi, lo.astype(np.float16), hi.astype(np.float16)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.sign.shape)

    def codes(self) -> np.ndarray:
        """Per-row palette index into [-hi, -lo, 0, +lo, +hi]: 2 + sign * (1 + is_hi), uint8 in 0..4."""
        return (2 + self.sign.astype(np.int8) * (1 + self.is_hi.astype(np.int8))).astype(np.uint8)

    def dense_fp16(self) -> np.ndarray:
        mag = np.where(self.is_hi, self.hi[:, None], self.lo[:, None])
        return (self.sign.astype(np.float16) * mag).astype(np.float16)


def load_phonon2(container: Path = DEFAULT_CONTAINER):
    """Returns (nemo_state_dict fp32, five_value {nemo_name: FiveValue}, int6 {nemo_name: (q int8, scale fp16)})."""
    tensors, index, raw = read_container(str(container), with_raw=True)
    state: Dict[str, torch.Tensor] = {}
    five: Dict[str, FiveValue] = {}
    int6: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for e in index:
        hf_name = e["n"] + ".weight" if e["k"] == "five_value" else e["n"]
        nemo_name = hf_to_nemo_name(hf_name)
        v = tensors[hf_name]
        if hf_name.endswith("num_batches_tracked"):
            x = float(np.asarray(v, dtype=np.float32).reshape(-1)[0])
            state[nemo_name] = torch.tensor(int(x) if np.isfinite(x) else 0, dtype=torch.int64)
            continue
        if e["k"] == "five_value":
            r = raw[e["n"]]
            fv = FiveValue(r["sign"], r["is_hi"], r["lo"], r["hi"])
            w = fv.dense_fp16().astype(np.float32)
            assert np.array_equal(w.astype(np.float16), v), hf_name
            if ".pointwise_conv" in hf_name:
                w = w[:, :, None]  # Conv1d k=1 -> [out, in, 1]
            five[nemo_name] = fv
            state[nemo_name] = torch.from_numpy(np.ascontiguousarray(w))
            continue
        if e["k"].startswith("int"):
            r = raw[e["n"]]
            int6[nemo_name] = (r["q"], r["scale"])
        state[nemo_name] = torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32)))
    return state, five, int6


def load_into_nemo(asr_model, container: Path = DEFAULT_CONTAINER, verbose: bool = True):
    """Strict in-place load into a NeMo parakeet-tdt-0.6b-v3 model. Returns (five_value dict, int6 dict, diff report)."""
    state, five, int6 = load_phonon2(container)
    orig = {k: v.detach().clone() for k, v in asr_model.state_dict().items()}
    for k in orig:  # the container ships no mel front-end; NeMo's v3 preprocessor buffers stay
        if k.startswith("preprocessor."):
            state[k] = orig[k]
    missing = sorted(set(orig) - set(state))
    unexpected = sorted(set(state) - set(orig))
    if missing or unexpected:
        raise KeyError(f"state dict mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    for k in orig:
        if tuple(orig[k].shape) != tuple(state[k].shape):
            raise ValueError(f"shape mismatch {k}: nemo {tuple(orig[k].shape)} vs phonon {tuple(state[k].shape)}")
    asr_model.load_state_dict(state, strict=True)

    report = {}
    for k in orig:
        if k.startswith("preprocessor.") or k.endswith("num_batches_tracked"):
            continue
        a, b = orig[k].float(), state[k].float()
        diff = (a - b).abs().max().item()
        report[k] = (diff, diff / (a.abs().max().item() + 1e-12))
    if verbose:
        changed = {k: v for k, v in report.items() if k not in five and v[0] > 0}
        print(f"five-value modules: {len(five)}, int6 tables: {len(int6)}")
        print(f"non-five-value tensors changed vs nvidia v3: {len(changed)} / {len(report) - len(five)}")
        by_group: Dict[str, list] = {}
        for k, (d, r) in changed.items():
            g = "encoder.layers" if k.startswith("encoder.layers") else ".".join(k.split(".")[:2])
            by_group.setdefault(g, []).append(r)
        for g, rs in sorted(by_group.items()):
            print(f"  {g:28s} n={len(rs):4d} max_rel_diff={max(rs):.4f} median_rel_diff={float(np.median(rs)):.4f}")
    return five, int6, report


if __name__ == "__main__":
    import argparse

    import nemo.collections.asr as nemo_asr

    ap = argparse.ArgumentParser()
    ap.add_argument("--container", type=Path, default=DEFAULT_CONTAINER)
    args = ap.parse_args()
    m = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
    m.eval()
    five, int6, _ = load_into_nemo(m, args.container)
    k0 = "encoder.layers.0.feed_forward1.linear1.weight"
    fv = five[k0]
    print("layer0 ff1.linear1 code histogram [-hi,-lo,0,+lo,+hi]:", np.bincount(fv.codes().ravel(), minlength=5))
    n = sum(f.sign.size for f in five.values())
    z = sum(int((f.sign == 0).sum()) for f in five.values())
    h = sum(int((f.is_hi & (f.sign != 0)).sum()) for f in five.values())
    print(f"five-value params {n / 1e6:.1f}M: zero {z / n:.3f}, hi {h / n:.3f}, lo {(n - z - h) / n:.3f}")
    print("int6 tables:", sorted(int6)[:6], "...")
