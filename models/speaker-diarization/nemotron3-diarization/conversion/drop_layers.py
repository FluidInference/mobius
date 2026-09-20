"""Zero-shot layer-drop sensitivity sweep: convert fast32-shape variants with k interior
transformer layers removed (no retraining) to map compression headroom for distillation.

Drops evenly spaced interior layers (keeps the first 6 and last 3 blocks, which
typically carry input featurization and output shaping in residual stacks).

Usage: uv run python drop_layers.py --drops 2 4 6 8
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import config
from convert import apply_variant, convert_variant, load_model
from export_patches import apply_patches


class _ZeroSublayer(torch.nn.Module):
    """Identity-out a residual sublayer: x + drop(0) == x, so the stream passes through."""

    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)


def drop_sublayers(model, kind: str, indices):
    for i in indices:
        setattr(model.encoder.layers[i], kind, _ZeroSublayer())
    return sorted(indices)


def drop_interior_layers(model, k: int, explicit=None):
    n = len(model.encoder.layers)
    if explicit is not None:
        drop = set(explicit)
    else:
        interior = list(range(6, n - 3))
        idx_to_drop = set(
            int(round(x)) for x in np.linspace(0, len(interior) - 1, k)
        )
        drop = {interior[i] for i in idx_to_drop}
    kept = [layer for i, layer in enumerate(model.encoder.layers) if i not in drop]
    model.encoder.layers = torch.nn.ModuleList(kept)
    return sorted(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", nargs="*", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--shape", default="s32")
    ap.add_argument("--explicit", default=None, help="name:l1,l2 explicit drop set")
    ap.add_argument(
        "--sub", action="append", default=[],
        help="name:kind:l1,l2 sublayer drop set (kind = ffn | attn); repeatable")
    ap.add_argument(
        "--combo", action="append", default=[],
        help="name:b1,b2:f1,f2 remove full blocks b* AND zero ffn in f* (original indices); repeatable")
    args = ap.parse_args()

    v = config.VARIANTS[args.shape]
    out = Path("build")
    explicit_jobs = []
    if args.explicit:
        nm, ls = args.explicit.split(":")
        explicit_jobs = [(nm, [int(x) for x in ls.split(",")])]
    jobs = [(f"ld{k}", None, k) for k in args.drops] + [(nm, ls, 0) for nm, ls in explicit_jobs]
    for name, explicit, k in jobs:
        model = apply_patches(load_model())
        dropped = drop_interior_layers(model, k, explicit=explicit)
        print(f"{name}: dropped layers {dropped} -> {len(model.encoder.layers)} remain")
        convert_variant(model, name, v, out)
        del model
    for spec in args.sub:
        name, kind, ls = spec.split(":")
        assert kind in ("ffn", "attn"), kind
        indices = [int(x) for x in ls.split(",")]
        model = apply_patches(load_model())
        dropped = drop_sublayers(model, kind, indices)
        print(f"{name}: zeroed {kind} in layers {dropped}")
        convert_variant(model, name, v, out)
        del model
    for spec in args.combo:
        name, bs, fs = spec.split(":")
        blocks = [int(x) for x in bs.split(",")]
        ffns = [int(x) for x in fs.split(",")]
        model = apply_patches(load_model())
        # Zero FFNs first (attribute swap, no renumbering), then remove blocks by original index.
        drop_sublayers(model, "ffn", ffns)
        drop_interior_layers(model, 0, explicit=blocks)
        print(f"{name}: removed blocks {sorted(blocks)}, zeroed ffn in {sorted(ffns)}")
        convert_variant(model, name, v, out)
        del model


if __name__ == "__main__":
    main()
