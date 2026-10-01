#!/usr/bin/env python3
"""Phase 2: replace the fp16 encoder weight consts with exact five-value palettes (iOS18 constexpr_lut_to_dense).

Each five-value tensor w[r, c] in {0, +-lo[r], +-hi[r]} is re-encoded as a grouped-channel LUT over G consecutive
output rows: the palette of a group is the sorted multiset of its rows' five values (5G entries, zero-padded to 2^nbits)
and every weight becomes the index of its own value in that palette. The reconstruction is bit-exact in fp16, so the
only difference from the fp16 encoder is the storage format.

  --mode lut3   G=1 row  / 8 palettes  (3 bits per weight)   smallest
  --mode lut4   G=2 rows / 16 palettes (4 bits per weight)
  --mode lut6   G=8 rows / 64 palettes (6 bits per weight)   same bit width as the shipped v3 encoder

hi/lo is not an integer (median 2.1, up to 100x), so no affine (shift-scale) encoding is exact here; the LUT is.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import coremltools as ct  # noqa: E402
from coremltools.converters.mil.frontend._utils import _construct_constexpr_lut_op  # noqa: E402
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass  # noqa: E402
from coremltools.converters.mil.mil.passes.helper import block_context_manager  # noqa: E402
from coremltools.models.utils import _apply_graph_pass  # noqa: E402

from phonon2_weights import DEFAULT_CONTAINER, FiveValue, load_phonon2  # noqa: E402

MODES = {"lut3": (1, 3), "lut4": (2, 4), "lut6": (8, 6)}  # mode -> (rows per group, nbits)


def const_name_for(nemo_key: str) -> str:
    # ct.convert names traced-parameter consts "module_<path with '.'->'_'>_to_fp16" (EncoderWrapper.module)
    assert nemo_key.startswith("encoder.") and nemo_key.endswith(".weight")
    return "module_" + nemo_key[len("encoder."):].replace(".", "_") + "_to_fp16"


def grouped_palette(fv: FiveValue, rows_per_group: int, nbits: int):
    """-> (indices uint8 [out, in], lut fp16 [out/G, 1, 2^nbits, 1]); lut[g, 0, indices[r, c], 0] == w[r, c] exactly."""
    out_f, in_f = fv.shape
    g = rows_per_group
    assert out_f % g == 0, (out_f, g)
    n_pal = 2 ** nbits
    assert 5 * g <= n_pal, (g, nbits)
    lo = fv.lo.astype(np.float32)
    hi = fv.hi.astype(np.float32)
    # per-row candidates [-hi, -lo, 0, +lo, +hi] -> per-group candidate list [G, 5g]
    cand = np.stack([-hi, -lo, np.zeros_like(lo), lo, hi], axis=1).reshape(out_f // g, 5 * g)
    order = np.argsort(cand, axis=1, kind="stable")  # sorted[j] = cand[order[j]]
    inv = np.empty_like(order)
    np.put_along_axis(inv, order, np.arange(5 * g)[None, :].repeat(out_f // g, axis=0), axis=1)
    palette = np.take_along_axis(cand, order, axis=1)  # [G, 5g], ascending
    lut = np.zeros((out_f // g, n_pal), dtype=np.float32)
    lut[:, : 5 * g] = palette
    # weight (r, c) with per-row code k (0..4) sits at unsorted candidate position (r % g) * 5 + k
    codes = fv.codes().astype(np.int64)  # [out, in]
    pos = (np.arange(out_f) % g)[:, None] * 5 + codes
    indices = np.take_along_axis(inv.repeat(g, axis=0), pos, axis=1).astype(np.uint8)
    lut16 = lut.astype(np.float16)
    return indices, lut16.reshape(out_f // g, 1, n_pal, 1)


class FiveValueLutPass(AbstractGraphPass):
    def __init__(self, five: dict, mode: str) -> None:
        super().__init__()
        self.rows_per_group, self.nbits = MODES[mode]
        self.by_const_name = {const_name_for(k): (k, fv) for k, fv in five.items()}
        self.replaced: list = []
        self.mismatched: list = []

    def apply(self, prog) -> None:
        for f in prog.functions.values():
            self._apply_block(f)

    @block_context_manager
    def _apply_block(self, block) -> None:
        for op in list(block.operations):
            for b in op.blocks:
                self._apply_block(b)
            if op.op_type != "const" or op.name not in self.by_const_name:
                continue
            key, fv = self.by_const_name[op.name]
            val = op.outputs[0].val
            expect = fv.dense_fp16().reshape(val.shape)
            if val.dtype != np.float16 or not np.array_equal(val, expect):
                self.mismatched.append(op.name)
                continue
            indices, lut = grouped_palette(fv, self.rows_per_group, self.nbits)
            # verify the palette reconstructs the const bit-exactly before touching the graph
            recon = np.take_along_axis(lut[:, 0, :, 0].repeat(self.rows_per_group, axis=0), indices.astype(np.int64), axis=1)
            assert np.array_equal(recon.reshape(val.shape), val), op.name
            if val.ndim == 3:  # pointwise conv1d k=1: [out, in, 1]
                assert val.shape[2] == 1
                indices = indices[:, :, None]
                lut = lut[:, :, None]
            new_var = _construct_constexpr_lut_op(indices, lut, None, name=op.name + "_fivevalue", before_op=op)
            block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0], new_var=new_var, no_check_var_types=True
            )
            block.remove_ops([op])
            self.replaced.append(op.name)


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", type=Path, default=DEFAULT_CONTAINER)
    ap.add_argument("--work", type=Path, default=Path("~/Documents/phonon2-work").expanduser())
    ap.add_argument("--fp16", type=Path, default=None, help="fp16 encoder mlpackage (default <work>/encoder_fp16_phonon2.mlpackage)")
    ap.add_argument("--mode", default="lut3", choices=sorted(MODES))
    ap.add_argument("--out-name", default=None, help="default Encoder_<mode>")
    args = ap.parse_args()
    fp16_path = args.fp16 or (args.work / "encoder_fp16_phonon2.mlpackage")
    out_name = args.out_name or f"Encoder_{args.mode}"

    _, five, _ = load_phonon2(args.container)
    # linear_pos acts on the fixed-window positional embedding, so ct.convert constant-folds it away; nothing to palettize
    five = {k: v for k, v in five.items() if k.startswith("encoder.") and ".linear_pos." not in k}
    print(f"{len(five)} five-value encoder tensors (linear_pos excluded: folded into pos-emb consts)")

    mlmodel = ct.models.MLModel(str(fp16_path), skip_model_load=True)
    p = FiveValueLutPass(five, args.mode)
    t0 = time.time()
    out = _apply_graph_pass(mlmodel, p, spec_version=ct._SPECIFICATION_VERSION_IOS_18, skip_model_load=True)
    print(f"pass + re-serialize took {time.time() - t0:.0f}s; replaced {len(p.replaced)}, mismatched {len(p.mismatched)}")
    if p.mismatched:
        print("MISMATCHED:", p.mismatched[:10])
    missing = set(p.by_const_name) - set(p.replaced) - set(p.mismatched)
    if missing:
        print("NOT FOUND in program:", sorted(missing)[:10], "...", len(missing))
    if p.mismatched or missing:
        raise SystemExit("five-value pass incomplete")

    g, nbits = MODES[args.mode]
    out.short_description = f"Phonon-2 encoder (exact five-value palette, {nbits}-bit LUT per {g} row(s), 15 s window)"
    out.author = "Fluid Inference"
    pkg = args.work / "components" / f"{out_name}.mlpackage"
    pkg.parent.mkdir(parents=True, exist_ok=True)
    if pkg.exists():
        shutil.rmtree(pkg)
    out.save(str(pkg))
    t0 = time.time()
    compiled = Path(ct.utils.compile_model(str(pkg)))
    dst = pkg.with_suffix(".mlmodelc")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(compiled), str(dst))
    print(f"compiled in {time.time() - t0:.0f}s -> {dst}")
    print(f"mlpackage {dir_size(pkg) / 1e6:.1f} MB, mlmodelc {dir_size(dst) / 1e6:.1f} MB (fp16 source {dir_size(fp16_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
