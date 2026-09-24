#!/usr/bin/env python3
"""Phase 2: replace the fp16 encoder weight consts with exact ternary blockwise LUTs (iOS18 constexpr_lut_to_dense).

Each ternary Linear/Conv1d weight w[r, c] = scale[r, c // 128] * (code - 1) becomes
  indices: uint2 [out, in(, 1)]           (the base-3 codes, 0/1/2; 3 unused)
  lut:     fp16  [out, in/128, (1,) 4, 1] = [-s, 0, +s, 0] per (row, 128-column block)
which the MIL op reconstructs bit-exactly; no re-quantization noise is introduced on top of the model's own ternary.
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
from coremltools.converters.mil.mil import Builder as mb  # noqa: E402
from coremltools.converters.mil.mil import types  # noqa: E402
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass  # noqa: E402
from coremltools.converters.mil.mil.passes.helper import block_context_manager  # noqa: E402
from coremltools.models.utils import _apply_graph_pass  # noqa: E402

from redux_weights import GROUP, load_redux  # noqa: E402


def const_name_for(nemo_key: str) -> str:
    # ct.convert names traced-parameter consts "module_<path with '.'->'_'>_to_fp16" (EncoderWrapper.module)
    assert nemo_key.startswith("encoder.") and nemo_key.endswith(".weight")
    return "module_" + nemo_key[len("encoder."):].replace(".", "_") + "_to_fp16"


class TernaryLutPass(AbstractGraphPass):
    def __init__(self, ternary: dict, mode: str = "joint") -> None:
        super().__init__()
        assert mode in ("joint", "blockwise-lut", "int4-block", "joint-perchannel", "lut-perrow")
        self.mode = mode
        self.by_const_name = {const_name_for(k): (k, codes, scales) for k, (codes, scales) in ternary.items()}
        self.replaced = []
        self.mismatched = []

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
            key, codes, scales = self.by_const_name[op.name]
            val = op.outputs[0].val
            out_f, in_f = codes.shape
            expect = np.repeat(scales.astype(np.float32), GROUP, axis=1) * (codes.astype(np.float32) - 1.0)
            expect = expect.astype(np.float16).reshape(val.shape)
            if val.dtype != np.float16 or not np.array_equal(val, expect):
                self.mismatched.append(op.name)
                continue
            if val.ndim == 2:  # linear [out, in]
                indices = codes.astype(np.uint8)
                block_shape = (out_f, in_f // GROUP)
            elif val.ndim == 3 and val.shape[2] == 1:  # conv1d k=1 [out, in, 1]
                indices = codes.astype(np.uint8)[:, :, None]
                block_shape = (out_f, in_f // GROUP, 1)
            else:
                raise ValueError(f"unexpected weight rank for {op.name}: {val.shape}")
            s = scales.astype(np.float16)
            if self.mode == "blockwise-lut":
                # one fp16 LUT [-s, 0, +s, 0] per (row, 128-col block); N-D blockwise constexpr_lut_to_dense
                lut = np.zeros(block_shape + (4, 1), dtype=np.float16)
                lut[..., 0, 0] = -s.reshape(block_shape)
                lut[..., 2, 0] = s.reshape(block_shape)
                new_var = _construct_constexpr_lut_op(indices, lut, None, name=op.name + "_ternary", before_op=op)
            elif self.mode == "joint":
                # iOS18 joint compression: per-tensor int8 LUT {-1, 0, +1, 0} -> blockwise fp16 scale [out, in/128]
                lut = np.array([-1, 0, 1, 0], dtype=np.int8).reshape((1,) * len(block_shape) + (4, 1))
                codes_var = _construct_constexpr_lut_op(indices, lut, None, name=op.name + "_ternary_codes", before_op=op)
                new_var = mb.constexpr_blockwise_shift_scale(
                    data=codes_var, scale=s.reshape(block_shape), offset=None, name=op.name + "_ternary", before_op=op
                )
            elif self.mode == "int4-block":
                # exact: int4 data in {-1, 0, +1} with the blockwise fp16 scale [out, in/128]
                int4_t = types.nptype_from_builtin(types.int4)
                data = (indices.astype(np.int8) - 1).astype(int4_t)
                new_var = mb.constexpr_blockwise_shift_scale(
                    data=data, scale=s.reshape(block_shape), offset=None, name=op.name + "_ternary", before_op=op
                )
            elif self.mode == "joint-perchannel":
                # DIAGNOSTIC (inexact): per-row scale only, to probe ANE support for the lut->scale chain
                lut = np.array([-1, 0, 1, 0], dtype=np.int8).reshape((1,) * len(block_shape) + (4, 1))
                codes_var = _construct_constexpr_lut_op(indices, lut, None, name=op.name + "_ternary_codes", before_op=op)
                row_scale = s.reshape(block_shape[0], -1)[:, :1].reshape((out_f, 1) + (1,) * (len(block_shape) - 2))
                new_var = mb.constexpr_blockwise_shift_scale(
                    data=codes_var, scale=row_scale, offset=None, name=op.name + "_ternary", before_op=op
                )
            elif self.mode == "lut-perrow":
                # DIAGNOSTIC (inexact): per-row fp16 LUT (grouped-channel palettization, group 1 on axis 0)
                lut = np.zeros((out_f, 1) + (1,) * (len(block_shape) - 2) + (4, 1), dtype=np.float16)
                row_scale = s.reshape(block_shape[0], -1)[:, 0]
                lut[..., 0, 0] = -row_scale.reshape(lut.shape[:-2])
                lut[..., 2, 0] = row_scale.reshape(lut.shape[:-2])
                new_var = _construct_constexpr_lut_op(indices, lut, None, name=op.name + "_ternary", before_op=op)
            block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0], new_var=new_var, no_check_var_types=True
            )
            block.remove_ops([op])
            self.replaced.append(op.name)


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, default=Path("~/Documents/parakeet-redux-work/hf").expanduser())
    ap.add_argument("--work", type=Path, default=Path("~/Documents/parakeet-redux-work").expanduser())
    ap.add_argument("--fp16", type=Path, default=None, help="fp16 encoder mlpackage (default <work>/encoder_fp16_redux.mlpackage)")
    ap.add_argument("--out-name", default="Encoder")
    ap.add_argument("--mode", default="joint", choices=["joint", "blockwise-lut", "int4-block", "joint-perchannel", "lut-perrow"])
    args = ap.parse_args()
    fp16_path = args.fp16 or (args.work / "encoder_fp16_redux.mlpackage")

    _, ternary, _ = load_redux(args.hf_dir)
    # linear_pos acts on the fixed-window positional embedding, so ct.convert constant-folds it away
    # (24 x [1, 8, 128, 375] fp16 consts); nothing left to palettize for those.
    ternary = {k: v for k, v in ternary.items() if k.startswith("encoder.") and ".linear_pos." not in k}
    print(f"{len(ternary)} ternary encoder tensors (linear_pos excluded: folded into pos-emb consts)")

    mlmodel = ct.models.MLModel(str(fp16_path), skip_model_load=True)
    p = TernaryLutPass(ternary, mode=args.mode)
    t0 = time.time()
    out = _apply_graph_pass(mlmodel, p, spec_version=ct._SPECIFICATION_VERSION_IOS_18, skip_model_load=True)
    print(f"pass + re-serialize took {time.time() - t0:.0f}s; replaced {len(p.replaced)}, mismatched {len(p.mismatched)}")
    if p.mismatched:
        print("MISMATCHED:", p.mismatched[:10])
    missing = set(p.by_const_name) - set(p.replaced) - set(p.mismatched)
    if missing:
        print("NOT FOUND in program:", sorted(missing)[:10], "...", len(missing))
    if p.mismatched or missing:
        raise SystemExit("ternary pass incomplete")

    out.short_description = f"parakeet-redux encoder (ternary 2-bit, {args.mode}, 15 s window)"
    out.author = "Fluid Inference"
    pkg = args.work / "components" / f"{args.out_name}.mlpackage"
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
