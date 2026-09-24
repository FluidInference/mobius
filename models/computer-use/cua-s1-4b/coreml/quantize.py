"""Post-training weight compression of the decoder parts (linear weights only).

Only `linear` ops whose weight is a model parameter (`*_weight_to_fp16`) are
compressed. The converter also lowers constant-operand matmuls to `linear`
(the delta-rule prefix/suffix/pair-sum masks); those must stay exact.

Modes:
  w8      int8 linear, per output channel
  w4      int4 linear, per block (32)
  p4/p3/p2/p1  k-means palettization, 2^k entries per grouped channel (group 16)
  m4      mixed: MLP (gate/up/down) int4 per block (32), all other linears int8 per channel
  m4b16   as m4 with int4 blocks of 16
  m4gu    as m4 but down_proj stays int8 (int4 only on gate/up)
Writes build/<modality>/L<len>-<mode>/ (config.json copied).
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

MIXED = {
    # mode: (int4 block size, weight-name substrings that get int4; everything else int8)
    "m4": (32, ("_mlp_",)),
    "m4b16": (16, ("_mlp_",)),
    "m4gu": (32, ("_mlp_gate_proj", "_mlp_up_proj")),
}

MODES = {
    "w8": lambda: cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"),
    "w4": lambda: cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32
    ),
    **{
        f"p{k}": (
            lambda k=k: cto.OpPalettizerConfig(
                mode="kmeans", nbits=k, granularity="per_grouped_channel", group_size=16, num_kmeans_workers=12
            )
        )
        for k in (1, 2, 3, 4)
    },
}


def weight_linear_ops(model) -> dict[str, str]:
    """{linear op name: weight const name} for weights that are parameters (not mask/table buffers)."""
    names = {}
    for block in model.get_spec().mlProgram.functions["main"].block_specializations.values():
        for op in block.operations:
            if op.type != "linear":
                continue
            weight = op.inputs["weight"].arguments[0].name
            if weight.endswith(("_weight_to_fp16", "_weight")):
                names[op.outputs[0].name] = weight
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="text")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--mode", choices=sorted([*MODES, *MIXED]), required=True)
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument(
        "--part",
        type=int,
        default=-1,
        help="only this part; k-means with workers can run once per process (coremltools closes its pool)",
    )
    args = ap.parse_args()

    src = args.build / args.modality / f"L{args.length}"
    dst = args.build / args.modality / f"L{args.length}-{args.mode}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "config.json", dst / "config.json")
    for part in sorted(src.glob("CuaS1Decoder_part*.mlpackage")):
        if args.part >= 0 and part.name != f"CuaS1Decoder_part{args.part}.mlpackage":
            continue
        t0 = time.time()
        op_cfg = (
            MODES[args.mode]() if args.mode in MODES else None
        )  # fresh per part: a reused config's k-means worker pool is already closed
        model = ct.models.MLModel(str(part), skip_model_load=True)
        # keyed by weight const name: configs keyed by linear op name conflict on the zero-bias
        # consts the converter shares between linears
        weights = weight_linear_ops(model).values()
        if args.mode in MIXED:
            block, patterns = MIXED[args.mode]
            int4 = cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=block
            )
            int8 = MODES["w8"]()
            config = cto.OptimizationConfig(
                op_name_configs={w: (int4 if any(p in w for p in patterns) else int8) for w in weights}
            )
        else:
            config = cto.OptimizationConfig(op_name_configs={w: op_cfg for w in weights})
        if args.mode.startswith(("w", "m")):
            model = cto.linear_quantize_weights(model, config)
        else:
            model = cto.palettize_weights(model, config)
        model.save(str(dst / part.name))
        print(f"{args.mode} {part.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
