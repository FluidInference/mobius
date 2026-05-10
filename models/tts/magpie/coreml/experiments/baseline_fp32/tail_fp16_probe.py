"""Magpie nanocodec tail-fp16 probe (Option 1, post-Phase-F.2 follow-up).

Phase F.2 sweep (`STATUS.md` §F.2) tested whole-graph fp16 islands by
op-class (convs only, activations only, Snake only). All islands failed
the audibility test. The one configuration F.2 did NOT cover is the
*tail*-fp16 split: keep the early HiFi-GAN stages (pre_conv, stages 0-1)
at fp32 where Snake + ConvTranspose1d round-off accumulates audibly,
push only the late stages to fp16. The hypothesis is that noise injected
late in the stack does not have downstream layers to amplify it before
the output, so a small tail fp16 region might be acoustically masked.

Variants:
  v1 — post_conv + out_activation fp16 (smallest possible region)
  v2 — v1 + up_sample_conv_layers.4 + res_layers.4 + activations.4
  v3 — v2 + up_sample_conv_layers.3 + res_layers.3 + activations.3

The selection is by `op.scopes[TORCHSCRIPT_MODULE_NAME]` substring match.
Probe-1 (Probe 1 in `OPTIONS.md`) confirmed `op.scopes` is populated
post-conversion on torch-frontend ops; that's the matching surface used
here.

Usage:
    uv run python -m experiments.baseline_fp32.tail_fp16_probe \\
        --variant v1 --output build/fp32/nanocodec_tail_fp16_v1.mlpackage
    uv run python -m experiments.baseline_fp32.tail_fp16_probe \\
        --variant v1 --inspect-scopes        # dump all unique scopes
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import coremltools as ct
import torch

# Allow running both as a module (-m experiments…) and as a script.
_HERE = Path(__file__).resolve().parent
_COREML_DIR = _HERE.parent.parent
if str(_COREML_DIR) not in sys.path:
    sys.path.insert(0, str(_COREML_DIR))

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.scope import ScopeSource


# ──────────────── variant definitions ────────────────
#
# Substring needles checked against the joined scope list. An op is
# elected for fp16 cast iff ANY needle in the variant's list appears as
# a substring in the joined scope string.
VARIANTS: Dict[str, Tuple[str, ...]] = {
    "v1": (
        "post_conv",
        "out_activation",
    ),
    "v2": (
        "post_conv",
        "out_activation",
        "up_sample_conv_layers.4",
        "res_layers.4",
        "activations.4",
    ),
    "v3": (
        "post_conv",
        "out_activation",
        "up_sample_conv_layers.4",
        "res_layers.4",
        "activations.4",
        "up_sample_conv_layers.3",
        "res_layers.3",
        "activations.3",
    ),
}

# Ops where a `mb.cast` rebuild would be unsafe / would break the graph.
# Keep this minimal — only structural ops that don't produce tensors,
# don't accept dtype variation, or are control-flow.
_SKIP_OPS = frozenset({
    "cast",
    "while_loop", "cond",
    "make_list", "list_gather", "list_scatter",
    "list_read", "list_write", "list_length",
    "read_state", "coreml_update_state",
    "const",  # constants are recasted by the cast_optimization pass
})


def _joined_scopes(op) -> str:
    names = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME, []) or []
    return "/".join(names)


# ──────────────── pass ────────────────
#
# coremltools' PASS_REGISTRY stores classes, not instances — every time
# the pipeline runs, a fresh `TailFP16Cast()` is constructed. So we
# can't configure a singleton; instead, the pass reads its needles from
# a module-level config that the driver sets before `ct.convert(...)`.

_PASS_NEEDLES: Tuple[str, ...] = ()
_PASS_RESULT: Dict[str, Any] = {}  # populated by the most-recent run


def _set_needles(needles: Iterable[str]) -> None:
    global _PASS_NEEDLES
    _PASS_NEEDLES = tuple(needles)
    _PASS_RESULT.clear()


def _last_pass_result() -> Dict[str, Any]:
    return dict(_PASS_RESULT)


@register_pass(namespace="magpie")
class TailFP16Cast(AbstractGraphPass):
    """Cast each fp32 op output to fp16 if its scope matches a needle.

    Boundary casts (fp32 → fp16 on inputs, fp16 → fp32 on outputs) are
    inserted around each elected op. The downstream `cast_optimization`
    pass folds adjacent fp32→fp16→fp32 pairs.
    """

    def __init__(self):
        super().__init__()
        self._needles: Tuple[str, ...] = _PASS_NEEDLES
        self._scope_counter: Counter = Counter()
        self._touched: int = 0
        self._skipped: List[str] = []

    def _should_cast(self, op) -> bool:
        if not self._needles:
            return False
        joined = _joined_scopes(op)
        return any(n in joined for n in self._needles)

    def apply(self, prog):
        self._scope_counter.clear()
        self._touched = 0
        self._skipped = []
        for f in prog.functions.values():
            self._apply_block(f)
        print(f"[TailFP16Cast] cast {self._touched} ops to fp16 "
              f"(needles={self._needles})", file=sys.stderr)
        _PASS_RESULT.clear()
        _PASS_RESULT.update({
            "touched": self._touched,
            "needles": list(self._needles),
            "scope_counter": dict(self._scope_counter),
            "skipped_ops": list(self._skipped),
        })

    @block_context_manager
    def _apply_block(self, block):
        for op in list(block.operations):
            for sub in op.blocks:
                self._apply_block(sub)
            self._scope_counter[_joined_scopes(op)] += 1
            if op.op_type in _SKIP_OPS:
                continue
            if not self._should_cast(op):
                continue
            self._downcast_op(op)

    def _downcast_op(self, op):
        block = op.enclosing_block
        new_inputs = {}
        modified = False
        for pname, vals in op.inputs.items():
            seq = vals if isinstance(vals, (list, tuple)) else [vals]
            new_seq = list(seq)
            for i, v in enumerate(seq):
                if v is None:
                    continue
                if not hasattr(v, "is_tensor_or_scalar_of"):
                    continue
                if v.is_tensor_or_scalar_of(dtype="fp32"):
                    new_seq[i] = mb.cast(
                        x=v, dtype="fp16",
                        name=f"{v.name}__to_fp16",
                        before_op=op,
                    )
                    modified = True
            new_inputs[pname] = (new_seq if isinstance(vals, (list, tuple))
                                 else new_seq[0])
        if not modified:
            return
        new_inputs["name"] = f"{op.name}__fp16"
        new_inputs["before_op"] = op
        try:
            new_op_outputs = getattr(mb, op.op_type)(**new_inputs)
        except Exception as e:  # noqa: BLE001
            # Some ops have parameters that fight a naïve rebuild; skip them
            # rather than abort the whole probe. Phase F.2 already covers
            # the worst-case (whole-graph fp16); a few skipped ops in the
            # tail region won't change the audibility verdict.
            msg = f"{op.op_type}({op.name}): {type(e).__name__}: {e}"
            print(f"[TailFP16Cast] skip {msg}", file=sys.stderr)
            self._skipped.append(msg)
            return
        new_outs = (new_op_outputs if isinstance(new_op_outputs, (list, tuple))
                    else [new_op_outputs])
        for old, new in zip(op.outputs, new_outs):
            if old.is_tensor_or_scalar_of(dtype="fp32"):
                cast_back = mb.cast(
                    x=new, dtype="fp32",
                    name=f"{new.name}__to_fp32",
                    before_op=op,
                )
                block.replace_uses_of_var_after_op(
                    anchor_op=op, old_var=old,
                    new_var=cast_back, force_replace=True,
                )
            else:
                block.replace_uses_of_var_after_op(
                    anchor_op=op, old_var=old,
                    new_var=new, force_replace=True,
                )
        block.remove_ops([op])
        self._touched += 1


# ──────────────── trace + convert driver ────────────────


def _trace_nanocodec(max_frames: int = 24,
                     nemo_path: str | None = None) -> Tuple[torch.jit.TracedModule, int, int]:
    """Reproduce the trace from convert_nanocodec.py at production shape."""
    from convert_nanocodec import (
        TraceableNanoCodecDecoder,
        replace_snake_activations,
    )
    from nemo.collections.tts.models import MagpieTTSModel

    print("[trace] loading NeMo MagpieTTSModel…", file=sys.stderr)
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    codec = model._codec_model
    codec.eval()
    num_codebooks = codec.num_codebooks
    codebook_size = codec.codebook_size

    print(f"[trace] replacing Snake activations…", file=sys.stderr)
    replace_snake_activations(codec)

    traceable = TraceableNanoCodecDecoder(codec, max_frames)
    traceable.eval()

    tokens = torch.randint(0, codebook_size, (1, num_codebooks, max_frames),
                           dtype=torch.long)
    print(f"[trace] tracing (T_in={max_frames}, num_codebooks={num_codebooks})…",
          file=sys.stderr)
    with torch.no_grad():
        traced = torch.jit.trace(traceable, (tokens,))
    return traced, num_codebooks, codebook_size


def _build_pipeline(needles: Tuple[str, ...]):
    """Build a PassPipeline that runs the default common passes plus our
    TailFP16Cast at the end, before lowering.

    `_PASS_NEEDLES` is set globally so each freshly-instantiated
    `TailFP16Cast()` sees the same needles. (`PASS_REGISTRY` stores the
    class, not an instance, so the pipeline calls `__init__` per run.)
    """
    _set_needles(needles)
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.append_pass("magpie::TailFP16Cast")
    return pipeline


def build_variant(
    variant: str,
    output: Path,
    max_frames: int = 24,
    nemo_path: str | None = None,
) -> Tuple[Path, int, int]:
    """Trace, convert at fp32 with TailFP16Cast appended, save mlpackage."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} (choices: {list(VARIANTS)})")
    needles = VARIANTS[variant]

    traced, num_codebooks, _ = _trace_nanocodec(max_frames=max_frames,
                                                nemo_path=nemo_path)
    pipeline = _build_pipeline(needles)

    print(f"[build] ct.convert with custom pipeline (variant={variant})…",
          file=sys.stderr)
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens",
                              shape=(1, num_codebooks, max_frames),
                              dtype=np.int32)],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        pass_pipeline=pipeline,
    )
    print(f"[build] ct.convert finished in {time.time() - t0:.1f}s",
          file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    print(f"[build] saved {output}", file=sys.stderr)
    return output, num_codebooks, max_frames


def inspect_scopes(max_frames: int = 24,
                   nemo_path: str | None = None) -> None:
    """Trace + convert at fp32 (no pass), dump unique scope strings.

    Useful for tuning the variant needles. The default pipeline still
    fires; we just don't elect any ops for casting. We use the singleton
    `TailFP16Cast` to walk the graph and record scope names as a side
    effect.
    """
    traced, num_codebooks, _ = _trace_nanocodec(max_frames=max_frames,
                                                nemo_path=nemo_path)
    # Empty needles → no ops cast, but the apply method still walks and
    # logs the scope counter.
    pipeline = _build_pipeline(())
    print("[inspect] running pass over fp32 program…", file=sys.stderr)
    ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens",
                              shape=(1, num_codebooks, max_frames),
                              dtype=np.int32)],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
        pass_pipeline=pipeline,
    )
    counts = _last_pass_result().get("scope_counter", {})
    print(f"[inspect] {len(counts)} unique scopes, "
          f"{sum(counts.values())} ops total", file=sys.stderr)
    for scope, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {n:5d}  {scope!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=list(VARIANTS) + ["inspect"],
                    default="inspect")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output mlpackage path (required for variant != inspect)")
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--nemo-path", type=str, default=None)
    args = ap.parse_args()

    if args.variant == "inspect":
        inspect_scopes(max_frames=args.max_frames, nemo_path=args.nemo_path)
        return 0

    if args.output is None:
        args.output = Path(f"build/fp32/nanocodec_tail_fp16_{args.variant}.mlpackage")
    build_variant(args.variant, args.output,
                  max_frames=args.max_frames,
                  nemo_path=args.nemo_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
