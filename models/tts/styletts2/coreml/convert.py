"""Trace + convert StyleTTS2 stages to CoreML fp32 .mlpackage.

Usage:

    cd models/tts/styletts2
    uv run python coreml/convert.py --stage all
    uv run python coreml/convert.py --stage text_encoder

Output: coreml/packages/<stage>.mlpackage (gitignored).

Per-stage results, blockers, and mitigations are appended to
`coreml/trials.md` by hand (this script is intentionally not chatty
about *why* a stage failed — see logs).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python coreml/convert.py` from the project root to import the
# `coreml` package: prepend the project root to sys.path before any
# `from coreml.* import` statements.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def _patch_coreml_int_cast() -> None:
    """coremltools's `_cast` op (used for `aten::Int`) calls
    `dtype(x.val)` directly. If `x.val` is a length-1 ndarray (e.g.
    `np.array([1])`) instead of a 0-d scalar, `int(arr)` raises
    `TypeError: only 0-dimensional arrays can be converted to Python
    scalars`. HF transformers (Albert) trips this in its embedding
    layer (`aten::size` -> `prim::NumToTensor` -> `aten::Int` chain).

    The patch unwraps length-1 arrays via `.item()` before casting.
    """
    from coremltools.converters.mil.frontend.torch import ops as _ops
    from coremltools.converters.mil.mil import Builder as mb

    if getattr(_ops, "_int_patched_for_styletts2", False):
        return

    _orig_cast = _ops._cast

    def _patched_cast(context, node, dtype, dtype_name):
        inputs = _ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")
        if x.can_be_folded_to_const():
            v = x.val
            if hasattr(v, "shape") and getattr(v, "shape", None):
                # Unwrap length-1 ndarrays / tensors to a Python scalar.
                v = np.asarray(v).reshape(()).item()
            if not isinstance(v, dtype):
                res = mb.const(val=dtype(v), name=node.name)
            else:
                res = mb.const(val=v, name=node.name)
        elif len(x.shape) > 0:
            x2 = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x2, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    _ops._cast = _patched_cast
    _ops._int_patched_for_styletts2 = True


_patch_coreml_int_cast()


def _register_aten_aliases() -> None:
    """coremltools registers `aten::mul` but not `aten::multiply` (the
    non-overloaded alias). HiFi-GAN's source filter uses `torch.multiply`
    in one place. Register a passthrough so the converter doesn't bail.
    """
    from coremltools.converters.mil.frontend.torch.torch_op_registry import (
        _TORCH_OPS_REGISTRY,
    )

    if _TORCH_OPS_REGISTRY.get_func("multiply") is None:
        mul_fn = _TORCH_OPS_REGISTRY.get_func("mul")
        if mul_fn is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(mul_fn, "multiply")


_register_aten_aliases()

# Ensure project paths and macOS espeak env are set before any imports
# that touch the model modules.
from coreml._runtime import (  # noqa: E402  (after path setup in module)
    HERE,
    Runtime,
    build_runtime,
    stage_example_inputs,
    stage_reference_outputs,
)
from coreml.wrappers import STAGE_NAMES, build_wrapper  # noqa: E402

PACKAGES_DIR = HERE / "coreml" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)


def _tensor_metric(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().to(torch.float64).cpu().numpy()
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {
            "shape_a": tuple(a.shape),
            "shape_b": tuple(b.shape),
            "mse": float("nan"),
            "max_abs_delta": float("nan"),
        }
    diff = a - b
    return {
        "shape": tuple(a.shape),
        "mse": float(np.mean(diff * diff)),
        "max_abs_delta": float(np.max(np.abs(diff))),
        "rms_a": float(np.sqrt(np.mean(a * a))),
        "rms_b": float(np.sqrt(np.mean(b * b))),
    }


def _trace_module(
    wrapper: torch.nn.Module, example_inputs: tuple, *, freeze: bool = False
) -> torch.jit.ScriptModule:
    """Trace.

    `torch.jit.freeze` was tried initially to constant-fold
    `aten::size`-derived `prim::NumToTensor` -> `aten::Int` chains, but
    it folds LSTM init hidden state into a 0-d tensor `prim::Constant`
    that coremltools' `Const` type-inference can't handle (recursive
    `any_symbolic` iterates it and torch raises). The `aten::Int`
    fallout is now handled directly by the `_patch_coreml_int_cast`
    monkey-patch above, so freeze is no longer needed.
    """
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False, strict=False)
        if freeze:
            traced = torch.jit.freeze(traced)
    return traced


def _ct_inputs_for_stage(stage: str, example_inputs: tuple) -> list:
    """Build the coremltools input descriptor list matching the trace inputs.

    Token-axis (T_text) and frame-axis (T_frames) dimensions are promoted
    to `ct.RangeDim` so the converted models accept any sentence length
    without padding. This avoids LSTM bidirectional contamination on the
    token axis (text_encoder / duration_predictor's pack_padded_sequence
    is dropped during tracing) and decoder convolution / AdaIN
    contamination on the frame axis when zero-padding the time axis.

    Bounds are generous (1..512 tokens, 1..2048 frames) to cover all
    plausible LibriTTS-style sentences. The default in `ct.Shape(default=)`
    is the trace shape so CoreML's optimizer can specialise.
    """
    import coremltools as ct  # local import so importing convert.py is cheap

    T_TOKEN = ct.RangeDim(lower_bound=1, upper_bound=512, default=57)
    T_FRAME = ct.RangeDim(lower_bound=1, upper_bound=2048, default=147)
    F0_LEN = ct.RangeDim(lower_bound=2, upper_bound=4096, default=294)            # = 2 * T_FRAME
    T_FRAME2 = ct.RangeDim(lower_bound=2, upper_bound=4096, default=294)          # = 2 * T_FRAME (decoder_upsample x_pre time axis)
    HAR_LEN = ct.RangeDim(lower_bound=600, upper_bound=1228800, default=88200)    # = 300 * F0_LEN = 600 * T_FRAME

    descs = []
    if stage == "text_encoder":
        tokens, lengths, mask = example_inputs
        descs.append(ct.TensorType(name="tokens", shape=ct.Shape(shape=(1, T_TOKEN)), dtype=np.int32))
        descs.append(ct.TensorType(name="input_lengths", shape=tuple(lengths.shape), dtype=np.int32))
        # text_mask is consumed multiplicatively in the wrapper, so fp32 IO is fine.
        descs.append(ct.TensorType(name="text_mask", shape=ct.Shape(shape=(1, T_TOKEN)), dtype=np.float32))
    elif stage == "bert":
        # HF Albert's MIL graph contains shape ops that the CPU MLProgram
        # backend rejects under RangeDim ("data-dependent shapes were
        # disabled"). Keep T fixed and let the caller pad tokens to 57 —
        # BERT respects attention_mask so contamination at real positions
        # is bounded and small.
        tokens, attn = example_inputs
        descs.append(ct.TensorType(name="tokens", shape=tuple(tokens.shape), dtype=np.int32))
        descs.append(ct.TensorType(name="attention_mask", shape=tuple(attn.shape), dtype=np.int32))
    elif stage == "ref_encoder":
        (mel_4d,) = example_inputs
        descs.append(ct.TensorType(name="mel", shape=tuple(mel_4d.shape), dtype=np.float32))
    elif stage == "duration_predictor":
        d_en, s, mask = example_inputs
        descs.append(
            ct.TensorType(name="d_en", shape=ct.Shape(shape=(1, d_en.shape[1], T_TOKEN)), dtype=np.float32)
        )
        descs.append(ct.TensorType(name="s", shape=tuple(s.shape), dtype=np.float32))
        descs.append(ct.TensorType(name="text_mask", shape=ct.Shape(shape=(1, T_TOKEN)), dtype=np.float32))
    elif stage == "f0n_predictor":
        en, s = example_inputs
        descs.append(
            ct.TensorType(name="en", shape=ct.Shape(shape=(1, en.shape[1], T_FRAME)), dtype=np.float32)
        )
        descs.append(ct.TensorType(name="s", shape=tuple(s.shape), dtype=np.float32))
    elif stage == "har_source":
        (f0,) = example_inputs
        descs.append(ct.TensorType(name="f0", shape=ct.Shape(shape=(1, F0_LEN)), dtype=np.float32))
    elif stage == "decoder":
        asr, f0, n, ref, har = example_inputs
        descs.append(
            ct.TensorType(name="asr", shape=ct.Shape(shape=(1, asr.shape[1], T_FRAME)), dtype=np.float32)
        )
        descs.append(ct.TensorType(name="f0", shape=ct.Shape(shape=(1, F0_LEN)), dtype=np.float32))
        descs.append(ct.TensorType(name="n", shape=ct.Shape(shape=(1, F0_LEN)), dtype=np.float32))
        descs.append(ct.TensorType(name="ref", shape=tuple(ref.shape), dtype=np.float32))
        descs.append(
            ct.TensorType(
                name="har_source", shape=ct.Shape(shape=(1, 1, HAR_LEN)), dtype=np.float32
            )
        )
    elif stage == "decoder_pre":
        asr, f0, n, ref = example_inputs
        descs.append(
            ct.TensorType(name="asr", shape=ct.Shape(shape=(1, asr.shape[1], T_FRAME)), dtype=np.float32)
        )
        descs.append(ct.TensorType(name="f0_pred", shape=ct.Shape(shape=(1, F0_LEN)), dtype=np.float32))
        descs.append(ct.TensorType(name="n_pred", shape=ct.Shape(shape=(1, F0_LEN)), dtype=np.float32))
        descs.append(ct.TensorType(name="ref", shape=tuple(ref.shape), dtype=np.float32))
    elif stage == "decoder_upsample":
        x_pre, ref, har = example_inputs
        descs.append(
            ct.TensorType(name="x_pre", shape=ct.Shape(shape=(1, x_pre.shape[1], T_FRAME2)), dtype=np.float32)
        )
        descs.append(ct.TensorType(name="ref", shape=tuple(ref.shape), dtype=np.float32))
        descs.append(
            ct.TensorType(
                name="har_source", shape=ct.Shape(shape=(1, 1, HAR_LEN)), dtype=np.float32
            )
        )
    elif stage == "diffusion_unet":
        # As with bert, the U-Net's cross-attention over the token axis
        # produces a shape op MLProgram CPU can't satisfy at runtime when
        # T is RangeDim. Keep fixed; rely on padded BERT output. The
        # padded positions in `embedding` are near-zero (BERT respects
        # attention_mask) so the U-Net's attention to them perturbs
        # `s_pred` only slightly.
        x_noisy, sigma, embedding, features = example_inputs
        descs.append(ct.TensorType(name="x_noisy", shape=tuple(x_noisy.shape), dtype=np.float32))
        descs.append(ct.TensorType(name="sigma", shape=tuple(sigma.shape), dtype=np.float32))
        descs.append(ct.TensorType(name="embedding", shape=tuple(embedding.shape), dtype=np.float32))
        descs.append(ct.TensorType(name="features", shape=tuple(features.shape), dtype=np.float32))
    else:
        raise NotImplementedError(stage)
    return descs


def convert_stage(stage: str, rt: Runtime, *, precision: str = "fp32") -> Path:
    import coremltools as ct

    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown stage {stage!r}")
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be fp32 or fp16, got {precision!r}")

    print(f"\n=== {stage} (precision={precision}) ===")
    wrapper = build_wrapper(stage, rt.model)
    example_inputs = stage_example_inputs(stage, rt)
    print("  trace inputs:")
    for i, t in enumerate(example_inputs):
        print(f"    [{i}] {tuple(t.shape)} {t.dtype}")

    # 1) Eager forward — what we expect the CoreML output to match.
    # Decoder's SineGen / SourceModuleHnNSF have their three
    # `torch.rand`/`randn_like` calls patched to zeros inside the
    # wrapper, so trace, eager, and parity reference are deterministic
    # by construction.
    with torch.no_grad():
        eager_out = wrapper(*example_inputs)
    if not isinstance(eager_out, tuple):
        eager_out = (eager_out,)
    for i, t in enumerate(eager_out):
        print(f"  eager out [{i}]: {tuple(t.shape)} {t.dtype}")

    # 2) Trace.
    t0 = time.time()
    traced = _trace_module(wrapper, example_inputs)
    print(f"  traced in {time.time() - t0:.2f}s")

    # 3) Sanity: traced forward matches eager forward.
    with torch.no_grad():
        traced_out = traced(*example_inputs)
    if not isinstance(traced_out, tuple):
        traced_out = (traced_out,)
    for i, (a, b) in enumerate(zip(eager_out, traced_out)):
        m = _tensor_metric(a, b)
        print(f"  trace parity [{i}]: mse={m['mse']:.3e} max|d|={m['max_abs_delta']:.3e}")

    # 4) Convert.
    t0 = time.time()
    inputs = _ct_inputs_for_stage(stage, example_inputs)
    ct_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct_precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    suffix = "_fp16" if precision == "fp16" else ""
    out_path = PACKAGES_DIR / f"{stage}{suffix}.mlpackage"
    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    print(f"  converted in {time.time() - t0:.2f}s -> {out_path.relative_to(HERE)}")
    return out_path


def quantize_stage(stage: str, *, source_precision: str = "fp16",
                   weight_threshold: int = 2048) -> Path:
    """Post-training int8 weight-only quantization on an existing mlpackage.

    Reads `coreml/packages/<stage>{_fp16}.mlpackage`, applies
    `coremltools.optimize.coreml.linear_quantize_weights` (per-channel
    symmetric int8, weights ≥ `weight_threshold` elements), saves as
    `<stage>_int8.mlpackage`. Activations stay fp16.

    HiFi-GAN-style decoders are bandwidth-bound: int8 weights cut DRAM
    traffic 4× per inference and typically deliver 1.5-2× warm speedup
    on CPU. Per-channel scales preserve accuracy on ConvTranspose1d
    (per-tensor scales lose ~0.5 dB of dynamic range across channels).

    Reality check (decoder_upsample, M-series CPU): linear int8 was
    *slower* than fp16 (385 vs 274 ms warm). CoreML's CPU backend has
    no native int8 ConvTranspose1d kernel and dequantizes per-call.
    Int8 only helps where ANE accepts the graph (this one doesn't).
    Kept here for stages that *do* land on ANE.
    """
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    suffix_in = "_fp16" if source_precision == "fp16" else ""
    src = PACKAGES_DIR / f"{stage}{suffix_in}.mlpackage"
    if not src.exists():
        raise FileNotFoundError(f"missing source {src} — convert it first")

    print(f"\n=== quantize {stage} (int8 from {source_precision}) ===")
    print(f"  source: {src.relative_to(HERE)}")
    print(f"  weight_threshold: {weight_threshold}")
    t0 = time.time()
    mlmodel = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f"  loaded in {time.time() - t0:.2f}s")

    op_cfg = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
        weight_threshold=weight_threshold,
    )
    cfg = OptimizationConfig(global_config=op_cfg)
    t0 = time.time()
    quantized = linear_quantize_weights(mlmodel, config=cfg)
    print(f"  quantized in {time.time() - t0:.2f}s")

    out = PACKAGES_DIR / f"{stage}_int8.mlpackage"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    quantized.save(str(out))
    src_size = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
    out_size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(
        f"  saved -> {out.relative_to(HERE)}  "
        f"({src_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB, "
        f"{out_size / src_size * 100:.0f}%)"
    )
    return out


def palettize_stage(stage: str, *, source_precision: str = "fp16",
                    nbits: int = 8, granularity: str = "per_tensor",
                    weight_threshold: int = 2048) -> Path:
    """Post-training k-means weight palettization on an existing mlpackage.

    Reads `coreml/packages/<stage>{_fp16}.mlpackage`, applies
    `coremltools.optimize.coreml.palettize_weights` (k-means with
    2**nbits centroids, weights ≥ `weight_threshold` elements), saves
    as `<stage>_int<nbits>pal.mlpackage`. Activations stay fp16.

    Why try this when linear int8 lost: at runtime, palette dequant is
    a fp16 LUT fetch (no per-weight multiply), which can hit faster
    Accelerate paths on CPU. ANE has historically had better support
    for palettized weights than per-channel linear int8 — there's a
    chance HiFi-GAN's ConvTranspose1d ups stack actually compiles for
    ANE in this representation (the linear-int8 graph hit
    ANECCompile() FAILED).

    Default `nbits=8 / per_tensor`: 256 centroids shared across the
    whole tensor, ~50% size, near-fp16 quality. Bump to per_channel
    granularity (group_size=1 grouped channel) for tighter per-output
    centroids if A/B reveals quality loss.
    """
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    suffix_in = "_fp16" if source_precision == "fp16" else ""
    src = PACKAGES_DIR / f"{stage}{suffix_in}.mlpackage"
    if not src.exists():
        raise FileNotFoundError(f"missing source {src} — convert it first")

    print(f"\n=== palettize {stage} (int{nbits}pal from {source_precision}) ===")
    print(f"  source: {src.relative_to(HERE)}")
    print(f"  nbits: {nbits}  granularity: {granularity}")
    print(f"  weight_threshold: {weight_threshold}")
    t0 = time.time()
    mlmodel = ct.models.MLModel(str(src), compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f"  loaded in {time.time() - t0:.2f}s")

    op_cfg = OpPalettizerConfig(
        mode="kmeans",
        nbits=nbits,
        granularity=granularity,
        weight_threshold=weight_threshold,
    )
    cfg = OptimizationConfig(global_config=op_cfg)
    t0 = time.time()
    palettized = palettize_weights(mlmodel, config=cfg)
    print(f"  palettized in {time.time() - t0:.2f}s")

    out = PACKAGES_DIR / f"{stage}_int{nbits}pal.mlpackage"
    if out.exists():
        import shutil
        shutil.rmtree(out)
    palettized.save(str(out))
    src_size = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
    out_size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(
        f"  saved -> {out.relative_to(HERE)}  "
        f"({src_size / 1e6:.1f} MB -> {out_size / 1e6:.1f} MB, "
        f"{out_size / src_size * 100:.0f}%)"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        help=f"one of {', '.join(STAGE_NAMES)} or 'all' (default).",
    )
    parser.add_argument("--text", default="StyleTTS 2 is a text to speech model.")
    parser.add_argument(
        "--reference",
        default=str(HERE / "reference_audio" / "696_92939_000016_000006.wav"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16", "int8", "int8pal"],
        help=(
            "fp32 / fp16: full convert + save with matching suffix. "
            "int8: post-training linear weight-only quantization of an "
            "existing fp16 mlpackage. "
            "int8pal: post-training k-means weight palettization "
            "(8-bit, 256 centroids per tensor) of an existing fp16 "
            "mlpackage. Both quant paths require the matching fp16 to "
            "already exist (no re-trace)."
        ),
    )
    parser.add_argument(
        "--weight-threshold",
        type=int,
        default=2048,
        help="int8 / int8pal only: skip tensors with fewer elements than this (default 2048).",
    )
    parser.add_argument(
        "--palette-granularity",
        default="per_tensor",
        choices=["per_tensor", "per_grouped_channel"],
        help="int8pal only: palette scope (default per_tensor — one 256-LUT per weight).",
    )
    args = parser.parse_args()

    # Quant paths don't need the eager runtime — they operate on
    # existing mlpackages.
    if args.precision in ("int8", "int8pal"):
        if args.stage == "all":
            raise SystemExit(
                f"--precision {args.precision} requires --stage <name>; "
                "process one stage at a time."
            )
        try:
            if args.precision == "int8":
                quantize_stage(args.stage, source_precision="fp16",
                               weight_threshold=args.weight_threshold)
            else:
                palettize_stage(args.stage, source_precision="fp16",
                                nbits=8, granularity=args.palette_granularity,
                                weight_threshold=args.weight_threshold)
            return 0
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {type(e).__name__}: {e}")
            return 1

    rt = build_runtime(text=args.text, reference=args.reference, seed=args.seed)
    print(f"\nRuntime ready: tokens={tuple(rt.captures.tokens.shape)} "
          f"frames={tuple(rt.captures.en.shape)}")

    if args.stage == "all":
        stages = list(STAGE_NAMES)
    else:
        stages = [args.stage]

    failures: list[tuple[str, str]] = []
    for stage in stages:
        try:
            convert_stage(stage, rt, precision=args.precision)
        except Exception as e:  # noqa: BLE001
            import traceback

            failures.append((stage, f"{type(e).__name__}: {e}"))
            print(f"  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n=== summary ===")
    for stage in stages:
        if any(f[0] == stage for f in failures):
            print(f"  {stage:<22s} FAIL")
        else:
            print(f"  {stage:<22s} ok")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
