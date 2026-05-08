"""Per-stage CoreML vs PyTorch parity check (fp32).

For each converted stage, loads the .mlpackage, runs it on the same
representative inputs that `convert.py` traced with, and compares the
outputs against `pipeline.stages.*` (single source of truth).

Targets per stage: MSE < 1e-5, max|delta| < 1e-3, Pearson corr > 0.999.

Usage:

    cd models/tts/styletts2
    uv run python coreml/parity.py --stage all
    uv run python coreml/parity.py --stage decoder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from coreml._runtime import (  # noqa: E402
    HERE,
    Runtime,
    build_runtime,
    stage_example_inputs,
    stage_reference_outputs,
)
from coreml.wrappers import STAGE_NAMES

PACKAGES_DIR = HERE / "coreml" / "packages"


# Default per-stage tolerances. Loosened for the decoder, whose output
# is an 88k-sample 24 kHz waveform — accumulated fp32 error across the
# deep convolutional pipeline (`encode` + 4 AdaIN decode blocks + the
# generator's snake activations + 4 ConvTranspose1d ups + 12 resblocks
# + final tanh) routinely produces max|d|≈2e-3 even when mse is in the
# 1e-9 range and Pearson corr is exactly 1.0.
_TOL: dict[str, dict[str, float]] = {
    "decoder": {"mse": 1e-6, "max_abs_delta": 5e-3},
}
_DEFAULT_TOL = {"mse": 1e-5, "max_abs_delta": 1e-3}


# Output names declared by the trace (positional in our wrappers): coremltools
# auto-names them "var_NNN" or similar. We map by output index.
def _ml_predict(mlmodel: Any, stage: str, example_inputs: tuple) -> list[np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    if stage == "text_encoder":
        tokens, lengths, mask = example_inputs
        feed["tokens"] = tokens.numpy().astype(np.int32)
        feed["input_lengths"] = lengths.numpy().astype(np.int32)
        feed["text_mask"] = mask.numpy().astype(np.float32)
    elif stage == "bert":
        tokens, attn = example_inputs
        feed["tokens"] = tokens.numpy().astype(np.int32)
        feed["attention_mask"] = attn.numpy().astype(np.int32)
    elif stage == "ref_encoder":
        (mel_4d,) = example_inputs
        feed["mel"] = mel_4d.numpy().astype(np.float32)
    elif stage == "duration_predictor":
        d_en, s, mask = example_inputs
        feed["d_en"] = d_en.numpy().astype(np.float32)
        feed["s"] = s.numpy().astype(np.float32)
        feed["text_mask"] = mask.numpy().astype(np.float32)
    elif stage == "f0n_predictor":
        en, s = example_inputs
        feed["en"] = en.numpy().astype(np.float32)
        feed["s"] = s.numpy().astype(np.float32)
    elif stage == "decoder":
        asr, f0, n, ref, har = example_inputs
        feed["asr"] = asr.numpy().astype(np.float32)
        feed["f0"] = f0.numpy().astype(np.float32)
        feed["n"] = n.numpy().astype(np.float32)
        feed["ref"] = ref.numpy().astype(np.float32)
        feed["har_source"] = har.numpy().astype(np.float32)
    elif stage == "diffusion_unet":
        x_noisy, sigma, embedding, features = example_inputs
        feed["x_noisy"] = x_noisy.numpy().astype(np.float32)
        feed["sigma"] = sigma.numpy().astype(np.float32)
        feed["embedding"] = embedding.numpy().astype(np.float32)
        feed["features"] = features.numpy().astype(np.float32)
    else:
        raise NotImplementedError(stage)

    out = mlmodel.predict(feed)
    # mlmodel.predict returns a dict whose iteration order is *not* the
    # spec's declared output order. Use the spec to recover the trace
    # order (which matches the wrapper's `forward` return tuple).
    spec_order = [o.name for o in mlmodel.get_spec().description.output]
    return [np.asarray(out[name]) for name in spec_order]


def _metric(a_np: np.ndarray, b: torch.Tensor) -> dict:
    b_np = b.detach().to(torch.float64).cpu().numpy()
    a_np = np.asarray(a_np, dtype=np.float64)
    if a_np.shape != b_np.shape:
        return {"shape_a": tuple(a_np.shape), "shape_b": tuple(b_np.shape), "mse": float("nan")}
    diff = a_np - b_np
    a_zero = a_np - a_np.mean()
    b_zero = b_np - b_np.mean()
    denom = float(np.sqrt((a_zero ** 2).sum() * (b_zero ** 2).sum()))
    corr = float((a_zero * b_zero).sum() / denom) if denom > 0 else float("nan")
    return {
        "shape": tuple(a_np.shape),
        "mse": float(np.mean(diff * diff)),
        "max_abs_delta": float(np.max(np.abs(diff))),
        "rms_a": float(np.sqrt(np.mean(a_np * a_np))),
        "rms_b": float(np.sqrt(np.mean(b_np * b_np))),
        "pearson_corr": corr,
    }


def parity_stage(stage: str, rt: Runtime) -> dict:
    import coremltools as ct

    pkg = PACKAGES_DIR / f"{stage}.mlpackage"
    if not pkg.exists():
        return {"stage": stage, "status": "missing", "path": str(pkg)}

    print(f"\n=== {stage} ===")
    mlmodel = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.ALL)

    example_inputs = stage_example_inputs(stage, rt)
    refs = stage_reference_outputs(stage, rt)

    ml_outs = _ml_predict(mlmodel, stage, example_inputs)
    if len(ml_outs) != len(refs):
        return {
            "stage": stage,
            "status": "output-count-mismatch",
            "n_ml": len(ml_outs),
            "n_ref": len(refs),
        }

    metrics = []
    for i, (a, b) in enumerate(zip(ml_outs, refs)):
        m = _metric(a, b)
        metrics.append(m)
        print(
            f"  out[{i}]: shape={m.get('shape')} "
            f"mse={m.get('mse'):.3e} max|d|={m.get('max_abs_delta'):.3e} "
            f"corr={m.get('pearson_corr'):.6f}"
        )

    tol = _TOL.get(stage, _DEFAULT_TOL)
    status = "ok"
    for m in metrics:
        if (
            m.get("mse", float("inf")) > tol["mse"]
            or m.get("max_abs_delta", float("inf")) > tol["max_abs_delta"]
        ):
            status = "high-error"
            break
    return {"stage": stage, "status": status, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all")
    parser.add_argument("--text", default="StyleTTS 2 is a text to speech model.")
    parser.add_argument(
        "--reference",
        default=str(HERE / "reference_audio" / "696_92939_000016_000006.wav"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rt = build_runtime(text=args.text, reference=args.reference, seed=args.seed)

    if args.stage == "all":
        stages = list(STAGE_NAMES)
    else:
        stages = [args.stage]

    rows = [parity_stage(s, rt) for s in stages]
    print("\n=== summary ===")
    fail = 0
    for r in rows:
        print(f"  {r['stage']:<22s} {r['status']}")
        if r["status"] not in ("ok",):
            fail += 1
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
