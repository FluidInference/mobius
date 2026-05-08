"""Dump every CoreML stage's input/output as .npy fixtures.

Wraps `coreml.inference._load_stage` + `_predict` to log feeds and
outputs to disk, then runs the existing Python pipeline. The Swift
side-loader (`Iter3TTS`) reads these fixtures, runs each stage's
predict in Swift, and writes a WAV.

Usage (from `models/tts/styletts2/`):

    uv run python iteration_3/swift/dump_intermediates.py \
        --text "StyleTTS 2 is a text to speech model." \
        --reference reference_audio/696_92939_000016_000006.wav

Writes to `iteration_3/swift/fixtures/<stage>/{in_*.npy, out_*.npy}`
plus a `manifest.json` describing shapes, dtypes, and the stage order.
Also writes `iteration_3/swift/fixtures_python.wav` for parity check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent  # iteration_3/swift
STYLETTS_ROOT = HERE.parent.parent       # models/tts/styletts2
FIXTURES = HERE / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

# Make `coreml.inference` importable.
if str(STYLETTS_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLETTS_ROOT))

import coreml.inference as inf  # noqa: E402

orig_load = inf._load_stage
orig_predict = inf._predict

# stage_name → {"compute": str, "precision": str, "inputs": [...], "outputs": [...]}
manifest: dict[str, dict] = {}
# stage_name → call index (some stages run >1 time, e.g. diffusion_unet
# in the unfused path; under iteration_3 every stage runs once)
call_counts: dict[str, int] = {}


def _np_dtype_str(arr: np.ndarray) -> str:
    return str(arr.dtype)


def patched_load(stage, *, precision=None, compute_units=None):
    m = orig_load(stage, precision=precision, compute_units=compute_units)
    setattr(m, "_dump_stage", stage)
    setattr(m, "_dump_precision", precision or inf._STAGE_PRECISION[stage])
    setattr(m, "_dump_compute", str(compute_units or inf._STAGE_COMPUTE[stage]).split(".")[-1])
    return m


def patched_predict(model, feed):
    stage = getattr(model, "_dump_stage", None)
    if stage is None:
        return orig_predict(model, feed)

    idx = call_counts.get(stage, 0)
    call_counts[stage] = idx + 1
    suffix = "" if idx == 0 else f"_call{idx}"

    out_dir = FIXTURES / f"{stage}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_meta = []
    for k, v in feed.items():
        arr = np.ascontiguousarray(np.asarray(v))
        np.save(out_dir / f"in_{k}.npy", arr)
        inputs_meta.append({"name": k, "shape": list(arr.shape), "dtype": _np_dtype_str(arr)})

    outs = orig_predict(model, feed)
    out_names = inf._spec_outputs_in_order(model)
    outputs_meta = []
    for n, arr in zip(out_names, outs):
        arr = np.ascontiguousarray(np.asarray(arr))
        np.save(out_dir / f"out_{n}.npy", arr)
        outputs_meta.append({"name": n, "shape": list(arr.shape), "dtype": _np_dtype_str(arr)})

    manifest.setdefault(stage, {
        "compute": getattr(model, "_dump_compute", "?"),
        "precision": getattr(model, "_dump_precision", "?"),
        "calls": [],
    })
    manifest[stage]["calls"].append({
        "index": idx,
        "dir": f"{stage}{suffix}",
        "inputs": inputs_meta,
        "outputs": outputs_meta,
    })
    return outs


inf._load_stage = patched_load
inf._predict = patched_predict


def main() -> int:
    # Default --output to a sibling WAV next to fixtures/.
    new_argv = ["dump_intermediates.py"]
    user_args = sys.argv[1:]
    if "--output" not in user_args:
        new_argv += ["--output", str(HERE / "fixtures_python.wav")]
    new_argv += user_args
    sys.argv = new_argv

    ret = inf.main()

    # Stage order is the order in which stages were first loaded (which
    # matches inference.py's load order: text_encoder, bert,
    # ref_encoder, fused_diffusion_sampler, duration_predictor,
    # fused_f0n_har_source, decoder_pre, decoder_upsample).
    ordered = list(manifest.keys())
    out = {
        "version": 1,
        "sample_rate": 24000,
        "stage_order": ordered,
        "stages": manifest,
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(out, indent=2))
    print(f"\nDumped {len(manifest)} stages to {FIXTURES}")
    print(f"Stage order: {ordered}")
    return ret


if __name__ == "__main__":
    sys.exit(main())
