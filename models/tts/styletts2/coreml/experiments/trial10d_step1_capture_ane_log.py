"""Trial 10d Step 1 — capture the actual ANE compiler error for
`decoder_upsample`.

Issue #59 framed Trial 10d as a drill-down into "ANECCompile() FAILED"
to identify the offending MIL op. Step 1 of the plan was: run the
existing iteration_3 fp16 mlpackage with `MLModelConfiguration.computeUnits
= .cpuAndNeuralEngine` while ANE logging is on, and capture the verbose
compile output.

This script is the throwaway probe used for that capture. It does NOT
ship in the inference path. Re-run it any time the iteration_3 fp16
artifact changes, or to re-confirm the rejection reason after a
hypothetical Trial 10e rewrite.

Inputs (HF cache, downloaded by phase4 setup):
  * iteration_3/packages/decoder_upsample_fp16.mlpackage  (production fp16 — what users actually run)
  * iteration_3/swift/fixtures/decoder_upsample/in_*.npy   (parity fixture inputs)

Output: prints the runtime stderr from `ct.models.MLModel(...)
.predict(...)` with `compute_units=CPU_AND_NE`. The errors that matter
land on stderr regardless of `MLLOG=1` / `OS_ACTIVITY_MODE=info` because
they come straight from CoreML's runtime, not the os_log subsystem.

Run:
    cd models/tts/styletts2
    MLLOG=1 OS_ACTIVITY_MODE=info \\
        uv run python coreml/experiments/trial10d_step1_capture_ane_log.py \\
        2>&1 | tee /tmp/trial10d_step1.log

Look for lines like:
    Error: Tensor width goes beyond limit supported (NNNN > 16384.

These are emitted by Apple's ANE Espresso layer when one of the model's
intermediate tensor dimensions exceeds the ANE hardware width limit
(16384). The number to the left of `>` is the MIL tensor width that
caused the rejection.

Companion: `coreml-cli --fallback --json` prints the same `MILCompilerForANE
error: ANECCompile() FAILED` after the JSON body (Trial 10d Step 1a
output is in `coreml/trials.md`).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import coremltools as ct

HF_SNAP = Path(
    "/Users/kikow/.cache/huggingface/hub/models--FluidInference--StyleTTS-2-coreml/"
    "snapshots/1316f73553ad7c6a63377928766b35ab423e2e3d/iteration_3"
)


def _resolve_mlpackage() -> Path:
    """coremltools' MLModel(...) compile step does NOT follow huggingface_hub
    symlinks (huggingface_hub stores blobs in the ~/.cache content-addressed
    store and links them in via symlinks). The compile step copies the package
    to /private/var/folders/... and the symlinks fail to resolve there,
    producing a misleading "weight.bin doesn't exist" error.

    Workaround: dereference the symlinks into a real on-disk copy before
    handing it to MLModel.
    """
    src = HF_SNAP / "packages" / "decoder_upsample_fp16.mlpackage"
    if not src.exists():
        sys.exit(
            f"missing iteration_3 fp16 mlpackage at {src}\n"
            "run `huggingface_hub.snapshot_download(..., allow_patterns=['iteration_3/packages/decoder_upsample_fp16.mlpackage/**'])` first"
        )
    dst = Path(tempfile.gettempdir()) / "decoder_upsample_fp16.mlpackage"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)  # follow symlinks
    return dst


def main() -> None:
    print(
        f"[trial10d-step1] env: MLLOG={os.environ.get('MLLOG')!r} "
        f"OS_ACTIVITY_MODE={os.environ.get('OS_ACTIVITY_MODE')!r}",
        file=sys.stderr,
    )

    mlp = _resolve_mlpackage()
    print(f"[trial10d-step1] dereferenced mlpackage at {mlp}", file=sys.stderr)

    print(
        "[trial10d-step1] loading with .cpuAndNeuralEngine to provoke ANECCompile ...",
        file=sys.stderr,
        flush=True,
    )
    m = ct.models.MLModel(str(mlp), compute_units=ct.ComputeUnit.CPU_AND_NE)

    fixtures = HF_SNAP / "swift" / "fixtures" / "decoder_upsample"
    feed = {
        "x_pre": np.load(fixtures / "in_x_pre.npy"),
        "ref": np.load(fixtures / "in_ref.npy"),
        "har_source": np.load(fixtures / "in_har_source.npy"),
    }
    for k, v in feed.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}", file=sys.stderr)

    print(
        "[trial10d-step1] predicting (ANE compile happens lazily on first predict) ...",
        file=sys.stderr,
        flush=True,
    )
    out = m.predict(feed)
    out_arr = list(out.values())[0]
    print(
        f"[trial10d-step1] predict OK (CPU fallback path). "
        f"output shape={out_arr.shape} dtype={out_arr.dtype}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
