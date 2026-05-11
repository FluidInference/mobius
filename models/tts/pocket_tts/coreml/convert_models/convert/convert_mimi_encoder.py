"""Convert Mimi audio encoder (stateless) to CoreML.

Traces `TraceableMimiEncoder` (encoder + encoder_transformer + _to_framerate
+ speaker_proj linear) against pocket-tts >= 2.0.0 weights and writes a
CoreML mlpackage.

Input:  audio        [1, 1, 240_000]  float32 (10s @ 24kHz, exactly 125 frames)
Output: conditioning [1, 125, 1024]   float32 (one row per 80ms frame)

This is the encoder counterpart to the deployed Apr 27 cond_step /
flowlm_step / flow_decoder mlpackages. The legacy
`voice_cloning/mimi_encoder.mlmodelc` was traced against a pre-2.0.0
pocket-tts and produces conditioning in the wrong latent space for the
deployed cond_step weights, causing immediate EOS (silent / garbled audio
— FluidAudio issue #592). Re-tracing here restores the pure-CoreML voice
cloning path (`export_voice_coreml.py`).
"""
import argparse
import os
import sys

import coremltools as ct
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
# NOTE: do NOT inject the parent pocket_tts/ directory onto sys.path. The
# local checkout under `mobius/models/tts/pocket_tts/pocket_tts/` is the
# 1.0.3-era source and would shadow the pip-installed `pocket-tts==2.0.0`
# whose weights the deployed mlpackages were traced against. Rely solely on
# the active venv to provide `pocket_tts` (`pip install 'pocket-tts==2.0.0'`).
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))

from _language_arg import (
    add_compute_args,
    add_language_arg,
    build_output_dir,
    resolve_compute_precision,
    resolve_compute_units,
)
from traceable_mimi_encoder import (
    AUDIO_LENGTH_SAMPLES,
    EMBEDDING_DIM,
    NUM_FRAMES,
    TraceableMimiEncoder,
)


def rename_single_output(mlpackage_path: str, desired_name: str) -> None:
    """Rename the auto-generated output (e.g. `var_NNN`/`castN`) to `desired_name`.

    `ct.convert(..., outputs=[ct.TensorType(dtype=np.float32)])` leaves the
    sole output with an anonymous SSA-derived name. The Swift loader needs
    a stable semantic name (`conditioning`).
    """
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()

    output_names = [out.name for out in spec.description.output]
    if len(output_names) != 1:
        raise RuntimeError(
            f"Expected exactly 1 output, found {len(output_names)}: {output_names}"
        )

    current = output_names[0]
    if current == desired_name:
        print(f"Output already named '{desired_name}'; nothing to rename.")
        return

    print(f"Renaming output: {current} -> {desired_name}")
    ct.utils.rename_feature(
        spec, current, desired_name, rename_inputs=False, rename_outputs=True
    )

    weights_dir = os.path.join(mlpackage_path, "Data", "com.apple.CoreML", "weights")
    updated = ct.models.MLModel(
        spec, weights_dir=weights_dir, compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    updated.save(mlpackage_path)


def force_fp32_output_dtype(mlpackage_path: str) -> None:
    """Flip the multiArray output's `dataType` from FLOAT16 to FLOAT32.

    With `compute_precision=fp16` the internal MIL function produces fp16
    values; the boundary cast op inserted by `ct.convert` already promotes
    to fp32, but the spec's port dtype enum may still read FLOAT16. Force
    it to FLOAT32 so Swift `MLMultiArray` allocations stay fp32-friendly
    (the macOS MLE5 binder rejects fp16 buffers from Swift heap allocs).
    """
    from coremltools.proto import FeatureTypes_pb2 as _ft

    FLOAT32 = _ft.ArrayFeatureType.FLOAT32
    FLOAT16 = _ft.ArrayFeatureType.FLOAT16

    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()

    flipped = []
    for out in spec.description.output:
        if not out.type.HasField("multiArrayType"):
            continue
        if out.type.multiArrayType.dataType == FLOAT16:
            out.type.multiArrayType.dataType = FLOAT32
            flipped.append(out.name)

    if not flipped:
        print("All multiArray outputs already FLOAT32; nothing to flip.")
        return

    print(f"Flipping {len(flipped)} output dtype(s) FLOAT16 -> FLOAT32: {flipped}")
    weights_dir = os.path.join(mlpackage_path, "Data", "com.apple.CoreML", "weights")
    updated = ct.models.MLModel(
        spec, weights_dir=weights_dir, compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    updated.save(mlpackage_path)


def convert(language: str, compute_precision: str = "fp16", compute_units: str = "ALL") -> str:
    print(f"Loading PocketTTS model (language={language})...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Creating traceable Mimi encoder...")
    traceable = TraceableMimiEncoder.from_tts_model(model)
    traceable.eval()

    print(f"Tracing on fixed input shape (1, 1, {AUDIO_LENGTH_SAMPLES})...")
    example_audio = torch.zeros(1, 1, AUDIO_LENGTH_SAMPLES)
    with torch.no_grad():
        traced = torch.jit.trace(traceable, (example_audio,), strict=False)

    # Anonymous output spec (renamed below) — naming here would alias an
    # internal SSA value and force a graph rewrite; matching the
    # decoder converter's pattern instead.
    ct_inputs = [ct.TensorType(name="audio", shape=(1, 1, AUDIO_LENGTH_SAMPLES), dtype=np.float32)]
    ct_outputs = [ct.TensorType(dtype=np.float32)]

    print(
        f"Converting to CoreML (precision={compute_precision}, "
        f"inputs=fp32, outputs=fp32 via cast op)..."
    )
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mimi_encoder.mlpackage")
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    rename_single_output(output_path, "conditioning")
    force_fp32_output_dtype(output_path)

    # Reload for the post-save smoke test.
    mlmodel = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))
    spec = mlmodel.get_spec()

    print(f"\n=== INPUTS ({len(spec.description.input)}) ===")
    for inp in spec.description.input:
        if inp.type.HasField("multiArrayType"):
            print(f"  {inp.name}: {list(inp.type.multiArrayType.shape)}")

    print(f"\n=== OUTPUTS ({len(spec.description.output)}) ===")
    for out in spec.description.output:
        if out.type.HasField("multiArrayType"):
            print(f"  {out.name}: {list(out.type.multiArrayType.shape)}")

    print("\nRunning inference smoke test on zero audio...")
    test_audio = np.zeros((1, 1, AUDIO_LENGTH_SAMPLES), dtype=np.float32)
    out = mlmodel.predict({"audio": test_audio})
    cond = out["conditioning"]
    print(f"  conditioning: shape={cond.shape} dtype={cond.dtype} std={cond.std():.6f}")
    assert cond.shape == (1, NUM_FRAMES, EMBEDDING_DIM), (
        f"Unexpected output shape {cond.shape}, expected (1, {NUM_FRAMES}, {EMBEDDING_DIM})"
    )

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    args = parser.parse_args()
    convert(args.language, args.compute_precision, args.compute_units)
