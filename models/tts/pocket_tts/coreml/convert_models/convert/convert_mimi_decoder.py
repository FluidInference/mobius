"""Convert Mimi streaming decoder to CoreML.

Traces the TraceableMimiDecoder (which bakes in denormalize + quantize)
and converts to CoreML .mlpackage.

v2 schema (pocket_tts >= 2.0.0): attention dropped `end_offset`, so the
total state count is 24 (was 26 in v1). The 3 zero-length state tensors
(`res{0,1,2}_conv1_prev` with T=0) are kept in the mlProgram spec —
stripping them causes "Model and main function must have same number of
inputs and states"; the Swift side provides them as empty MLMultiArrays.

Input:  latent [1, 32]       +  24 state tensors
Output: audio  [1, 1, 1920]  +  24 updated state tensors
"""
import argparse
import torch
import numpy as np
import coremltools as ct
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, _SCRIPT_DIR)  # for: from _language_arg import ...
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))  # for: from traceable_* import ...

from _language_arg import (
    add_compute_args,
    add_language_arg,
    build_output_dir,
    resolve_compute_precision,
    resolve_compute_units,
)
from traceable_mimi_decoder import TraceableMimiDecoder, MIMI_STATE_SPEC


def rename_outputs_semantic(mlpackage_path, state_input_names):
    """Rename auto-generated output names (var_NNN, cast_NN) to semantic names.

    CoreML's tracer assigns outputs anonymous names from the SSA dump.
    Pair them positionally with the inputs they update so the Swift loader
    can discover the I/O mapping without hardcoding numeric suffixes that
    change every conversion run.

    Convention:
      output[0]   → "audio"                  (the [1, 1, 1920] waveform)
      output[i+1] → "<state_input_name>_out" (paired with input[i+1])

    PASS-THROUGH HANDLING: a few state outputs share an SSA value with their
    corresponding input parameter — the `*_first` scalars (only mutated on
    first frame) and the zero-length `res*_conv1_prev` tensors (never
    written when the conv layer's kernel is empty). For those, the trace
    emits an output whose name matches the input. Renaming such an output
    via `ct.utils.rename_feature(rename_outputs=True)` would also rename
    the underlying SSA, which is the input parameter — desyncing the spec
    and breaking the model with "ML Program is missing MLModel
    Specification input <name>" at predict time.

    To avoid that, we DETECT pass-throughs by intersecting current output
    names with the input-name set, and SKIP renaming those. The Swift
    schema loader (`PocketTtsMimiSchema.discover` Path A) is taught to
    accept the bare `<input_name>` as the output for pass-throughs when
    `<input_name>_out` is absent.
    """
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()

    output_names = [out.name for out in spec.description.output]
    input_name_set = {inp.name for inp in spec.description.input}
    expected = 1 + len(state_input_names)
    if len(output_names) != expected:
        raise RuntimeError(
            f"Output count mismatch: spec has {len(output_names)} outputs but expected "
            f"1 audio + {len(state_input_names)} state outputs = {expected}"
        )

    desired = ["audio"] + [f"{n}_out" for n in state_input_names]
    rename_pairs = []
    skipped_passthrough = []
    for old, new in zip(output_names, desired):
        if old == new:
            continue
        if old in input_name_set:
            # Pass-through output aliased to an input SSA; renaming would
            # corrupt the input parameter. Leave it with its original name.
            skipped_passthrough.append(old)
            continue
        rename_pairs.append((old, new))

    if skipped_passthrough:
        print(f"Skipping {len(skipped_passthrough)} pass-through outputs "
              f"(aliased to input SSA): {skipped_passthrough}")

    if not rename_pairs:
        print("No non-pass-through outputs need renaming.")
        return

    print(f"Renaming {len(rename_pairs)} output names to semantic form:")
    for old, new in rename_pairs:
        print(f"  {old} → {new}")
        ct.utils.rename_feature(spec, old, new, rename_inputs=False, rename_outputs=True)

    weights_dir = os.path.join(mlpackage_path, "Data", "com.apple.CoreML", "weights")
    updated = ct.models.MLModel(
        spec, weights_dir=weights_dir, compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    updated.save(mlpackage_path)


def force_fp32_output_dtype(mlpackage_path):
    """Flip every multiArray output's `dataType` from FLOAT16 to FLOAT32.

    The MIL function still produces fp16 internally — this only edits the
    spec's output port `dataType` enum. The CoreML runtime inserts implicit
    fp16→fp32 casts at the boundary so callers see fp32 buffers.

    Why we do this here (not via `outputs=` in `ct.convert`):
        Specifying `outputs=[ct.TensorType(name=...)]` at convert time tries
        to rename the MIL block's output SSA values, which breaks for
        pass-through outputs that share their SSA value with an input
        parameter. A spec-only protobuf edit avoids the SSA layer entirely.

    Why we want fp32 outputs:
        Swift's `MLMultiArray` allocator at `dataType=.float16` produces
        buffers that the macOS MLE5 binder rejects ("Invalid heap allocated
        handle"). Forcing fp32 IO keeps the model Swift-friendly while
        still letting `compute_precision=fp16` give us the fp16 internal
        speed-up.
    """
    from coremltools.proto import FeatureTypes_pb2 as _ft

    FLOAT32 = _ft.ArrayFeatureType.FLOAT32  # 65568
    FLOAT16 = _ft.ArrayFeatureType.FLOAT16  # 65552

    mlmodel = ct.models.MLModel(
        mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()

    flipped = []
    for out in spec.description.output:
        if not out.type.HasField('multiArrayType'):
            continue
        if out.type.multiArrayType.dataType == FLOAT16:
            out.type.multiArrayType.dataType = FLOAT32
            flipped.append(out.name)

    if not flipped:
        print("All multiArray outputs already FLOAT32; nothing to flip.")
        return

    print(f"Flipping {len(flipped)} output dtype(s) FLOAT16 → FLOAT32:")
    for name in flipped:
        print(f"  {name}")

    weights_dir = os.path.join(
        mlpackage_path, "Data", "com.apple.CoreML", "weights")
    updated = ct.models.MLModel(
        spec, weights_dir=weights_dir,
        compute_units=ct.ComputeUnit.CPU_AND_GPU)
    updated.save(mlpackage_path)


def strip_zero_length_io(mlpackage_path):
    """Remove zero-length tensor inputs/outputs from a saved CoreML mlpackage.

    Three Mimi state tensors have a zero-length dimension (kernel_size=1
    streaming conv layers with 0 padding). CoreML Espresso crashes on
    zero-element blobs, so we strip them from the spec.

    Must operate on a saved .mlpackage (not in-memory) because mlProgram
    models require the weights directory when loading from spec.
    """
    # NOTE: zero-length stripping uses CPU_AND_GPU explicitly because we just
    # want to round-trip the spec — the output mlpackage is re-loaded with the
    # caller-selected compute_units below for the inference smoke test.
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = mlmodel.get_spec()

    # Find zero-length input/output names
    zero_inputs = set()
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = list(inp.type.multiArrayType.shape)
            if 0 in shape:
                zero_inputs.add(inp.name)

    zero_outputs = set()
    for out in spec.description.output:
        if out.type.HasField('multiArrayType'):
            shape = list(out.type.multiArrayType.shape)
            if 0 in shape:
                zero_outputs.add(out.name)

    if not zero_inputs and not zero_outputs:
        print("No zero-length tensors to strip.")
        return

    print(f"Stripping {len(zero_inputs)} zero-length inputs: {zero_inputs}")
    print(f"Stripping {len(zero_outputs)} zero-length outputs: {zero_outputs}")

    # Remove from spec
    inputs_to_keep = [inp for inp in spec.description.input
                      if inp.name not in zero_inputs]
    outputs_to_keep = [out for out in spec.description.output
                       if out.name not in zero_outputs]

    del spec.description.input[:]
    spec.description.input.extend(inputs_to_keep)

    del spec.description.output[:]
    spec.description.output.extend(outputs_to_keep)

    # Save modified spec back (with weights dir from the mlpackage)
    weights_dir = os.path.join(mlpackage_path, "Data",
                               "com.apple.CoreML", "weights")
    updated = ct.models.MLModel(spec, weights_dir=weights_dir,
                                compute_units=ct.ComputeUnit.CPU_AND_GPU)  # round-trip only
    updated.save(mlpackage_path)


def convert(language: str, compute_precision: str = "fp16", compute_units: str = "ALL"):
    # NOTE: Mimi codec weights are shared across all languages. We still accept
    # a --language flag so the orchestrator can run this per language for
    # isolated `build/<lang>/` directories; the resulting mlpackage is
    # byte-identical across languages.
    print(f"Loading PocketTTS model (language={language})...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Creating traceable Mimi decoder (with denormalize + quantize baked in)...")
    traceable = TraceableMimiDecoder.from_tts_model(model)
    traceable.eval()

    # Build example inputs from MIMI_STATE_SPEC
    #
    # NOTE: every TensorType is declared with `dtype=np.float32` even when
    # `compute_precision=fp16`. Without this, CoreML's converter promotes the
    # model's IO contract to `tensor<fp16, ...>` to match the compute
    # precision, and the macOS MLE5 buffer binder then refuses to accept
    # `MLMultiArray`s allocated as `.float16` from Swift ("Invalid heap
    # allocated handle"). Forcing fp32 IO inserts implicit fp32↔fp16 casts at
    # the boundary so internal ops still run in fp16 (the perf win we want)
    # while the IO contract stays Swift-friendly.
    print("Creating example inputs...")
    latent = torch.randn(1, 32)
    state_tensors = []
    ct_inputs = [ct.TensorType(name="latent", shape=(1, 32), dtype=np.float32)]

    for name, shape in MIMI_STATE_SPEC:
        t = torch.zeros(*shape)
        state_tensors.append(t)
        ct_inputs.append(
            ct.TensorType(name=name, shape=tuple(shape), dtype=np.float32))

    example_inputs = (latent,) + tuple(state_tensors)

    print(f"Tracing with {len(state_tensors)} state tensors...")
    with torch.no_grad():
        traced = torch.jit.trace(traceable, example_inputs)

    # NOTE: we pass anonymous `outputs=[ct.TensorType(dtype=np.float32), ...]`
    # (no `name=`) so coremltools inserts an fp16→fp32 cast op at every
    # output boundary. This matters because:
    #   1. With `compute_precision=fp16`, internal ops produce fp16 SSA values.
    #   2. Without the cast, the MIL function returns fp16; if we then flip
    #      the spec output dtype to fp32 via protobuf edit, the validator
    #      rejects the model: "Model output 'audio' has a different type than
    #      its corresponding return value to main".
    #   3. NAMING the outputs (e.g. `name='conv0_first_out'`) trips
    #      `sanitize_input_output_names`: pass-through outputs that alias an
    #      input SSA fail the assertion "Main block's input name 'conv0_first'
    #      is different from its corresponding var's name 'conv0_first_out'".
    # Anonymous outputs sidestep that assertion — the inserted cast op breaks
    # the input-aliasing SSA chain, and the auto-generated names get
    # rewritten to semantic form by `rename_outputs_semantic` below.
    n_state_outputs = len(MIMI_STATE_SPEC)
    n_outputs = 1 + n_state_outputs  # audio + 24 state outputs
    ct_outputs = [ct.TensorType(dtype=np.float32) for _ in range(n_outputs)]
    print(f"Converting to CoreML (precision={compute_precision}, "
          f"inputs=fp32, outputs=fp32 via cast ops)...")
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mimi_decoder.mlpackage")
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    # Rename auto-generated outputs (var_NNN, cast_NN) to semantic names so
    # the Swift loader can discover the input ↔ output mapping by position.
    state_input_names = [name for name, _ in MIMI_STATE_SPEC]
    rename_outputs_semantic(output_path, state_input_names)

    # Flip output multiArray dataType FLOAT16 → FLOAT32 in the spec so the
    # IO contract is fp32 (matches the fp32 input dtype declared via
    # `ct.TensorType(..., dtype=np.float32)` above).
    force_fp32_output_dtype(output_path)

    # Reload the mlmodel for the inference smoke-test below (the in-memory
    # `mlmodel` object still references the pre-edit spec).
    mlmodel = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)

    # NOTE: Zero-length I/O stripping is skipped for mlProgram format.
    # The 3 zero-length state tensors (res{0,1,2}_conv1_prev) are kept in the
    # model spec. The Swift side provides them as empty MLMultiArrays.
    # Stripping from the spec description causes "Model and main function must
    # have same number of inputs and states" because the MIL function still
    # references them.

    # Print I/O summary
    spec = mlmodel.get_spec()
    print(f"\n=== INPUTS ({len(spec.description.input)}) ===")
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = list(inp.type.multiArrayType.shape)
            print(f"  {inp.name}: {shape}")

    print(f"\n=== OUTPUTS ({len(spec.description.output)}) ===")
    for out in spec.description.output:
        if out.type.HasField('multiArrayType'):
            shape = list(out.type.multiArrayType.shape)
            print(f"  {out.name}: {shape}")

    # Quick inference test
    print("\nRunning inference test...")
    test_inputs = {}
    for inp in spec.description.input:
        if inp.type.HasField('multiArrayType'):
            shape = list(inp.type.multiArrayType.shape)
            test_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))
    out = coreml_model.predict(test_inputs)
    print(f"  Inference succeeded — {len(out)} outputs")

    for key, val in out.items():
        if hasattr(val, 'shape') and len(val.shape) == 3 and val.shape[-1] == 1920:
            print(f"  Audio output '{key}': {val.shape} (1920 samples = 80ms at 24kHz)")
            break

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    args = parser.parse_args()
    convert(args.language, args.compute_precision, args.compute_units)
