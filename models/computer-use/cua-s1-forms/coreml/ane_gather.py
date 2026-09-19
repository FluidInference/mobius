"""Use unsigned byte indices and shared floating inputs for ANE embedding lookup."""

from __future__ import annotations

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.scope import ScopeInfo, ScopeSource


def rewrite_byte_gathers(converted: ct.models.MLModel) -> ct.models.MLModel:
    """Rewrite only the two byte embedding gathers in the pinned fixed-shape scorer.

    The public int32 inputs remain unchanged. Valid byte IDs are 0...256, all
    exactly representable by float16 and uint16. Reuse the mask's float16 input
    cast, reshape there, and cast to uint16 for gather. This avoids signed-index
    correction and leaves only the three public input casts on CPU in the M5 plan.
    """
    program = converted._mil_program
    if program is None:
        raise ValueError("Rewrite must follow conversion, before serialization")
    function = program.functions["main"]
    gathers = [operation for operation in function.operations if operation.op_type == "gather"]
    if len(gathers) != 2 or any(len(op.x.shape) != 2 or op.x.shape[0] != 257 for op in gathers):
        raise ValueError("Expected exactly two gathers over the original 257-entry byte embedding")

    def float_input(name: str):
        matches = [
            operation.outputs[0]
            for operation in function.operations
            if operation.op_type == "cast"
            and operation.x is function.inputs[name]
            and operation.dtype.val == "fp16"
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one shared float16 input cast for {name}")
        return matches[0]

    context = float_input("context_ids")
    options = float_input("option_ids")
    for operation in gathers:
        scopes = [
            ScopeInfo(source=source, data=data[-1:] if source == ScopeSource.COREMLTOOLS_GRAPH_PASS else data)
            for source, data in operation.scopes.items()
        ]
        with function, mb.scope(*scopes):
            if tuple(operation.indices.shape) == tuple(context.shape):
                source = context
            else:
                expected = (options.shape[0] * options.shape[1], options.shape[2])
                if tuple(operation.indices.shape) != expected:
                    raise ValueError("Unexpected option embedding shape")
                source = mb.reshape(x=options, shape=list(expected), before_op=operation)
            indices = mb.cast(x=source, dtype="uint16", before_op=operation)
            replacement = mb.gather(
                x=operation.x,
                indices=indices,
                axis=int(operation.axis.val),
                batch_dims=int(operation.batch_dims.val),
                validate_indices=False,
                before_op=operation,
            )
        function.replace_uses_of_var_after_op(
            anchor_op=operation, old_var=operation.outputs[0], new_var=replacement
        )
        function.remove_ops([operation])
    program.validate()
    # The program is already FP16. FLOAT32 disables a second precision rewrite;
    # it does not upcast the existing operations or trained weights.
    return ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        pass_pipeline=ct.PassPipeline(["common::dead_code_elimination"]),
    )
