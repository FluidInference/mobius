"""Public MLComputePlan API wrapper for per-operation device assignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import CoreML
from Foundation import NSURL

# Use framework constants — values differ across macOS versions
COMPUTE_UNITS = {
    "all": CoreML.MLComputeUnitsAll,
    "cpu_only": CoreML.MLComputeUnitsCPUOnly,
    "cpu_and_gpu": CoreML.MLComputeUnitsCPUAndGPU,
    "cpu_and_neural_engine": CoreML.MLComputeUnitsCPUAndNeuralEngine,
}


def _device_name(device: Any) -> str:
    """Extract device name string from an MLComputeDevice instance."""
    if device is None:
        return "unknown"
    cls_name = type(device).__name__
    if "CPU" in cls_name:
        return "cpu"
    if "GPU" in cls_name:
        return "gpu"
    if "NeuralEngine" in cls_name:
        return "ane"
    return cls_name


def _walk_operations(block: Any) -> list[Any]:
    """Recursively walk a program block to collect all operations."""
    ops = []
    for op in block.operations():
        ops.append(op)
        # Some ops have nested blocks (e.g., cond, while_loop)
        for sub_block in (op.blocks() or []):
            ops.extend(_walk_operations(sub_block))
    return ops


def get_compute_plan(model_path: Path, compute_units: str) -> dict:
    """Load compute plan for a model with given compute units.

    Returns dict with summary and per-operation breakdown.
    """
    url = NSURL.fileURLWithPath_(str(model_path))
    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(COMPUTE_UNITS[compute_units])

    import threading

    result_holder: dict = {}
    event = threading.Event()

    def completion(loaded_plan: Any, load_error: Any) -> None:
        result_holder["plan"] = loaded_plan
        result_holder["error"] = load_error
        event.set()

    CoreML.MLComputePlan.loadContentsOfURL_configuration_completionHandler_(
        url, config, completion
    )
    event.wait(timeout=30)
    plan = result_holder.get("plan")
    error = result_holder.get("error")

    if error is not None or plan is None:
        err_msg = str(error) if error else "unknown error"
        raise RuntimeError(f"Failed to load compute plan: {err_msg}")

    structure = plan.modelStructure()
    program = structure.program()

    if program is None:
        # Might be a neural network model, not an ML Program
        return _handle_neural_network(plan, structure, compute_units)

    main_fn = program.functions().get("main")
    if main_fn is None:
        raise RuntimeError("No 'main' function in model program")

    all_ops = _walk_operations(main_fn.block())

    operations = []
    device_costs = {"cpu": 0.0, "gpu": 0.0, "ane": 0.0, "unknown": 0.0}

    for op in all_ops:
        device_usage = plan.computeDeviceUsageForMLProgramOperation_(op)
        cost_obj = plan.estimatedCostOfMLProgramOperation_(op)

        device = _device_name(device_usage.preferredComputeDevice() if device_usage else None)
        cost = cost_obj.weight() * 100 if cost_obj else 0.0

        op_type = str(op.operatorName()) if op.operatorName() else ""

        # Get output variable name from MLModelStructureProgramNamedValueType
        outputs = op.outputs() or []
        output_name = ""
        if len(outputs) > 0:
            out0 = outputs[0]
            if hasattr(out0, "name"):
                output_name = str(out0.name())

        # Skip const/identity ops with no device assignment (not compute ops)
        if device == "unknown" and cost == 0.0 and op_type in ("const", "identity"):
            continue

        operations.append({
            "name": output_name or op_type,
            "type": op_type,
            "device": device,
            "cost_percent": round(cost, 4),
        })
        device_costs[device] += cost

    total = sum(device_costs.values())
    if total > 0:
        summary = {
            "cpu_percent": round(device_costs["cpu"] / total * 100, 2),
            "gpu_percent": round(device_costs["gpu"] / total * 100, 2),
            "ane_percent": round(device_costs["ane"] / total * 100, 2),
        }
    else:
        summary = {"cpu_percent": 0.0, "gpu_percent": 0.0, "ane_percent": 0.0}

    return {
        "compute_units": compute_units,
        "summary": summary,
        "operations": operations,
    }


def _handle_neural_network(plan: Any, structure: Any, compute_units: str) -> dict:
    """Handle NeuralNetwork-type models (not ML Programs)."""
    nn = structure.neuralNetwork()
    if nn is None:
        return {
            "compute_units": compute_units,
            "summary": {"cpu_percent": 0.0, "gpu_percent": 0.0, "ane_percent": 0.0},
            "operations": [],
            "error": "Model has no program or neural network structure",
        }

    operations = []
    device_costs = {"cpu": 0.0, "gpu": 0.0, "ane": 0.0, "unknown": 0.0}

    for layer in nn.layers():
        device_usage = plan.computeDeviceUsageForNeuralNetworkLayer_(layer)
        device = _device_name(device_usage.preferredComputeDevice() if device_usage else None)

        layer_name = str(layer.name()) if layer.name() else ""
        layer_type = str(layer.type()) if layer.type() else ""

        operations.append({
            "name": layer_name,
            "type": layer_type,
            "device": device,
            "cost_percent": 0.0,  # NeuralNetwork API doesn't expose cost
        })
        device_costs[device] += 1  # Count-based for NN

    total = sum(device_costs.values())
    if total > 0:
        summary = {
            "cpu_percent": round(device_costs["cpu"] / total * 100, 2),
            "gpu_percent": round(device_costs["gpu"] / total * 100, 2),
            "ane_percent": round(device_costs["ane"] / total * 100, 2),
        }
    else:
        summary = {"cpu_percent": 0.0, "gpu_percent": 0.0, "ane_percent": 0.0}

    return {
        "compute_units": compute_units,
        "summary": summary,
        "operations": operations,
    }
