"""Private MLE5Engine API wrapper for detailed backend profiling.

Uses undocumented CoreML internals discovered via runtime introspection.
These APIs may break across macOS versions.

Based on: freedomtan/coreml_modelc_profling (coreml_profiling_without_compute_plan_2.m)

API path on macOS 26:
  model.valueForKey_("program") → MLE5Engine
  engine.valueForKey_("programLibrary") → MLE5ProgramLibrary
  progLib.segmentationAnalyticsAndReturnError_(None) → NSDictionary

The NSDictionary keys are analytics.mil file paths, values are NSDictionaries with:
  DebugName, OpType, SelectedBackend, BackendSupport, EstimatedRunTime,
  ValidationMessages, OpIndex, OpPath (array with Output name)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import CoreML
from Foundation import NSURL

from .compute_plan import COMPUTE_UNITS


def get_detailed_profile(model_path: Path, compute_units: str) -> dict | None:
    """Get detailed per-operation profiling via private MLE5Engine APIs.

    Returns enriched operation data or None if private APIs are unavailable.
    """
    url = NSURL.fileURLWithPath_(str(model_path))
    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(COMPUTE_UNITS[compute_units])

    # Private API: enable profiling options
    try:
        config.setValue_forKey_(1, "profilingOptions")
    except Exception:
        pass

    # Load the model
    model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(
        url, config, None
    )
    if error or model is None:
        err_msg = str(error) if error else "unknown error"

        return None

    # Navigate: model → MLE5Engine → MLE5ProgramLibrary → segmentationAnalytics
    analytics = _extract_segmentation_analytics(model)
    if analytics is None:
        return None

    return _parse_analytics(analytics, compute_units)


def _try_kvc(obj: Any, key: str) -> Any:
    """Safely get a KVC attribute from an ObjC object."""
    try:
        return obj.valueForKey_(key)
    except Exception:
        return None


def _extract_segmentation_analytics(model: Any) -> Any:
    """Navigate the private object graph to get segmentation analytics."""
    # model.program returns MLE5Engine on macOS 26
    engine = _try_kvc(model, "program")
    if engine is None:

        return None

    # MLE5Engine.programLibrary → MLE5ProgramLibrary
    prog_lib = _try_kvc(engine, "programLibrary")
    if prog_lib is None:

        return None

    # MLE5ProgramLibrary.segmentationAnalyticsAndReturnError: → NSDictionary
    try:
        analytics = prog_lib.segmentationAnalyticsAndReturnError_(None)
        if analytics is not None:
            return analytics
    except Exception as e:
        pass

    return None


def _extract_output_name(entry: Any) -> str:
    """Extract the operation output name from OpPath array."""
    op_path = entry.get("OpPath")
    if op_path:
        for item in op_path:
            output = item.get("Output") if hasattr(item, "get") else None
            if output:
                return str(output)
    return ""


def _parse_analytics(analytics: Any, compute_units: str) -> dict:
    """Parse the NSDictionary from segmentationAnalytics into our JSON schema."""
    operations = []

    for key in analytics:
        entry = analytics[key]
        if not hasattr(entry, "get"):
            continue

        op_type = str(entry.get("OpType") or "")
        selected_raw = str(entry.get("SelectedBackend") or "")
        # SelectedBackend comes with embedded quotes like '"bnns"'
        selected_backend = selected_raw.strip('"')

        output_name = _extract_output_name(entry)
        debug_name = str(entry.get("DebugName") or "")

        # Parse backend support dict
        supported_backends = []
        backend_support = entry.get("BackendSupport")
        if backend_support and hasattr(backend_support, "__iter__"):
            for backend_name in backend_support:
                supported_backends.append(str(backend_name))

        # Parse estimated runtimes
        estimated_runtime = {}
        runtime_data = entry.get("EstimatedRunTime")
        if runtime_data and hasattr(runtime_data, "__iter__"):
            for backend_name in runtime_data:
                try:
                    estimated_runtime[str(backend_name)] = float(runtime_data[backend_name])
                except (ValueError, TypeError):
                    pass

        # Parse validation messages
        validation_messages = {}
        val_msgs = entry.get("ValidationMessages")
        if val_msgs and hasattr(val_msgs, "__iter__"):
            for backend_name in val_msgs:
                msg = val_msgs[backend_name]
                if msg:
                    validation_messages[str(backend_name)] = str(msg)

        op_index = int(entry.get("OpIndex") or 0)

        operations.append({
            "name": output_name or debug_name,
            "type": op_type,
            "device": _normalize_backend(selected_backend),
            "op_index": op_index,
            "detailed": {
                "selected_backend": selected_backend,
                "supported_backends": sorted(supported_backends),
                "estimated_runtime_ms": estimated_runtime,
                "validation_messages": validation_messages,
            },
        })

    # Sort by op_index for consistent ordering
    operations.sort(key=lambda o: o["op_index"])

    # Remove op_index from output (internal use only)
    for op in operations:
        del op["op_index"]

    return {
        "compute_units": compute_units,
        "operations": operations,
    }


def _normalize_backend(name: str) -> str:
    """Normalize backend name to cpu/gpu/ane."""
    lower = name.lower()
    if "bnns" in lower or "cpu" in lower or "classic_cpu" in lower:
        return "cpu"
    if "gpu" in lower or "metal" in lower:
        return "gpu"
    if "ane" in lower or "neural" in lower:
        return "ane"
    return lower or "unknown"
