"""Analyze CPU fallback ops and group by rejection reason."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .compute_plan import COMPUTE_UNITS, get_compute_plan
from .private_profiler import get_detailed_profile


def analyze_fallback(model_path: Path, compute_units: str = "cpu_and_neural_engine") -> dict:
    """Analyze which ops fall back to CPU and why.

    Returns a fallback_summary dict with grouped reasons.
    """
    # Get public compute plan for device assignments
    plan = get_compute_plan(model_path, compute_units)

    # Get private profiler data for validation messages
    detail = get_detailed_profile(model_path, compute_units)

    # Build lookup of private data by op name
    private_by_name: dict[str, dict] = {}
    if detail and "operations" in detail:
        for op in detail["operations"]:
            private_by_name[op["name"]] = op.get("detailed", {})

    # Categorize ops
    all_ops = plan["operations"]
    cpu_ops = []
    ane_ops = 0
    gpu_ops = 0

    for op in all_ops:
        if op["device"] == "ane":
            ane_ops += 1
        elif op["device"] == "gpu":
            gpu_ops += 1
        else:
            # CPU or unknown — these are the fallback ops
            priv = private_by_name.get(op["name"], {})
            cpu_ops.append({
                "name": op["name"],
                "type": op["type"],
                "cost_percent": op["cost_percent"],
                "supported_backends": priv.get("supported_backends", []),
                "validation_messages": priv.get("validation_messages", {}),
                "estimated_runtime_ms": priv.get("estimated_runtime_ms", {}),
            })

    # Group by reason
    reasons: dict[str, list[dict]] = defaultdict(list)

    for op in cpu_ops:
        ane_validation = op["validation_messages"].get("ane", "")
        ane_supported = "ane" in op["supported_backends"]

        if ane_validation:
            reasons[ane_validation].append(op)
        elif ane_supported:
            reasons["ANE supported but scheduler chose CPU"].append(op)
        else:
            reasons["ANE not available for this op"].append(op)

    # Build structured output
    reason_list = []
    for reason, ops in sorted(reasons.items(), key=lambda x: -len(x[1])):
        type_counts: dict[str, int] = defaultdict(int)
        for op in ops:
            type_counts[op["type"]] += 1

        total_runtime = sum(
            op["estimated_runtime_ms"].get("bnns", 0) or op["estimated_runtime_ms"].get("classic_cpu", 0)
            for op in ops
        )

        reason_list.append({
            "reason": reason,
            "count": len(ops),
            "estimated_cpu_runtime_ms": round(total_runtime, 4),
            "op_types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
            "ops": [op["name"] for op in ops],
        })

    total = len(all_ops)
    return {
        "compute_units": compute_units,
        "total_ops": total,
        "ane_ops": ane_ops,
        "gpu_ops": gpu_ops,
        "cpu_ops": len(cpu_ops),
        "ane_percent": round(ane_ops / total * 100, 1) if total else 0,
        "reasons": reason_list,
    }
