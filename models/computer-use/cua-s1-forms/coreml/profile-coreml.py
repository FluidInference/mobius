"""Profile the shipped model's compute plan and latency using a small real-demo manifest."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

from assets import ROOT, load_demo, sha256, verify_assets
from preprocessing import InputLimits, prepare_inputs
from verify import check_package, latency_summary, validate_prediction

DEFAULT_ROWS = (0, 68, 130)
DEFAULT_UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")


def select_rows(rows: list[dict], indices: list[int] | tuple[int, ...]) -> list[dict]:
    """Require an explicit, nonempty manifest without duplicate or out-of-range rows."""
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Choose at least one distinct demo row")
    if any(index < 0 or index >= len(rows) for index in indices):
        raise ValueError("A selected row is outside the pinned demo")
    return [rows[index] for index in indices]


def compute_plan(model: ct.models.MLModel, units: ct.ComputeUnit) -> dict:
    """Record the public scheduler plan; operation counts are not measured runtime shares."""
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=units)
    program = plan.model_structure.program
    if program is None:
        raise ValueError("Expected the converted ML Program")
    names = {"MLCPUComputeDevice": "cpu", "MLGPUComputeDevice": "gpu", "MLNeuralEngineComputeDevice": "ane"}
    operations = []
    unassigned = Counter()

    def walk(block) -> None:
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is None:
                unassigned[operation.operator_name] += 1
            else:
                device = names.get(type(usage.preferred_compute_device).__name__, "unknown")
                operations.append({
                    "name": operation.outputs[0].name if operation.outputs else operation.operator_name,
                    "type": operation.operator_name,
                    "device": device,
                })
            for nested in operation.blocks:
                walk(nested)

    for function in program.functions.values():
        walk(function.block)
    if not operations:
        raise ValueError("Compute plan contains no assigned operations")
    counts = Counter(operation["device"] for operation in operations)
    return {
        "kind": "MLComputePlan preferred device; not a runtime utilization or power trace",
        "assigned_operation_count": len(operations),
        "device_counts": {name: counts[name] for name in ("cpu", "gpu", "ane", "unknown")},
        "unassigned_operation_types": dict(unassigned),
        "operations": operations,
    }


def predict_checked(model: ct.models.MLModel, arrays: dict, row: dict, capacity: int) -> float:
    """Time only prediction, then reject invalid outputs or a wrong labeled choice."""
    started = time.perf_counter()
    output = model.predict(arrays)
    elapsed = 1000 * (time.perf_counter() - started)
    _, probabilities = validate_prediction(output, len(row["options"]), capacity)
    if int(probabilities.argmax()) != row["label"]:
        raise ValueError(f"Wrong option for {row['meta']['page']}")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/ane-profile.json")
    parser.add_argument("--rows", nargs="+", type=int, default=list(DEFAULT_ROWS))
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--timed-passes", type=int, default=10)
    parser.add_argument("--compute-units", nargs="+", choices=DEFAULT_UNITS, default=list(DEFAULT_UNITS))
    args = parser.parse_args()
    if args.warmup_passes < 1 or args.timed_passes < 1:
        parser.error("Warmup and timed passes must be positive")
    lock = verify_assets()
    package, conversion = check_package(args.build_dir)
    limits = InputLimits(**conversion["limits"])
    rows = select_rows(load_demo(), args.rows)
    inputs = [prepare_inputs(row["context"], row["options"], limits) for row in rows]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Bounded real-input latency and scheduler placement; not a general accuracy benchmark",
        "model_revision": lock["model_revision"],
        "package_files": conversion["package_files"],
        "dataset_revision": lock["dataset_revision"],
        "dataset_sha256": sha256(ROOT / lock["evaluation_file"]),
        "environment": {
            "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
            "macos": platform.mac_ver()[0],
            "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
            "python": platform.python_version(),
            "coremltools": ct.__version__,
        },
        "protocol": {
            "rows": args.rows,
            "pages": [row["meta"]["page"] for row in rows],
            "option_counts": [len(row["options"]) for row in rows],
            "first_call_row": args.rows[0],
            "warmup_passes": args.warmup_passes,
            "timed_passes": args.timed_passes,
            "backend_order": args.compute_units,
            "timing_scope": "Synchronous Python MLModel.predict; pre-encoded inputs; output checks outside timer",
            "load_scope": "MLModel construction; system caches retained, not first-install cold start",
            "first_call_scope": "First predict after this load; may use existing system/ANE caches",
            "plan_order": "Compute plan loaded after timing each backend",
        },
        "backends": {},
    }
    for name in args.compute_units:
        units = getattr(ct.ComputeUnit, name)
        started = time.perf_counter()
        model = ct.models.MLModel(str(package), compute_units=units)
        load_ms = 1000 * (time.perf_counter() - started)

        first_ms = predict_checked(model, inputs[0], rows[0], limits.max_options)
        for _ in range(args.warmup_passes):
            for index in range(len(rows)):
                predict_checked(model, inputs[index], rows[index], limits.max_options)
        calls = []
        for _ in range(args.timed_passes):
            for index, row_id in enumerate(args.rows):
                elapsed = predict_checked(model, inputs[index], rows[index], limits.max_options)
                calls.append({"row": row_id, "milliseconds": elapsed})
        backend = {
            "load_ms": load_ms,
            "first_prediction_ms": first_ms,
            "warm_prediction": latency_summary([call["milliseconds"] for call in calls]),
            "timed_predictions_correct": len(calls),
            "calls": calls,
            "compute_plan": compute_plan(model, units),
        }
        report["backends"][name] = backend
        print(f"{name}: {backend['warm_prediction']}; devices={backend['compute_plan']['device_counts']}", flush=True)
        del model
    report["script_sha256"] = sha256(Path(__file__))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
