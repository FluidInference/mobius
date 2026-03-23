"""CLI entry point."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from .compute_plan import COMPUTE_UNITS, get_compute_plan
from .fallback import analyze_fallback
from .latency import measure_cold_compile, measure_latency
from .metadata import get_model_metadata
from .model_loader import discover_models
from .output import emit_fallback_table, emit_json, emit_table
from .private_profiler import get_detailed_profile

app = typer.Typer(add_completion=False)

_debug = False


def _log(msg: str) -> None:
    if _debug:
        print(msg, file=sys.stderr)


def _get_hardware_info() -> dict[str, str]:
    chip = platform.processor() or platform.machine()
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass

    mac_ver, _, _ = platform.mac_ver()

    try:
        ram_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
        ram_gb = f"{ram_bytes / (1024 ** 3):.0f}GB"
    except Exception:
        ram_gb = "?"

    return {
        "device": platform.machine(),
        "chip": chip,
        "ram": ram_gb,
        "os_version": f"macOS {mac_ver}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class ComputeUnitChoice(str, Enum):
    all = "all"
    cpu_only = "cpu_only"
    cpu_and_gpu = "cpu_and_gpu"
    cpu_and_neural_engine = "cpu_and_neural_engine"


@app.command()
def bench(
    model_path: Path = typer.Argument(..., help="Path to .mlmodelc, .mlpackage, or directory"),
    units: Optional[ComputeUnitChoice] = typer.Option(
        None, "--units", "-u", help="Compute units to profile (default: all four)"
    ),
    ops: bool = typer.Option(
        False, "--ops", help="Include per-operation breakdown"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Per-op private API data (implies --ops)"
    ),
    fallback: bool = typer.Option(
        False, "--fallback", "-f", help="Show CPU fallback ops grouped by reason (ANE optimization)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output JSON instead of table"
    ),
    iterations: int = typer.Option(10, "--iterations", "-n", help="Number of timed iterations"),
    debug: bool = typer.Option(False, "--debug", help="Print progress to stderr"),
) -> None:
    """Profile CoreML model compute device assignments and latency."""
    global _debug
    _debug = debug

    # Suppress noisy CoreML/OS system logs unless debugging
    if not debug:
        os.environ["OS_ACTIVITY_DT_MODE"] = "disable"
        os.environ["COREML_LOGGING"] = "0"

    if detailed:
        ops = True

    models = discover_models(model_path)
    _log(f"Found {len(models)} model(s)")

    # Fallback analysis mode — separate path
    if fallback:
        cu = units.value if units else "cpu_and_neural_engine"
        all_fb = []
        for model in models:
            _log(f"Analyzing fallback for {model.name}...")
            fb = analyze_fallback(model, cu)
            all_fb.append({
                "model_path": str(model),
                "model_name": model.stem,
                "fallback": fb,
            })
        output = {"hardware": _get_hardware_info(), "models": all_fb}
        if json_output:
            emit_json(output)
        else:
            emit_fallback_table(output)
        return

    if units is not None:
        unit_configs = [units.value]
    else:
        unit_configs = list(COMPUTE_UNITS.keys())

    all_results = []

    for model in models:
        _log(f"Profiling {model.name}...")

        # Cold compile — once per model (bypass E5 cache via private API)
        _log(f"  measuring cold compile...")
        cold_compile_ms = measure_cold_compile(model)
        if cold_compile_ms >= 0:
            _log(f"  cold_compile={cold_compile_ms:.1f}ms")

        model_results = []

        for unit_config in unit_configs:
            _log(f"  compute_units={unit_config}")

            result = get_compute_plan(model, unit_config)

            if detailed:
                detail = get_detailed_profile(model, unit_config)
                if detail and "operations" in detail:
                    result = _merge_detailed(result, detail)

            if not ops:
                del result["operations"]

            _log(f"    measuring latency (5 warmup + {iterations} iterations)...")
            result["latency"] = measure_latency(
                model, unit_config, iterations=iterations
            )
            lat = result["latency"]
            if "compile_ms" in lat:
                _log(f"    compile={lat['compile_ms']:.1f}ms")
            if "median_ms" in lat:
                _log(f"    predict median={lat['median_ms']:.1f}ms")
            elif "error" in lat:
                _log(f"    latency error: {lat['error']}")

            model_results.append(result)

        all_results.append({
            "model_path": str(model),
            "model_name": model.stem,
            "metadata": get_model_metadata(model),
            "cold_compile_ms": round(cold_compile_ms, 3) if cold_compile_ms >= 0 else None,
            "results": model_results,
        })

    output = {"hardware": _get_hardware_info(), "models": all_results}
    if json_output:
        emit_json(output)
    else:
        emit_table(output)


def _merge_detailed(public: dict, private: dict) -> dict:
    """Merge private API detailed data into public compute plan results."""
    private_ops = {}
    for op in private.get("operations", []):
        private_ops[op["name"]] = op.get("detailed", {})

    for op in public.get("operations", []):
        key = op["name"]
        if key in private_ops:
            op["detailed"] = private_ops[key]

    return public
