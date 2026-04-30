#!/usr/bin/env python3
"""Per-compute-unit latency / device-residency sweep for Parakeet-TDT-v3.

Drives ``coreml-cli`` (without ``--fallback``) and aggregates the latency +
device-assignment results across all four ``MLComputeUnits`` configurations:

  * ``all``
  * ``cpu_only``
  * ``cpu_and_gpu``
  * ``cpu_and_neural_engine``

For each component in the input directory (any ``.mlmodelc``, or ``.mlpackage``
that gets compiled to ``.mlmodelc``) we capture:

  * cold compile time (one-shot, populated by ``coreml-cli`` via private API)
  * cached compile time per CU
  * predict latency (median ms) per CU
  * % of operations dispatched to CPU / GPU / ANE per CU

Output: ``<input_dir>/compute_unit_sweep.json`` with the aggregated results.

Notes
-----

* This script is a thin orchestrator on top of ``coreml-cli``; it shells out
  to ``uv run coreml-cli ... --json`` and parses the structured output.
* Models that are still ``.mlpackage`` are compiled to ``.mlmodelc`` first
  (cached under ``<input_dir>/.compiled/`` to avoid recompiling between runs).
* Use ``--component`` (repeatable) to limit the sweep to a subset.

"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import typer

import coremltools as ct


BASE_DIR = Path(__file__).resolve().parent
COMPUTE_UNITS = ("all", "cpu_only", "cpu_and_gpu", "cpu_and_neural_engine")

# coreml-cli is checked into the repo under tools/coreml-cli; resolve at runtime.
REPO_ROOT_HINT = BASE_DIR.parents[3]  # .../models/stt/parakeet-tdt-v3-0.6b/coreml -> repo root
COREML_CLI_DIR = REPO_ROOT_HINT / "tools" / "coreml-cli"


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@dataclass
class ComponentTarget:
    name: str
    compiled_path: Path  # .mlmodelc on disk to feed into coreml-cli


def _run_coreml_cli(
    target: Path,
    units: str,
    iterations: int,
    plan_timeout: float,
    cli_dir: Path,
) -> Dict[str, object]:
    """Invoke ``coreml-cli`` for a single (model, units) and return parsed JSON."""
    cmd = [
        "uv",
        "run",
        "coreml-cli",
        str(target),
        "--units",
        units,
        "--iterations",
        str(iterations),
        "--plan-timeout",
        str(plan_timeout),
        "--json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cli_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "error": "coreml-cli exited non-zero",
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"non-JSON stdout: {e}", "stdout": proc.stdout[-2000:]}


def _compile_if_needed(src: Path, cache_dir: Path) -> Path:
    """If ``src`` is an mlpackage, compile to mlmodelc under ``cache_dir`` and return it.

    If ``src`` is already an mlmodelc, returns it as-is.
    """
    if src.suffix == ".mlmodelc":
        return src
    if src.suffix != ".mlpackage":
        raise typer.BadParameter(f"Unsupported model suffix: {src}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / (src.stem + ".mlmodelc")
    if target.exists():
        return target
    typer.echo(f"  Compiling {src.name} -> {target.name} ...")
    compiled = ct.utils.compile_model(str(src))
    compiled_path = Path(compiled)
    # Move into our cache so subsequent runs reuse it.
    if compiled_path != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(compiled_path), str(target))
    return target


def _discover_components(
    input_dir: Path,
    explicit: Optional[List[str]],
    cache_dir: Path,
) -> List[ComponentTarget]:
    """Discover candidate components and ensure each is compiled to .mlmodelc."""
    candidates: List[Path] = []
    for entry in sorted(input_dir.iterdir()):
        if entry.suffix in {".mlmodelc", ".mlpackage"}:
            candidates.append(entry)

    if explicit:
        wanted = {name.strip() for name in explicit if name.strip()}
        candidates = [p for p in candidates if p.stem in wanted or p.name in wanted]

    targets: List[ComponentTarget] = []
    for cand in candidates:
        compiled = _compile_if_needed(cand, cache_dir)
        targets.append(ComponentTarget(name=cand.stem, compiled_path=compiled))
    return targets


def _coreml_cli_dir(override: Optional[Path]) -> Path:
    if override is not None:
        return override.resolve()
    if COREML_CLI_DIR.exists():
        return COREML_CLI_DIR
    raise typer.BadParameter(
        f"coreml-cli not found at {COREML_CLI_DIR}; pass --coreml-cli-dir explicitly."
    )


@app.command()
def run(
    input_dir: Path = typer.Option(
        Path("parakeet_coreml"),
        help="Directory containing .mlmodelc / .mlpackage Parakeet components.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Where to write the aggregated JSON (default: <input_dir>/compute_unit_sweep.json).",
    ),
    component: Optional[List[str]] = typer.Option(
        None,
        "--component",
        help="Restrict to a subset of component stems (e.g. parakeet_encoder). Repeatable.",
    ),
    iterations: int = typer.Option(20, help="Timed iterations per (component, units) pair."),
    plan_timeout: float = typer.Option(
        120.0,
        help="Per-config MLComputePlan load timeout passed through to coreml-cli.",
    ),
    coreml_cli_dir: Optional[Path] = typer.Option(
        None,
        help="Override the path to the coreml-cli checkout (must contain pyproject.toml).",
    ),
    keep_compiled: bool = typer.Option(
        True,
        help="Keep <input_dir>/.compiled/ around for re-runs. Disable to clean up after.",
    ),
) -> None:
    """Run the per-compute-unit latency / residency sweep."""
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise typer.BadParameter(f"input_dir does not exist: {input_dir}")
    cli_dir = _coreml_cli_dir(coreml_cli_dir)
    cache_dir = input_dir / ".compiled"
    output_path = output.resolve() if output is not None else (input_dir / "compute_unit_sweep.json")

    targets = _discover_components(input_dir, component, cache_dir)
    if not targets:
        raise typer.BadParameter(f"No .mlmodelc or .mlpackage found in {input_dir}")

    typer.echo(f"Sweeping {len(targets)} component(s) across {len(COMPUTE_UNITS)} compute units")
    typer.echo(f"  cli_dir = {cli_dir}")
    typer.echo(f"  output  = {output_path}")

    aggregated: Dict[str, object] = {
        "_meta": {
            "input_dir": str(input_dir),
            "iterations": iterations,
            "plan_timeout": plan_timeout,
            "compute_units": list(COMPUTE_UNITS),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "components": {},
    }

    for target in targets:
        typer.echo(f"\n=== {target.name} ===")
        per_unit: Dict[str, object] = {}
        for cu in COMPUTE_UNITS:
            typer.echo(f"  units={cu} ...")
            result = _run_coreml_cli(
                target.compiled_path,
                units=cu,
                iterations=iterations,
                plan_timeout=plan_timeout,
                cli_dir=cli_dir,
            )
            per_unit[cu] = result

            # Pretty-print the headline numbers if present.
            try:
                models = result.get("models", []) if isinstance(result, dict) else []
                if models and isinstance(models[0], dict):
                    runs = models[0].get("compute_units", [])
                    if runs and isinstance(runs[0], dict):
                        latency = runs[0].get("latency", {})
                        device = runs[0].get("device_assignment", {})
                        median = latency.get("median_ms")
                        cpu_pct = device.get("cpu_percent")
                        gpu_pct = device.get("gpu_percent")
                        ane_pct = device.get("ane_percent")
                        typer.echo(
                            f"    latency={median} ms | "
                            f"CPU={cpu_pct}% GPU={gpu_pct}% ANE={ane_pct}%"
                        )
            except Exception:
                pass

        aggregated["components"][target.name] = {
            "compiled_path": str(target.compiled_path),
            "per_unit": per_unit,
        }

    aggregated["_meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregated, indent=2))
    typer.echo(f"\nWrote {output_path}")

    if not keep_compiled and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        typer.echo(f"Removed {cache_dir}")


if __name__ == "__main__":
    app()
