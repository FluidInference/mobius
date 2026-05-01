#!/usr/bin/env python3
"""Per-component ANE fallback analysis for Parakeet-TDT-v3.

Runs ``coreml-cli --fallback --json`` on every component in a directory and
aggregates the CPU-fallback breakdown into a single ``fallback.json``. Useful
for understanding *why* parts of an mlpackage end up on CPU instead of the
Neural Engine after a quantization or pruning pass.

Output
------

``<input_dir>/fallback.json`` shape::

    {
      "_meta": {
        "input_dir": "...",
        "compute_units": "cpu_and_neural_engine",
        ...
      },
      "components": {
        "parakeet_encoder": {
          "compiled_path": "<input_dir>/.compiled/parakeet_encoder.mlmodelc",
          "fallback": { ... raw coreml-cli output ... },
          "summary": {
            "total_ops": 612,
            "fallback_ops": 110,
            "fallback_percent": 17.97,
            "by_reason": { "<reason>": <count>, ... }
          }
        },
        ...
      }
    }

mlpackage inputs are auto-compiled to mlmodelc into
``<input_dir>/.compiled/<component>.mlmodelc`` and kept on disk (matching the
behaviour of ``compute_unit_sweep.py``) so repeated runs don't pay the cold
compile cost.

"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import typer

import coremltools as ct


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT_HINT = BASE_DIR.parents[3]
COREML_CLI_DIR = REPO_ROOT_HINT / "tools" / "coreml-cli"


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@dataclass
class ComponentTarget:
    name: str
    compiled_path: Path


def _coreml_cli_dir(override: Optional[Path]) -> Path:
    if override is not None:
        return override.resolve()
    if COREML_CLI_DIR.exists():
        return COREML_CLI_DIR
    raise typer.BadParameter(
        f"coreml-cli not found at {COREML_CLI_DIR}; pass --coreml-cli-dir explicitly."
    )


def _compile_if_needed(src: Path, cache_dir: Path) -> Path:
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


def _run_fallback_cli(
    target: Path,
    units: str,
    plan_timeout: float,
    cli_dir: Path,
) -> Dict[str, object]:
    cmd = [
        "uv",
        "run",
        "coreml-cli",
        str(target),
        "--units",
        units,
        "--fallback",
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


def _summarize_fallback(raw: Dict[str, object]) -> Dict[str, object]:
    """Reduce coreml-cli's fallback JSON into a flat summary.

    The raw shape is ``{"hardware": ..., "models": [{"fallback": {...}}]}``.
    Different versions of coreml-cli emit slightly different keys; we keep the
    extraction defensive and return whatever we can.
    """
    summary: Dict[str, object] = {
        "total_ops": None,
        "fallback_ops": None,
        "fallback_percent": None,
        "by_reason": {},
    }
    if not isinstance(raw, dict):
        return summary
    models = raw.get("models", [])
    if not isinstance(models, list) or not models:
        return summary
    fb = models[0].get("fallback") if isinstance(models[0], dict) else None
    if not isinstance(fb, dict):
        return summary

    # coreml-cli emits {"total_ops": int, "cpu_ops": int, "reasons": [{"reason", "count", ...}]}.
    # Use direct gets (not `or`) so legitimate 0 values aren't treated as falsy.
    total_ops = fb.get("total_ops")
    if total_ops is None:
        total_ops = fb.get("op_count")
    fallback_ops = fb.get("cpu_ops")
    if fallback_ops is None:
        fallback_ops = fb.get("fallback_ops")
        if fallback_ops is None:
            fallback_ops = fb.get("cpu_op_count")
    summary["total_ops"] = total_ops
    summary["fallback_ops"] = fallback_ops
    if isinstance(total_ops, (int, float)) and isinstance(fallback_ops, (int, float)) and total_ops:
        summary["fallback_percent"] = round(100.0 * float(fallback_ops) / float(total_ops), 2)

    # Try a handful of likely shapes for the per-reason breakdown.
    reasons = fb.get("reasons")
    if reasons is None:
        reasons = fb.get("by_reason")
    if reasons is None:
        reasons = fb.get("groups")
    counter: Counter[str] = Counter()
    if isinstance(reasons, list):
        # coreml-cli shape: list of {"reason": str, "count": int, ...}
        for entry in reasons:
            if not isinstance(entry, dict):
                continue
            reason = entry.get("reason")
            count = entry.get("count")
            if isinstance(reason, str) and isinstance(count, (int, float)):
                counter[reason] += int(count)
    elif isinstance(reasons, dict):
        for key, value in reasons.items():
            if isinstance(value, (int, float)):
                counter[str(key)] += int(value)
            elif isinstance(value, list):
                counter[str(key)] += len(value)
            elif isinstance(value, dict):
                count = value.get("count")
                if isinstance(count, (int, float)):
                    counter[str(key)] += int(count)
                else:
                    ops = value.get("ops") or value.get("operations")
                    if isinstance(ops, list):
                        counter[str(key)] += len(ops)
    elif isinstance(reasons, list):
        for entry in reasons:
            if isinstance(entry, dict):
                key = entry.get("reason") or entry.get("name") or "unknown"
                count = entry.get("count")
                if isinstance(count, (int, float)):
                    counter[str(key)] += int(count)
                else:
                    ops = entry.get("ops") or entry.get("operations")
                    if isinstance(ops, list):
                        counter[str(key)] += len(ops)

    summary["by_reason"] = dict(counter.most_common())
    return summary


@app.command()
def run(
    input_dir: Path = typer.Option(
        Path("parakeet_coreml"),
        help="Directory containing .mlmodelc / .mlpackage Parakeet components.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Where to write fallback.json (default: <input_dir>/fallback.json).",
    ),
    component: Optional[List[str]] = typer.Option(
        None,
        "--component",
        help="Restrict to a subset of component stems. Repeatable.",
    ),
    units: str = typer.Option(
        "cpu_and_neural_engine",
        help="MLComputeUnits config to ask coreml-cli to analyze (default matches the ANE-priority path).",
    ),
    plan_timeout: float = typer.Option(
        120.0,
        help="Per-component MLComputePlan load timeout (seconds).",
    ),
    coreml_cli_dir: Optional[Path] = typer.Option(
        None,
        help="Override the path to the coreml-cli checkout.",
    ),
    keep_compiled: bool = typer.Option(
        True,
        help="Keep <input_dir>/.compiled/ around for re-runs.",
    ),
) -> None:
    """Run ANE fallback analysis on every component."""
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise typer.BadParameter(f"input_dir does not exist: {input_dir}")
    cli_dir = _coreml_cli_dir(coreml_cli_dir)
    cache_dir = input_dir / ".compiled"
    output_path = output.resolve() if output is not None else (input_dir / "fallback.json")

    targets = _discover_components(input_dir, component, cache_dir)
    if not targets:
        raise typer.BadParameter(f"No .mlmodelc or .mlpackage found in {input_dir}")

    typer.echo(f"Analyzing fallback for {len(targets)} component(s) on {units}")

    aggregated: Dict[str, object] = {
        "_meta": {
            "input_dir": str(input_dir),
            "compute_units": units,
            "plan_timeout": plan_timeout,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "components": {},
    }

    for target in targets:
        typer.echo(f"\n=== {target.name} ===")
        raw = _run_fallback_cli(
            target.compiled_path,
            units=units,
            plan_timeout=plan_timeout,
            cli_dir=cli_dir,
        )
        summary = _summarize_fallback(raw)
        aggregated["components"][target.name] = {
            "compiled_path": str(target.compiled_path),
            "fallback": raw,
            "summary": summary,
        }
        typer.echo(
            "  total={total} fallback={fb} ({pct}%); top reasons: {top}".format(
                total=summary.get("total_ops"),
                fb=summary.get("fallback_ops"),
                pct=summary.get("fallback_percent"),
                top=", ".join(f"{k}={v}" for k, v in list(summary.get("by_reason", {}).items())[:3])
                or "(none)",
            )
        )

    aggregated["_meta"]["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregated, indent=2))
    typer.echo(f"\nWrote {output_path}")

    if not keep_compiled and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        typer.echo(f"Removed {cache_dir}")


if __name__ == "__main__":
    app()
