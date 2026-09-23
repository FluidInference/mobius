"""Profile a small set of pinned, real Kai/Lex Core ML requests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

UNITS = {
    "all": "ALL",
    "cpu-ne": "CPU_AND_NE",
    "cpu-gpu": "CPU_AND_GPU",
    "cpu": "CPU_ONLY",
}


def percentile_ms(samples_ns: list[int], percentile: float) -> float:
    """Nearest-rank percentile, with the count reported alongside results."""
    ordered = sorted(samples_ns)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)] / 1e6


def profile(
    repo_root: Path, package: Path, kind: str, units: str, warmup: int, iterations: int
) -> dict:
    if warmup < 0 or iterations < 2:
        raise ValueError("Require nonnegative warmup and at least two timed requests")
    root = repo_root.resolve()
    sys.path.append(str(root / "conversion"))
    import coremltools as ct
    from run_coreml import CoreMLSystemOne

    rows = []
    for source in ("upstream-decisions.jsonl", "upstream-system-one-rows.jsonl"):
        rows.extend(
            json.loads(line)
            for line in (root / "conversion" / source).read_text().splitlines()
            if line
        )
    selected = [row for row in rows if row["question"]["type"].lower() == kind]
    if len(selected) != 2:
        raise ValueError(f"Expected exactly two pinned real {kind} requests")

    runtime = CoreMLSystemOne(root)
    started = time.perf_counter_ns()
    model = ct.models.MLModel(
        str(package), compute_units=getattr(ct.ComputeUnit, UNITS[units])
    )
    load_ms = (time.perf_counter_ns() - started) / 1e6
    runtime.models[kind] = model
    for index in range(warmup):
        runtime.predict_row(selected[index % len(selected)])
    times_ns = []
    outputs = []
    for index in range(iterations):
        row = selected[index % len(selected)]
        started = time.perf_counter_ns()
        result = runtime.predict_row(row)
        times_ns.append(time.perf_counter_ns() - started)
        outputs.append(result["probabilities"])
    return {
        "repo": runtime.model_name,
        "kind": kind,
        "units": units,
        "package": package.name,
        "package_bytes": sum(
            path.stat().st_size for path in package.rglob("*") if path.is_file()
        ),
        "fixture_ids": [row["id"] for row in selected],
        "warmup": warmup,
        "iterations": iterations,
        "load_ms": load_ms,
        "full_request_p50_ms": statistics.median(times_ns) / 1e6,
        "full_request_p95_ms": percentile_ms(times_ns, 0.95),
        "probabilities_first_two": outputs[:2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--kind", choices=("choice", "noul", "score"), required=True)
    parser.add_argument("--units", choices=UNITS, default="cpu-ne")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = profile(
        args.repo_root,
        args.package,
        args.kind,
        args.units,
        args.warmup,
        args.iterations,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
