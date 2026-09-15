"""Reduce an existing parity run to portable evidence; never execute a model."""

import argparse
import json
from pathlib import Path

from kokoro_training.artifacts import sha256, write_json


def summarize(run: Path) -> dict:
    def read(name):
        return json.loads((run / name).read_text())

    summary = read("summary.json")
    mapping = read("checkpoint-map.json")
    states = read("loaded-state-parity.json")
    cases = []
    for case in summary["cases"]:
        detail = read(f"cases/{case['id']}.json")
        checks = detail.get("candidate", {}).get("checks", {})
        repeat = detail.get("oracle_repeat", {}).get("checks", {})
        cases.append(
            {
                **case,
                "compared_tensors": len(checks),
                "all_intermediates_exact": bool(checks)
                and all(c["passed"] for c in checks.values()),
                "oracle_repeat_exact": bool(repeat) and all(c["passed"] for c in repeat.values()),
                "max_absolute_error": max(
                    (c.get("max_absolute_error", 0) for c in checks.values()), default=None
                ),
                "checked_tensors": sorted(checks),
            }
        )
    return {
        "schema_version": 1,
        "run_label": run.name,
        "passed": summary["passed"]
        and mapping["passed"]
        and states["passed"]
        and all(c["all_intermediates_exact"] and c["oracle_repeat_exact"] for c in cases),
        "scope": summary["scope"],
        "environment": read("environment.json"),
        "resolved_config": read("resolved-config.json"),
        "inputs": read("inputs.json"),
        "checkpoint_mapping": {
            k: v for k, v in mapping.items() if k not in {"entries", "identity_constants"}
        },
        "identity_constant_tensors": len(mapping["identity_constants"]),
        "loaded_state_tensors_compared": len(states["checks"]),
        "cases": cases,
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
        "raw_report_sha256": {
            str(p.relative_to(run)): sha256(p) for p in sorted(run.rglob("*.json"))
        },
        "remaining_gates": summary["remaining_gates"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    write_json(args.output, result)
    if args.mapping_output:
        mapping = json.loads((args.run / "checkpoint-map.json").read_text())
        args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
        with args.mapping_output.open("w") as stream:
            for entry in mapping["entries"]:
                stream.write(json.dumps(entry, separators=(",", ":")) + "\n")
            for entry in mapping["identity_constants"]:
                stream.write(json.dumps({"identity_constant": entry}, separators=(",", ":")) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
