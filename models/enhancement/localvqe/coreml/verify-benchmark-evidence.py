#!/usr/bin/env python3
"""Verify committed LocalVQE benchmark evidence without rerunning inference.

Checks every CSV hash, exact 800-stem coverage, scenario membership, DNSMOS
availability and all stored aggregate values. It also prints the published
far-end targets beside both ERLE definitions so a gated reconstruction cannot
be mistaken for the model card's declared plain-energy metric.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALIDATION = ROOT / "validation"
NUMERIC_FIELDS = ("echo", "deg", "erle", "erle_gated", "sig", "bak", "ovrl", "p808", "seconds")
SUMMARY_FIELDS = ("echo", "deg", "erle", "ovrl", "sig", "bak", "p808")
DNSMOS_FIELDS = ("sig", "bak", "ovrl", "p808")
FAR_END_SCENARIOS = ("farend-singletalk", "farend-singletalk-with-movement")


def fail(message: str) -> None:
    raise SystemExit(f"benchmark evidence verification failed: {message}")


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"{path} is not a JSON object")
    return value


def parse_number(raw: str, *, path: Path, row: int, field: str) -> float | None:
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        fail(f"{path}:{row} has non-numeric {field}={raw!r}")
    if not math.isfinite(value):
        fail(f"{path}:{row} has non-finite {field}={raw!r}")
    return value


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def assert_close(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        if actual is expected:
            return
        fail(f"{label}: expected {expected}, got {actual}")
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-10):
        fail(f"{label}: expected {expected:.12g}, got {actual:.12g}")


def verify_run(run: dict, index: dict, manifest: set[str]) -> list[dict]:
    path = VALIDATION / run["file"]
    if not path.is_file():
        fail(f"missing {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != run["sha256"]:
        fail(f"{path} SHA256 is {digest}, expected {run['sha256']}")

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != index["expected_unique_clips"]:
        fail(f"{path} has {len(rows)} rows, expected {index['expected_unique_clips']}")
    stems = [row["stem"] for row in rows]
    if len(stems) != len(set(stems)):
        fail(f"{path} contains duplicate stems")
    if set(stems) != manifest:
        fail(f"{path} does not exactly match {index['manifest']}")

    for row_number, row in enumerate(rows, start=2):
        stem = row["stem"]
        if not stem.endswith("_" + row["scenario"]):
            fail(f"{path}:{row_number} scenario does not match stem {stem}")
        for field in NUMERIC_FIELDS:
            parse_number(row[field], path=path, row=row_number, field=field)
        dns_present = [row[field] != "" for field in DNSMOS_FIELDS]
        if run["dnsmos_scored"] and not all(dns_present):
            fail(f"{path}:{row_number} is missing a scored DNSMOS value")
        if not run["dnsmos_scored"] and any(dns_present):
            fail(f"{path}:{row_number} has DNSMOS values but dnsmos_scored=false")

    scenarios = {row["scenario"] for row in rows}
    if scenarios != set(run["summary"]):
        fail(f"{path} scenarios do not match its summary")
    for scenario, expected in run["summary"].items():
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if len(scenario_rows) != expected["n"]:
            fail(f"{path} {scenario} has {len(scenario_rows)} rows, expected {expected['n']}")
        for field in SUMMARY_FIELDS:
            values = [
                value
                for row_number, row in enumerate(scenario_rows, start=2)
                if (value := parse_number(row[field], path=path, row=row_number, field=field)) is not None
            ]
            assert_close(mean(values), expected[field], f"{path} {scenario}.{field}")
        gated = [
            value
            for row_number, row in enumerate(scenario_rows, start=2)
            if (value := parse_number(row["erle_gated"], path=path, row=row_number, field="erle_gated"))
            is not None
        ]
        if len(gated) != expected["erle_gated_n"]:
            fail(f"{path} {scenario}.erle_gated_n is {len(gated)}, expected {expected['erle_gated_n']}")
        assert_close(mean(gated), expected["erle_gated"], f"{path} {scenario}.erle_gated")
    return rows


def print_reference_comparison(index: dict, reference: dict) -> None:
    runs = {run["file"]: run for run in index["runs"]}
    print("\nPublished far-end reference versus saved current-GGML scores")
    print("(plain ERLE is the model card's declared formula; gated ERLE is a separate reconstruction)\n")
    for version in ("v1.3", "v1.2"):
        run = runs[f"blind/upstream-ggml-{version}.csv"]
        print(version)
        for scenario in FAR_END_SCENARIOS:
            target = reference["models"][version][scenario]
            actual = run["summary"][scenario]
            print(
                f"  {scenario}: card echo={target['echo']:.2f}, observed={actual['echo']:.2f}; "
                f"card ERLE={target['erle']:.1f}, plain={actual['erle']:.1f}, "
                f"gated={actual['erle_gated']:.1f} dB"
            )


def main() -> None:
    index = load_json(VALIDATION / "benchmark-index.json")
    reference = load_json(VALIDATION / "upstream-model-card.json")
    manifest_list = [line.strip() for line in (VALIDATION / index["manifest"]).read_text().splitlines() if line.strip()]
    if len(manifest_list) != index["expected_unique_clips"]:
        fail(f"manifest has {len(manifest_list)} rows, expected {index['expected_unique_clips']}")
    manifest = set(manifest_list)
    if len(manifest) != len(manifest_list):
        fail("manifest contains duplicate stems")

    seen_files: set[str] = set()
    for run in index["runs"]:
        if run["file"] in seen_files:
            fail(f"duplicate run entry {run['file']}")
        seen_files.add(run["file"])
        verify_run(run, index, manifest)

    print(f"verified {len(index['runs'])} runs x {len(manifest)} unique clips")
    print_reference_comparison(index, reference)


if __name__ == "__main__":
    main()
