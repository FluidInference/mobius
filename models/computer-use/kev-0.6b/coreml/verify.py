"""Compare a converted Kev package with the merged PyTorch checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from assets import LOCK_PATH, ROOT, load_model
from export_model import KevExport
from preprocessing import Shape, prepare_inputs
from release_integrity import package_file_hashes, sha256

COMPUTE_UNITS = {
    "all": ct.ComputeUnit.ALL,
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
}


def fixtures() -> list[dict]:
    return [
        {
            "state": "The piece leaves one hole beneath it and creates a small bump on top.",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Classify the placement.",
                    "criteria": {
                        "clean": "No buried holes and a flat surface",
                        "risky": "Creates a cavity or awkward surface",
                    },
                    "label": "risky",
                    "src": "coreml-fixture",
                }
            },
        },
        {
            "state": "URGENT: verify your account at http://unknown.example and enter your password.",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Is this message phishing?",
                    "criteria": {"false": "legitimate", "true": "phishing"},
                    "label": True,
                    "src": "coreml-fixture",
                }
            },
        },
        {
            "state": "The customer was charged twice and wants the duplicate transaction reversed.",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Route this support ticket.",
                    "criteria": {
                        "billing": "Payments and charges",
                        "technical": "Product malfunction",
                        "sales": "Buying a product",
                    },
                    "label": "billing",
                    "src": "coreml-fixture",
                }
            },
        },
        {
            "state": "The order arrived two weeks late and the outer box was damaged.",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "Rate the delivery issue severity.",
                    "criteria": ["Low impact", "Moderate impact", "High impact"],
                    "label": 2,
                    "src": "coreml-fixture",
                }
            },
        },
    ]


def suite_requests(path: Path, limit: int) -> list[dict]:
    requests = []
    per_suite: dict[str, int] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if per_suite.get(row["suite"], 0) >= 2:
            continue
        if row["type"] == "choice":
            question = {
                "type": "choice",
                "instructions": row["instructions"],
                "criteria": {key: description for key, description in row["options"]},
                "label": row["options"][row["gold"]][0],
            }
        else:
            question = {
                "type": "noul",
                "instructions": row["instructions"],
                "criteria": ({key: description for key, description in row["options"]} if row["options"] else None),
                "label": bool(row["gold"]),
            }
        question["src"] = row["suite"]
        requests.append({"state": json.loads(row["state"]), "questions": {"q": question}})
        per_suite[row["suite"]] = per_suite.get(row["suite"], 0) + 1
        if len(requests) == limit:
            break
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--units", choices=COMPUTE_UNITS, default="all")
    parser.add_argument("--max-probability-error", type=float, default=0.02)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--suite-cases", type=int, default=20)
    parser.add_argument("--report", type=Path, help="write parity evidence after a passing run")
    args = parser.parse_args()
    _, tokenizer, decision_model = load_model()
    decision_model.eval()
    shape = Shape(args.length, args.max_options)
    wrapper = KevExport(decision_model, shape.length, shape.max_options).eval()
    started = time.perf_counter()
    coreml = ct.models.MLModel(str(args.package), compute_units=COMPUTE_UNITS[args.units])
    load_seconds = time.perf_counter() - started
    errors: list[float] = []
    times: list[float] = []
    agreements = 0
    evaluated = 0
    fixture_evaluated = 0
    fixture_agreements = 0
    requests = fixtures()
    if args.suite:
        requests.extend(suite_requests(args.suite, args.suite_cases))
    for index, request in enumerate(requests):
        try:
            arrays, encoded = prepare_inputs(decision_model, tokenizer, request, shape)
        except ValueError as error:
            print(f"fixture {index}: skipped ({error})")
            continue
        evaluated += 1
        tensors = tuple(torch.from_numpy(value) for value in arrays.values())
        with torch.no_grad():
            _, reference = wrapper(*tensors)
        started = time.perf_counter()
        output = coreml.predict(arrays)
        times.append((time.perf_counter() - started) * 1000)
        options = len(encoded["opt_idx"][0])
        expected = reference[0, :options].numpy()
        actual = np.asarray(output["probabilities"])[0, :options]
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise ValueError(f"fixture {index}: native or Core ML probabilities are not finite")
        error = float(np.max(np.abs(expected - actual)))
        errors.append(error)
        agreement = int(expected.argmax()) == int(actual.argmax())
        agreements += agreement
        if index < 4:
            fixture_evaluated += 1
            fixture_agreements += agreement
        print(f"fixture {index}: options={options} argmax={agreement} max_probability_error={error:.6f}")
    if not times:
        raise ValueError("No Kev requests fit the selected Core ML package")
    p95 = sorted(times)[max(0, int(0.95 * len(times)) - 1)]
    print(
        f"{agreements}/{evaluated} argmax; max_probability_error={max(errors):.6f}; "
        f"p50={statistics.median(times):.2f} ms; p95={p95:.2f} ms; load={load_seconds:.2f} s"
    )
    if agreements != evaluated or max(errors) > args.max_probability_error:
        raise SystemExit(1)
    report = {
        "package": args.package.name,
        "package_files_sha256": package_file_hashes(args.package),
        "source_lock_sha256": sha256(LOCK_PATH),
        "length": shape.length,
        "max_options": shape.max_options,
        "compute_units": args.units,
        "evaluated": evaluated,
        "agreements": agreements,
        "fixture_evaluated": fixture_evaluated,
        "fixture_agreements": fixture_agreements,
        "max_probability_error": max(errors),
        "passed": True,
    }
    target = args.report or ROOT / "reports" / f"verification-{args.package.stem}-{args.units}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote parity report {target}")


if __name__ == "__main__":
    main()
