"""Compare the pinned upstream PyTorch model and two specified Core ML exports on test.jsonl."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from assets import ROOT, load_reference, sha256, verify_assets
from preprocessing import InputLimits, prepare_inputs
from synthetic_test import MANIFEST, calibration, inspect_rows, load_test, paired_outcomes
from verify import PROBABILITY_TOLERANCE, check_package, latency_summary, validate_prediction


def validate_variants(conversions: dict, candidate_name: str) -> None:
    """Reject mislabeled or unrelated models before evaluating the held-out split."""
    baseline, candidate = conversions["baseline"], conversions[candidate_name]
    if baseline.get("optimization", "baseline") != "baseline" or "quantization" in baseline:
        raise ValueError("Expected the original baseline")
    if candidate_name in ("int8-weights", "int4-weights"):
        quantization = candidate.get("quantization", {})
        if (
            quantization.get("name") != candidate_name
            or quantization.get("source_package_files") != baseline["package_files"]
            or quantization.get("settings", {}).get("dtype") != candidate_name.split("-")[0]
            or candidate["minimum_target"] != baseline["minimum_target"]
        ):
            raise ValueError("Quantized candidate must derive from this exact baseline with the stated dtype")
        if candidate_name == "int4-weights" and candidate["minimum_target"] != "iOS18/macOS15":
            raise ValueError("This packed INT4 trial requires an iOS18/macOS15 baseline")
    elif candidate_name == "ane-gather":
        if candidate.get("optimization") != candidate_name or "quantization" in candidate:
            raise ValueError("Expected the unquantized ANE-gather candidate")
    else:
        raise ValueError("Unknown benchmark candidate")
    for key in ("limits", "model_revision", "source_revision", "assets_lock_sha256"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Baseline and candidate disagree on {key}")


def read_scorer():
    spec = importlib.util.spec_from_file_location("score_report", ROOT / "score-report.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the upstream metric adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_for_comparison(output: dict, count: int, capacity: int) -> tuple[np.ndarray, list[str]]:
    """Retain an imperfect probability sum as a failed gate, without changing the output.

    Structural/nonfinite/range/padding failures still abort. A finite softmax sum
    outside the original tolerance can be scored by argmax for comparison, while
    the report remains failed. Probabilities are never renormalized.
    """
    try:
        _, probabilities = validate_prediction(output, count, capacity)
        return probabilities, []
    except ValueError as error:
        if str(error) != "Live option probabilities do not sum to one":
            raise
        if np.any(np.asarray(output["probabilities"])[0, count:] != 0):
            raise ValueError("Padding received probability mass") from error
        return np.asarray(output["probabilities"])[0, :count], [str(error)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "build")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--candidate-name", choices=["ane-gather", "int8-weights", "int4-weights"], default="ane-gather"
    )
    parser.add_argument("--report", type=Path, default=ROOT / "reports/synthetic-test.json")
    parser.add_argument("--trace", type=Path, default=ROOT / "build/synthetic-test-decisions.jsonl.gz")
    parser.add_argument("--require-parity", action="store_true", help="Exit nonzero if original conversion gates fail")
    args = parser.parse_args()
    names = ("upstream_pytorch", "baseline", args.candidate_name)
    candidate_dir = args.candidate or ROOT / "build" / args.candidate_name
    if args.report.exists() or args.trace.exists():
        raise FileExistsError("Use fresh report/trace paths to preserve previous evaluation runs")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    lock = verify_assets()
    rows, manifest = load_test(download=True)
    packages, conversions = {}, {}
    for name, directory in (("baseline", args.baseline), (args.candidate_name, candidate_dir)):
        packages[name], conversions[name] = check_package(directory)
    validate_variants(conversions, args.candidate_name)
    limits = InputLimits(**conversions["baseline"]["limits"])
    census = inspect_rows(rows, limits)
    print(f"Pinned synthetic test: {len(rows)} rows; all fit unchanged; maxima={census['maxima']}", flush=True)
    started = time.perf_counter()
    reference, collator, _ = load_reference()
    load_ms = {"upstream_pytorch": (time.perf_counter() - started) * 1000}
    from cua_s1.model import validate_example

    examples = [validate_example(row) for row in rows]
    models = {}
    for name in names[1:]:
        started = time.perf_counter()
        models[name] = ct.models.MLModel(str(packages[name]), compute_units=ct.ComputeUnit.CPU_AND_NE)
        load_ms[name] = (time.perf_counter() - started) * 1000
    first_call_ms = {}
    with torch.no_grad():
        for name in names:
            for index in range(3):
                batch = (
                    collator([examples[index]])
                    if name == names[0]
                    else prepare_inputs(rows[index]["context"], rows[index]["options"], limits)
                )
                started = time.perf_counter()
                if name == names[0]:
                    reference(batch)[0].softmax(-1)
                else:
                    models[name].predict(batch)
                if index == 0:
                    first_call_ms[name] = (time.perf_counter() - started) * 1000
    stats = {
        name: {key: [] for key in ("predictions", "confidence", "gold_probability", "correct", "latency_ms")}
        for name in names
    }
    preprocessing = {"upstream_collator_ms": [], "shared_coreml_encoding_ms": []}
    errors = {name: [] for name in names[1:]}
    violations = {name: [] for name in names[1:]}
    output_issues = {name: {} for name in names[1:]}
    records = []
    test_started = time.perf_counter()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    partial = args.trace.with_suffix(args.trace.suffix + ".partial")
    with torch.no_grad(), gzip.open(partial, "wt", encoding="utf-8") as trace:
        for index, (row, example) in enumerate(zip(rows, examples)):
            started = time.perf_counter()
            batch = collator([example])
            collator_ms = (time.perf_counter() - started) * 1000
            preprocessing["upstream_collator_ms"].append(collator_ms)
            started = time.perf_counter()
            reference_probabilities = reference(batch)[0].softmax(-1).numpy()
            reference_ms = (time.perf_counter() - started) * 1000
            if not np.isfinite(reference_probabilities).all():
                raise ValueError(f"Nonfinite reference output in row {index}")
            started = time.perf_counter()
            arrays = prepare_inputs(row["context"], row["options"], limits)
            encoding_ms = (time.perf_counter() - started) * 1000
            preprocessing["shared_coreml_encoding_ms"].append(encoding_ms)
            probabilities = {names[0]: reference_probabilities}
            latencies = {names[0]: reference_ms}
            # Counterbalance order, without rerunning the test split or selecting favorable calls.
            order = list(names[1:]) if index % 2 == 0 else list(reversed(names[1:]))
            for name in order:
                started = time.perf_counter()
                output = models[name].predict(arrays)
                latencies[name] = (time.perf_counter() - started) * 1000
                probabilities[name], issues = prediction_for_comparison(output, len(row["options"]), limits.max_options)
                if issues:
                    output_issues[name][index] = issues
                delta = float(np.max(np.abs(probabilities[name] - reference_probabilities)))
                errors[name].append(delta)
                if delta > PROBABILITY_TOLERANCE:
                    violations[name].append(index)
            record = {
                "row": index,
                "label": row["label"],
                "action": row["meta"]["action"],
                "role": row["meta"]["role"],
                "seed": row["meta"]["seed"],
                "options": len(row["options"]),
                "coreml_order": order,
                "upstream_collator_ms": collator_ms,
                "shared_coreml_encoding_ms": encoding_ms,
                "outputs": {},
            }
            compact = {"row": index}
            for name in names:
                values = probabilities[name]
                selected = int(values.argmax())
                confidence, gold_probability = float(values[selected]), float(values[row["label"]])
                correct = selected == row["label"]
                for key, value in (
                    ("predictions", selected),
                    ("confidence", confidence),
                    ("gold_probability", gold_probability),
                    ("correct", correct),
                    ("latency_ms", latencies[name]),
                ):
                    stats[name][key].append(value)
                compact[name] = selected
                record["outputs"][name] = {
                    "selected": selected,
                    "correct": correct,
                    "confidence": confidence,
                    "gold_probability": gold_probability,
                    "latency_ms": latencies[name],
                    "probability_sum": float(values.sum()),
                }
                if name != names[0]:
                    record["outputs"][name]["max_abs_probability_error"] = errors[name][-1]
                    record["outputs"][name]["validation_issues"] = output_issues[name].get(index, [])
            records.append(compact)
            trace.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            if (index + 1) % 1000 == 0 or index + 1 == len(rows):
                scores = ", ".join(f"{name}={sum(stats[name]['correct'])}/{index + 1}" for name in names)
                print(
                    f"{index + 1}/{len(rows)}: {scores}; elapsed {time.perf_counter() - test_started:.1f}s", flush=True
                )
    partial.replace(args.trace)
    labels = [row["label"] for row in rows]
    scorer = read_scorer()
    results, failures = {}, []
    for name in names:
        measured = stats[name]
        metrics = scorer.score_decisions(rows, records, name)
        correct = int(sum(measured["correct"]))
        entry = {
            "correct": correct,
            "rows": len(rows),
            "accuracy": correct / len(rows),
            "model_call_latency_ms": latency_summary(measured["latency_ms"]),
            "load_ms": load_ms[name],
            "first_call_ms": first_call_ms[name],
            "calibration": calibration(measured["confidence"], measured["gold_probability"], measured["correct"]),
            "upstream_action_metrics": metrics,
            "macro_action_accuracy": float(np.mean([value["accuracy"] for value in metrics["per_action"].values()])),
            "wrong_rows": [i for i, hit in enumerate(measured["correct"]) if not hit],
        }
        if name != names[0]:
            paired = paired_outcomes(labels, stats[names[0]]["predictions"], measured["predictions"])
            passed = (
                paired["argmax_agreement"] == len(rows)
                and not violations[name]
                and not output_issues[name]
                and correct >= sum(stats[names[0]]["correct"])
            )
            entry.update(
                {
                    "paired_with_upstream": paired,
                    "max_abs_probability_error": max(errors[name]),
                    "mean_row_max_abs_probability_error": float(np.mean(errors[name])),
                    "probability_tolerance_violation_rows": violations[name],
                    "conversion_parity_passed": passed,
                    "output_validation_failures": output_issues[name],
                }
            )
        results[name] = entry
    for index, row in enumerate(rows):
        if all(stats[name]["correct"][index] for name in names) and all(
            errors[name][index] <= PROBABILITY_TOLERANCE and index not in output_issues[name] for name in names[1:]
        ):
            continue
        failures.append(
            {
                "row": index,
                "context": row["context"],
                "gold_option": row["options"][row["label"]],
                "selected_options": {name: row["options"][stats[name]["predictions"][index]] for name in names},
            }
        )
    report = {
        "purpose": "Full published synthetic test; no fitting, filtering, resampling or test-based tuning",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": manifest,
        "manifest_sha256": sha256(MANIFEST),
        "input_census": census,
        "model_revision": lock["model_revision"],
        "source_revision": lock["source_revision"],
        "conversions": conversions,
        "environment": {
            "machine": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
            "os": platform.mac_ver()[0],
            "os_build": subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "torch_cpu_threads": 2,
            "torch_interop_threads": 1,
            "torch_mha_fastpath": False,
        },
        "protocol": {
            "batch_size": 1,
            "warmups_per_model": 3,
            "timed_passes": 1,
            "coreml_compute_units": "CPU_AND_NE",
            "coreml_order": "AB on even rows, BA on odd rows",
            "timing": (
                "PyTorch CPU forward+softmax / synchronous Core ML predict; "
                "excludes encoding, validation, GUI and network"
            ),
            "preprocessing": (
                "Upstream variable-length collator; independent fixed-shape Core ML encoder shared by both exports"
            ),
            "load_note": "Process/model creation can use existing system caches; not a cold-start experiment",
            "output_audit": (
                "Imperfect finite probability sums are retained unchanged and fail the original normalization gate"
            ),
            "calibration_note": "NLL/ECE use raw emitted scores; no renormalization or calibration was applied",
        },
        "thresholds": {
            "argmax_agreement": 1.0,
            "max_abs_probability_error": PROBABILITY_TOLERANCE,
            "allow_accuracy_loss": False,
            "probability_sum_atol": 0.001,
            "probability_sum_rtol": 0.00001,
        },
        "host_preprocessing_ms": {name: latency_summary(values) for name, values in preprocessing.items()},
        "results": results,
        "failures": failures,
        "trace": {"file": args.trace.name, "sha256": sha256(args.trace), "rows": len(rows)},
        "upstream_evaluator_sha256": scorer.METRICS_SHA256,
        "harness_sha256": {
            file: sha256(ROOT / file) for file in ("benchmark-synthetic.py", "synthetic_test.py", "score-report.py")
        },
        "conversion_parity_passed": all(results[name]["conversion_parity_passed"] for name in names[1:]),
        "limitations": [
            (
                "The release file has 24370 rows; the model card claims approximately 15000. "
                "This is not a reconstruction of that unspecified manifest."
            ),
            (
                "Upstream describes form-disjoint synthetic data; "
                "this is not an unseen real-world GUI or task-completion benchmark."
            ),
            (
                "No fresh hosted Jev calls are included. "
                "Published hosted-Jev percentages are not a matched baseline for this run."
            ),
            (
                "Local timings use CPU for PyTorch and CPU+ANE for Core ML; "
                "they are backend/deployment measurements, not equal-hardware algorithm comparisons."
            ),
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, result in results.items():
        print(
            f"{name}: {result['correct']}/{len(rows)} ({100 * result['accuracy']:.6f}%), "
            f"p50={result['model_call_latency_ms']['median_ms']:.3f} ms",
            flush=True,
        )
    print(f"Conversion parity gates passed: {report['conversion_parity_passed']}; report: {args.report}", flush=True)
    if args.require_parity and not report["conversion_parity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
