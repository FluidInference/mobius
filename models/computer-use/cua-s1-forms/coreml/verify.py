"""Verify Core ML against the real upstream model on the pinned 196-decision demo."""

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
import numpy as np
import torch

from assets import LOCK_PATH, ROOT, load_demo, load_reference, sha256, verify_assets
from export_model import ExportScorer
from preprocessing import InputLimits, prepare_inputs

PROBABILITY_TOLERANCE = 0.005


def latency_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def validate_prediction(output: dict, count: int, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(output["logits"])
    probabilities = np.asarray(output["probabilities"])
    if logits.shape != (1, capacity) or probabilities.shape != (1, capacity):
        raise ValueError(f"Unexpected output shapes: {logits.shape}, {probabilities.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite model outputs")
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Invalid probabilities")
    if not np.isclose(probabilities[0, :count].sum(), 1.0, atol=0.001):
        raise ValueError("Live option probabilities do not sum to one")
    if np.any(probabilities[0, count:] != 0):
        raise ValueError("Padding received probability mass")
    return logits[0, :count], probabilities[0, :count]


def check_package(build_dir: Path) -> tuple[Path, dict]:
    manifest = json.loads((build_dir / "conversion.json").read_text())
    if sha256(LOCK_PATH) != manifest["assets_lock_sha256"]:
        raise ValueError("The model was exported from a different asset lock")
    package = build_dir / manifest["model"]
    actual = {str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*")) if p.is_file()}
    if actual != manifest["package_files"]:
        raise ValueError("Core ML package checksum mismatch")
    return package, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/verification.json")
    parser.add_argument(
        "--compute-units",
        nargs="+",
        default=["ALL", "CPU_AND_NE"],
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    lock = verify_assets()
    package, conversion = check_package(args.build_dir)
    limits = InputLimits(**conversion["limits"])
    reference, collator, _ = load_reference()
    from cua_s1.model import validate_example

    wrapper = ExportScorer(reference, optimization=conversion.get("optimization", "baseline")).eval()
    rows = load_demo()
    examples = [validate_example(row) for row in rows]
    inputs, expected, preprocessing_ms, reference_ms = [], [], [], []
    wrapper_max_delta = 0.0
    wrapper_agreement = 0
    # Upstream uses its own variable-length collator; the export adapter uses
    # separate fixed-shape preprocessing. This detects padding/encoding errors.
    with torch.no_grad():
        for row, example in zip(rows, examples):
            started = time.perf_counter()
            arrays = prepare_inputs(row["context"], row["options"], limits)
            preprocessing_ms.append(1000 * (time.perf_counter() - started))
            inputs.append(arrays)
            batch = collator([example])
            started = time.perf_counter()
            logits = reference(batch)[0]
            reference_ms.append(1000 * (time.perf_counter() - started))
            probabilities = logits.softmax(-1)
            padded_logits, padded_probabilities = wrapper(*(torch.from_numpy(v) for v in arrays.values()))
            _, actual = validate_prediction(
                {"logits": padded_logits.numpy(), "probabilities": padded_probabilities.numpy()},
                len(row["options"]),
                limits.max_options,
            )
            expected.append({"logits": logits.numpy(), "probabilities": probabilities.numpy()})
            wrapper_max_delta = max(wrapper_max_delta, float(np.max(np.abs(actual - probabilities.numpy()))))
            wrapper_agreement += int(actual.argmax() == probabilities.argmax().item())
    upstream_correct = sum(int(item["probabilities"].argmax()) == row["label"] for item, row in zip(expected, rows))
    report = {
        "purpose": "Local conversion parity on the upstream demo; not a generalization or live GUI benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_revision": lock["model_revision"],
        "source_revision": lock["source_revision"],
        "dataset_revision": lock["dataset_revision"],
        "dataset_file": "demo.jsonl",
        "dataset_sha256": sha256(ROOT / lock["evaluation_file"]),
        "conversion": conversion,
        "environment": {
            "machine": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "os": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "rows": len(rows),
        "upstream": {
            "correct": upstream_correct,
            "accuracy": upstream_correct / len(rows),
            "inference_latency_ms": latency_summary(reference_ms),
        },
        "export_adapter_fp32": {"argmax_agreement": wrapper_agreement, "max_abs_probability_error": wrapper_max_delta},
        "host_preprocessing_latency_ms": latency_summary(preprocessing_ms),
        "thresholds": {
            "argmax_agreement": 1.0,
            "max_abs_probability_error": PROBABILITY_TOLERANCE,
            "export_adapter_max_abs_probability_error": 0.0001,
            "allow_accuracy_loss": False,
        },
        "backends": {},
    }
    all_passed = wrapper_agreement == len(rows) and wrapper_max_delta <= 0.0001
    print(
        f"Upstream: {upstream_correct}/{len(rows)} correct; FP32 adapter max probability error {wrapper_max_delta:.3g}",
        flush=True,
    )
    for units in args.compute_units:
        started = time.perf_counter()
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, units))
        load_seconds = time.perf_counter() - started
        # Warm up with real inputs; timing below excludes model loading and encoding.
        for arrays in inputs[:3]:
            model.predict(arrays)
        latencies, details = [], []
        correct = agreement = 0
        errors, logit_errors = [], []
        per_action = {}
        for index, (row, arrays, target) in enumerate(zip(rows, inputs, expected)):
            started = time.perf_counter()
            output = model.predict(arrays)
            latencies.append(1000 * (time.perf_counter() - started))
            logits, probabilities = validate_prediction(output, len(row["options"]), limits.max_options)
            prediction = int(probabilities.argmax())
            upstream_prediction = int(target["probabilities"].argmax())
            error = float(np.max(np.abs(probabilities - target["probabilities"])))
            logit_error = float(np.max(np.abs(logits - target["logits"])))
            correct += prediction == row["label"]
            agreement += prediction == upstream_prediction
            errors.append(error)
            logit_errors.append(logit_error)
            action = row.get("meta", {}).get("action", "unknown")
            counts = per_action.setdefault(action, Counter())
            counts["rows"] += 1
            counts["correct"] += prediction == row["label"]
            details.append(
                {
                    "row": index,
                    "label": row["label"],
                    "upstream": upstream_prediction,
                    "coreml": prediction,
                    "max_abs_probability_error": error,
                    "max_abs_logit_error": logit_error,
                }
            )
        # Metamorphic check: reordering real choices must preserve scores by identity.
        permutation_errors = []
        for index in (0, 31, 63, 95, 127, 195):
            row = rows[index]
            arrays = prepare_inputs(row["context"], list(reversed(row["options"])), limits)
            _, probabilities = validate_prediction(model.predict(arrays), len(row["options"]), limits.max_options)
            permutation_errors.append(float(np.max(np.abs(probabilities[::-1] - expected[index]["probabilities"]))))
        passed = (
            correct >= upstream_correct
            and agreement == len(rows)
            and max(errors) <= PROBABILITY_TOLERANCE
            and max(permutation_errors) <= PROBABILITY_TOLERANCE
        )
        backend = {
            "passed": passed,
            "correct": correct,
            "accuracy": correct / len(rows),
            "argmax_agreement": agreement,
            "max_abs_probability_error": max(errors),
            "mean_row_max_abs_probability_error": float(np.mean(errors)),
            "max_abs_logit_error": max(logit_errors),
            "load_seconds": load_seconds,
            "warm_inference_latency_ms": latency_summary(latencies),
            "reversed_option_order": {"rows": 6, "max_abs_probability_error": max(permutation_errors)},
            "per_action": per_action,
            "decisions": details,
        }
        report["backends"][units] = backend
        all_passed = all_passed and passed
        print(
            f"{units}: {correct}/{len(rows)} correct; {agreement}/{len(rows)} agree; "
            f"max probability error {max(errors):.3g}; p50 {np.median(latencies):.2f} ms; passed={passed}",
            flush=True,
        )
        del model
    report["passed"] = all_passed
    report["code_sha256"] = {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.report}", flush=True)
    if not all_passed:
        raise SystemExit("Conversion verification FAILED; inspect the report")


if __name__ == "__main__":
    main()
