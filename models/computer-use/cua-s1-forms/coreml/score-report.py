"""Apply Cua's unmodified offline evaluation harness to saved model decisions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from assets import ROOT, load_demo, sha256, verify_assets
from verify import check_package

METRICS_SHA256 = "49dccdeb4e2b456187cdc4491a6d78fabac6a45384272f8e3ec802a58872aa6e"


def load_evaluator():
    path = ROOT / "vendor/cua_metrics.py"
    if sha256(path) != METRICS_SHA256:
        raise ValueError("Upstream evaluation source checksum mismatch")
    spec = importlib.util.spec_from_file_location("cua_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load pinned Cua evaluation harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_predictions


def normalize_choice(index: int, row: dict, choice: int) -> dict:
    if not isinstance(choice, int) or isinstance(choice, bool) or not 0 <= choice < len(row["options"]):
        raise ValueError(f"Invalid option index for row {index}")
    option = row["options"][choice]
    if option.startswith("fill "):
        return {"id": str(index), "action": "fill", "target": choice}
    if option in {"check", "click", "skip"}:
        return {"id": str(index), "action": option}
    raise ValueError(f"Unsupported action option in row {index}")


def score_decisions(rows: list[dict], decisions: list[dict], prediction_key: str) -> dict:
    indices = [decision["row"] for decision in decisions]
    if len(indices) != len(rows) or set(indices) != set(range(len(rows))):
        raise ValueError("Decisions must cover every demo row exactly once")
    gold = [normalize_choice(index, row, row["label"]) for index, row in enumerate(rows)]
    predictions = [normalize_choice(d["row"], rows[d["row"]], d[prediction_key]) for d in decisions]
    return load_evaluator()(gold, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/verification.json")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/upstream-metrics.json")
    args = parser.parse_args()
    lock = verify_assets()
    _, conversion = check_package(args.build_dir)
    report = json.loads(args.report.read_text())
    if report["conversion"] != conversion or report["dataset_sha256"] != sha256(ROOT / lock["evaluation_file"]):
        raise ValueError("Verification report does not match the pinned model and dataset")
    rows = load_demo()
    results = {}
    for backend, result in report["backends"].items():
        results[backend] = score_decisions(rows, result["decisions"], "coreml")
    first = next(iter(report["backends"].values()))
    results["upstream_pytorch"] = score_decisions(rows, first["decisions"], "upstream")
    payload = {
        "verification_report_sha256": sha256(args.report),
        "evaluator_source_revision": lock["source_revision"],
        "evaluator_source_path": "libs/cua-s1/evals/metrics.py",
        "evaluator_sha256": METRICS_SHA256,
        "adapter_sha256": sha256(Path(__file__)),
        "scope": "Offline action/target classification, not GUI execution or task completion",
        "normalization": "Stable row IDs; fill target is entity option index; upstream treats skip as abstention",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for name, metrics in results.items():
        print(
            f"{name}: {metrics['counts']['correct']}/{metrics['examples']} correct; "
            f"wrong actions={metrics['wrong_action_rate']}, wrong targets={metrics['wrong_target_rate']}"
        )
    if not report["passed"] or any(metrics["accuracy"] != 1.0 for metrics in results.values()):
        raise SystemExit("Saved decisions failed the pinned demo check")


if __name__ == "__main__":
    main()
