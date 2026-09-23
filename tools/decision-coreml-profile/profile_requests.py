"""Small real-request latency profiles for the converted Jeff and Kev models.

Run with the chosen model toolkit's pinned Python environment. Model loading is
timed separately; the request measurement includes rendering/tokenization,
Core ML prediction, and typed-answer decoding. No source weights are loaded.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLKITS = {
    "kev05": ROOT / "models/computer-use/kev-0.5b/coreml",
    "kev06": ROOT / "models/computer-use/kev-0.6b/coreml",
    "jeff": ROOT / "models/computer-use/jeff/coreml",
}
UNITS = {
    "all": ct.ComputeUnit.ALL,
    "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
}


def package_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def percentile_ms(samples: list[float], fraction: float) -> float:
    if not samples:
        raise ValueError("at least one sample is required")
    return sorted(samples)[min(len(samples) - 1, math.ceil(fraction * len(samples)) - 1)]


def unlabelled(request: dict) -> dict:
    copy_request = copy.deepcopy(request)
    for question in copy_request["questions"].values():
        question.pop("label", None)
        question.pop("src", None)
    return copy_request


def kev05_runner(toolkit, assets: Path, package: Path, units: str):
    runtime = importlib.import_module("runtime")
    verify = importlib.import_module("verify")
    session = runtime.KevCoreML(assets, units=units, package=package)
    fixtures = [(f"fixture-{i}", unlabelled(request)) for i, request in enumerate(verify.fixtures())]
    return fixtures, session.predict


def kev06_runner(toolkit, assets: Path, model):
    from kev.api import output_tokens, to_answers
    from transformers import AutoTokenizer

    preprocessing = importlib.import_module("preprocessing")
    verify = importlib.import_module("verify")
    training = json.loads((assets / "config/training_config.json").read_text())
    if training["args"]["option_isolation"] not in (0, False):
        raise ValueError("the published Kev 0.6B package requires option_isolation=False")
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True)
    fixtures = [(f"fixture-{i}", unlabelled(request)) for i, request in enumerate(verify.fixtures())]

    def run(request):
        arrays, encoded, metadata, _ = preprocessing.prepare_runtime_inputs(
            tokenizer, request, preprocessing.Shape()
        )
        result = np.asarray(model.predict(arrays)["probabilities"], dtype=np.float64)
        count = len(metadata[0]["keys"])
        probabilities = result[0, :count].tolist()
        answers = to_answers([probabilities], metadata)
        return {
            "answers": answers,
            "probabilities": probabilities,
            "input_tokens": len(encoded["ids"]),
            "output_tokens": output_tokens(tokenizer, answers),
        }

    return fixtures, run


def jeff_runner(toolkit, assets: Path, package: Path, units):
    export = importlib.import_module("export")
    runtime = importlib.import_module("runtime")
    engine = runtime.JeffCoreML(assets, package, compute_units=units)
    fixtures = [(name, (text, group)) for name, text, group in export.FIXTURES]

    def run(item):
        text, group = item
        probabilities = engine.score(text, list(group.labels), name=group.name, description=group.description)
        return {"chosen": group.labels[int(np.argmax(probabilities))], "probabilities": probabilities}

    return fixtures, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=TOOLKITS, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--units", choices=UNITS, default="cpu-ne")
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.package.exists():
        parser.error(f"package does not exist: {args.package}")
    toolkit = TOOLKITS[args.model]
    sys.path.insert(0, str(toolkit))
    started = time.perf_counter()
    if args.model == "jeff":
        fixtures, run = jeff_runner(toolkit, args.assets, args.package, UNITS[args.units])
    elif args.model == "kev05":
        fixtures, run = kev05_runner(toolkit, args.assets, args.package, args.units)
    else:
        model = ct.models.MLModel(str(args.package), compute_units=UNITS[args.units])
        fixtures, run = kev06_runner(toolkit, args.assets, model)
    load_ms = (time.perf_counter() - started) * 1000
    # One warmup prevents compile and first-call setup from contaminating steady-state timing.
    run(fixtures[0][1])
    cases = []
    latencies = []
    for _ in range(args.rounds):
        for name, request in fixtures:
            t0 = time.perf_counter_ns()
            result = run(request)
            latency_ms = (time.perf_counter_ns() - t0) / 1e6
            latencies.append(latency_ms)
            cases.append({"name": name, "latency_ms": latency_ms, "result": result})
    report = {
        "model": args.model,
        "package": str(args.package.resolve()),
        "package_bytes": package_bytes(args.package),
        "compute_units": args.units,
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "power_state": subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "timing_scope": "exploratory local battery-powered real-request smoke test",
        "load_and_prepare_ms": load_ms,
        "warmups": 1,
        "rounds": args.rounds,
        "request_count": len(latencies),
        "full_request_p50_ms": statistics.median(latencies),
        "full_request_p95_ms": percentile_ms(latencies, 0.95),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
