"""Compare a converted RLCD package with its pinned native full-value scorer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from decision import SOURCE_REVISION, CandidateScorer, load_native, source_path
from fixtures import CONTEXT, SCHEMA
from preprocessing import Shape, batch_arrays, prepare_candidates, select_values
from release_integrity import package_file_hashes


def score_case(case_id, context, schema, shape, tokenizer, scorer, coreml):
    candidates = prepare_candidates(tokenizer, context, schema, shape)
    expected, actual, call_ms = [], [], []
    for offset in range(0, len(candidates), shape.candidates):
        group = candidates[offset : offset + shape.candidates]
        arrays = batch_arrays(tokenizer, group, shape)
        with torch.inference_mode():
            tensors = [torch.from_numpy(array) for array in arrays.values()]
            expected.extend(scorer(*tensors).float().numpy().tolist()[: len(group)])
        start = time.perf_counter()
        result = coreml.predict(arrays)
        call_ms.append((time.perf_counter() - start) * 1000)
        actual.extend(np.asarray(result["scores"]).reshape(-1).tolist()[: len(group)])

    errors = [abs(left - right) for left, right in zip(expected, actual)]
    return {
        "case_id": case_id,
        "candidates": len(candidates),
        "max_log_likelihood_error": max(errors),
        "same_selected_values": select_values(candidates, expected) == select_values(candidates, actual),
        "model_call_ms": call_ms,
        "expected": expected,
        "actual": actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-value-tokens", type=int, default=16)
    parser.add_argument("--compute-units", choices=["cpu", "cpu-ane", "all"], default="cpu-ane")
    parser.add_argument("--upstream-cases", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)

    shape = Shape(args.length, args.batch, args.max_value_tokens)
    tokenizer, native = load_native("cpu")
    scorer = CandidateScorer(native, shape.length).eval()
    units = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-ane": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }[args.compute_units]
    coreml = ct.models.MLModel(str(args.package), compute_units=units)
    fixtures = [("coreml-fixture", SCHEMA, CONTEXT)]
    if args.upstream_cases:
        sys.path.insert(0, str(source_path()))
        from rlcd.tasks import CASES  # noqa: E402

        fixtures = [(case_id, schema, context) for case_id, schema, context, _ in CASES[: args.upstream_cases]]
    cases = [
        score_case(case_id, context, schema, shape, tokenizer, scorer, coreml) for case_id, schema, context in fixtures
    ]
    report = {
        "package": args.package.name,
        "source_revision": SOURCE_REVISION,
        "package_files_sha256": package_file_hashes(args.package),
        "compute_units": args.compute_units,
        "fixture_set": "upstream-rlcd-tasks" if args.upstream_cases else "coreml-fixture",
        "max_log_likelihood_error": max(case["max_log_likelihood_error"] for case in cases),
        "same_selected_values": all(case["same_selected_values"] for case in cases),
        "cases": cases,
    }
    output = Path(f"build/coreml-parity-{args.package.stem}-{args.compute_units}-cases{args.upstream_cases}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["same_selected_values"] or report["max_log_likelihood_error"] > 0.5:
        raise SystemExit("Core ML scorer diverged from native")


if __name__ == "__main__":
    main()
