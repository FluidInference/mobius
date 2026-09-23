"""Compare real Verdict PyTorch decisions with the converted Core ML package."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import coremltools as ct
import torch
from gliclass import GLiClassModel
from transformers import AutoTokenizer

from assets import LOCK, ROOT, sha256, verify_assets
from decision_index_engine import adapt_question, as_text
from native_reference import build_request, decode, load_calibrator
from preprocessing import prepare

CASES = [
    (
        "lost card",
        "I lost my wallet yesterday and need to stop my debit card immediately.",
        {
            "type": "choice",
            "question": "What is the primary customer inquiry?",
            "options": [
                {"id": "card_lost", "description": "Reporting a lost or stolen card"},
                {"id": "pin_reset", "description": "Requesting a PIN reset"},
            ],
        },
    ),
    (
        "binary inquiry",
        "The account holder reports a stolen debit card.",
        {"type": "noul", "proposition": "The customer needs a card blocked"},
    ),
    (
        "urgency score",
        "The customer reports a stolen card and unknown charges.",
        {
            "type": "score",
            "question": "How urgent is the request?",
            "levels": [
                {"id": "low", "description": "Low urgency", "value": 0},
                {"id": "medium", "description": "Medium urgency", "value": 1},
                {"id": "high", "description": "High urgency", "value": 2},
            ],
        },
    ),
    (
        "insufficient evidence",
        "No information is available.",
        {
            "type": "choice",
            "question": "Which card was stolen?",
            "options": [
                {"id": "visa", "description": "a Visa debit card"},
                {"id": "mastercard", "description": "a Mastercard debit card"},
            ],
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    source = verify_assets()
    package = args.package or ROOT / "build" / f"verdict_fp16_L{args.length}_candidates25.mlpackage"
    model = GLiClassModel.from_pretrained(source).eval()
    tokenizer = AutoTokenizer.from_pretrained(source)
    calibrator = load_calibrator(source)
    coreml = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
    rows = []
    cases = list(CASES)
    if args.length >= 512:
        fixture = json.loads((ROOT / "fixtures" / "decision-index-long.json").read_text())
        benchmark_question = next(iter(fixture["questions"].values()))
        cases.append((fixture["id"], as_text(fixture["state"]), adapt_question(benchmark_question)))
    for name, context, question in cases:
        request = build_request(context, question)
        arrays = prepare(tokenizer, model.config.class_token_index, request.text, args.length, 25)
        if int(arrays["class_marker_map"].sum()) != len(request.ids):
            raise ValueError(f"{name}: lost a candidate marker")
        with torch.inference_mode():
            native_logits = (
                model(
                    input_ids=torch.from_numpy(arrays["input_ids"]).long(),
                    attention_mask=torch.from_numpy(arrays["attention_mask"]).long(),
                    max_num_classes=len(request.ids),
                )
                .logits[0]
                .numpy()
            )
        native = decode(native_logits, request, calibrator)
        for _ in range(3):
            coreml.predict(arrays)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            output = coreml.predict(arrays)
            times.append((time.perf_counter() - start) * 1000)
        result = decode(output["logits"], request, calibrator)
        error = max(abs(native["probabilities"][k] - result["probabilities"][k]) for k in request.ids)
        rows.append(
            {
                "name": name,
                "type": question["type"],
                "tokens": int(arrays["attention_mask"].sum()),
                "candidates_including_abstention": len(request.ids),
                "native": native,
                "coreml": result,
                "max_calibrated_probability_error": error,
                "selected_id_agrees": native["selected_id"] == result["selected_id"],
                "median_model_call_ms": statistics.median(times),
            }
        )
    report = {
        "source_repo": LOCK["source_repo"],
        "source_revision": LOCK["source_revision"],
        "package": package.name,
        "package_files_sha256": {
            str(path.relative_to(package)): sha256(path) for path in sorted(package.rglob("*")) if path.is_file()
        },
        "chip": platform.processor(),
        "macos": platform.mac_ver()[0],
        "max_calibrated_probability_error": max(r["max_calibrated_probability_error"] for r in rows),
        "selected_id_agreements": sum(r["selected_id_agrees"] for r in rows),
        "questions": len(rows),
        "rows": rows,
    }
    report["passed"] = (
        report["selected_id_agreements"] == len(rows) and report["max_calibrated_probability_error"] <= 0.02
    )
    standard = f"verdict_fp16_L{args.length}_candidates25.mlpackage"
    name = f"verification-L{args.length}.json" if package.name == standard else f"verification-{package.stem}.json"
    target = args.report or ROOT / "reports" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
