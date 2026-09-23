"""Check native scorer versus fixed K16 Core ML over real upstream demo requests."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from assets import ROOT, check_base_access
from native_reference import choose, encode_options, pad_batch
from scorer import score_coreml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT / "build" / "system_one_gemma_fp16_L256_K16.mlpackage")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    check_base_access()

    import coremltools as ct
    import torch

    from export_model import load_trained_scorer

    tokenizer, native = load_trained_scorer()
    coreml = ct.models.MLModel(str(args.package), compute_units=ct.ComputeUnit.ALL)
    demos = json.loads((ROOT / "build" / "source" / "demos.json").read_text())
    rows = []
    for demo in demos:
        sequences = encode_options(tokenizer, demo["state"], demo["question"], demo["options"])
        ids, masks = pad_batch(sequences, tokenizer.pad_token_id)
        with torch.inference_mode():
            native_logits = (
                native(
                    input_ids=torch.tensor(ids, dtype=torch.long),
                    attention_mask=torch.tensor(masks, dtype=torch.long),
                )
                .logits.reshape(-1)
                .float()
                .tolist()
            )
        native_choice = choose(demo["options"], native_logits)
        times = []
        converted = None
        for _ in range(args.repeats):
            start = time.perf_counter()
            converted = score_coreml(coreml, tokenizer, demo["state"], demo["question"], demo["options"])
            times.append((time.perf_counter() - start) * 1000)
        assert converted is not None
        error = max(abs(a - b) for a, b in zip(native_choice["probabilities"], converted["probabilities"]))
        rows.append(
            {
                "demo": demo["name"],
                "candidates": len(demo["options"]),
                "native_selected_index": native_choice["selected_index"],
                "coreml_selected_index": converted["selected_index"],
                "max_calibrated_probability_error": error,
                "median_end_to_end_ms": statistics.median(times),
            }
        )
    report = {
        "package": args.package.name,
        "source": "pinned upstream demos.json",
        "questions": len(rows),
        "selected_id_agreements": sum(r["native_selected_index"] == r["coreml_selected_index"] for r in rows),
        "max_calibrated_probability_error": float(np.max([r["max_calibrated_probability_error"] for r in rows])),
        "rows": rows,
    }
    report["passed"] = (
        report["selected_id_agreements"] == len(rows) and report["max_calibrated_probability_error"] <= 0.02
    )
    target = ROOT / "reports" / "verification-L256-K16.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
