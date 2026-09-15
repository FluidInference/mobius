"""Freeze a development-selected checkpoint before the test set is evaluated."""

import argparse
import json
from pathlib import Path
import torch

from kokoro_training.artifacts import ROOT, sha256, write_json
from kokoro_training.bundle import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path(".artifacts/baseline"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.record.exists():
        raise ValueError("Refusing to rewrite a selection record")
    policy = json.loads((ROOT / "evaluation-policy.json").read_text())
    validations = {event["step"]: event["development"] for event in
                   (json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines())
                   if "development" in event}
    candidates = []
    for step in policy["candidate_steps"]:
        result_file = args.evaluations / f"mf5-v2-step{step}-eval/results.json"
        result = json.loads(result_file.read_text())
        if result["step"] != step or not all(e["suite"] in {"dev", "controls"} for e in result["entries"]):
            raise ValueError("Selection may use development data and controls only")
        signals = [entry["signal"] for entry in result["entries"] if entry["variant"] == "candidate"]
        signal_pass = all(s["rms"] > 0.001 and s["clipping_fraction"] < 0.001 for s in signals)
        english = result["aggregates"]["dev/en/candidate"]["rate"]
        mandarin = result["aggregates"]["dev/zh/candidate"]["rate"]
        candidates.append({"step": step, "score": (english + mandarin) / 2,
                           "english_wer": english, "mandarin_cer": mandarin,
                           "development_loss": validations[step]["total"], "signal_pass": signal_pass,
                           "evaluation_sha256": sha256(result_file), "evaluated_model_sha256": result["model_sha256"]})
    eligible = [entry for entry in candidates if entry["signal_pass"]]
    if not eligible:
        raise ValueError("No candidate passed signal checks")
    selected = min(eligible, key=lambda entry: (entry["score"], entry["development_loss"]))
    checkpoint = args.run / f"snapshot-{selected['step']}.pt"
    bundle = export(checkpoint, args.assets, args.output)
    if bundle["files"]["model.pth"]["sha256"] != selected["evaluated_model_sha256"]:
        raise ValueError("Re-exported weights differ from the evaluated checkpoint")
    record = {"policy": policy, "policy_sha256": sha256(ROOT / "evaluation-policy.json"),
              "candidates": candidates, "selected": selected,
              "checkpoint_sha256": sha256(checkpoint), "bundle_manifest_sha256": sha256(args.output / "bundle.json"),
              "test_used_for_selection": False}
    write_json(args.record, record)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
