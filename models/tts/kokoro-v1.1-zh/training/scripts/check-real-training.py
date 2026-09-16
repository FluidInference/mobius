"""Explicit GPU integration check using acquired recordings, never fixture audio."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from kokoro_training.artifacts import write_json
from kokoro_training.features import dtw_durations
from kokoro_training.model import CompatibleGenerator, select_style
from kokoro_training.supervised import AdaptationModel, SupervisedLoss
from kokoro_training.train import batch_for, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path(".artifacts/baseline"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new evidence path")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rows, _ = load_data(args.data)
    g = CompatibleGenerator(json.loads((args.assets / "config.json").read_text()))
    g.load_checkpoint(torch.load(args.assets / "kokoro-v1_1-zh.pth", weights_only=True))
    model = AdaptationModel(g, torch.load(args.assets / "voices/zf_001.pt", weights_only=True), train_style=True).cuda()
    criterion = SupervisedLoss().cuda()
    cases, controls = [], []
    for language in ("en", "zh"):
        subset = [r for r in rows if r["language"] == language and r["split"] == "train"][:4]
        for index, row in enumerate(subset):
            batch = batch_for(row, args.data, "cuda")
            model.eval()
            style, _ = select_style(model.voices, batch["ids"].shape[1] - 2)
            torch.manual_seed(3456)
            audio, durations = g(batch["ids"], style)
            size = int(durations.sum())
            parity_batch = {**batch, "durations": durations[None], "audio": audio[None],
                            "f0": torch.zeros(size * 2, device="cuda"), "energy": torch.zeros(1, size * 2, device="cuda")}
            torch.manual_seed(3456)
            with torch.no_grad():
                output = model(parity_batch)
            delta = float((output["audio"].reshape(-1) - audio).abs().max())
            assert delta == 0, delta
            model.train()
            model.zero_grad(set_to_none=True)
            losses = criterion(model(batch, crop_frames=96))
            losses["total"].backward()
            coverage = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    group = name.split(".")[1] if name.startswith("generator.") else name
                    coverage[group] = coverage.get(group, 0.0) + float(parameter.grad.norm())
                else:
                    assert parameter.grad is None, name
            assert all(v > 0 for v in coverage.values()), coverage
            cases.append({"id": row["id"], "language": language, "supervised_parity_max_abs": delta,
                          "gradient_group_norm_sums": coverage, "loss": float(losses["total"].detach())})
            teacher = audio.cpu().numpy()
            real = batch["audio"].cpu().numpy().reshape(-1)
            wrong = batch_for(subset[(index + 1) % len(subset)], args.data, "cpu")["audio"].numpy().reshape(-1)
            # Wrong-text negative control is real speech from the same speaker/language.
            report = {"id": row["id"]}
            for label, target in (("self", teacher), ("matched", real), ("wrong_text", wrong)):
                try:
                    transferred, detail = dtw_durations(teacher, target, durations.cpu().numpy())
                    if label == "self":
                        assert np.array_equal(transferred, durations.cpu().numpy())
                    report[label] = detail["mean_path_cost"]
                except ValueError as exc:
                    report[label] = {"rejected": str(exc)}
            controls.append(report)
    passed = all(isinstance(c["matched"], float) and isinstance(c["wrong_text"], float)
                 and c["matched"] < c["wrong_text"] for c in controls)
    write_json(args.output, {"gradient_and_supervised_parity_passed": True, "cases": cases,
                            "alignment_controls": controls, "all_matched_costs_below_wrong_text": passed,
                            "limitations": "DTW control discrimination is not phoneme-boundary annotation accuracy"})
    print(json.dumps({"cases": len(cases), "all_matched_costs_below_wrong_text": passed}))


if __name__ == "__main__":
    main()
