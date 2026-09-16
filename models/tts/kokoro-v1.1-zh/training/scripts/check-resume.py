"""Compare uninterrupted real-data optimization with save/restart/resume."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from kokoro_training.artifacts import write_json


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, "-m", "kokoro_training.train", "--data", str(args.data),
              "--micro", "2", "--train-style", "--validate-every", "1", "--accumulation", "1"]
    for name, steps, resume in [("continuous", 3, False), ("resumed", 1, False), ("resumed", 3, True)]:
        with (args.output / f"{name}-{steps}.log").open("w") as log:
            subprocess.run([*common, "--output", str(args.output / name), "--steps", str(steps),
                            *(["--resume"] if resume else [])], stdout=log, stderr=subprocess.STDOUT, check=True)
    continuous = torch.load(args.output / "continuous/last.pt", weights_only=True, map_location="cpu")
    resumed = torch.load(args.output / "resumed/last.pt", weights_only=True, map_location="cpu")
    checks = {key: equal(continuous[key], resumed[key]) for key in
              ("model", "optimizer", "cpu_rng", "cuda_rng", "sample_rng", "step", "best")}
    write_json(args.output / "result.json", {"passed": all(checks.values()), "checks": checks,
               "protocol": "3 uninterrupted real-data optimizer updates vs 1 update + new process + 2 updates"})
    print(json.dumps(checks))
    assert all(checks.values()), checks


if __name__ == "__main__":
    main()
