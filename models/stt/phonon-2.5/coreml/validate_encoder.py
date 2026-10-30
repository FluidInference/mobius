"""Validate converted encoder numerics without recompiling a CPU compute plan."""
import argparse
import json
from pathlib import Path
import time

import coremltools as ct
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("--units", choices=["cpu", "ane", "gpu"], default="ane")
args = parser.parse_args()
units = {"cpu": ct.ComputeUnit.CPU_ONLY, "ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU}[args.units]
reference = np.load(args.reference)
started = time.perf_counter()
model = ct.models.CompiledMLModel(str(args.model), compute_units=units)
load_seconds = time.perf_counter() - started
inputs = {"mel": reference["mel"].astype(np.float32), "mel_length": reference["mel_length"].astype(np.int32)}
output = model.predict(inputs)
actual, expected = output["encoder"], reference["encoder"]
assert actual.shape == expected.shape and np.isfinite(actual).all()
assert np.array_equal(output["encoder_length"], reference["encoder_length"])
cosine = float(np.dot(actual.ravel(), expected.ravel()) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
nrmse = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
times = []
for _ in range(3):
    started = time.perf_counter()
    model.predict(inputs)
    times.append(time.perf_counter() - started)
report = {"units": args.units, "load_seconds": load_seconds, "cosine": cosine,
          "nrmse": nrmse, "encoder_frames": int(output["encoder_length"].ravel()[0]),
          "median_prediction_seconds": float(np.median(times))}
args.report.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)

assert cosine > 0.999 and nrmse < 0.05, (cosine, nrmse)
