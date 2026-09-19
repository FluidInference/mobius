"""The exported reference is a complete, pinned cross-runtime fixture."""

import json
import subprocess
import sys

import numpy as np

from assets import ROOT, load_demo, sha256, verify_assets


def test_reference_export_covers_the_pinned_demo(tmp_path):
    output = tmp_path / "reference.json"
    subprocess.run([sys.executable, str(ROOT / "export-reference.py"), "--output", str(output)], check=True)
    reference = json.loads(output.read_text())
    lock = verify_assets()
    rows = load_demo()
    assert reference["model_revision"] == lock["model_revision"]
    assert reference["dataset_sha256"] == sha256(ROOT / lock["evaluation_file"])
    assert len(reference["probabilities"]) == len(rows) == 196
    for row, values in zip(rows, reference["probabilities"]):
        probabilities = np.asarray(values)
        assert len(values) == len(row["options"])
        assert np.isfinite(probabilities).all()
        assert np.all(probabilities >= 0)
        assert np.isclose(probabilities.sum(), 1.0, atol=0.000001)
        assert probabilities.argmax() == row["label"]
