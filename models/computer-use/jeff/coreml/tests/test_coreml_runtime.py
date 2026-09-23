"""Standalone Core ML runtime integration check with bundled tokenizer/config."""

from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import snapshot_download

from export import FIXTURES, REVISION, SOURCE
from runtime import JeffCoreML

PACKAGE = Path("build/JeffDecision-L128-FP16.mlpackage")


@pytest.mark.skipif(not PACKAGE.exists(), reason="local converted Core ML package is absent")
def test_standalone_coreml_scores_match_native_reference():
    assets = snapshot_download(SOURCE, revision=REVISION, local_files_only=True)
    engine = JeffCoreML(assets, PACKAGE)
    native_logits = {
        "billing": [10.187458038330078, -6.71131706237793],
        "technical": [-12.322779655456543, 13.379047393798828],
        "three_way": [5.41853141784668, -12.076669692993164, -9.080745697021484],
        "boolean": [1.0098832845687866, -1.0589547157287598],
    }
    for name, text, group in FIXTURES:
        observed = np.asarray(engine.score(text, list(group.labels), name=group.name), dtype=np.float32)
        expected = 1.0 / (1.0 + np.exp(-np.asarray(native_logits[name], dtype=np.float32)))
        assert int(np.argmax(observed)) == int(np.argmax(expected))
        assert float(np.max(np.abs(observed - expected))) < 0.03
