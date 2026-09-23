"""Inspect the trained embedding selected for weight-only quantization."""

import os
import runpy
from pathlib import Path

import coremltools as ct
import pytest


def test_quantizer_selects_real_word_embedding():
    package = os.environ.get("GLINER2_EXTRACTION_FEATURE_PACKAGE")
    if not package:
        pytest.skip("Set GLINER2_EXTRACTION_FEATURE_PACKAGE to a pinned real Core ML feature package")
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "quantize-extraction-coreml.py"))
    model = ct.models.MLModel(package, skip_model_load=True)
    name, shape, dtype = namespace["embedding_weight_name"](model)
    assert name.startswith("encoder_embeddings_word_embeddings_weight")
    assert shape[0] >= 250_112
    assert shape[1] == 768
    assert dtype in ("float16", "float32")
