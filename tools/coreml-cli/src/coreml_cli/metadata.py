"""Extract model metadata from .mlmodelc."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import CoreML
from Foundation import NSURL


def get_model_metadata(model_path: Path) -> dict:
    """Extract metadata from a compiled CoreML model."""
    result = _from_metadata_json(model_path)
    result.update(_from_model_description(model_path))
    return result


def _from_metadata_json(model_path: Path) -> dict:
    meta_file = model_path / "metadata.json"
    if not meta_file.exists():
        return {}

    with open(meta_file) as f:
        raw = json.load(f)

    if not raw or not isinstance(raw, list):
        return {}

    m = raw[0]
    inputs = [
        {"name": i["name"], "type": i["formattedType"]}
        for i in m.get("inputSchema", [])
    ]
    outputs = [
        {"name": o["name"], "type": o["formattedType"]}
        for o in m.get("outputSchema", [])
    ]

    return {
        "model_type": m.get("modelType", {}).get("name", ""),
        "compute_precision": m.get("computePrecision", ""),
        "storage_precision": m.get("storagePrecision", ""),
        "spec_version": m.get("specificationVersion"),
        "availability": m.get("availability", {}),
        "op_histogram": m.get("mlProgramOperationTypeHistogram", {}),
        "coremltools_version": m.get("userDefinedMetadata", {}).get("com.github.apple.coremltools.version", ""),
        "source": m.get("userDefinedMetadata", {}).get("com.github.apple.coremltools.source", ""),
        "inputs": inputs,
        "outputs": outputs,
    }


def _from_model_description(model_path: Path) -> dict:
    """Extract author, description, license, version from MLModel metadata."""
    url = NSURL.fileURLWithPath_(str(model_path))
    config = CoreML.MLModelConfiguration.alloc().init()
    model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
    if error or model is None:
        return {}

    meta = model.modelDescription().metadata()
    if not meta:
        return {}

    result = {}
    for objc_key, out_key in [
        ("MLModelDescriptionKey", "description"),
        ("MLModelAuthorKey", "author"),
        ("MLModelLicenseKey", "license"),
        ("MLModelVersionStringKey", "version"),
    ]:
        val = meta.get(objc_key)
        if val and str(val).strip():
            result[out_key] = str(val).strip()

    return result
