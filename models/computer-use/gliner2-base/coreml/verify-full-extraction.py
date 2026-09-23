"""Selected end-to-end native/Core ML checks across GLiNER2 schema paths."""

import argparse
import json
from pathlib import Path

import torch
from gliner2 import AttributeGroup, AutoExtractor, Schema
from huggingface_hub import snapshot_download

from extraction_runtime import CoreMLBoundaryExtractor

MODEL_ID = "fastino/gliner2.5-base-v1"
MODEL_REVISION = "1a8bc24e00dc7300b9017c81d63e3dcdabb26596"


def structure(mode):
    schema = Schema()
    builder = schema.structure("employment", mode=mode, anchor="person" if mode == "natural" else None)
    builder.field("person", dtype="str")
    builder.field("company", dtype="str")
    return schema


def fixtures():
    attribute = Schema().entities(["person", "organization"])
    attribute.entity_attributes({"role": AttributeGroup(labels=["founder", "employee"])})
    mixed = Schema().entities(["person", "organization"])
    mixed.classification("sentiment", ["positive", "negative"])
    choice = Schema()
    choice.structure("product").field("category", dtype="str", choices=["electronics", "clothing"])
    return [
        (
            "entities",
            "Alice founded Acme in Toronto in 2020.",
            Schema().entities(["person", "organization", "location"]),
        ),
        (
            "entities_multi",
            "Marie Curie was born in Warsaw and worked in Paris.",
            Schema().entities(["person", "city"]),
        ),
        ("relations", "Alice founded Acme in Toronto.", Schema().relations(["founded"])),
        ("relations_two", "Steve Jobs founded Apple.", Schema().relations(["founded"])),
        ("record_natural", "Alice works at Acme. Bob works at Beta.", structure("natural")),
        ("record_latent", "Alice works at Acme. Bob works at Beta.", structure("latent")),
        ("record_anchorless", "Alice works at Acme. Bob works at Beta.", structure("anchorless")),
        ("attributes", "Alice founded Acme.", attribute),
        ("mixed_classification", "Alice founded Acme.", mixed),
        (
            "classification_only",
            "The rocket launched successfully.",
            Schema().classification("topic", ["science", "sports", "politics"]),
        ),
        ("enum_choice", "The item is electronics.", choice),
    ]


def without_confidence(value):
    if isinstance(value, dict):
        return {key: without_confidence(item) for key, item in value.items() if key != "confidence"}
    if isinstance(value, list):
        return [without_confidence(item) for item in value]
    return value


def confidence_errors(reference, actual):
    if isinstance(reference, dict) and isinstance(actual, dict):
        errors = []
        for key, value in reference.items():
            if key == "confidence" and key in actual:
                errors.append(abs(float(value) - float(actual[key])))
            elif key in actual:
                errors.extend(confidence_errors(value, actual[key]))
        return errors
    if isinstance(reference, list) and isinstance(actual, list):
        return [error for a, b in zip(reference, actual) for error in confidence_errors(a, b)]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    native = AutoExtractor.from_pretrained(str(source), map_location="cpu").eval()
    runtime = CoreMLBoundaryExtractor(args.model_dir, precision=args.precision)
    cases = []
    for name, text, schema in fixtures():
        expected = native.extract(text, schema, include_confidence=True, include_spans=True)
        actual = runtime.extract(text, schema, include_confidence=True, include_spans=True)
        errors = confidence_errors(expected, actual)
        cases.append(
            {
                "name": name,
                "text": text,
                "native": expected,
                "coreml": actual,
                "structure_match": without_confidence(expected) == without_confidence(actual),
                "maximum_confidence_error": max(errors, default=None),
            }
        )
    result = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "precision": args.precision,
        "selected_manifest": "eleven fixed real-text schema fixtures, not a Decision Index score",
        "matched": sum(case["structure_match"] for case in cases),
        "total": len(cases),
        "cases": cases,
    }
    path = Path(args.model_dir) / f"verify-full-{args.precision}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "matched": result["matched"],
                "total": result["total"],
                "failed": [case["name"] for case in cases if not case["structure_match"]],
            },
            indent=2,
        )
    )
    if result["matched"] != result["total"] and not args.allow_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
