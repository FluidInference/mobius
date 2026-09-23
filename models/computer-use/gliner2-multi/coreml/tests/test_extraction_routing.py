"""Schema routing for the quantized multilingual extraction runtime."""

from gliner2 import Schema

from extraction_runtime import CoreMLAdaptiveBoundaryExtractor


def record_schema(mode: str):
    schema = Schema()
    schema.structure("employment", mode=mode).field("person", dtype="str").field("company", dtype="str")
    return schema


def test_latent_record_uses_fp32():
    assert CoreMLAdaptiveBoundaryExtractor.selected_precision(record_schema("latent")) == "fp32"


def test_other_selected_schemas_use_fp16():
    assert CoreMLAdaptiveBoundaryExtractor.selected_precision(Schema().entities(["person"])) == "fp16"
    assert CoreMLAdaptiveBoundaryExtractor.selected_precision(record_schema("natural")) == "fp16"
    assert CoreMLAdaptiveBoundaryExtractor.selected_precision(record_schema("anchorless")) == "fp16"
