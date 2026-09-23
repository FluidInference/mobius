"""Integration checks against the real pinned GLiNER2 checkpoint."""
from pathlib import Path

import pytest
from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download

from preprocessing import load_processor, prepare_classification, prepare_with_processor
from runtime import classify


def test_standalone_tokenizer_matches_native():
    source = snapshot_download(
        "fastino/gliner2.5-base-v1",
        revision="1a8bc24e00dc7300b9017c81d63e3dcdabb26596",
        allow_patterns=[
            "config.json", "encoder_config/*", "model.safetensors", "tokenizer.json", "tokenizer_config.json"
        ],
    )
    native = AutoExtractor.from_pretrained(source, map_location="cpu").eval()
    processor = load_processor(source)
    for text, task, labels in [
        ("The rocket launched successfully.", "topic", ["science", "sports", "politics"]),
        ("The team won the championship.", "topic", ["science", "sports", "politics"]),
        ("I feel happy.", "emotion", ["joy", "sadness"]),
    ]:
        reference = prepare_classification(native, text, task, labels, 128, 8)
        standalone = prepare_with_processor(processor, text, task, labels, 128, 8)
        for key in reference:
            assert (reference[key] == standalone[key]).all(), key


def test_compressed_runtime_matches_fp16_on_real_request(tmp_path):
    build = Path(__file__).resolve().parents[1] / "build"
    packages = [
        build / f"gliner2_base_classification_{precision}_L128_K8.mlpackage"
        for precision in ("fp16", "embedding_w8")
    ]
    tokenizer = build / "hub" / "tokenizer"
    if not tokenizer.exists() or any(not package.exists() for package in packages):
        pytest.skip("local Core ML packages and tokenizer are required")
    (tmp_path / "tokenizer").symlink_to(tokenizer, target_is_directory=True)
    for package in packages:
        (tmp_path / package.name).symlink_to(package, target_is_directory=True)
    labels = ["science", "sports", "politics"]
    outputs = [classify(str(tmp_path), "The rocket launched successfully.", "topic", labels,
                        precision=precision) for precision in ("fp16", "embedding_w8")]
    assert outputs[0]["label"] == outputs[1]["label"]
    assert max(abs(outputs[0]["probabilities"][label] - outputs[1]["probabilities"][label])
               for label in labels) <= 0.02
