"""Integration checks against the real pinned GLiNER2 checkpoint."""
from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download

from preprocessing import load_processor, prepare_classification, prepare_with_processor


def test_standalone_tokenizer_matches_native():
    source = snapshot_download(
        "fastino/gliner2.5-small-v1",
        revision="7e6f537f10337497069276892a5ef435028252ce",
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
