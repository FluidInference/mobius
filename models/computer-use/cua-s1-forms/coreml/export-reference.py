"""Export pinned PyTorch probabilities for independent Swift/runtime parity checks."""

import argparse
import json
from pathlib import Path

import torch

from assets import ROOT, load_demo, load_reference, sha256, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/reference-probabilities.json")
    args = parser.parse_args()
    lock = verify_assets()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    model, collator, _ = load_reference()
    from cua_s1.model import validate_example

    with torch.inference_mode():
        probabilities = [model(collator([validate_example(row)]))[0].softmax(-1).tolist() for row in load_demo()]
    report = {
        "model_revision": lock["model_revision"],
        "source_revision": lock["source_revision"],
        "dataset_revision": lock["dataset_revision"],
        "dataset_sha256": sha256(ROOT / lock["evaluation_file"]),
        "probabilities": probabilities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(probabilities)} real upstream predictions to {args.output}")


if __name__ == "__main__":
    main()
