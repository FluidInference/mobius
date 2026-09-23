"""Compare NanoJev Core ML outputs with its pinned trained PyTorch checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from assets import ROOT, load_model
from fixtures import requests
from preprocessing import prepare_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--max-probability-error", type=float, default=0.02)
    args = parser.parse_args()
    root, tokenizer, model = load_model()
    encoder = ct.models.MLModel(
        str(args.build_dir / f"nanojev_encoder_fp16_L{args.length}_K{args.candidates}.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    head = ct.models.MLModel(
        str(args.build_dir / f"nanojev_heads_fp16_K{args.candidates}.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    max_error = 0.0
    for request in requests():
        inputs, candidate_mask, example = prepare_request(root, tokenizer, request, args.length, args.candidates)
        typ = example["type"]
        with torch.no_grad():
            native = model([example], tokenizer.pad_token_id)[0][0, : len(example["candidate_ids"])]
        expected = torch.softmax(native.float(), dim=-1).numpy()
        embeddings = encoder.predict(inputs)["embeddings"]
        output = head.predict(
            {
                "embeddings": np.asarray(embeddings, dtype=np.float32),
                "candidate_mask": candidate_mask,
                "use_set_head": np.array([[typ == "choice"]], dtype=np.float32),
                "is_boolean": np.array([[typ == "boolean"]], dtype=np.float32),
            }
        )
        actual = np.asarray(output["probabilities"])[0, : len(example["candidate_ids"])]
        error = float(np.max(np.abs(expected - actual)))
        max_error = max(error, max_error)
        same = int(np.argmax(expected)) == int(np.argmax(actual))
        print(f"{typ}: argmax={same} max_probability_error={error:.6f}")
        if not same or error > args.max_probability_error:
            raise SystemExit(1)
    print(f"3/3 request types agreed; max_probability_error={max_error:.6f}")


if __name__ == "__main__":
    main()
