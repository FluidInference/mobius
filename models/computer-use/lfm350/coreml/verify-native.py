"""Compare the no-cache Core ML boundary with RLCD's cached native engine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from decision import CandidateScorer, load_native, source_path
from fixtures import CONTEXT, SCHEMA
from preprocessing import Shape, batch_arrays, prepare_candidates, select_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(2)
    snapshot = source_path()
    sys.path.insert(0, str(snapshot))
    from rlcd.engine import Engine  # noqa: E402

    tokenizer, model = load_native(args.device)
    upstream = Engine.__new__(Engine)
    upstream.model = model
    upstream.tokenizer = tokenizer
    upstream.device = args.device
    upstream.dtype = "float32"
    shape = Shape(length=args.length, candidates=args.batch)
    candidates = prepare_candidates(tokenizer, CONTEXT, SCHEMA, shape)
    scorer = CandidateScorer(model, shape.length).eval()

    started = time.perf_counter()
    with torch.inference_mode():
        native = upstream.constrained(CONTEXT, SCHEMA)
        ours = []
        for offset in range(0, len(candidates), shape.candidates):
            group = candidates[offset : offset + shape.candidates]
            arrays = batch_arrays(tokenizer, group, shape)
            tensors = [torch.from_numpy(value).to(args.device) for value in arrays.values()]
            ours.extend(scorer(*tensors).float().cpu().tolist()[: len(group)])

    expected = [entry["log_likelihood"] for entries in native["scores"].values() for entry in entries]
    errors = [abs(left - right) for left, right in zip(ours, expected)]
    same_answer = select_values(candidates, ours) == json.loads(native["text"])
    report = {
        "source_repo": "notnotsamuel/LFM2.5-350M-RLCD",
        "source_revision": "deb589d803d141cabd158ef55f6617b128529f36",
        "device": args.device,
        "shape": vars(shape),
        "candidates": len(candidates),
        "max_log_likelihood_error": max(errors),
        "same_selected_values": same_answer,
        "seconds": round(time.perf_counter() - started, 2),
        "expected": expected,
        "actual": ours,
    }
    output = Path("build/native-parity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not same_answer or max(errors) > 1e-2:
        raise SystemExit("native cached/no-cache parity failed")


if __name__ == "__main__":
    main()
