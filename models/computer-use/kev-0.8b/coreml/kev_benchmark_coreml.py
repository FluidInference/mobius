"""Run Kev's own benchmark (kev.benchmark, unchanged scoring) against the Core ML packages.

kev.predictors.LocalPredictor loads a checkpoint and calls model.encode / model.forward and reads model.head.temperature.
This swaps the loaded model for CoreMLKev, which answers every question row with the smallest Core ML bucket that fits.

    cd <kev repo> && PYTHONPATH=<this dir> uv run --with coremltools==9.0 python <this dir>/kev_benchmark_coreml.py \
        --build <this dir>/build -- --run jaredpalmer/kev-0.8b --suite evals/v4/transfer-v4 --out runs/coreml-transfer-v4
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import coremltools as ct
import numpy as np
import torch

import kev.predictors
from kev.checkpoint import Checkpoint
from kev.model import encode, load_tokenizer, rows_of
from kev_export import row_inputs
from qwen35_export import TextConfig

NAMES = ["hidden", "cos", "sin", "decide_onehot", "option_onehot", "option_mask"]


class CoreMLKev:
    backend, hybrid, option_isolation, device = "coreml", True, False, "cpu"

    def __init__(self, build: Path, merged: Path):
        meta = json.loads((merged / "meta.json").read_text())
        self.cfg = TextConfig(meta["text_config"])
        self.pad_id = meta["pad_id"]
        self.head = SimpleNamespace(temperature=meta["temperature"])
        self.buckets = []
        for directory in sorted(build.glob("L*_K*"), key=lambda p: json.loads((p / "config.json").read_text())["length"]):
            config = json.loads((directory / "config.json").read_text())
            package = next(directory.glob("KevRow_*.mlpackage"))
            model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
            self.buckets.append((config["length"], config["max_options"], model))
            vocab, hidden = config["vocab_size"], config["hidden_size"]
            self.embed = torch.from_numpy(np.fromfile(directory / "embeddings.f16", dtype=np.float16).reshape(vocab, hidden)
                                          .astype(np.float32))
        self.usage = {length: 0 for length, _, _ in self.buckets}

    def eval(self):
        return self

    def encode(self, tok, rec, **kw):
        return encode(tok, rec, option_isolation=False, **kw)

    def forward(self, enc):
        state_ids, state_pos, rows = rows_of(enc)
        logits = []
        for r in rows:
            n, k = len(state_ids) + len(r["ids"]), len(r["opts"])
            length, options, model = next((b for b in self.buckets if n <= b[0] and k <= b[1]), (None, None, None))
            if model is None:
                raise ValueError(f"row of {n} tokens / {k} options exceeds every bucket")
            self.usage[length] += 1
            inputs = row_inputs(self.cfg, self.embed, state_ids + r["ids"], state_pos + r["pos"], len(state_ids) + r["decide"],
                                [len(state_ids) + o for o in r["opts"]], length, options, self.pad_id)
            out = model.predict({name: t.numpy().astype(np.float32) for name, t in zip(NAMES, inputs)})
            logits.append(torch.from_numpy(np.asarray(out["logits"])[:k].astype(np.float32)))
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--merged", type=Path)
    args, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]
    coreml = CoreMLKev(args.build, args.merged or args.build / "merged")

    def load_coreml(self, device, opts=None):
        """Same tokenizer as Checkpoint.load (the base's, at the pinned revision); the Core ML model instead of torch."""
        meta = self.meta
        return load_tokenizer(meta.base, revision=meta.base_revision), coreml

    kev.predictors.Checkpoint.load = load_coreml
    sys.argv = ["kev.benchmark"] + rest
    from kev.benchmark import main as benchmark_main

    benchmark_main()
    print("question rows per bucket:", coreml.usage)


if __name__ == "__main__":
    main()
