"""Run Kev's own benchmark (kev.benchmark, unchanged scoring) against the Core ML packages.

kev.predictors.LocalPredictor loads a checkpoint and calls model.encode / model.forward and reads model.head.temperature.
This swaps the loaded model for CoreMLKev, which answers every question row with the smallest Core ML bucket that fits,
or with --fused, CoreMLFusedKev: one fused call (state + packed questions) per record, rows only for questions that do
not fit a lane.

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
from kev_stages import fused_inputs, pack_groups
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


class CoreMLFusedKev(CoreMLKev):
    state_buckets = [32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 6144]
    packed_buckets = [32, 64, 128, 256]
    lane, readouts, max_options = 128, 16, 16

    def __init__(self, build: Path, merged: Path, fused: Path):
        super().__init__(build, merged)
        self.fused, self.functions = fused, {}
        self.usage = {"fused calls": 0, "row fallbacks": 0}

    def _function(self, name):
        if name not in self.functions:
            self.functions[name] = ct.models.MLModel(str(self.fused), compute_units=ct.ComputeUnit.CPU_AND_GPU,
                                                     function_name=name)
        return self.functions[name]

    def forward(self, enc):
        state_ids, state_pos, rows = rows_of(enc)
        state_len = next((s for s in self.state_buckets if len(state_ids) <= s), None)
        fits = [i for i, r in enumerate(rows) if state_len and len(r["ids"]) <= self.lane and len(r["opts"]) <= self.max_options]
        logits = [None] * len(rows)
        rest = [i for i in range(len(rows)) if i not in fits]
        if rest:
            self.usage["row fallbacks"] += len(rest)
            for i in rest:
                logits[i] = self._row(state_ids, state_pos, rows[i])
        for group in pack_groups([len(rows[i]["ids"]) for i in fits], self.packed_buckets[-1], self.readouts, self.lane):
            chosen = [rows[fits[g]] for g in group]
            start, extent = 0, 0
            for r in chosen:
                n = len(r["ids"])
                if start // self.lane != (start + n - 1) // self.lane:
                    start = (start // self.lane + 1) * self.lane
                start += n
                extent = start
            packed_len = next(p for p in self.packed_buckets if extent <= p)
            inputs = fused_inputs(self.cfg, self.embed, state_ids, [(r["ids"], r["decide"], r["opts"]) for r in chosen],
                                  state_len, packed_len, self.readouts, self.max_options, self.pad_id, lane=self.lane)
            names = ["hidden", "cos", "sin", "valid", "tail_onehot", "segment", "lag_keep", "lag_tail", "decide_onehot",
                     "option_onehot", "option_mask"]
            model = self._function(f"fused_S{state_len}_P{packed_len}_B{self.readouts}_K{self.max_options}")
            out = np.asarray(model.predict({n: t.numpy().astype(np.float16) for n, t in zip(names, inputs)})["logits"])
            self.usage["fused calls"] += 1
            for slot, g in enumerate(group):
                logits[fits[g]] = torch.from_numpy(out[slot, : len(rows[fits[g]]["opts"])].astype(np.float32))
        return logits

    def _row(self, state_ids, state_pos, r):
        n, k = len(state_ids) + len(r["ids"]), len(r["opts"])
        length, options, model = next((b for b in self.buckets if n <= b[0] and k <= b[1]), (None, None, None))
        if model is None:
            raise ValueError(f"row of {n} tokens / {k} options exceeds every bucket")
        inputs = row_inputs(self.cfg, self.embed, state_ids + r["ids"], state_pos + r["pos"], len(state_ids) + r["decide"],
                            [len(state_ids) + o for o in r["opts"]], length, options, self.pad_id)
        out = model.predict({name: t.numpy().astype(np.float32) for name, t in zip(NAMES, inputs)})
        return torch.from_numpy(np.asarray(out["logits"])[:k].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--merged", type=Path)
    ap.add_argument("--fused", type=Path, help="multifunction KevFused.mlpackage")
    args, rest = ap.parse_known_args()
    rest = [a for a in rest if a != "--"]
    merged = args.merged or args.build / "merged"
    coreml = CoreMLFusedKev(args.build, merged, args.fused) if args.fused else CoreMLKev(args.build, merged)

    def load_coreml(self, device, opts=None):
        """Same tokenizer as Checkpoint.load (the base's, at the pinned revision); the Core ML model instead of torch."""
        meta = self.meta
        return load_tokenizer(meta.base, revision=meta.base_revision), coreml

    kev.predictors.Checkpoint.load = load_coreml
    sys.argv = ["kev.benchmark"] + rest
    from kev.benchmark import main as benchmark_main

    benchmark_main()
    print("usage:", coreml.usage)


if __name__ == "__main__":
    main()
