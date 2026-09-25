"""Kev's own serving cases on this Mac, set up exactly as kev.serve.main() does (MLX backend, bf16)."""
import json, sys
from dataclasses import replace

import torch

sys.path.insert(0, "scripts")
from serving_bench import latency  # his cases and timing (median of 20, new vs repeated state)
from kev.checkpoint import Checkpoint, LoadOptions
from kev.device import default_device
from kev.serve import Server

run, out = sys.argv[1], sys.argv[2]
dev = default_device()
opts = LoadOptions.from_env()
if dev == "mps" and opts.attn is None: opts = replace(opts, attn="sdpa")
if dev != "cpu" and opts.dtype is None: opts = replace(opts, dtype=torch.bfloat16)
if opts.backend is None: opts = replace(opts, backend="auto")
ck = Checkpoint(run)
tok, model = ck.load(dev, opts)
server = Server(ck, tok, model, dev)
print(f"{run} on {dev} via {model.backend} ({model.dtype})", flush=True)
result = {"run": run, "device": dev, "backend": str(model.backend), "dtype": str(model.dtype), "cases": latency(server, 20)}
server.close()
json.dump(result, open(out, "w"), indent=2)
print(json.dumps(result, indent=2))
