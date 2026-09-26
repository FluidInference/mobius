"""Kev's own PyTorch model (Checkpoint.load, default dtype) through Kev's serving path (api.to_record -> encode ->
forward) on a yes/no workload, for comparison with the Core ML fused path (FluidUse `KevCheck bios`). Run from a Kev
clone; wrap in `/usr/bin/time -l` for peak memory.

    KEV_BACKEND=torch uv run python bench_original_bios.py <workload.json> <out.json> [--device mps]
"""
import argparse
import json
import os
import time

os.environ.setdefault("KEV_BACKEND", "torch")

import torch  # noqa: E402

from kev.api import SystemOneRequest, to_record  # noqa: E402
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.device import sync  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("out")
    ap.add_argument("--run", default="jaredpalmer/kev-0.8b")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    records = json.load(open(args.workload))
    start = time.perf_counter()
    tok, model = Checkpoint(args.run).load(args.device, LoadOptions.from_env())
    load = time.perf_counter() - start

    @torch.no_grad()
    def answer(record):
        rec, _ = to_record(SystemOneRequest.model_validate({"state": record["state"], "questions": record["questions"]}))
        logits = model.forward(model.encode(tok, rec))
        p = [torch.softmax(z.float(), -1).cpu() for z in logits]
        sync(args.device)
        return [float(x[1]) for x in p]

    start = time.perf_counter()
    for record in records[:3]:
        answer(record)
    warm = time.perf_counter() - start
    times, p_yes = [], []
    run_start = time.perf_counter()
    for record in records:
        call = time.perf_counter()
        p_yes.append(answer(record))
        times.append(1000 * (time.perf_counter() - call))
    total = time.perf_counter() - run_start
    median = sorted(times)[len(times) // 2]
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    dtype = "/".join(dtypes)
    json.dump({"load_s": load, "warm_s": warm, "total_s": total, "median_ms": median, "per_call_ms": times,
               "p_yes": p_yes, "dtype": dtype, "device": args.device}, open(args.out, "w"))
    print(f"{dtype} on {args.device}: load {load:.1f} s, warm {warm:.1f} s, {len(p_yes)} states x {len(p_yes[0])} "
          f"questions in {total:.2f} s, median {median:.1f} ms")


if __name__ == "__main__":
    main()
