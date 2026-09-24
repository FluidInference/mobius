"""Parity fixtures from Cua's own cua-bench-s1 task generator (no hand-made data).

Writes fixtures/tasks.jsonl (+ fixtures/screens/*.png) covering every synthetic
app family, both modalities available on each task.

Needs the Cua source tree on PYTHONPATH (see README):
  PYTHONPATH=$CUA/libs/cua-bench-s1/python/src:$CUA/libs/cua-s1/python/src
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cua_bench_s1.datagen.generator import generate_dataset
from cua_bench_s1.datagen.specs import EXAMPLE_APPS
from cua_bench_s1.task import dataset_hash, save_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("fixtures"))
    ap.add_argument("--per-app", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    screens = args.out / "screens"
    tasks = generate_dataset(
        EXAMPLE_APPS, args.per_app, args.seed, ("text", "multimodal"), screens
    )
    save_jsonl(tasks, args.out / "tasks.jsonl")
    families = sorted({t.family for t in tasks})
    print(f"{len(tasks)} tasks, families={families}, hash={dataset_hash(tasks)}")


if __name__ == "__main__":
    main()
