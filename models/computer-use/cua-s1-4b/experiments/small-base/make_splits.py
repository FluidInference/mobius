"""Train / val / test CuaTask splits for retraining the Cua-S1 recipe on a smaller Qwen3.5 base.

Cua's own SFT split (`crossdataset_hard_v2`) is not published, so this rebuilds an open one with
Cua's converters:

- train + val: GUI-360 *train* episodes (vyokky/GUI-360 `train/data/*/*/success`), text modality,
  hard distractors on (the setting of the test split), split 90/10 BY EPISODE so no screen
  sequence straddles the two; plus `cua_bench_s1` generator tasks (all synthetic families) at a
  seed disjoint from the Core ML fixtures (20260923) and the GPTQ calibration set (7777777).
- test: the 613-task GUI-360 *test* split from ../../coreml/make_gui360.py, copied verbatim
  (dataset_hash e305de96...). The trainer refuses a --val whose name starts with "test".

Writes splits/{train,val,test}.jsonl and splits/manifest.json (counts, families, dataset hashes).
Needs Cua's source on PYTHONPATH (see README).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from cua_bench_s1.datagen.generator import generate_dataset
from cua_bench_s1.datagen.gui360 import convert_episode
from cua_bench_s1.datagen.specs import EXAMPLE_APPS
from cua_bench_s1.task import dataset_hash, load_jsonl, save_jsonl
from cua_s1.four_b import LETTERS

SYNTHETIC_SEED = 424_242
SPLIT_SEED = 0
EXPECTED_TEST_HASH = "e305de962fae21523cd972fa53f0984879f25554c1519f001ad25470c67ac2f6"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui360-train", type=Path, default=Path("../../coreml/data/gui360/train/data"))
    ap.add_argument("--test", type=Path, default=Path("../../coreml/data/gui360-test-text.jsonl"))
    ap.add_argument("--synthetic-per-app", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("splits"))
    args = ap.parse_args()

    episodes = sorted(args.gui360_train.glob("*/*/success/*.jsonl"))
    if not episodes:
        raise SystemExit(f"no GUI-360 train episodes under {args.gui360_train} (see README for the download)")
    rng = random.Random(SPLIT_SEED)
    shuffled = episodes[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_eps, train_eps = set(shuffled[:n_val]), set(shuffled[n_val:])

    def convert(paths: set[Path]) -> list:
        tasks = []
        for path in sorted(paths):
            tasks += convert_episode(
                path, images_root=Path("/nonexistent"), out_dir=Path("/nonexistent"),
                modality_available=("text",), hard_distractor=True,
            )
        return [t for t in tasks if len(t.options) <= len(LETTERS)]

    scratch = args.out / "_synthetic"
    synthetic = generate_dataset(EXAMPLE_APPS, args.synthetic_per_app, SYNTHETIC_SEED, ("text",), scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    rng.shuffle(synthetic)
    n_syn_val = int(len(synthetic) * args.val_frac)

    train = convert(train_eps) + synthetic[n_syn_val:]
    val = convert(val_eps) + synthetic[:n_syn_val]
    test = load_jsonl(args.test)
    if dataset_hash(test) != EXPECTED_TEST_HASH:
        raise SystemExit(f"{args.test} is not the 613-task GUI-360 test split (hash mismatch)")

    test_eps = {t.provenance.get("episode_id") for t in test}
    leaked = {t.provenance.get("episode_id") for t in train + val} & test_eps
    if leaked:
        raise SystemExit(f"test episodes leaked into train/val: {sorted(leaked)[:5]}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    rng.shuffle(train)
    rng.shuffle(val)
    for name, tasks in (("train", train), ("val", val), ("test", test)):
        save_jsonl(tasks, args.out / f"{name}.jsonl")
        manifest[name] = {
            "tasks": len(tasks),
            "families": dict(Counter(t.family for t in tasks)),
            "sources": dict(Counter(t.provenance.get("source", "synthetic") for t in tasks)),
            "dataset_hash": dataset_hash(tasks),
        }
    manifest["recipe"] = {
        "gui360_train_episodes": {"train": len(train_eps), "val": len(val_eps)},
        "synthetic_seed": SYNTHETIC_SEED,
        "synthetic_per_app": args.synthetic_per_app,
        "split_seed": SPLIT_SEED,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v["tasks"] for k, v in manifest.items() if k != "recipe"}))


if __name__ == "__main__":
    main()
