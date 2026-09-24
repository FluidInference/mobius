"""GUI-360 test split via Cua's own converter (cua_bench_s1.datagen.gui360).

Cua's published "hard cross-dataset (GUI-360)" split is frozen by hash only (615 text /
168 multimodal tasks); its selection recipe is not in the repo. This builds the closest
open analogue: every GUI-360 *test* success episode (in_app + online), up to 5 steps each,
hard-distractor options on, text modality. Protocol-adjacent, not a byte-identical split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cua_bench_s1.datagen.gui360 import convert_episode
from cua_bench_s1.task import dataset_hash, save_jsonl
from cua_s1.four_b import LETTERS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/gui360/test/data"))
    ap.add_argument("--out", type=Path, default=Path("data/gui360-test-text.jsonl"))
    args = ap.parse_args()

    tasks, too_many = [], 0
    for path in sorted(args.root.glob("*/*/success/*.jsonl")):
        for task in convert_episode(
            path, images_root=Path("/nonexistent"), out_dir=Path("/nonexistent"),
            modality_available=("text",), hard_distractor=True,
        ):
            if len(task.options) > len(LETTERS):
                too_many += 1
                continue
            tasks.append(task)
    save_jsonl(tasks, args.out)
    families = {}
    for t in tasks:
        families[t.family] = families.get(t.family, 0) + 1
    print(f"{len(tasks)} tasks ({too_many} dropped: >26 options), families={families}")
    print(f"dataset_hash={dataset_hash(tasks)}")


if __name__ == "__main__":
    main()
