"""Fixtures for the Swift runtime parity check (FluidUse `fluiduse-cua-s1 parity`).

One JSON per modality with, per task: the state fields the Swift prompt builder takes, the
reference chat string and token ids (tokenizer parity), the image grid (preprocessing parity)
and the fp32 reference letter logits (end-to-end parity).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cua_bench_s1.task import load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    args = ap.parse_args()

    tasks = {t.id: t for t in load_jsonl(args.fixtures / "tasks.jsonl")}
    for modality in ("text", "multimodal"):
        ref = json.loads((args.fixtures / f"reference-{modality}-fp32.json").read_text())
        rows = []
        for r in ref["records"]:
            t = tasks[r["id"]]
            rows.append({
                "id": t.id,
                "app": t.app,
                "task_family": t.family,
                "goal": t.goal,
                "ax_tree": t.ax_tree if modality == "text" else None,
                "screenshot": t.screenshot if modality == "multimodal" else None,
                "options": [
                    {"element_id": o.element_id, "role": o.role, "label": o.label, "action": o.action,
                     "entity_id": o.entity_id}
                    for o in t.options
                ],
                "expected": t.expected,
                "chat": r["chat"],
                "input_ids": r["input_ids"],
                "image_grid_thw": r.get("image_grid_thw"),
                "letter_logits": r["letter_logits"],
            })
        out = args.fixtures / f"swift-{modality}.json"
        out.write_text(json.dumps({"modality": modality, "reference": "fp32-hf-layers", "tasks": rows}))
        print(f"{out}: {len(rows)} tasks")


if __name__ == "__main__":
    main()
