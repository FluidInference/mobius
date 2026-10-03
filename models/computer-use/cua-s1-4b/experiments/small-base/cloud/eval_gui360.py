"""Score a Cua-S1 adapter on the 613-task GUI-360 text test split (CUDA, bf16, FourBModel readout).

Same prompt, letter readout and task scoring as the Core ML benchmark (coreml/bench_gui360.py):
per element, argmax over that element's own options; a task counts only if every element is right.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from cua_bench_s1.task import load_jsonl
from cua_s1.four_b import LETTERS, Option, assign_letters, build_prompt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (omit to score the zero-shot base)")
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows, by_family, t0 = [], defaultdict(lambda: [0, 0]), time.time()
    for task in load_jsonl(args.tasks):
        assignment = assign_letters([Option(o.element_id, o.role, o.label, o.action, o.entity_id) for o in task.options])
        msgs = build_prompt(assignment, app=task.app, task_family=task.family, ax_tree=task.ax_tree, goal=task.goal)
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt")
        with torch.no_grad():
            logits = model(**ids.to("cuda")).logits[0, -1]
        logits = logits[torch.tensor(letter_ids[: len(assignment.letters)], device="cuda")].float().cpu().tolist()
        by_el = defaultdict(list)
        for i, opt in enumerate(assignment.options):
            by_el[opt.element_id].append(i)
        correct = all(
            assignment.options[max(idx, key=lambda i: logits[i])].action == task.expected.get(el)
            for el, idx in by_el.items()
        )
        by_family[task.family][0] += correct
        by_family[task.family][1] += 1
        rows.append({"id": task.id, "family": task.family, "correct": correct, "logits": logits})
    acc = sum(r["correct"] for r in rows) / len(rows)
    summary = {
        "base_model": args.base_model, "adapter": args.adapter, "tasks": len(rows), "task_accuracy": acc,
        "by_family": {k: {"correct": c, "n": n, "acc": c / n} for k, (c, n) in sorted(by_family.items())},
        "seconds": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
