"""Task accuracy on the GUI-360 text split: PyTorch (FourBModel-equivalent) vs Core ML.

Readout = Cua's eval readout (train_4b_v2.evaluate_val): letter logits at the last
position, argmax over each element's own options, task correct only if every element
matches its gold action. Writes per-task logits so backends can be compared row by row.

  --backend torch   HF Qwen3.5-4B bf16 + PEFT text adapter (unmerged), MPS -- the shipped runtime
  --backend coreml  Core ML decoder parts (+ --variant), host embedding gather
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from cua_bench_s1.task import load_jsonl
from cua_s1.four_b import LETTERS, Option, assign_letters, build_prompt
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from qwen35_export import TextConfig, rope_cos_sin

BASE = "Qwen/Qwen3.5-4B"
ADAPTER = "cua-ai/cua-s1-4b-0.2"


class CoreMLRunner:
    def __init__(self, model_dir: Path, compute_units: str):
        import coremltools as ct

        self.meta = json.loads((model_dir / "config.json").read_text())
        self.L, D = self.meta["seq_len"], self.meta["hidden_size"]
        raw = json.loads(Path(hf_hub_download(BASE, "config.json")).read_text())["text_config"]
        cos, sin = rope_cos_sin(TextConfig(raw), torch.arange(self.L)[None].expand(3, self.L))
        self.cos, self.sin = cos.numpy().astype(np.float16), sin.numpy().astype(np.float16)
        self.emb = np.memmap(model_dir.parent / "embeddings.f16", dtype=np.float16, mode="r").reshape(-1, D)
        cu = getattr(ct.ComputeUnit, compute_units)
        self.parts = [
            ct.models.MLModel(str(model_dir / f"CuaS1Decoder_part{i}.mlpackage"), compute_units=cu)
            for i in range(len(self.meta["parts"]))
        ]

    def letter_logits(self, ids: list[int]) -> np.ndarray | None:
        if len(ids) > self.L:
            return None
        h = np.zeros((1, self.L, self.emb.shape[1]), dtype=np.float16)
        h[0, : len(ids)] = self.emb[np.array(ids)]
        onehot = np.zeros((1, self.L), dtype=np.float16)
        onehot[0, len(ids) - 1] = 1
        for i, part in enumerate(self.parts):
            feed = {"hidden": h, "cos": self.cos, "sin": self.sin}
            if i == len(self.parts) - 1:
                feed["last_onehot"] = onehot
                return np.asarray(part.predict(feed)["letter_logits"], dtype=np.float32).reshape(-1)
            h = part.predict(feed)["hidden_out"]
        raise AssertionError


class TorchRunner:
    def __init__(self, device: str):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(model, str(Path(snapshot_download(ADAPTER)) / "text")).eval().to(device)
        self.device = device

    def letter_logits(self, ids: list[int], letter_ids: list[int]) -> np.ndarray:
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0, -1]
        return out[torch.tensor(letter_ids, device=self.device)].float().cpu().numpy()


def score(task, assignment, logits: np.ndarray) -> tuple[bool, int, int]:
    """(task correct, elements correct, elements) with per-element argmax over its own options."""
    by_el = defaultdict(list)
    for i, opt in enumerate(assignment.options):
        by_el[opt.element_id].append(i)
    ok = 0
    for el, idxs in by_el.items():
        best = max(idxs, key=lambda i: logits[i])
        ok += assignment.options[best].action == task.expected.get(el)
    return ok == len(by_el), ok, len(by_el)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=Path("data/gui360-test-text.jsonl"))
    ap.add_argument("--backend", choices=["torch", "coreml"], required=True)
    ap.add_argument("--variant", default="")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--compute-units", default="CPU_AND_GPU")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    tasks = load_jsonl(args.tasks)[: args.limit or None]
    if args.backend == "coreml":
        model_dir = Path("build/text") / (f"L{args.length}" + (f"-{args.variant}" if args.variant else ""))
        runner = CoreMLRunner(model_dir, args.compute_units)
    else:
        runner = TorchRunner(args.device)

    rows, times, by_family = [], [], defaultdict(lambda: [0, 0])
    for n, task in enumerate(tasks):
        assignment = assign_letters([Option(o.element_id, o.role, o.label, o.action, o.entity_id) for o in task.options])
        msgs = build_prompt(assignment, app=task.app, task_family=task.family, ax_tree=task.ax_tree, goal=task.goal)
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))["input_ids"]
        t0 = time.perf_counter()
        if args.backend == "coreml":
            logits = runner.letter_logits(ids)
        else:
            logits = runner.letter_logits(ids, letter_ids)
        times.append((time.perf_counter() - t0) * 1000)
        if logits is None:
            rows.append({"id": task.id, "family": task.family, "tokens": len(ids), "skipped": True, "correct": False})
            by_family[task.family][1] += 1
            continue
        logits = logits[: len(assignment.letters)]
        correct, el_ok, el_n = score(task, assignment, logits)
        by_family[task.family][0] += correct
        by_family[task.family][1] += 1
        rows.append({
            "id": task.id, "family": task.family, "tokens": len(ids), "correct": correct,
            "elements_ok": el_ok, "elements": el_n, "logits": logits.tolist(),
        })
        if (n + 1) % 50 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"{n + 1}/{len(tasks)} acc={acc:.3f} median={np.median(times):.0f} ms", flush=True)

    summary = {
        "backend": args.backend,
        "variant": args.variant or ("fp16" if args.backend == "coreml" else "bf16-peft"),
        "compute_units": args.compute_units if args.backend == "coreml" else args.device,
        "tasks": len(rows),
        "skipped_too_long": sum(r.get("skipped", False) for r in rows),
        "task_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "by_family": {k: {"correct": v[0], "n": v[1], "acc": v[0] / v[1]} for k, v in sorted(by_family.items())},
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_tokens": max(r["tokens"] for r in rows),
    }
    print(json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}))


if __name__ == "__main__":
    main()
