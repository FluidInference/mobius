"""Core ML parity and latency against the unmodified PyTorch reference."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import coremltools as ct
import laya
import numpy as np
import torch
from laya.common import collate_items

from assets import ROOT, checkpoint_dir, load_lock, sha256, verify_assets
from preprocessing import Shape, encode, package_name, prepare_arrays

GATES = {"max_probability_error": 0.02, "max_action_probability_error": 0.02}


def compiled_path(package: Path) -> Path:
    compiled = package.with_suffix(".mlmodelc")
    if not compiled.exists():
        ct.utils.compile_model(str(package), destination_path=str(compiled))
    return compiled


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values), q))


def run(agent, package: Path, shape: Shape, units, cases: list[dict], warmup: int, repeats: int) -> dict:
    started = time.perf_counter()
    model = ct.models.CompiledMLModel(str(compiled_path(package)), compute_units=units)
    load_seconds = time.perf_counter() - started
    rows, timings = [], []
    worst_p = worst_a = 0.0
    agree = total = 0
    skipped = []
    for case in cases:
        for qid, question in case["questions"].items():
            try:
                ids, markers, qtype = encode(agent.tok, case["state"], question, shape, agent.cfg["head_max_len"])
            except ValueError as error:
                skipped.append({"case": case["name"], "question": qid, "reason": str(error)})
                continue
            b = collate_items([[{"ids": ids, "markers": markers, "qtype": qtype}]], agent.tok.pad_token_id)
            with torch.no_grad():
                ref_logits, ref_act = agent.model(
                    b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
                )
            k = len(markers)
            ref_p = torch.softmax(ref_logits[0, :k], -1).numpy()
            ref_a = torch.softmax(ref_act[0], -1).numpy()
            arrays = prepare_arrays(ids, markers, qtype, shape, agent.tok.pad_token_id)
            for _ in range(warmup):
                model.predict(arrays)
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = model.predict(arrays)
                samples.append((time.perf_counter() - t0) * 1000)
            p = out["probabilities"][0, :k]
            a = out["action_probabilities"][0]
            pad = out["logits"][0, k:]
            dp = float(np.abs(p - ref_p).max())
            da = float(np.abs(a - ref_a).max())
            worst_p, worst_a = max(worst_p, dp), max(worst_a, da)
            same = int(np.argmax(p)) == int(np.argmax(ref_p))
            agree += same
            total += 1
            timings.extend(samples)
            rows.append(
                {
                    "case": case["name"],
                    "question": qid,
                    "type": question["type"],
                    "tokens": len(ids),
                    "options": k,
                    "argmax_agrees": same,
                    "coreml_argmax": int(np.argmax(p)),
                    "reference_argmax": int(np.argmax(ref_p)),
                    "max_probability_error": dp,
                    "max_action_probability_error": da,
                    "finite": bool(np.isfinite(out["logits"]).all()),
                    "padding_masked": bool((pad <= -9999).all()) if k < shape.max_options else True,
                    "probability_sum": float(p.sum()),
                    "median_ms": float(np.median(samples)),
                }
            )
    report = {
        "compute_units": str(units).split(".")[-1],
        "load_seconds": load_seconds,
        "questions": total,
        "argmax_agreements": agree,
        "max_probability_error": worst_p,
        "max_action_probability_error": worst_a,
        "latency_ms": {
            "p50": percentile(timings, 50),
            "p95": percentile(timings, 95),
            "min": min(timings),
            "max": max(timings),
        },
        "skipped": skipped,
        "rows": rows,
    }
    report["passed"] = (
        agree == total
        and worst_p <= GATES["max_probability_error"]
        and worst_a <= GATES["max_action_probability_error"]
        and all(r["finite"] and r["padding_masked"] for r in rows)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="multilingual")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--precision", default="fp16", help="package precision tag, e.g. fp16, w8, w4")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--cases", type=Path, default=ROOT / "fixtures" / "cases.json")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    verify_assets(args.variant)
    lock = load_lock()
    agent = laya.load(str(checkpoint_dir(args.variant)), device="cpu")
    agent.model.eval()
    shape = Shape(args.length, args.max_options)
    package = (
        args.build_dir / f"{package_name(args.variant, shape.length, shape.max_options, args.precision)}.mlpackage"
    )
    cases = json.loads(args.cases.read_text())
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    report = {
        "package": package.name,
        "package_files_sha256": {
            str(p.relative_to(package)): sha256(p) for p in sorted(package.rglob("*")) if p.is_file()
        },
        "source_repo": lock["repo"],
        "source_revision": lock["revision"],
        "variant": args.variant,
        "length": shape.length,
        "max_options": shape.max_options,
        "precision": args.precision,
        "package_bytes": sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
        "environment": {
            "chip": chip,
            "os": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
        },
        "gates": {"argmax_agreement": "100%", **GATES, "finite": True, "padding_masked": True},
        "cases_sha256": sha256(args.cases),
        "runs": {},
    }
    for name, units in (("ALL", ct.ComputeUnit.ALL), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)):
        result = run(agent, package, shape, units, cases, args.warmup, args.repeats)
        report["runs"][name] = result
        print(
            f"{name:10s} questions={result['questions']} agree={result['argmax_agreements']} "
            f"max_dp={result['max_probability_error']:.5f} max_da={result['max_action_probability_error']:.5f} "
            f"p50={result['latency_ms']['p50']:.2f}ms p95={result['latency_ms']['p95']:.2f}ms "
            f"load={result['load_seconds']:.2f}s passed={result['passed']} skipped={len(result['skipped'])}"
        )
    report["passed"] = all(r["passed"] for r in report["runs"].values())
    suffix = "" if args.precision == "fp16" else f"-{args.precision}"
    out = args.report or (ROOT / "reports" / f"verification-{args.variant}-L{shape.length}{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
