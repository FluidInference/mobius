"""Convert each diagnostic nn.Module in modules.py to .mlmodelc and probe ANE %.

Pipeline per spec:
    build module -> torch.jit.trace -> ct.convert (mlprogram, fp16, iOS17)
        -> save .mlpackage -> xcrun coremlcompiler compile -> .mlmodelc
        -> coreml-cli --fallback --json (CPU+ANE config) -> record

Results are aggregated to results/ledger.json with per-spec ANE %, total ops,
CPU ops, top rejection reasons, and the path to the raw fallback JSON.

Run:
    uv run python per_module/analyze.py
    uv run python per_module/analyze.py --only snake_learned snake_poly_taylor
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import coremltools as ct

from per_module import modules as mods


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE / "build" / "packages"
MLC_DIR = HERE / "build" / "compiled"
RESULTS_DIR = HERE / "results"
RAW_DIR = RESULTS_DIR / "raw"
LEDGER_PATH = RESULTS_DIR / "ledger.json"

COREML_CLI_DIR = HERE.parent.parent.parent.parent.parent / "tools" / "coreml-cli"


@dataclass
class Spec:
    """Single diagnostic specification.

    name:       short id, used for output filenames + ledger keys
    builder:    callable -> nn.Module (already in eval mode)
    inputs:     list of (name, torch.Tensor) used both as trace example and ct input
    input_dtypes: per-input numpy dtype passed to ct.TensorType
    output_names: list of CoreML output tensor names
    notes:      one-line human description for the report
    """
    name: str
    builder: Callable[[], torch.nn.Module]
    inputs: list[tuple[str, torch.Tensor]]
    input_dtypes: list[Any]
    output_names: list[str]
    notes: str = ""


def _f32(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32)


def _zeros(*shape: int) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.float32)


def _pos(p: float = 0.0) -> torch.Tensor:
    return torch.tensor([p], dtype=torch.float32)


def build_specs() -> list[Spec]:
    """Production shapes for the magpie 357m model."""
    # codec activations: ~256 channels, ~256 frames is representative
    snake_C = 256
    snake_T = 256

    # decoder_step shapes (12-layer, 768-d, 12-head, max_seq=512, d_head=64)
    d_model = 768
    n_heads = 12
    d_head = 64
    max_seq = 512

    # weight_norm conv: representative codec block
    wn_C = 256
    wn_K = 7
    wn_T = 256

    specs: list[Spec] = []

    # ---- Snake variants ----
    for cls, sub in [
        (mods.SnakeLearned,    "snake_learned"),
        (mods.SnakePolyTaylor, "snake_poly_taylor"),
        (mods.SnakeNoSinPow,   "snake_no_sin_pow"),
    ]:
        specs.append(Spec(
            name=sub,
            builder=lambda c=cls: c(snake_C).eval(),
            inputs=[("x", _f32(1, snake_C, snake_T))],
            input_dtypes=[np.float32],
            output_names=["y"],
            notes=f"{cls.__name__} (B=1,C={snake_C},T={snake_T})",
        ))

    # ---- KV cache write variants ----
    # rank-4: cache (1, max_seq, H, D); new (1, 1, H, D); pos (1,)
    specs.append(Spec(
        name="kv_write_rank4_onehot",
        builder=lambda: mods.KVCacheWriteRank4OneHot(max_seq).eval(),
        inputs=[
            ("kv_k",     _zeros(1, max_seq, n_heads, d_head)),
            ("kv_v",     _zeros(1, max_seq, n_heads, d_head)),
            ("k_new",    _f32(1, 1, n_heads, d_head)),
            ("v_new",    _f32(1, 1, n_heads, d_head)),
            ("position", _pos(0.0)),
        ],
        input_dtypes=[np.float32] * 5,
        output_names=["new_k", "new_v"],
        notes=f"rank-4 one-hot blend (max_seq={max_seq}, H={n_heads}, D={d_head})",
    ))

    # rank-3: collapse H*D
    specs.append(Spec(
        name="kv_write_rank3_onehot",
        builder=lambda: mods.KVCacheWriteRank3OneHot(max_seq).eval(),
        inputs=[
            ("kv_k",     _zeros(max_seq, n_heads * d_head)),
            ("kv_v",     _zeros(max_seq, n_heads * d_head)),
            ("k_new",    _f32(1, n_heads * d_head)),
            ("v_new",    _f32(1, n_heads * d_head)),
            ("position", _pos(0.0)),
        ],
        input_dtypes=[np.float32] * 5,
        output_names=["new_k", "new_v"],
        notes=f"rank-3 one-hot blend (max_seq={max_seq}, C={n_heads*d_head})",
    ))

    # host-concat: just emit k_new/v_new
    specs.append(Spec(
        name="kv_write_host_concat",
        builder=lambda: mods.KVCacheWriteHostConcat(d_model, n_heads, d_head).eval(),
        inputs=[("x", _f32(1, 1, d_model))],
        input_dtypes=[np.float32],
        output_names=["k_new", "v_new"],
        notes=f"host-concat: emit (k_new,v_new) only (d_model={d_model})",
    ))

    # ---- Full attention subgraph at decoder_step shape ----
    specs.append(Spec(
        name="causal_self_attn_rank4_cache",
        builder=lambda: mods.CausalSelfAttnRank4Cache(d_model, n_heads, max_seq).eval(),
        inputs=[
            ("x",        _f32(1, 1, d_model)),
            ("kv_k",     _zeros(1, max_seq, n_heads, d_head)),
            ("kv_v",     _zeros(1, max_seq, n_heads, d_head)),
            ("position", _pos(0.0)),
        ],
        input_dtypes=[np.float32] * 4,
        output_names=["out", "new_k", "new_v", "new_position"],
        notes="full attn: qkv-proj + onehot KV write + causal softmax + o-proj",
    ))

    # ---- weight_norm Conv1d ----
    # ---- Wrapped Snake variants (Conv1d -> Snake -> Conv1d) ----
    for cls, sub in [
        (mods.SnakeLearned,        "snake_learned_block"),
        (mods.SnakePolyTaylor,     "snake_poly_taylor_block"),
        (mods.SnakeTaylor5,        "snake_taylor5_block"),
        (mods.SnakeTaylor5Clipped, "snake_taylor5_clipped_block"),
        (mods.SnakeTaylor7,        "snake_taylor7_block"),
        (mods.SnakeNoSinPow,       "snake_no_sin_pow_block"),
    ]:
        specs.append(Spec(
            name=sub,
            builder=lambda c=cls: mods.SnakeBlock(c, snake_C, kernel=3).eval(),
            inputs=[("x", _f32(1, snake_C, snake_T))],
            input_dtypes=[np.float32],
            output_names=["y"],
            notes=f"Conv1d -> {cls.__name__} -> Conv1d (C={snake_C},T={snake_T})",
        ))

    # ---- Wrapped KV-write rank4 (linear -> kv-write -> attn -> linear) ----
    specs.append(Spec(
        name="kv_attn_rank4_block",
        builder=lambda: mods.KVAttnBlock(
            mods.KVCacheWriteRank4OneHot, d_model, n_heads, max_seq).eval(),
        inputs=[
            ("x",        _f32(1, 1, d_model)),
            ("kv_k",     _zeros(1, max_seq, n_heads, d_head)),
            ("kv_v",     _zeros(1, max_seq, n_heads, d_head)),
            ("position", _pos(0.0)),
        ],
        input_dtypes=[np.float32] * 4,
        output_names=["out", "new_k", "new_v"],
        notes="qkv-proj -> rank4-onehot KV-write -> causal attn -> o-proj",
    ))

    specs.append(Spec(
        name="wn_conv1d_unfolded",
        builder=lambda: mods.WeightNormConv1dUnfolded(wn_C, wn_C, wn_K).eval(),
        inputs=[("x", _f32(1, wn_C, wn_T))],
        input_dtypes=[np.float32],
        output_names=["y"],
        notes=f"weight_norm parametrization left in place (C={wn_C}, K={wn_K})",
    ))
    specs.append(Spec(
        name="wn_conv1d_folded",
        builder=lambda: mods.WeightNormConv1dFolded(wn_C, wn_C, wn_K).eval(),
        inputs=[("x", _f32(1, wn_C, wn_T))],
        input_dtypes=[np.float32],
        output_names=["y"],
        notes=f"weight_norm folded via remove_weight_norm() (C={wn_C}, K={wn_K})",
    ))

    return specs


def trace_and_convert(spec: Spec, pkg_path: Path) -> None:
    """torch.jit.trace -> ct.convert -> save mlpackage."""
    model = spec.builder()
    example_inputs = tuple(t for _, t in spec.inputs)

    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs, strict=False)

    ct_inputs = [
        ct.TensorType(name=name, shape=tuple(t.shape), dtype=dtype)
        for (name, t), dtype in zip(spec.inputs, spec.input_dtypes)
    ]
    ct_outputs = [ct.TensorType(name=n) for n in spec.output_names]

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    if pkg_path.exists():
        shutil.rmtree(pkg_path)
    pkg_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(pkg_path))


def compile_to_mlmodelc(pkg_path: Path, mlc_dir: Path) -> Path:
    """xcrun coremlcompiler compile <pkg> <out_dir>. Returns the .mlmodelc path."""
    if mlc_dir.exists():
        shutil.rmtree(mlc_dir)
    mlc_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["xcrun", "coremlcompiler", "compile", str(pkg_path), str(mlc_dir)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = mlc_dir / f"{pkg_path.stem}.mlmodelc"
    if not out.exists():
        raise RuntimeError(f"compile produced no mlmodelc at {out}")
    return out


def run_fallback_json(mlmodelc: Path, plan_timeout: float = 600.0) -> dict[str, Any]:
    """Run mobius coreml-cli --fallback --json. Returns the parsed JSON dict."""
    cmd = [
        "uv", "run", "coreml-cli",
        str(mlmodelc),
        "--fallback",
        "--json",
        "--plan-timeout", str(plan_timeout),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(COREML_CLI_DIR),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"coreml-cli failed for {mlmodelc.name}\n"
            f"stdout: {proc.stdout.decode(errors='ignore')}\n"
            f"stderr: {proc.stderr.decode(errors='ignore')}"
        )
    text = proc.stdout.decode(errors="ignore").strip()
    return json.loads(text)


def summarize(fb: dict[str, Any]) -> dict[str, Any]:
    """Pull the headline numbers out of coreml-cli --fallback --json output.

    Top-level shape: { hardware: {...}, models: [ { fallback: {...} } ] }
    fallback shape: { total_ops, ane_ops, gpu_ops, cpu_ops, ane_percent, reasons: [{reason,count,...}] }
    """
    models = fb.get("models") or []
    fallback = (models[0] if models else {}).get("fallback") or {}

    total = fallback.get("total_ops")
    ane_ops = fallback.get("ane_ops")
    cpu_ops = fallback.get("cpu_ops")
    gpu_ops = fallback.get("gpu_ops")
    ane_pct = fallback.get("ane_percent")
    if ane_pct is None and total:
        ane_pct = round(100.0 * (ane_ops or 0) / total, 2)

    reasons = fallback.get("reasons") or []
    reasons_top = [
        (r.get("reason"), r.get("count"), r.get("op_types"))
        for r in reasons[:5]
    ]

    return {
        "ane_percent": ane_pct,
        "total_ops": total,
        "ane_ops": ane_ops,
        "gpu_ops": gpu_ops,
        "cpu_ops": cpu_ops,
        "top_rejections": reasons_top,
    }


def run_one(spec: Spec) -> dict[str, Any]:
    pkg = PKG_DIR / f"{spec.name}.mlpackage"
    mlc_root = MLC_DIR / spec.name
    raw_path = RAW_DIR / f"{spec.name}.json"

    print(f"\n[{spec.name}] {spec.notes}")
    t0 = time.time()
    trace_and_convert(spec, pkg)
    t1 = time.time()
    mlmodelc = compile_to_mlmodelc(pkg, mlc_root)
    t2 = time.time()
    fb = run_fallback_json(mlmodelc)
    t3 = time.time()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(fb, indent=2))

    summary = summarize(fb)
    summary.update({
        "name": spec.name,
        "notes": spec.notes,
        "mlmodelc": str(mlmodelc.relative_to(HERE)),
        "raw": str(raw_path.relative_to(HERE)),
        "timings_s": {
            "convert": round(t1 - t0, 2),
            "compile": round(t2 - t1, 2),
            "fallback": round(t3 - t2, 2),
        },
    })
    pct = summary["ane_percent"]
    print(f"  -> ANE {pct}%  cpu_ops={summary['cpu_ops']}/{summary['total_ops']}  "
          f"convert={summary['timings_s']['convert']}s "
          f"compile={summary['timings_s']['compile']}s "
          f"fallback={summary['timings_s']['fallback']}s")
    if summary["top_rejections"]:
        for reason, n, op_types in summary["top_rejections"]:
            ops_short = ", ".join(f"{k}:{v}" for k, v in (op_types or {}).items())
            print(f"     - {n} x {reason}  [{ops_short}]")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only the listed spec names")
    parser.add_argument("--skip", nargs="*", default=None,
                        help="Skip the listed spec names")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_specs()
    if args.only:
        specs = [s for s in specs if s.name in args.only]
    if args.skip:
        specs = [s for s in specs if s.name not in args.skip]

    if not specs:
        print("No specs selected.", file=sys.stderr)
        return 1

    ledger: dict[str, Any] = {"runs": []}
    if LEDGER_PATH.exists():
        try:
            ledger = json.loads(LEDGER_PATH.read_text())
        except Exception:
            ledger = {"runs": []}
    runs_by_name = {r.get("name"): r for r in ledger.get("runs", [])}

    for spec in specs:
        try:
            summary = run_one(spec)
            runs_by_name[spec.name] = summary
        except Exception as e:
            print(f"[{spec.name}] FAILED: {e}", file=sys.stderr)
            runs_by_name[spec.name] = {
                "name": spec.name,
                "notes": spec.notes,
                "error": str(e),
            }

    ledger["runs"] = list(runs_by_name.values())
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2))
    print(f"\nWrote {LEDGER_PATH}")

    print("\n=== ANE summary ===")
    fmt = "{:32s}  {:>7s}  {:>10s}  {}"
    print(fmt.format("name", "ANE %", "cpu/total", "notes"))
    for r in ledger["runs"]:
        if "error" in r:
            print(fmt.format(r["name"], "ERR", "-", r.get("error", "")[:60]))
            continue
        pct = r.get("ane_percent")
        cpu = r.get("cpu_ops")
        total = r.get("total_ops")
        print(fmt.format(
            r["name"],
            f"{pct}" if pct is not None else "?",
            f"{cpu}/{total}" if cpu is not None else "?",
            r.get("notes", ""),
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
