#!/usr/bin/env python3
"""PocketTTS ANE-residency + RTFx benchmark/verification harness.

Answers "did we get more on the ANE?" and "did RTFx improve?" in one shot, on
an Apple Silicon box, by driving `coreml-cli` (device split + fallback reasons)
and optionally the FluidAudio `fluidaudiocli tts` CLI (end-to-end RTFx A/B).

Two parts, run independently or together:

  PART A — ANE residency (coreml-cli):
    For every PocketTTS mlpackage/mlmodelc in a build dir, report the CPU/GPU/ANE
    op split under each compute-unit config, plus the top CPU-fallback reasons.
    With --baseline-dir, prints a before→after ANE% diff per model. This is the
    direct answer to "more on ANE": compare flow_decoder vs flow_decoder_fused
    and flowlm_step ANE% under the `ALL` config.

  PART B — RTFx A/B (optional, --tts-cmd-a / --tts-cmd-b):
    Runs two TTS commands interleaved (A,B,A,B,...) with warmup, parses the
    `Audio duration` and `Generated N frames in Xs` log lines, computes
    RTFx = audio_seconds / generate_seconds, and reports mean±std. Interleaving
    + warmup follow the Trial 15 methodology (cancels thermal/cache drift).

Examples:
    # ANE residency of the new build, vs the old build:
    python bench_ane_rtfx.py ane --build-dir build/english \\
        --baseline-dir build/english_baseline

    # RTFx A/B: baseline models dir vs new models dir, via env-swap on the CLI.
    python bench_ane_rtfx.py rtfx \\
        --tts-cmd-a 'env POCKET_TTS_DIR=$HOME/.cache/pt_old swift run -c release fluidaudiocli tts "the quick brown fox jumps over the lazy dog" --output /tmp/a.wav' \\
        --tts-cmd-b 'env POCKET_TTS_DIR=$HOME/.cache/pt_new swift run -c release fluidaudiocli tts "the quick brown fox jumps over the lazy dog" --output /tmp/b.wav' \\
        --iters 5 --warmup 2

    # Everything:
    python bench_ane_rtfx.py all --build-dir build/english --baseline-dir build/english_baseline \\
        --tts-cmd-a '...' --tts-cmd-b '...'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # .../pocket_tts/coreml
_DEFAULT_COREML_CLI_PROJECT = (_HERE / "../../../../tools/coreml-cli").resolve()

# Model artifacts and their before/after pairing. `baseline` is the old artifact
# the change replaces (None when there's no separate old file). `expect_ane`
# documents the intended end state so the report can flag surprises.
MODEL_SPECS = [
    # key,            new_basename,            baseline_basename,   expect_ane
    ("flow_decoder",  "flow_decoder_fused",    "flow_decoder",      True),
    ("flowlm_step",   "flowlm_step",           "flowlm_step",       True),
    ("flowlm_stepv2", "flowlm_stepv2",         "flowlm_stepv2",     True),
    ("cond",          "cond_prefill",          "cond_step",         False),  # rank-5 KV → CPU/GPU
    ("mimi_decoder",  "mimi_decoder",          "mimi_decoder",      False),  # streaming → cpuOnly
]

# Compute-unit config name (as emitted by coreml-cli) we treat as the headline
# "what CoreML places on ANE when allowed to" number.
HEADLINE_CU = "ALL"


# ----------------------------------------------------------------------------- coreml-cli plumbing
def _resolve_model_path(build_dir: Path, basename: str) -> Path | None:
    """Prefer compiled .mlmodelc, fall back to .mlpackage."""
    for ext in (".mlmodelc", ".mlpackage"):
        p = build_dir / f"{basename}{ext}"
        if p.exists():
            return p
    return None


def _run_coreml_cli(cli_project: Path, model_path: Path, fallback: bool, debug: bool) -> dict | None:
    cmd = [
        "uv", "run", "--project", str(cli_project),
        "coreml-cli", str(model_path), "--json",
    ]
    if fallback:
        cmd.append("--fallback")
    env = dict(os.environ)
    env.setdefault("OS_ACTIVITY_DT_MODE", "disable")
    if debug:
        print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"  ! coreml-cli timed out on {model_path.name}", file=sys.stderr)
        return None
    if out.returncode != 0:
        print(f"  ! coreml-cli failed on {model_path.name} (exit {out.returncode})", file=sys.stderr)
        if debug:
            print(out.stderr, file=sys.stderr)
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        print(f"  ! could not parse coreml-cli JSON for {model_path.name}", file=sys.stderr)
        if debug:
            print(out.stdout[:2000], file=sys.stderr)
        return None


def _headline_split(bench: dict) -> dict | None:
    """Pull the CPU/GPU/ANE split for the headline compute-unit config."""
    try:
        model = bench["models"][0]
    except (KeyError, IndexError):
        return None
    results = model.get("results", [])
    chosen = None
    for r in results:
        if r.get("compute_units") == HEADLINE_CU:
            chosen = r
            break
    if chosen is None and results:
        chosen = results[0]  # fall back to whatever config ran
    if chosen is None:
        return None
    s = chosen.get("summary", {})
    lat = chosen.get("latency", {})
    return {
        "compute_units": chosen.get("compute_units", "?"),
        "cpu": s.get("cpu_percent"),
        "gpu": s.get("gpu_percent"),
        "ane": s.get("ane_percent"),
        "predict_ms": lat.get("median_ms"),
    }


def _fallback_summary(fb_data: dict) -> dict | None:
    try:
        fb = fb_data["models"][0]["fallback"]
    except (KeyError, IndexError):
        return None
    reasons = [
        {
            "reason": r.get("reason", "?"),
            "count": r.get("count", 0),
            "cpu_ms": r.get("estimated_cpu_runtime_ms", 0.0),
        }
        for r in fb.get("reasons", [])
    ]
    reasons.sort(key=lambda r: r["cpu_ms"], reverse=True)
    return {
        "ane_ops": fb.get("ane_ops"),
        "total_ops": fb.get("total_ops"),
        "cpu_ops": fb.get("cpu_ops"),
        "ane_percent": fb.get("ane_percent"),
        "reasons": reasons,
    }


def _fmt_pct(v) -> str:
    return f"{v:5.1f}%" if isinstance(v, (int, float)) else "   ?  "


def cmd_ane(args) -> int:
    cli_project = Path(args.coreml_cli_project).resolve()
    build_dir = Path(args.build_dir).resolve()
    baseline_dir = Path(args.baseline_dir).resolve() if args.baseline_dir else None

    if not build_dir.is_dir():
        print(f"build-dir not found: {build_dir}", file=sys.stderr)
        return 2
    if not cli_project.is_dir():
        print(f"coreml-cli project not found: {cli_project}", file=sys.stderr)
        return 2

    print(f"Device profiling via coreml-cli @ {cli_project}")
    print(f"Build dir:    {build_dir}")
    if baseline_dir:
        print(f"Baseline dir: {baseline_dir}")
    print()
    header = f"{'model':<20s} {'config':<12s} {'CPU':>6s} {'GPU':>6s} {'ANE':>6s} {'predict':>9s}  {'ANE ops':>9s}"
    if baseline_dir:
        header += f"   {'Δ ANE%':>8s}"
    print(header)
    print("─" * len(header))

    fallback_blocks: list[str] = []
    regressions: list[str] = []

    for key, new_base, base_base, expect_ane in MODEL_SPECS:
        new_path = _resolve_model_path(build_dir, new_base)
        if new_path is None:
            print(f"{key:<20s} (not found: {new_base}.mlmodelc/.mlpackage)")
            continue

        bench = _run_coreml_cli(cli_project, new_path, fallback=False, debug=args.debug)
        split = _headline_split(bench) if bench else None
        fb = _run_coreml_cli(cli_project, new_path, fallback=True, debug=args.debug)
        fbs = _fallback_summary(fb) if fb else None

        base_ane = None
        if baseline_dir:
            base_path = _resolve_model_path(baseline_dir, base_base)
            if base_path is not None:
                bbench = _run_coreml_cli(cli_project, base_path, fallback=False, debug=args.debug)
                bsplit = _headline_split(bbench) if bbench else None
                base_ane = bsplit["ane"] if bsplit else None

        if split is None:
            print(f"{new_path.name:<20s} (profiling failed)")
            continue

        ane_ops_str = (
            f"{fbs['ane_ops']}/{fbs['total_ops']}" if fbs and fbs.get("total_ops") else "  ?  "
        )
        pm = split["predict_ms"]
        predict_str = f"{pm:7.2f}ms" if pm is not None else "n/a"
        line = (
            f"{new_path.stem:<20s} {split['compute_units']:<12s} "
            f"{_fmt_pct(split['cpu'])} {_fmt_pct(split['gpu'])} {_fmt_pct(split['ane'])} "
            f"{predict_str:>9s} "
            f"{ane_ops_str:>9s}"
        )
        if baseline_dir:
            if isinstance(split["ane"], (int, float)) and isinstance(base_ane, (int, float)):
                delta = split["ane"] - base_ane
                line += f"   {delta:+7.1f}%"
            else:
                line += f"   {'  n/a':>8s}"
        print(line)

        # Flag a model whose intended ANE state didn't materialize.
        ane_val = split["ane"] if isinstance(split["ane"], (int, float)) else 0.0
        if expect_ane and ane_val < 50.0:
            regressions.append(
                f"  · {new_path.stem}: expected mostly-ANE, got {ane_val:.1f}% — "
                f"check fallback reasons below"
            )

        # Stash fallback detail for models we wanted on ANE.
        if expect_ane and fbs and fbs["reasons"]:
            block = [f"\n── {new_path.stem}: {fbs['ane_ops']}/{fbs['total_ops']} ops on ANE "
                     f"({fbs['ane_percent']}%), {fbs['cpu_ops']} on CPU"]
            for r in fbs["reasons"][:6]:
                ms = f" (~{r['cpu_ms']:.2f}ms CPU)" if r["cpu_ms"] > 0.001 else ""
                block.append(f"   {r['count']:>3d}× {r['reason']}{ms}")
            fallback_blocks.append("\n".join(block))

    if fallback_blocks:
        print("\nCPU-fallback reasons (models targeted for ANE):")
        for b in fallback_blocks:
            print(b)

    print()
    if regressions:
        print("⚠️  ANE placement did NOT meet expectations:")
        print("\n".join(regressions))
        print("   → these are still CPU/GPU; the ANE win is unconfirmed for them.")
    else:
        print("✓ All ANE-targeted models cleared the 50% ANE bar at the ALL config.")
    print("\nReminder: cond_* and mimi_decoder are expected at 0% ANE by design")
    print("(rank-5 KV cache and fp16 streaming feedback respectively).")
    return 0


# ----------------------------------------------------------------------------- RTFx A/B
_DURATION_RE = re.compile(r"Audio duration:\s*([0-9.]+)\s*s", re.IGNORECASE)
_GEN_RE = re.compile(r"Generated\s+\d+\s+frames in\s*([0-9.]+)\s*s", re.IGNORECASE)


def _run_tts_once(cmd: str, debug: bool) -> float | None:
    """Run a TTS command, return RTFx = audio_seconds / generate_seconds."""
    env = dict(os.environ)
    env.setdefault("OS_ACTIVITY_DT_MODE", "disable")
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=1200)
    except subprocess.TimeoutExpired:
        print("  ! tts command timed out", file=sys.stderr)
        return None
    blob = out.stdout + "\n" + out.stderr
    if debug:
        print(blob[-1500:], file=sys.stderr)
    dur = _DURATION_RE.search(blob)
    gen = _GEN_RE.search(blob)
    if not dur or not gen:
        print("  ! could not parse 'Audio duration' / 'Generated ... frames in' from output",
              file=sys.stderr)
        if not debug:
            print("    (re-run with --debug to see the CLI output)", file=sys.stderr)
        return None
    audio_s = float(dur.group(1))
    gen_s = float(gen.group(1))
    if gen_s <= 0:
        return None
    return audio_s / gen_s


def _report_series(label: str, rtfx: list[float]) -> None:
    if not rtfx:
        print(f"  {label:<10s} no successful runs")
        return
    mean = statistics.mean(rtfx)
    sd = statistics.pstdev(rtfx) if len(rtfx) > 1 else 0.0
    vals = ", ".join(f"{v:.2f}" for v in rtfx)
    print(f"  {label:<10s} RTFx {mean:.2f} ± {sd:.2f}  (n={len(rtfx)})   [{vals}]")


def cmd_rtfx(args) -> int:
    if not args.tts_cmd_a or not args.tts_cmd_b:
        print("rtfx needs both --tts-cmd-a and --tts-cmd-b", file=sys.stderr)
        return 2

    print(f"RTFx A/B — interleaved, warmup={args.warmup}, iters={args.iters}")
    print(f"  A: {args.tts_cmd_a}")
    print(f"  B: {args.tts_cmd_b}\n")

    for w in range(args.warmup):
        print(f"  warmup {w + 1}/{args.warmup} ...", file=sys.stderr)
        _run_tts_once(args.tts_cmd_a, args.debug)
        _run_tts_once(args.tts_cmd_b, args.debug)

    a_vals: list[float] = []
    b_vals: list[float] = []
    for i in range(args.iters):
        print(f"  iter {i + 1}/{args.iters} ...", file=sys.stderr)
        a = _run_tts_once(args.tts_cmd_a, args.debug)  # interleave A then B
        b = _run_tts_once(args.tts_cmd_b, args.debug)
        if a is not None:
            a_vals.append(a)
        if b is not None:
            b_vals.append(b)

    print("\nResults (RTFx = audio_seconds / generate_seconds; higher is faster):")
    _report_series("A (base)", a_vals)
    _report_series("B (new)", b_vals)
    if a_vals and b_vals:
        ratio = statistics.mean(b_vals) / statistics.mean(a_vals)
        print(f"\n  B/A speedup: {ratio:.2f}×  ({(ratio - 1) * 100:+.1f}% RTFx)")
        print("  (drop the highest+lowest per side if variance is high — see Trial 15.)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_ane_args(p):
        p.add_argument("--build-dir", required=True, help="Dir with the NEW mlpackages/mlmodelc")
        p.add_argument("--baseline-dir", default=None, help="Dir with the OLD artifacts for a Δ ANE%% diff")
        p.add_argument("--coreml-cli-project", default=str(_DEFAULT_COREML_CLI_PROJECT),
                       help="Path to the coreml-cli uv project")

    def add_rtfx_args(p):
        p.add_argument("--tts-cmd-a", default=None, help="Baseline TTS shell command")
        p.add_argument("--tts-cmd-b", default=None, help="New-pipeline TTS shell command")
        p.add_argument("--iters", type=int, default=5)
        p.add_argument("--warmup", type=int, default=2)

    p_ane = sub.add_parser("ane", help="ANE residency + fallback reasons")
    add_ane_args(p_ane)
    p_ane.add_argument("--debug", action="store_true")

    p_rtfx = sub.add_parser("rtfx", help="End-to-end RTFx A/B")
    add_rtfx_args(p_rtfx)
    p_rtfx.add_argument("--debug", action="store_true")

    p_all = sub.add_parser("all", help="ane then rtfx")
    add_ane_args(p_all)
    add_rtfx_args(p_all)
    p_all.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.mode == "ane":
        return cmd_ane(args)
    if args.mode == "rtfx":
        return cmd_rtfx(args)
    if args.mode == "all":
        rc = cmd_ane(args)
        print("\n" + "=" * 72 + "\n")
        rc2 = cmd_rtfx(args) if (args.tts_cmd_a and args.tts_cmd_b) else 0
        return rc or rc2
    return 1


if __name__ == "__main__":
    sys.exit(main())
