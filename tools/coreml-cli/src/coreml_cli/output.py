"""Output formatting."""

from __future__ import annotations

import json
import os
import sys
from typing import Any


def emit_json(data: dict[str, Any]) -> None:
    json.dump(data, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except (ValueError, OSError):
        return 80


def _format_io_item(item: dict) -> str:
    t = item["type"]
    t = t.replace("MultiArray (", "").rstrip(")")
    t = t.replace(" × ", "×")
    return f"{item['name']}({t})"


def _wrap_items(label: str, items: list[str], width: int) -> str:
    """Wrap comma-separated items with aligned continuation lines."""
    prefix = f"  {label} "
    indent = " " * len(prefix)

    lines: list[str] = []
    current = prefix
    for i, item in enumerate(items):
        addition = f", {item}" if i > 0 else item
        if i > 0 and len(current) + len(addition) > width:
            lines.append(current + ",")
            current = indent + item
        else:
            current += addition
    lines.append(current)
    return "\n".join(lines)


def emit_table(data: dict[str, Any]) -> None:
    width = _terminal_width()

    hw = data.get("hardware", {})
    if hw:
        print(f"Device:    {hw.get('chip', '?')} ({hw.get('device', '?')})")
        print(f"RAM:       {hw.get('ram', '?')}")
        print(f"OS:        {hw.get('os_version', '?')}")
        print(f"Timestamp: {hw.get('timestamp', '?')}")

    for m in data["models"]:
        meta = m.get("metadata", {})
        name = m["model_name"]

        # ── model_name ────────────────────
        rule = "─" * max(width - len(name) - 5, 10)
        print(f"\n── {name} {rule}")

        # Description
        desc = meta.get("description", "")
        if desc:
            author = meta.get("author", "")
            line = desc
            if author:
                line += f" ({author})"
            print(f"  {line}")

        # Compact info line
        info_parts = []
        if meta.get("compute_precision"):
            info_parts.append(meta["compute_precision"])
        if meta.get("source"):
            info_parts.append(meta["source"])
        if meta.get("coremltools_version"):
            info_parts.append(f"coremltools {meta['coremltools_version']}")
        if info_parts:
            print(f"  {' | '.join(info_parts)}")

        # I/O with wrapping
        if meta.get("inputs"):
            items = [_format_io_item(i) for i in meta["inputs"]]
            print(_wrap_items("inputs: ", items, width))
        if meta.get("outputs"):
            items = [_format_io_item(i) for i in meta["outputs"]]
            print(_wrap_items("outputs:", items, width))

        # Cold compile (once per model)
        cold_ms = m.get("cold_compile_ms")
        if cold_ms is not None:
            print(f"  cold compile: {cold_ms:.0f}ms")

        # Benchmark table
        print()
        print(f"  {'Compute Unit':<25s} {'CPU':>6s} {'GPU':>6s} {'ANE':>6s} {'Compile':>9s} {'Predict':>9s}")
        print(f"  {'─' * 64}")
        for r in m["results"]:
            s = r["summary"]
            lat = r.get("latency", {})
            compile_str = f"{lat['compile_ms']:>7.0f}ms" if "compile_ms" in lat else f"{'err':>9s}"
            if "median_ms" in lat:
                predict_str = f"{lat['median_ms']:>7.2f}ms"
            elif "error" in lat:
                predict_str = f"{'err':>9s}"
            else:
                predict_str = f"{'n/a':>9s}"
            print(
                f"  {r['compute_units']:<25s} "
                f"{s['cpu_percent']:>5.1f}% "
                f"{s['gpu_percent']:>5.1f}% "
                f"{s['ane_percent']:>5.1f}% "
                f"{compile_str} "
                f"{predict_str}"
            )


def emit_fallback_table(data: dict[str, Any]) -> None:
    width = _terminal_width()

    hw = data.get("hardware", {})
    if hw:
        print(f"Device:  {hw.get('chip', '?')} ({hw.get('device', '?')})")
        print(f"OS:      {hw.get('os_version', '?')}")

    for m in data["models"]:
        name = m["model_name"]
        fb = m["fallback"]

        rule = "─" * max(width - len(name) - 5, 10)
        print(f"\n── {name} {rule}")

        total = fb["total_ops"]
        ane = fb["ane_ops"]
        cpu = fb["cpu_ops"]
        print(f"  {ane}/{total} ops on ANE ({fb['ane_percent']}%), {cpu} on CPU")

        if not fb["reasons"]:
            print(f"  No CPU fallback ops.")
            continue

        print()
        for r in fb["reasons"]:
            # Header: reason + count
            type_summary = ", ".join(
                f"{t}×{c}" for t, c in r["op_types"].items()
            )
            print(f"  {r['reason']}")
            print(f"    {r['count']} ops: {type_summary}")
            if r["estimated_cpu_runtime_ms"] > 0.001:
                print(f"    est. CPU cost: {r['estimated_cpu_runtime_ms']:.3f}ms")
            # List op names compactly
            ops_str = ", ".join(r["ops"])
            if len(ops_str) > width - 6:
                ops_str = ops_str[: width - 9] + "..."
            print(f"    [{ops_str}]")
            print()
    print()
