#!/usr/bin/env python3
"""Compile Core ML packages into ``.mlmodelc`` bundles via ``xcrun``.

Finds all ``*.mlpackage`` bundles under ``./build`` and compiles each
with ``xcrun coremlcompiler`` into ``./compiled``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BUILD_DIR = BASE_DIR / "build"
OUTPUT_ROOT = BASE_DIR / "compiled"


def ensure_coremlcompiler() -> None:
    """Ensure ``xcrun coremlcompiler`` is available for the active Xcode."""
    xcrun_path = shutil.which("xcrun")
    if xcrun_path is None:
        print("Error: 'xcrun' not found on PATH. Install Xcode command line tools.", file=sys.stderr)
        sys.exit(1)

    try:
        subprocess.run([
            xcrun_path,
            "--find",
            "coremlcompiler",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Error: 'coremlcompiler' not found via xcrun. Check your Xcode installation.", file=sys.stderr)
        sys.exit(1)


def gather_packages() -> list[Path]:
    """Return a list of all ``*.mlpackage`` bundles under the build dir."""
    if not BUILD_DIR.exists():
        print(f"Warning: {BUILD_DIR.relative_to(BASE_DIR)} does not exist; run convert-to-coreml.py first.", file=sys.stderr)
        return []
    return list(BUILD_DIR.rglob("*.mlpackage"))


def compile_package(package: Path) -> None:
    """Compile a single ``.mlpackage`` bundle using ``xcrun coremlcompiler``."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{package.stem}.mlmodelc"

    if output_path.exists():
        shutil.rmtree(output_path)

    cmd = [
        "xcrun",
        "coremlcompiler",
        "compile",
        str(package),
        str(OUTPUT_ROOT),
    ]

    print(f"Compiling {package.relative_to(BASE_DIR)} -> compiled/{package.stem}.mlmodelc")
    subprocess.run(cmd, check=True)


def main() -> None:
    ensure_coremlcompiler()
    packages = gather_packages()

    if not packages:
        print("No .mlpackage bundles found to compile.")
        return

    for package in packages:
        try:
            compile_package(package)
        except subprocess.CalledProcessError as exc:
            print(f"Failed to compile {package}: {exc}", file=sys.stderr)
            sys.exit(exc.returncode)

    print(f"Finished compiling {len(packages)} package(s) into {OUTPUT_ROOT.relative_to(BASE_DIR)}/.")


if __name__ == "__main__":
    main()
