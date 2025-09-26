#!/usr/bin/env python3
"""Compile Core ML packages into ``.mlmodelc`` bundles via ``xcrun``.

This script walks through the default Parakeet CoreML directories, finds
all ``*.mlpackage`` bundles, and compiles each of them with
``xcrun coremlcompiler`` into ``./compiled`` while preserving the
relative directory structure.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "compiled"
SOURCE_DIRS = [BASE_DIR / "parakeet_coreml", BASE_DIR / "parakeet_coreml_quantized"]


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
    """Return a list of all ``*.mlpackage`` bundles under the source dirs."""
    packages: list[Path] = []
    for source in SOURCE_DIRS:
        if not source.exists():
            print(f"Warning: {source.relative_to(BASE_DIR)} does not exist; skipping", file=sys.stderr)
            continue
        packages.extend(source.rglob("*.mlpackage"))
    return packages


def compile_package(package: Path) -> None:
    """Compile a single ``.mlpackage`` bundle using ``xcrun coremlcompiler``."""
    relative_pkg = package.relative_to(BASE_DIR)
    output_dir = OUTPUT_ROOT / relative_pkg.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{package.stem}.mlmodelc"

    if output_path.exists():
        shutil.rmtree(output_path)

    cmd = [
        "xcrun",
        "coremlcompiler",
        "compile",
        str(package),
        str(output_dir),
    ]

    print(f"Compiling {relative_pkg} -> {output_path.relative_to(BASE_DIR)}")
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

    print(f"Finished compiling {len(packages)} package(s) into {OUTPUT_ROOT.relative_to(BASE_DIR)}.")


if __name__ == "__main__":
    main()
