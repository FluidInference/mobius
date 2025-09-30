"""Compile Core ML model packages for a specific Apple platform."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def compile_model(
    model: Path,
    output_dir: Path,
    platform: str,
    deployment_target: str,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "xcrun",
        "coremlcompiler",
        "compile",
        str(model),
        str(output_dir),
        "--platform",
        platform,
        "--deployment-target",
        deployment_target,
    ]
    print("Running:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Core ML models for a target platform",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("coreml_models"),
        help="Directory containing .mlmodel or .mlpackage files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/compiled"),
        help="Directory where compiled .mlmodelc bundles will be written",
    )
    parser.add_argument(
        "--platform",
        default="iOS",
        help="Target platform for compatibility checks",
    )
    parser.add_argument(
        "--deployment-target",
        default="17.0",
        help="Minimum OS version to target",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_dir.exists():
        print(f"Input directory {args.input_dir} does not exist", file=sys.stderr)
        return 1

    models = list(args.input_dir.glob("*.mlpackage")) + list(args.input_dir.glob("*.mlmodel"))
    if not models:
        print(f"No Core ML models found in {args.input_dir}", file=sys.stderr)
        return 1

    for model in models:
        try:
            compile_model(
                model=model,
                output_dir=args.output_dir,
                platform=args.platform,
                deployment_target=args.deployment_target,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Failed to compile {model}: {exc}", file=sys.stderr)
            return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
