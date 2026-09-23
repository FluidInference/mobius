"""Compile one published package and record Core ML ANE fallback placement.

Uses the repository's coreml-cli because Core ML's model plan is a runtime
result, not something that can be inferred from a MIL op inventory alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def profile(package: Path, cli_dir: Path, timeout: int) -> dict:
    import coremltools as ct

    package = package.resolve()
    cli_dir = cli_dir.resolve()
    if not package.is_dir() or package.suffix != ".mlpackage":
        raise ValueError("Expected an existing .mlpackage directory")
    if not (cli_dir / "pyproject.toml").is_file():
        raise ValueError("Expected the coreml-cli project directory")
    compiled = Path(ct.utils.compile_model(str(package)))
    try:
        run = subprocess.run(
            [
                "uv",
                "run",
                "coreml-cli",
                str(compiled),
                "--fallback",
                "--json",
                "--plan-timeout",
                str(timeout),
            ],
            cwd=cli_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        raw = json.loads(run.stdout)
        if len(raw["models"]) != 1:
            raise ValueError("Expected one compute plan")
        return {
            "package": package.name,
            "package_bytes": sum(
                item.stat().st_size for item in package.rglob("*") if item.is_file()
            ),
            "hardware": raw["hardware"],
            "fallback": raw["models"][0]["fallback"],
        }
    finally:
        if compiled.suffix == ".mlmodelc" and compiled != package:
            shutil.rmtree(compiled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument(
        "--coreml-cli",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "tools" / "coreml-cli",
    )
    parser.add_argument("--plan-timeout", type=int, default=45)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan_timeout < 1:
        parser.error("--plan-timeout must be positive")
    rendered = (
        json.dumps(profile(args.package, args.coreml_cli, args.plan_timeout), indent=2)
        + "\n"
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
