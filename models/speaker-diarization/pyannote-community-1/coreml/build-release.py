#!/usr/bin/env python3
"""Build a pinned, provenance-carrying Community-1 Core ML release."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from provenance import (
    MATERIAL_INPUTS,
    UPSTREAM_REPOSITORY,
    build_manifest,
    require_disjoint_paths,
    require_commit_sha,
    verified_git_revision,
    write_manifest,
)


PACKAGE_NAMES = {
    "segmentation-community-1.mlpackage": "Segmentation.mlpackage",
    "fbank-community-1.mlpackage": "FBank.mlpackage",
    "embedding-community-1.mlpackage": "Embedding.mlpackage",
    "plda-community-1.mlpackage": "PLDA.mlpackage",
    "plda_rho-community-1.mlpackage": "PldaRho.mlpackage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-revision",
        required=True,
        help="Full immutable commit SHA for pyannote/speaker-diarization-community-1",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--selective-fp16", action="store_true")
    return parser.parse_args()


def require_empty_or_missing(path: Path, label: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"{label} must be empty or absent: {path}")


def run(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    args = parse_args()
    upstream_revision = require_commit_sha(args.upstream_revision, "upstream revision")
    script_dir = Path(__file__).resolve().parent
    repository_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=script_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    conversion_revision = verified_git_revision(repository_root)

    work_dir = args.work_dir.resolve()
    release_dir = args.release_dir.resolve()
    require_disjoint_paths(work_dir, release_dir)
    require_empty_or_missing(work_dir, "work directory")
    require_empty_or_missing(release_dir, "release directory")
    work_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    model_root = Path(
        snapshot_download(
            repo_id=UPSTREAM_REPOSITORY,
            revision=upstream_revision,
            allow_patterns=list(MATERIAL_INPUTS),
        )
    )
    converted_dir = work_dir / "converted"
    compiled_dir = work_dir / "compiled"

    convert_command = [
        sys.executable,
        "convert-coreml.py",
        "--model-root",
        str(model_root),
        "--output-dir",
        str(converted_dir),
        "--source-revision",
        upstream_revision,
        "--conversion-revision",
        conversion_revision,
    ]
    if args.selective_fp16:
        convert_command.append("--selective-fp16")
    run(convert_command, script_dir)

    compile_command = [
        sys.executable,
        "compile-coreml.py",
        "--input-dir",
        str(converted_dir),
        "--output-dir",
        str(compiled_dir),
        "--platform",
        "iOS",
        "--deployment-target",
        "17.0",
    ]
    run(compile_command, script_dir)

    packages_dir = release_dir / "mlpackages"
    packages_dir.mkdir(parents=True)
    for source_name, release_name in PACKAGE_NAMES.items():
        source_package = converted_dir / source_name
        source_compiled = compiled_dir / source_name.replace(".mlpackage", ".mlmodelc")
        if not source_package.is_dir() or not source_compiled.is_dir():
            raise FileNotFoundError(f"Expected converted outputs for {source_name}")
        shutil.copytree(source_package, packages_dir / release_name)
        shutil.copytree(source_compiled, release_dir / release_name.replace(".mlpackage", ".mlmodelc"))

    for resource_name in ("plda-parameters.json", "xvector-transform.json"):
        shutil.copy2(converted_dir / "resources" / resource_name, release_dir / resource_name)

    commands = (convert_command, compile_command)
    manifest = build_manifest(
        model_root=model_root,
        release_root=release_dir,
        upstream_repository=UPSTREAM_REPOSITORY,
        upstream_revision=upstream_revision,
        conversion_revision=conversion_revision,
        commands=commands,
    )
    write_manifest(manifest, release_dir / "provenance.json")
    print(f"Release staged at {release_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
