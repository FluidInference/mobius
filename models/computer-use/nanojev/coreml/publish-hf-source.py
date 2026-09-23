"""Publish NanoJev conversion source only while weight redistribution is unresolved."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent
REPO = "FluidInference/nanojev-coreml"
FILES = (
    ".gitignore",
    "LICENSE-CODE",
    "README.md",
    "RESULTS.md",
    "STATUS.md",
    "assets.lock.json",
    "assets.py",
    "convert-coreml.py",
    "export_model.py",
    "fixtures.py",
    "preprocessing.py",
    "pyproject.toml",
    "publish-hf-source.py",
    "tests/test_native_renderer.py",
    "uv.lock",
    "verify.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=ROOT / "hf/README.md"),
        CommitOperationAdd(path_in_repo="LICENSE-CODE", path_or_fileobj=ROOT / "LICENSE-CODE"),
    ]
    for name in FILES:
        operations.append(CommitOperationAdd(path_in_repo=f"source/{name}", path_or_fileobj=ROOT / name))
    if args.dry_run:
        for operation in operations:
            print(operation.path_in_repo)
        return
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    result = api.create_commit(
        repo_id=REPO,
        repo_type="model",
        operations=operations,
        commit_message="Publish NanoJev Core ML conversion source without trained weights",
    )
    print(result.commit_url)


if __name__ == "__main__":
    main()
