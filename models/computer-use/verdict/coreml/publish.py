"""Publish only a verified Verdict bucket to its dedicated FluidInference Hub repo."""

from __future__ import annotations

import argparse
import json

from huggingface_hub import CommitOperationAdd, HfApi

from assets import LOCK, ROOT, sha256, verify_assets

REPO = "FluidInference/verdict-coreml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    args = parser.parse_args()
    source = verify_assets()
    package = ROOT / "build" / f"verdict_fp16_L{args.length}_candidates25.mlpackage"
    conversion = json.loads((ROOT / "build" / f"{package.stem}.conversion.json").read_text())
    verification_path = ROOT / "reports" / f"verification-L{args.length}.json"
    verification = json.loads(verification_path.read_text())
    if (
        not verification["passed"]
        or conversion["package"] != package.name
        or verification["package"] != package.name
        or verification.get("package_files_sha256") != conversion["package_files_sha256"]
        or verification["source_revision"] != LOCK["source_revision"]
        or conversion["source_revision"] != LOCK["source_revision"]
    ):
        raise ValueError("matching conversion and passing native parity are required before publishing")
    if not package.exists():
        raise FileNotFoundError(package)
    for relative, expected in conversion["package_files_sha256"].items():
        if sha256(package / relative) != expected:
            raise ValueError(f"Core ML package changed since conversion: {relative}")
    paths = {
        "README.md": ROOT / "hf" / "README.md",
        "LICENSE": ROOT / "LICENSE",
        "assets.lock.json": ROOT / "assets.lock.json",
        "calibrator.json": source / "calibrator.json",
        "tokenizer.json": source / "tokenizer.json",
        "tokenizer_config.json": source / "tokenizer_config.json",
        "native_reference.py": ROOT / "native_reference.py",
        "preprocessing.py": ROOT / "preprocessing.py",
        verification_path.name: verification_path,
        f"{package.stem}.conversion.json": ROOT / "build" / f"{package.stem}.conversion.json",
    }
    if args.length == 128:
        paths["profile-L128.json"] = ROOT / "reports" / "profile-L128.json"
    for item in package.rglob("*"):
        if item.is_file():
            paths[f"{package.name}/{item.relative_to(package)}"] = item
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local) for remote, local in paths.items()]
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    result = api.create_commit(
        repo_id=REPO,
        repo_type="model",
        operations=operations,
        commit_message=f"Add verified Verdict Core ML L{args.length} FP16",
    )
    print(f"{REPO} revision {result.oid}")


if __name__ == "__main__":
    main()
