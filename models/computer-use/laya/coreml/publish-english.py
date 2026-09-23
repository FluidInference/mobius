"""Publish a verified Laya English bucket to its own public FluidInference Hub repo."""

from __future__ import annotations

import argparse
import json

from huggingface_hub import CommitOperationAdd, HfApi

from assets import ROOT, checkpoint_dir, load_lock, sha256, verify_assets
from preprocessing import package_name

REPO = "FluidInference/laya-english-coreml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=128)
    args = parser.parse_args()
    verify_assets("english")
    lock = load_lock()
    source = checkpoint_dir("english")
    package = ROOT / "build" / f"{package_name('english', args.length, 32)}.mlpackage"
    manifest_path = ROOT / "build" / f"{package.stem}.conversion.json"
    report_path = ROOT / "reports" / f"verification-english-L{args.length}.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    if (
        not report["passed"]
        or report["validated_release_units"] != "ALL"
        or report["variant"] != "english"
        or manifest["model"] != package.name
        or report["package"] != package.name
        or report["package_files_sha256"] != manifest["package_files"]
        or (report["length"], report["max_options"], report["precision"]) != (args.length, 32, "fp16")
        or (manifest["length"], manifest["max_options"], manifest["precision"]) != (args.length, 32, "float16")
    ):
        raise ValueError("matching passing English conversion/parity report required")
    if manifest["source_revision"] != lock["revision"] or report["source_revision"] != lock["revision"]:
        raise ValueError("source revision mismatch")
    for relative, expected in manifest["package_files"].items():
        if sha256(package / relative) != expected:
            raise ValueError(f"Core ML package changed since conversion: {relative}")
    files = {
        "README.md": ROOT / "hf" / "README-english.md",
        "LICENSE": ROOT / "LICENSE",
        "assets.lock.json": ROOT / "assets.lock.json",
        "rl_agent_config.json": source / "rl_agent_config.json",
        "encoder/config.json": source / "encoder" / "config.json",
        "tokenizer/tokenizer.json": source / "tokenizer" / "tokenizer.json",
        "tokenizer/tokenizer_config.json": source / "tokenizer" / "tokenizer_config.json",
        "preprocessing.py": ROOT / "preprocessing.py",
        "calibration.py": ROOT / "calibration.py",
        report_path.name: report_path,
        manifest_path.name: manifest_path,
    }
    for item in package.rglob("*"):
        if item.is_file():
            files[f"{package.name}/{item.relative_to(package)}"] = item
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local) for remote, local in files.items()]
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    result = api.create_commit(
        repo_id=REPO,
        repo_type="model",
        operations=operations,
        commit_message=f"Add verified Laya English Core ML L{args.length} FP16",
    )
    print(f"{REPO} revision {result.oid}")


if __name__ == "__main__":
    main()
