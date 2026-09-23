"""Publish the verified Kev FP16 package and reproducibility files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

from assets import LOCK_PATH
from release_integrity import package_file_hashes, sha256, validate_release_report

ROOT = Path(__file__).resolve().parent
REPO = "FluidInference/kev-0.6b-coreml"
FP16_PACKAGE = ROOT / "build/kev_0_6b_fp16_L128_options32.mlpackage"
W8_PACKAGE = ROOT / "build/kev_0_6b_w8_L128_options32.mlpackage"
SOURCE_FILES = (
    "README.md",
    "RESULTS.md",
    "STATUS.md",
    "assets.lock.json",
    "assets.py",
    "convert-coreml.py",
    "export_model.py",
    "preprocessing.py",
    "runtime.py",
    "release_integrity.py",
    "publish-hf.py",
    "quantize.py",
    "verify.py",
    "pyproject.toml",
    "uv.lock",
    "tests/test_real_model_parity.py",
    "tests/test_release_integrity.py",
    "tests/test_standalone_runtime.py",
    "reports/standalone-runtime.json",
)
TOKENIZER_FILES = (
    "added_tokens.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--w8-only", action="store_true", help="add the verified W8 package to an existing repository")
    parser.add_argument(
        "--metadata-only", action="store_true", help="refresh card and source without uploading weights"
    )
    args = parser.parse_args()
    if args.w8_only and args.metadata_only:
        parser.error("choose one publication mode")
    package = W8_PACKAGE if args.w8_only else FP16_PACKAGE
    if (not args.metadata_only and not package.is_dir()) or not (ROOT / "hf/README.md").is_file():
        raise FileNotFoundError("verified package and model card are required")

    report_path = ROOT / "reports" / f"verification-{package.stem}-cpu-ne.json"
    if not args.metadata_only:
        report = json.loads(report_path.read_text())
        validate_release_report(report, package.name, package_file_hashes(package), sha256(LOCK_PATH))

    operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=ROOT / "hf/README.md")]
    for name in SOURCE_FILES:
        operations.append(CommitOperationAdd(path_in_repo=f"source/{name}", path_or_fileobj=ROOT / name))
    if not args.metadata_only:
        operations.append(CommitOperationAdd(path_in_repo=f"reports/{report_path.name}", path_or_fileobj=report_path))
        for file in package.rglob("*"):
            if file.is_file():
                operations.append(
                    CommitOperationAdd(path_in_repo=str(file.relative_to(ROOT / "build")), path_or_fileobj=file)
                )
    lock = json.loads(LOCK_PATH.read_text())
    checkpoint = Path(
        snapshot_download(
            lock["checkpoint"]["repo"],
            revision=lock["checkpoint"]["revision"],
            allow_patterns=[*TOKENIZER_FILES, "training_config.json"],
        )
    )
    config = checkpoint / "training_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"pinned Kev configuration missing: {config}")
    operations.append(CommitOperationAdd(path_in_repo="config/training_config.json", path_or_fileobj=config))
    base = Path(snapshot_download(lock["base"]["repo"], revision=lock["base"]["revision"], allow_patterns=["LICENSE"]))
    license_file = base / "LICENSE"
    if not license_file.is_file():
        raise FileNotFoundError(f"pinned Qwen Apache license missing: {license_file}")
    operations.append(CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=license_file))
    if not args.w8_only and not args.metadata_only:
        for name in TOKENIZER_FILES:
            file = checkpoint / name
            if not file.is_file():
                raise FileNotFoundError(f"pinned tokenizer file missing: {file}")
            operations.append(CommitOperationAdd(path_in_repo=f"tokenizer/{name}", path_or_fileobj=file))

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
        commit_message=(
            "Refresh Kev 0.6B Core ML card and source"
            if args.metadata_only
            else ("Add verified Kev 0.6B W8 Core ML variant" if args.w8_only else "Publish Kev 0.6B FP16 Core ML")
        ),
    )
    print(result.commit_url)


if __name__ == "__main__":
    main()
