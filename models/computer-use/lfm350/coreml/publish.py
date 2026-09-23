"""Stage and upload only the validated LFM RLCD Core ML variant."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from decision import SOURCE_REPO, SOURCE_REVISION
from release_integrity import package_file_hashes, validate_release_reports

ROOT = Path(__file__).resolve().parent
REPO = "FluidInference/lfm2-5-350m-rlcd-coreml"
PACKAGE = "lfm350_rlcd_fp16_L256_B8_V16.mlpackage"
FILES = [
    "assets.lock.json",
    "convert-coreml.py",
    "decision.py",
    "fixtures.py",
    "preprocessing.py",
    "publish.py",
    "pyproject.toml",
    "runtime-requirements.txt",
    "runtime.py",
    "release_integrity.py",
    "uv.lock",
    "verify-coreml.py",
    "verify-native.py",
]
SOURCE_FILES = ["chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json"]


def stage() -> Path:
    source = Path(
        snapshot_download(
            SOURCE_REPO, revision=SOURCE_REVISION, allow_patterns=SOURCE_FILES + ["LICENSE", "LICENSE-CODE"]
        )
    )
    destination = ROOT / "build" / "hub-stage"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(ROOT / "MODEL_CARD.md", destination / "README.md")
    shutil.copy2(ROOT / "README.md", destination / "CONVERSION.md")
    for name in FILES:
        shutil.copy2(ROOT / name, destination / name)
    for name in SOURCE_FILES:
        shutil.copy2(source / name, destination / name)
    shutil.copy2(source / "LICENSE", destination / "LICENSE-LFM")
    shutil.copy2(source / "LICENSE-CODE", destination / "LICENSE-CODE")
    reports = destination / "reports"
    shutil.copytree(ROOT / "reports", reports)
    package = ROOT / "build" / PACKAGE
    if not package.exists():
        raise FileNotFoundError(package)
    conversion = json.loads((ROOT / "reports" / "fp16-conversion.json").read_text())
    parity = json.loads((ROOT / "reports" / "fp16-all-upstream9.json").read_text())
    validate_release_reports(conversion, parity, PACKAGE, SOURCE_REVISION)
    if package_file_hashes(package) != conversion["package_files_sha256"]:
        raise ValueError("Core ML package differs from the validated artifact")
    shutil.copytree(package, destination / PACKAGE, copy_function=os.link)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    folder = stage()
    files = [path for path in folder.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    print(f"Staged {len(files)} files, {total:,} bytes at {folder}", flush=True)
    if not args.upload:
        return
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    result = api.upload_folder(
        repo_id=REPO,
        repo_type="model",
        folder_path=folder,
        commit_message="Publish validated LFM2.5-350M-RLCD FP16 Core ML conversion",
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
