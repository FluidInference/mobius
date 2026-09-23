"""Publish the locally validated Jeff Core ML artifact to its own HF repo."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from export import REVISION, SOURCE

REPO_ID = "FluidInference/jeff-coreml"
W8_PACKAGE = "JeffDecision-L128-W8.mlpackage"
REQUIRED = (
    "README.md",
    "LICENSE",
    "assets.lock.json",
    "jeff_decision.py",
    "trace_compat.py",
    "export.py",
    "verify.py",
    "runtime.py",
    "quantize.py",
    "probe-native.py",
    "pyproject.toml",
    "uv.lock",
)
TOKENIZER = ("gliner_config.json", "tokenizer.json", "tokenizer_config.json")


def package_file_hashes(package: Path) -> dict[str, str]:
    hashes = {}
    for file in sorted(package.rglob("*")):
        if not file.is_file():
            continue
        sha = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        hashes[file.relative_to(package).as_posix()] = sha.hexdigest()
    return hashes


def validate_w8_report(report: dict) -> None:
    if report.get("source_revision") != REVISION or report.get("package") != W8_PACKAGE:
        raise ValueError("W8 report does not identify the pinned checkpoint and package")
    if report.get("native_fixture_count") != 4 or report.get("native_choice_agreement") != 4:
        raise ValueError("W8 report does not preserve all four native decisions")
    if report.get("max_logit_error", float("inf")) > 0.25:
        raise ValueError("W8 report exceeds the 0.25 logit-error gate")
    if len(report.get("package_files_sha256", {})) != 3:
        raise ValueError("W8 report lacks the complete package file hashes")


def stage() -> Path:
    source = Path(snapshot_download(SOURCE, revision=REVISION, local_files_only=True))
    root = Path(__file__).parent
    stage_dir = root / "build" / "hub-stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for name in REQUIRED:
        shutil.copy2(root / name, stage_dir / name)
    for name in TOKENIZER:
        shutil.copy2(source / name, stage_dir / name)
    shutil.copy2(root / "build/native-parity.json", stage_dir / "native-parity.json")
    shutil.copy2(root / "build/coreml-parity-fp16.json", stage_dir / "coreml-parity-fp16.json")
    shutil.copytree(root / "reports", stage_dir / "reports")
    package = root / "build/JeffDecision-L128-FP16.mlpackage"
    shutil.copytree(package, stage_dir / package.name)
    w8_package = root / "build" / W8_PACKAGE
    report = json.loads((root / "reports/w8-validation.json").read_text())
    validate_w8_report(report)
    if package_file_hashes(w8_package) != report["package_files_sha256"]:
        raise ValueError("W8 package does not match the verified release report")
    shutil.copytree(w8_package, stage_dir / W8_PACKAGE)
    return stage_dir


def main() -> None:
    native = json.loads(Path("build/native-parity.json").read_text())
    coreml = json.loads(Path("build/coreml-parity-fp16.json").read_text())
    if len(native) != 4 or len(coreml) != 4 or not all(row["top_label_agreement"] for row in coreml):
        raise RuntimeError("the four-case trained-model/Core ML parity evidence is incomplete")
    if max(row["max_logit_error"] for row in coreml) > 0.25:
        raise RuntimeError("Core ML parity exceeds the published tolerance")
    stage_dir = stage()
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    result = api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(stage_dir),
        commit_message="Publish validated Jeff GLiFormer Large L128 FP16 Core ML classifier",
    )
    print(result, flush=True)
    print(json.dumps(api.list_repo_files(REPO_ID), indent=2), flush=True)


if __name__ == "__main__":
    main()
