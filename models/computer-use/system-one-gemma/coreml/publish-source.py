"""Publish the blocked toolkit's allowlisted source files, never third-party weights."""

from __future__ import annotations

from huggingface_hub import CommitOperationAdd, HfApi

from assets import ROOT

REPO = "FluidInference/system-one-gemma-coreml"
ALLOWLIST = {
    "README.md": ROOT / "hf" / "README.md",
    "LICENSE-CODE": ROOT / "LICENSE-CODE",
    "README-toolkit.md": ROOT / "README.md",
    "STATUS.md": ROOT / "STATUS.md",
    "assets.lock.json": ROOT / "assets.lock.json",
    "assets.py": ROOT / "assets.py",
    "native_reference.py": ROOT / "native_reference.py",
    "export_model.py": ROOT / "export_model.py",
    "convert-coreml.py": ROOT / "convert-coreml.py",
    "scorer.py": ROOT / "scorer.py",
    "verify-parity.py": ROOT / "verify-parity.py",
    "publish-source.py": ROOT / "publish-source.py",
    "pyproject.toml": ROOT / "pyproject.toml",
    "uv.lock": ROOT / "uv.lock",
    "tests/test_source_contract.py": ROOT / "tests" / "test_source_contract.py",
    "reports/source-audit.json": ROOT / "reports" / "source-audit.json",
}


def main() -> None:
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=path) for remote, path in ALLOWLIST.items()]
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    result = api.create_commit(
        repo_id=REPO,
        repo_type="model",
        operations=operations,
        commit_message="Publish pinned blocked Gemma conversion toolkit without weights",
    )
    print(f"{REPO} revision {result.oid}")


if __name__ == "__main__":
    main()
