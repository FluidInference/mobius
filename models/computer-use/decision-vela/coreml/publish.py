"""Publish a validated, pinned Kai or Lex Core ML artifact to its own HF repo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from typed_coreml import KINDS, package_sha256

REPOS = {
    "decision-kai": "FluidInference/decision-1.0-kai-coreml",
    "decision-lex": "FluidInference/decision-1.0-lex-coreml",
}


def _validated(toolkit: Path, artifacts: Path) -> tuple[dict, dict]:
    lock = json.loads((toolkit / "assets.lock.json").read_text())
    if lock["engine"] not in REPOS:
        raise ValueError("Unknown Decision release")
    reports = {}
    for kind in KINDS:
        package = artifacts / (kind + ".mlpackage")
        conversion = json.loads((artifacts / (kind + ".json")).read_text())
        validation = json.loads((artifacts / (kind + "-validation.json")).read_text())
        if not package.is_dir() or conversion["package_sha256"] != package_sha256(package):
            raise ValueError(f"{kind} package hash mismatch")
        if any(item["source_revision"] != lock["source_revision"] or item["kind"] != kind
               for item in (conversion, validation)):
            raise ValueError(f"{kind} source or type mismatch")
        if (conversion["native_parameter_count"] != 571_909_635
                or conversion["wrapper_parity"]["max_abs_logit_error"] > 1e-4
                or conversion["trace_max_abs_logit_error"] > 1e-4
                or not conversion["coreml_choice_agreement"]):
            raise ValueError(f"{kind} export failed parity")
        if validation["request_count"] < 2 or not validation["choice_agreement"]:
            raise ValueError(f"{kind} real-fixture validation incomplete")
        reports[kind] = {"conversion": conversion, "validation": validation}
    return lock, reports


def _card(lock: dict, reports: dict, repo: str, embedding_w8: dict | None = None) -> str:
    source = lock["source_repo"]
    lines = [
        "---",
        "license: other",
        "license_name: apache-2.0-plus-inherited-tokenizer-terms",
        f"license_link: https://huggingface.co/{repo}/blob/main/DISTRIBUTION_TERMS.md",
        "tags:",
        "- coreml",
        "- decision-making",
        "- on-device",
        "---",
        "",
        f"# {lock['engine']} Core ML",
        "",
        (f"FP16 Core ML conversion of [{source}](https://huggingface.co/{source}) "
        f"at revision `{lock['source_revision']}`. The complete native model has "
        "571,909,635 unique parameters and three trained decision paths. "
        "`choice.mlpackage`, `noul.mlpackage`, and `score.mlpackage` jointly implement "
        "those paths; calling only Choice is not the full native model."),
        "",
        ("This is an initial fixed-shape release. Supported shapes are listed below. "
        "Longer requests or larger candidate sets must be rejected or served by a later "
        "variant; silently truncating them changes the model. The source model's public "
        "input budget is 1,024 tokens. `conversion/run_coreml.py` supplies the pinned "
        "System One renderer, tokenizer, candidate masks, softmax, and output mapping."),
        "",
        ("| Decision path | Tokens | Candidate slots | FP16 package MB | Validated source requests | "
        "Max probability error vs native |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in KINDS:
        report = reports[kind]
        shape = report["conversion"]["shape"]
        check = report["validation"]
        lines.append(
            f"| {kind} | {shape['tokens']} | {shape['candidates']} | "
            f"{report['conversion']['package_bytes'] / 1e6:.1f} | "
            f"{check['request_count']} | {check['max_abs_probability_error']:.6g} |"
        )
    total_bytes = sum(reports[kind]["conversion"]["package_bytes"] for kind in KINDS)
    lines.extend(
        [
            "",
            (f"All three installed packages total {total_bytes / 1e6:.1f} MB. The three "
            "graphs each carry an embedding copy; this exceeds the unique native "
            "parameter payload. Package size does not include Core ML compilation."),
            "",
            ("Each package was compared against the complete pinned native model on the "
            "source repository's real decision and System One examples. This small "
            "parity check establishes conversion behavior on those inputs; it is not "
            "an official Decision Index score or a broad quality evaluation."),
            "",
            ("The tracker lists an estimated 307.8M encoder for this entry. This release "
            "contains about 572M unique native parameters, including all trained paths "
            "and heads. Its exact historical serving adapter is not publicly verified, "
            "so the tracker score is not attributed to this Core ML build."),
            "",
            ("The Decision contributions are Apache 2.0. The inherited tokenizer "
            "has additional Gemma-origin terms. See `LICENSE`, `NOTICE`, "
            "`LICENSING_STATUS.md`, `DISTRIBUTION_TERMS.md`, and `LICENSES/` included "
            "here. The tokenizer and source runtime were not retrained or modified."),
            "",
            ("Conversion code, source locks, fixed real fixtures, and reports are in "
            "`conversion/` and `reports/`. Source artifacts remain at the upstream "
            "model link above."),
            "",
            "```bash",
            "uv sync --project conversion",
            ("uv run --project conversion python conversion/run_coreml.py "
            "--repo-root . --request-json conversion/upstream-system-one.json"),
            "```",
            "",
            f"Repository: https://huggingface.co/{repo}",
            "",
        ]
    )
    if embedding_w8:
        lines.extend([
            "## Optional embedding-only W8 packages",
            "",
            "These packages quantize only the token embedding to per-channel int8; the encoder and heads remain FP16. They are size options validated on two pinned real requests per typed path, not a Decision Index score or a measured speedup. Use `--embedding-w8-kinds` with `conversion/run_coreml.py` to select the listed paths; other paths keep FP16.",
            "",
            "| Path | Package MB | Max probability error vs native |",
            "|---|---:|---:|",
        ])
        for kind, result in embedding_w8.items():
            lines.append(f"| {kind} | {result['conversion']['package_bytes'] / 1e6:.1f} | "
                         f"{result['validation']['max_abs_probability_error']:.6g} |")
        if lock["engine"] == "decision-lex":
            lines.extend(["", "Lex Choice remains FP16: both symmetric and asymmetric embedding W8 changed one pinned real Choice decision."])
        lines.extend(["", "Forced CPU+ANE placed 926/938 executable operations on ANE in the profiled W8 path, with int32 CPU boundaries. This is operation placement, not a runtime percentage. Exact package hashes and validation reports are in `reports/`.", ""])
    return "\n".join(lines)


def _validated_embedding_w8(toolkit: Path, artifacts: Path | None) -> dict:
    if artifacts is None:
        return {}
    lock = json.loads((toolkit / "assets.lock.json").read_text())
    expected = ("choice", "noul", "score") if lock["engine"] == "decision-kai" else ("noul", "score")
    results = {}
    for kind in expected:
        package = artifacts / (kind + ".mlpackage")
        conversion = json.loads((artifacts / (kind + ".json")).read_text())
        validation = json.loads((artifacts / (kind + "-validation.json")).read_text())
        if not package.is_dir() or conversion.get("package_sha256") != package_sha256(package):
            raise ValueError(f"{kind} embedding-W8 package hash mismatch")
        if conversion.get("source_revision") != lock["source_revision"] or validation.get("source_revision") != lock["source_revision"]:
            raise ValueError(f"{kind} embedding-W8 source revision mismatch")
        compression = conversion.get("compression", {})
        if (compression.get("scope"), compression.get("dtype"), compression.get("granularity")) != ("embedding", "int8", "per_channel"):
            raise ValueError(f"{kind} package is not embedding-only W8")
        if conversion.get("validated") is not True:
            raise ValueError(f"{kind} embedding-W8 package lacks a passing validation mark")
        if (validation.get("kind") != kind or validation.get("request_count") != 2
                or not validation.get("choice_agreement")
                or validation.get("max_abs_probability_error", float("inf")) > 0.02):
            raise ValueError(f"{kind} embedding-W8 native parity failed")
        results[kind] = {"conversion": conversion, "validation": validation}
    return results


def stage(toolkit: Path, artifacts: Path, destination: Path,
          embedding_w8_artifacts: Path | None = None) -> tuple[str, dict]:
    lock, reports = _validated(toolkit, artifacts)
    embedding_w8 = _validated_embedding_w8(toolkit, embedding_w8_artifacts)
    repo = REPOS[lock["engine"]]
    source = Path(snapshot_download(
        repo_id=lock["source_repo"],
        revision=lock["source_revision"],
        allow_patterns=["LICENSE", "NOTICE", "LICENSES/**", "DISTRIBUTION_TERMS.md",
                        "LICENSING_STATUS.md", "native/tokenizer/**", "native/packing.py",
                        "decision_inference/_system_one.py"],
    ))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "README.md").write_text(_card(lock, reports, repo, embedding_w8))
    for name in ("LICENSE", "NOTICE", "DISTRIBUTION_TERMS.md", "LICENSING_STATUS.md"):
        shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "LICENSES", destination / "LICENSES")
    shutil.copytree(source / "native" / "tokenizer", destination / "tokenizer")
    conversion = destination / "conversion"
    conversion.mkdir()
    shared = toolkit.parents[1] / "decision-vela" / "coreml"
    for name in ("typed_coreml.py", "run.py", "run_coreml.py", "quantize_weights.py"):
        shutil.copy2(shared / name, conversion / name)
    shutil.copy2(source / "native" / "packing.py", conversion / "packing.py")
    shutil.copy2(source / "decision_inference" / "_system_one.py", conversion / "upstream_system_one.py")
    for name in ("assets.lock.json", "pyproject.toml", "uv.lock", "upstream-decisions.jsonl",
                 "upstream-system-one.json", "upstream-system-one-rows.jsonl"):
        shutil.copy2(toolkit / name, conversion / name)
    (conversion / "README.md").write_text(
        "# Conversion and runtime\n\n"
        "The original native weights and runtime are in the pinned upstream model "
        "named by `assets.lock.json`. `typed_coreml.py` exports its complete "
        "Choice, Noul, and Score paths. `packing.py` and "
        "`upstream_system_one.py` are unmodified upstream source files.\n\n"
        "From the repository root, run `uv sync --project conversion` and then "
        "`uv run --project conversion python conversion/run_coreml.py --repo-root . "
        "--request-json conversion/upstream-system-one.json` to serve a request "
        "with the fixed-shape packages. The host rejects requests beyond each "
        "package's token and candidate capacity.\n\n"
        "To repeat native parity after fetching and materializing the locked "
        "upstream checkpoint, run `uv run --project conversion python "
        "conversion/run.py --toolkit conversion --native-dir /path/to/native "
        "--verify-only`.\n"
    )
    packages = destination / "coreml"
    packages.mkdir()
    results = destination / "reports"
    results.mkdir()
    for kind in KINDS:
        # Hard links save local staging space; uploaded Hub files are ordinary files.
        shutil.copytree(artifacts / (kind + ".mlpackage"),
                        packages / (kind + ".mlpackage"), copy_function=os.link)
        for name in (kind + ".json", kind + "-validation.json"):
            shutil.copy2(artifacts / name, results / name)
    for kind in embedding_w8:
        shutil.copytree(embedding_w8_artifacts / (kind + ".mlpackage"),
                        packages / (kind + "-embedding-w8.mlpackage"), copy_function=os.link)
        for suffix in (".json", "-validation.json"):
            shutil.copy2(embedding_w8_artifacts / (kind + suffix),
                         results / (kind + "-embedding-w8" + suffix))
    return repo, {kind: reports[kind]["validation"] for kind in KINDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--embedding-w8-artifacts", type=Path)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    toolkit = args.toolkit.resolve()
    artifacts = args.artifacts.resolve()
    with tempfile.TemporaryDirectory(prefix="decision-coreml-publish-") as temp:
        repo, validation = stage(toolkit, artifacts, Path(temp) / "repo", args.embedding_w8_artifacts)
        print(json.dumps({"repo": repo, "validation": validation}))
        if args.upload:
            api = HfApi()
            api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
            commit = api.upload_folder(
                repo_id=repo, repo_type="model", folder_path=str(Path(temp) / "repo"),
                commit_message="Publish pinned Decision Core ML conversion",
            )
            (toolkit / "published.json").write_text(
                json.dumps({"repo": repo, "commit": commit.oid, "url": commit.commit_url}, indent=2) + "\n"
            )
            print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
