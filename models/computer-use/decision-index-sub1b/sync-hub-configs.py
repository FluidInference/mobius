"""Publish pinned, hash-checked architecture configs to existing Core ML Hub repos.

This only uploads small metadata files. It never stages model weights or changes a
conversion's validation status. Run without --upload to verify sources first.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ConfigSource:
    name: str
    hub_repo: str
    toolkit: str
    lock_key: str
    source_repo: str
    source_revision: str
    source_path: str
    sha256: str


SOURCES = (
    ConfigSource(
        "GLiNER2 small",
        "FluidInference/gliner2-5-small-coreml",
        "gliner2-small",
        "source",
        "fastino/gliner2.5-small-v1",
        "7e6f537f10337497069276892a5ef435028252ce",
        "config.json",
        "0b7d9e1401ceeb83e992ec66d2f93bff7e5646428f1b4706ec527cf88f53578a",
    ),
    ConfigSource(
        "Verdict",
        "FluidInference/verdict-coreml",
        "verdict",
        "source_repo",
        "heman10x/rlcd-modernbert-151m",
        "8af2496eb63c7fa66d7d234e1f62629380030eb4",
        "config.json",
        "303f8eef1009cfdcb0cfba3e653247e625f16a4501e2351bad1a633f1f644695",
    ),
    ConfigSource(
        "Laya English",
        "FluidInference/laya-english-coreml",
        "laya",
        "repo",
        "convaiinnovations/laya",
        "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "encoder/config.json",
        "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
    ),
    ConfigSource(
        "Kev 0.5B",
        "FluidInference/kev-0-5b-coreml",
        "kev-0.5b",
        "base.repo",
        "Qwen/Qwen2.5-0.5B",
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "config.json",
        "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b",
    ),
    ConfigSource(
        "Kev 0.6B",
        "FluidInference/kev-0.6b-coreml",
        "kev-0.6b",
        "base.repo",
        "Qwen/Qwen3-0.6B-Base",
        "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "config.json",
        "504a6b58c4271583724e66584b6b7698aea18450209df6b2f7582df0e89cee59",
    ),
    ConfigSource(
        "Decision Kai",
        "FluidInference/decision-1.0-kai-coreml",
        "decision-kai",
        "source_repo",
        "llm-semantic-router/Decision-1.0-Kai-0.6B",
        "7185f514f54b8f93c55998b1e8f9c5cc67f0d029",
        "native/encoder/config.json",
        "7aff915e9f159305e0bef3eb0206416f99b8560260b8969b35f1dcd54aaad1a5",
    ),
    ConfigSource(
        "Decision Lex",
        "FluidInference/decision-1.0-lex-coreml",
        "decision-lex",
        "source_repo",
        "llm-semantic-router/Decision-1.0-Lex-0.6B",
        "ee8e74d912fca8328a353c11d174b44da3f91781",
        "native/encoder/config.json",
        "7aff915e9f159305e0bef3eb0206416f99b8560260b8969b35f1dcd54aaad1a5",
    ),
    ConfigSource(
        "Jeff",
        "FluidInference/jeff-coreml",
        "jeff",
        "checkpoint.repo",
        "knowledgator/gliformer-large-v1",
        "d0a4e53d09cebe6bc963dd9be319d4279084bb2d",
        "gliner_config.json",
        "80aec1c8824bd58d6733f27d78a8271420f2c7008e4b0c81313d8da09624eea4",
    ),
)


def nested(data: dict, key: str) -> object:
    for part in key.split("."):
        data = data[part]
    return data


def verify_lock(source: ConfigSource) -> None:
    lock_path = (
        ROOT / "models/computer-use" / source.toolkit / "coreml/assets.lock.json"
    )
    lock = json.loads(lock_path.read_text())
    if nested(lock, source.lock_key) != source.source_repo:
        raise ValueError(f"{source.name}: source repo differs from {lock_path}")
    revision_key = (
        "base.revision"
        if source.lock_key == "base.repo"
        else (
            "checkpoint.revision"
            if source.lock_key == "checkpoint.repo"
            else (
                "revision"
                if source.lock_key in ("source", "repo")
                else "source_revision"
            )
        )
    )
    if nested(lock, revision_key) != source.source_revision:
        raise ValueError(f"{source.name}: source revision differs from {lock_path}")


def verified_config(source: ConfigSource) -> bytes:
    verify_lock(source)
    path = hf_hub_download(
        source.source_repo, source.source_path, revision=source.source_revision
    )
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != source.sha256:
        raise ValueError(f"{source.name}: source config SHA-256 mismatch")
    if not isinstance(json.loads(content), dict):
        raise TypeError(f"{source.name}: source config must be a JSON object")
    return content


def provenance(source: ConfigSource) -> bytes:
    component = (
        "full upstream model"
        if source.source_path == "config.json" and source.lock_key not in ("base.repo",)
        else "architecture component"
    )
    return (
        "# Root config provenance\n\n"
        f"The root `config.json` is an unmodified copy of the {component} config from "
        f"[`{source.source_repo}`](https://huggingface.co/{source.source_repo}/tree/{source.source_revision}) "
        f"at `{source.source_revision}`, path `{source.source_path}`.\n\n"
        f"SHA-256: `{source.sha256}`.\n\n"
        "This file describes the source architecture. It does not make the Core ML package "
        "a standalone Transformers AutoModel checkpoint. Use the repository's Core ML "
        "runtime and its included tokenizer, decision heads, and calibration where applicable.\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="commit missing config and provenance files to HF",
    )
    args = parser.parse_args()
    api = HfApi()
    for source in SOURCES:
        config = verified_config(source)
        print(
            f"{source.hub_repo}: verified {source.source_path} ({len(config)} bytes)",
            flush=True,
        )
        if not args.upload:
            continue
        info = api.model_info(source.hub_repo)
        existing = {item.rfilename for item in info.siblings}
        if "config.json" in existing:
            remote = Path(
                hf_hub_download(source.hub_repo, "config.json", revision=info.sha)
            ).read_bytes()
            if remote != config:
                raise ValueError(
                    f"{source.hub_repo}: root config differs from pinned source; inspect before updating"
                )
        operations = []
        if "config.json" not in existing:
            operations.append(
                CommitOperationAdd(
                    path_in_repo="config.json", path_or_fileobj=io.BytesIO(config)
                )
            )
        if "CONFIG_PROVENANCE.md" in existing:
            remote_note = Path(
                hf_hub_download(
                    source.hub_repo, "CONFIG_PROVENANCE.md", revision=info.sha
                )
            ).read_bytes()
            if remote_note != provenance(source):
                raise ValueError(
                    f"{source.hub_repo}: provenance differs; inspect before updating"
                )
        else:
            operations.append(
                CommitOperationAdd(
                    path_in_repo="CONFIG_PROVENANCE.md",
                    path_or_fileobj=io.BytesIO(provenance(source)),
                )
            )
        if not operations:
            print("  already current", flush=True)
            continue
        commit = api.create_commit(
            repo_id=source.hub_repo,
            repo_type="model",
            operations=operations,
            parent_commit=info.sha,
            commit_message="Add pinned source architecture config and provenance",
        )
        print(f"  published {commit.oid}", flush=True)


if __name__ == "__main__":
    main()
