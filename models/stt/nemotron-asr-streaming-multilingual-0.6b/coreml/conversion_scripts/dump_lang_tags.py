#!/usr/bin/env python3
"""Dump language-tag token IDs from a .nemo model_config.yaml.

The multilingual joint vocabulary contains 39 language tags like
`<en-US>`, `<zh-CN>`, `<bg-BG>`, ... scattered across the full 13,087
token vocabulary (NOT bunched at the start — confirmed by inspection,
they sit at ids 1, 256, 397, …, 9847, 12944). Greedy decoding will
emit these tokens; the Swift detokenizer must filter them (matches
NeMo's `strip_lang_tags=true` flag).

This script extracts the IDs without needing torch/NeMo installed:
just unpacks the .nemo tarball and parses the YAML.

Run:
    uv run python dump_lang_tags.py \
        --nemo-path /path/to/multilingual.nemo \
        --output lang_tag_token_ids.json
"""
from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import typer
import yaml

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# A language tag looks like <bg-BG>, <en-US>, <zh-CN>, <fr-CA>, ...
# The vocabulary uses strict ISO-639 + region form only. `<unk>` is NOT a
# language tag (it's the unknown-token special symbol) and is excluded.
_LANG_TAG_RE = re.compile(r"^<[A-Za-z]{2,4}-[A-Za-z]{2,4}>$")


def _extract_member(nemo_path: Path, name: str) -> bytes:
    with tarfile.open(nemo_path, "r:gz") as tar:
        member = next((m for m in tar.getmembers() if m.name == name), None)
        if member is None:
            raise FileNotFoundError(f"{name!r} not in {nemo_path}")
        f = tar.extractfile(member)
        assert f is not None
        return f.read()


def _scan_vocab(cfg: dict) -> List[Tuple[int, str]]:
    """Return [(id, token), ...] for the joint vocabulary."""
    joint = cfg.get("joint", {})
    vocab = joint.get("vocabulary")
    if vocab is None:
        labels = cfg.get("labels")
        if labels is None:
            raise RuntimeError("No `joint.vocabulary` or `labels` in config")
        vocab = labels
    return list(enumerate(vocab))


@app.command()
def dump(
    nemo_path: Path = typer.Option(
        ..., "--nemo-path", help="Path to the .nemo file"
    ),
    output: Path = typer.Option(
        Path("lang_tag_token_ids.json"), "--output"
    ),
) -> None:
    """Write a JSON list of language-tag token IDs."""

    typer.echo(f"Reading {nemo_path}...")
    raw = _extract_member(nemo_path, "model_config.yaml")
    cfg = yaml.safe_load(raw)

    entries = _scan_vocab(cfg)
    lang_ids: List[int] = []
    matched: Dict[int, str] = {}
    for tid, tok in entries:
        if isinstance(tok, str) and _LANG_TAG_RE.match(tok):
            lang_ids.append(tid)
            matched[tid] = tok

    typer.echo(f"Found {len(lang_ids)} language-tag tokens")
    for tid in lang_ids:
        typer.echo(f"  {tid:5d}  {matched[tid]!r}")

    output.write_text(
        json.dumps(
            {
                "lang_tag_token_ids": lang_ids,
                "lang_tag_tokens": matched,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
