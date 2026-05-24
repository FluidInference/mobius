"""Pre-conversion vocab pruning for English-only deployment.

Slices the decoder embedding and joint output projection to keep only
a subset of the 13,088-token vocabulary. The full vocab covers 38
languages; for English-only deployment ~1k tokens suffice.

Usage:
    # Inside the converter, before tracing:
    from patch_vocab_prune import prune_vocab_english
    keep_ids = build_english_keep_set(model.tokenizer, corpus_jsonl)
    id_map = prune_vocab_english(model, keep_ids)
    # id_map is old_id -> new_id; use to rewrite tokenizer.json.

Effect on model:
  decoder.prediction.embed.weight:  (13088, 640) -> (N_keep, 640)
  joint.joint_net[2].weight:        (13088, 640) -> (N_keep, 640)
  joint.joint_net[2].bias:          (13088,)     -> (N_keep,)

Where N_keep = len(keep_ids).

CRITICAL: blank_idx changes! Original blank is at 13087; new blank is at
N_keep-1 (we always put blank last in the kept set). Downstream Swift
code reading metadata.json must use the NEW blank_idx.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def build_english_keep_set(
    tokenizer,
    text_sources: Iterable[str],
    lang_tag_ids: list[int],
    old_blank_idx: int,
    padding_for_safety: int = 0,
) -> list[int]:
    """Build a sorted list of vocab IDs to keep.

    - Encodes every text in text_sources with the SentencePiece tokenizer
    - Unions with lang_tag_ids (model may emit these in auto-detect mode)
    - Appends old_blank_idx (=13087) as the LAST element so the new
      blank_idx = len(keep)-1
    - Includes a few additional safety tokens if padding_for_safety > 0
      (e.g., common punctuation pieces that might emerge OOD)

    Returns sorted list with blank guaranteed last.
    """
    keep = set()
    for text in text_sources:
        if not text:
            continue
        ids = tokenizer.text_to_ids(text)
        keep.update(ids)
    keep.update(lang_tag_ids)
    # Don't include blank in the "sorted vocab" portion — we want blank
    # at the END so the new blank_idx is exactly N_keep-1.
    keep.discard(old_blank_idx)
    sorted_vocab = sorted(keep)
    # Append blank last
    sorted_vocab.append(old_blank_idx)
    return sorted_vocab


def prune_vocab_english(model: torch.nn.Module, keep_ids: list[int]) -> dict[int, int]:
    """In-place slice decoder embed + joint output_proj to keep_ids.

    Returns old_id -> new_id map (only for old IDs in keep_ids).
    """
    n_keep = len(keep_ids)
    old_blank = keep_ids[-1]
    new_blank = n_keep - 1

    id_map = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}

    # ── Decoder embedding ────────────────────────────────────────────
    # nn.Embedding(num_embeddings, embedding_dim, padding_idx=...)
    embed = model.decoder.prediction["embed"]
    old_emb = embed.weight.data  # (13088, 640)
    new_emb = old_emb[keep_ids, :].clone()  # (n_keep, 640)
    # Replace with a fresh Embedding so padding_idx is correct
    new_embed = torch.nn.Embedding(
        n_keep,
        embed.embedding_dim,
        padding_idx=new_blank,
    )
    with torch.no_grad():
        new_embed.weight.copy_(new_emb)
    model.decoder.prediction["embed"] = new_embed

    # ── Joint output projection ──────────────────────────────────────
    # joint_net is Sequential(ReLU, Dropout, Linear(640, 13088))
    # The output Linear is the last module
    out_proj = model.joint.joint_net[-1]
    old_w = out_proj.weight.data  # (13088, 640)
    old_b = out_proj.bias.data    # (13088,)
    new_w = old_w[keep_ids, :].clone()  # (n_keep, 640)
    new_b = old_b[keep_ids].clone()     # (n_keep,)

    new_out = torch.nn.Linear(out_proj.in_features, n_keep, bias=True)
    with torch.no_grad():
        new_out.weight.copy_(new_w)
        new_out.bias.copy_(new_b)
    model.joint.joint_net[-1] = new_out

    # Update blank_idx wherever it's settable. Some NeMo properties
    # (e.g., num_classes_with_blank) are read-only @property — set their
    # backing field directly instead of the property.
    def _try_set(obj, name, value):
        try:
            setattr(obj, name, value)
        except AttributeError:
            # Try the backing private attribute by convention
            for cand in (f"_{name}", f"_{name}_val"):
                if hasattr(obj, cand):
                    try:
                        setattr(obj, cand, value)
                        return
                    except AttributeError:
                        pass

    _try_set(model.decoder, "blank_idx", new_blank)
    _try_set(model.joint, "blank_idx", new_blank)
    _try_set(model.joint, "num_classes", n_keep)

    return id_map


def rewrite_tokenizer_json(old_tokenizer_path: Path, new_tokenizer_path: Path,
                            id_map: dict[int, int]) -> None:
    """Rewrite tokenizer.json so new IDs map to BPE pieces.

    The original tokenizer.json format: {"old_id": "bpe_piece"} as a flat
    dict (string keys). We produce {"new_id": "bpe_piece"} for kept IDs only.
    """
    with open(old_tokenizer_path) as f:
        old_tok = json.load(f)
    # old_tok keys are stringified old IDs
    new_tok = {}
    for old_id_str, piece in old_tok.items():
        try:
            old_id = int(old_id_str)
        except ValueError:
            new_tok[old_id_str] = piece  # pass through non-numeric keys
            continue
        if old_id in id_map:
            new_tok[str(id_map[old_id])] = piece
    with open(new_tokenizer_path, "w") as f:
        json.dump(new_tok, f, indent=2, ensure_ascii=False)
