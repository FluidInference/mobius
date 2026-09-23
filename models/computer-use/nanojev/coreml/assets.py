"""Fetch and load the exact NanoJev checkpoint selected for conversion."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "assets.lock.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(*, with_weights: bool) -> Path:
    patterns = list(LOCK["source_files"])
    if with_weights:
        patterns.append(LOCK["checkpoint"]["selected_weight"])
    path = snapshot_download(
        LOCK["checkpoint"]["repo"],
        revision=LOCK["checkpoint"]["revision"],
        allow_patterns=patterns,
        token=False,
    )
    return Path(path)


def upstream_module(root: Path, name: str):
    path = root / "source/scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nanojev_pinned_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load audited NanoJev source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model() -> tuple[Path, AutoTokenizer, torch.nn.Module]:
    root = snapshot(with_weights=True)
    weights = root / LOCK["checkpoint"]["selected_weight"]
    if sha256(weights) != LOCK["checkpoint"]["sha256"]:
        raise ValueError("NanoJev trained checkpoint SHA256 mismatch")
    run_config = json.loads((root / "config.json").read_text())
    if run_config["set_head"] != "attention":
        raise ValueError("Expected trained attention set head")
    if run_config["resolved_model_revision"] != LOCK["upstream_base"]["revision"]:
        raise ValueError("NanoJev base revision mismatch")
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(root / "backbone_config", local_files_only=True, trust_remote_code=False)
    config.use_cache = False
    backbone = AutoModel.from_config(config, attn_implementation="sdpa", trust_remote_code=False).float()
    model = upstream_module(root, "train_toy_decisions").DecisionModel(backbone, run_config["set_head"])
    parameters = load_file(str(weights), device="cpu")
    model.load_state_dict(parameters, strict=True)
    del parameters
    model.eval()
    return root, tokenizer, model
