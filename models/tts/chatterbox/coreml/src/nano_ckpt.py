"""Locate the Chatterbox Nano checkpoint directory.

``CHATTERBOX_NANO_CKPT`` env var (a local dir with t3_nano_v1.safetensors,
s3gen_meanflow.safetensors, ve.safetensors, conds.pt + tokenizer json/txt)
takes precedence over the HF snapshot download — useful when the HF CDN is
throttled and the weights were fetched out-of-band.
"""
from __future__ import annotations

import os
from pathlib import Path


def nano_ckpt_dir(require=("t3_nano_v1.safetensors", "s3gen_meanflow.safetensors",
                           "ve.safetensors", "conds.pt", "vocab.json")) -> str:
    override = os.environ.get("CHATTERBOX_NANO_CKPT")
    if override:
        p = Path(override)
        missing = [f for f in require if not (p / f).exists()]
        if missing:
            raise FileNotFoundError(f"CHATTERBOX_NANO_CKPT={p} missing {missing}")
        return str(p)

    from chatterbox.tts_turbo import NANO_REPO_ID
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=NANO_REPO_ID,
        allow_patterns=["ve.safetensors", "t3_nano_v1.safetensors",
                        "s3gen_meanflow.safetensors", "conds.pt",
                        "*.json", "*.txt"],
    )


def load_nano():
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxTurboTTS.from_local(nano_ckpt_dir(), "cpu", nano=True)


class NanoT3Only:
    """T3 + tokenizer + conds without the S3Gen weights (partial checkpoint)."""

    def __init__(self, t3, tokenizer, conds):
        self.t3 = t3
        self.tokenizer = tokenizer
        self.conds = conds


def load_nano_t3() -> NanoT3Only:
    """Replicates the T3 portion of ChatterboxTurboTTS.from_local(nano=True)."""
    from pathlib import Path as _P

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.tts_turbo import Conditionals

    ckpt_dir = _P(nano_ckpt_dir(require=("t3_nano_v1.safetensors", "conds.pt",
                                         "vocab.json")))
    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_small"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False

    t3 = T3(hp)
    t3_state = load_file(ckpt_dir / "t3_nano_v1.safetensors")
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    del t3.tfmr.wte
    t3.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    conds = Conditionals.load(ckpt_dir / "conds.pt",
                              map_location=torch.device("cpu")).to("cpu")
    return NanoT3Only(t3, tokenizer, conds)
