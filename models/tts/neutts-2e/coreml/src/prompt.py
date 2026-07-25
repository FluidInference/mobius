"""Prompt construction for NeuTTS-2E (BPE input format).

Reimplements ``neutts.NeuTTS._apply_chat_template`` for the 2e/BPE path so the
conversion project does not depend on the ``neutts`` package (which drags in
phonemizer/espeak that the BPE model never uses).

Token layout produced (matches upstream exactly):

    <|TEXT_PROMPT_START|> {ref_text tokens} [<|EMOTION|>] {input_text tokens}
    <|TEXT_PROMPT_END|> <|SPEECH_GENERATION_START|> {ref speech-code tokens}

Generation then continues with sampled ``<|speech_N|>`` tokens until
``<|SPEECH_GENERATION_END|>``.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import torch

SAMPLE_DIR = Path(__file__).parents[1] / "samples"
SPEAKERS = ("emily", "paul", "sophie", "steven")
EMOTIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")

MAX_CONTEXT = 2048
SAMPLE_RATE = 24_000
HOP_LENGTH = 480  # audio samples per speech code at 24 kHz (50 codes/s)

_QUOTE_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.translate(_QUOTE_MAP))


def load_speaker(name: str) -> tuple[list[int], str]:
    """Return (ref_codes, ref_text) for one of the four fixed speakers."""
    if name not in SPEAKERS:
        raise ValueError(f"Unknown speaker '{name}'. Available: {list(SPEAKERS)}")
    codes = torch.load(SAMPLE_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
    text = (SAMPLE_DIR / f"{name}.txt").read_text().strip()
    return [int(c) for c in codes.reshape(-1).tolist()], text


def build_prompt_ids(
    tokenizer,
    text: str,
    speaker: str = "emily",
    emotion: str = "neutral",
) -> list[int]:
    """Token ids for the full generation prompt (mirrors upstream)."""
    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion '{emotion}'. Supported: {list(EMOTIONS)}")
    ref_codes, ref_text = load_speaker(speaker)

    text_prompt_start = tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_START|>")
    text_prompt_end = tokenizer.convert_tokens_to_ids("<|TEXT_PROMPT_END|>")
    speech_gen_start = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")

    ref_text = normalize_text(ref_text)
    text = normalize_text(text)
    if emotion == "neutral":
        # Single-pass encode so BPE resolves the boundary the same way upstream does.
        input_ids = tokenizer.encode(f"{ref_text} {text}", add_special_tokens=False)
    else:
        emotion_id = tokenizer.convert_tokens_to_ids(f"<|{emotion.upper()}|>")
        input_ids = (
            tokenizer.encode(ref_text, add_special_tokens=False)
            + [emotion_id]
            + tokenizer.encode(text, add_special_tokens=False)
        )

    codes_str = "".join(f"<|speech_{i}|>" for i in ref_codes)
    code_ids = tokenizer.encode(codes_str, add_special_tokens=False)

    return (
        [text_prompt_start]
        + input_ids
        + [text_prompt_end]
        + [speech_gen_start]
        + code_ids
    )


def extract_speech_codes(tokenizer, token_ids: list[int]) -> list[int]:
    """Map generated token ids back to NeuCodec code indices.

    Speech tokens occupy a contiguous id range, so decode via the id of
    ``<|speech_0|>`` rather than string round-tripping.
    """
    speech_0 = tokenizer.convert_tokens_to_ids("<|speech_0|>")
    speech_end = speech_0 + 65_536
    return [t - speech_0 for t in token_ids if speech_0 <= t < speech_end]
