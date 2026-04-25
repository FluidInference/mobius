"""Shared CLI helpers for per-language PocketTTS CoreML conversion.

Upstream `kyutai/pocket-tts` publishes language packs under
`languages/<id>/` on HuggingFace. The list below matches exactly what
upstream ships (see kyutai-labs/pocket-tts issue #118 and the `languages/`
tree on HF). Keep in sync with the Swift `PocketTtsLanguage` enum in
FluidAudio.
"""
from __future__ import annotations

import argparse
import os


# Exact folder names published under kyutai/pocket-tts/languages/ on HF.
# Ordered: English first (default), then 6-layer variants, then 24-layer.
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "english",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "french_24l",
    "german_24l",
    "italian_24l",
    "portuguese_24l",
    "spanish_24l",
)

DEFAULT_LANGUAGE = "english"


def add_language_arg(parser: argparse.ArgumentParser) -> None:
    """Register the shared `--language` flag on a converter's argparser."""
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default=DEFAULT_LANGUAGE,
        help=(
            "Which upstream language pack to convert. Must match a folder "
            "under kyutai/pocket-tts/languages/ on HuggingFace. "
            f"Default: {DEFAULT_LANGUAGE}."
        ),
    )


def parse_language() -> str:
    """Parse only `--language` (for scripts with no other args)."""
    parser = argparse.ArgumentParser()
    add_language_arg(parser)
    args, _ = parser.parse_known_args()
    return args.language


def build_output_dir(coreml_dir: str, language: str) -> str:
    """`<coreml_dir>/build/<language>` — created on demand by the caller."""
    return os.path.join(coreml_dir, "build", language)
