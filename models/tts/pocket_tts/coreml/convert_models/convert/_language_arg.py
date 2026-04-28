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


# --- compute precision / units -------------------------------------------------
#
# Defaults switched from (FLOAT32, CPU_AND_GPU) → (FLOAT16, ALL) so the
# converted mlpackages can be dispatched to the Apple Neural Engine where
# possible. ANE requires FLOAT16; ops it can't run fall back to GPU/CPU
# automatically when compute_units=ALL. FP32+CPU_AND_GPU is preserved as an
# escape hatch via flags, since it was the original release behavior.

_PRECISION_CHOICES = ("fp16", "fp32")
_DEFAULT_PRECISION = "fp16"

_UNITS_CHOICES = ("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY")
_DEFAULT_UNITS = "ALL"


def add_compute_args(parser: argparse.ArgumentParser) -> None:
    """Register `--compute-precision` and `--compute-units` flags."""
    parser.add_argument(
        "--compute-precision",
        choices=_PRECISION_CHOICES,
        default=_DEFAULT_PRECISION,
        help=(
            "CoreML compute precision. fp16 is required for the Apple "
            f"Neural Engine. Default: {_DEFAULT_PRECISION}."
        ),
    )
    parser.add_argument(
        "--compute-units",
        choices=_UNITS_CHOICES,
        default=_DEFAULT_UNITS,
        help=(
            "CoreML compute units used for the post-save load test. ALL "
            "lets CoreML route ops to ANE when supported and fall back to "
            f"GPU/CPU otherwise. Default: {_DEFAULT_UNITS}."
        ),
    )


def resolve_compute_precision(precision: str):
    """Map our CLI string → coremltools.precision enum (lazy import)."""
    import coremltools as ct  # noqa: WPS433  (function-local import for CLI helper)
    if precision == "fp16":
        return ct.precision.FLOAT16
    if precision == "fp32":
        return ct.precision.FLOAT32
    raise ValueError(f"unknown compute precision: {precision}")


def resolve_compute_units(units: str):
    """Map our CLI string → coremltools.ComputeUnit enum (lazy import)."""
    import coremltools as ct  # noqa: WPS433
    return getattr(ct.ComputeUnit, units)
