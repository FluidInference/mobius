"""Transcribe a generated PocketTTS wav with Whisper and print the result.

Used after ``convert_all_languages.sh`` + ``generate_coreml_v4.py`` to sanity
check that a non-English language pack produces intelligible speech.

Usage:
    uv run --extra verify python coreml/verify_with_whisper.py \
        --audio build/spanish/verify.wav --language es \
        [--reference "Hola, este es un sistema de síntesis de voz."]

Prefers ``mlx-whisper`` (fast on Apple Silicon); falls back to
``openai-whisper`` if mlx-whisper isn't installed.
"""

from __future__ import annotations

import argparse
import os
import sys


DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
# Fallback model id for openai-whisper when mlx-whisper is unavailable.
FALLBACK_MODEL = "large-v3"


def _transcribe_mlx(audio_path: str, language: str, model: str) -> dict:
    import mlx_whisper  # type: ignore

    return mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model,
        language=language,
        task="transcribe",
        fp16=True,
        verbose=False,
    )


def _transcribe_openai(audio_path: str, language: str, model: str) -> dict:
    import whisper  # type: ignore

    wh_model = whisper.load_model(model)
    return wh_model.transcribe(audio_path, language=language, task="transcribe")


def transcribe(audio_path: str, language: str, model: str | None) -> str:
    """Try mlx-whisper first, fall back to openai-whisper."""
    chosen = model or DEFAULT_MODEL
    try:
        result = _transcribe_mlx(audio_path, language, chosen)
    except ImportError:
        print("[verify] mlx-whisper not installed, falling back to openai-whisper")
        result = _transcribe_openai(audio_path, language, model or FALLBACK_MODEL)
    return result.get("text", "").strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Whisper-transcribe a PocketTTS wav.")
    parser.add_argument("--audio", required=True, help="Path to the generated .wav")
    parser.add_argument(
        "--language",
        required=True,
        help="ISO-639-1 code (es, fr, de, it, pt, en, ...) for Whisper language hint",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the Whisper model id. Default: "
            f"'{DEFAULT_MODEL}' for mlx-whisper, '{FALLBACK_MODEL}' for openai-whisper"
        ),
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference text to print alongside the transcription",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if not os.path.isfile(args.audio):
        print(f"[verify] ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    print(f"[verify] audio    : {args.audio}")
    print(f"[verify] language : {args.language}")
    if args.reference:
        print(f"[verify] reference: {args.reference}")

    text = transcribe(args.audio, args.language, args.model)
    print(f"[verify] whisper  : {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
