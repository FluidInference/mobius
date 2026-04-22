#!/usr/bin/env python3
"""Complete inference example for Cohere Transcribe CoreML models.

Fixed version addressing:
1. Feature extraction matches the official CohereAsrFeatureExtractor
   (was a broken from-scratch reimplementation; see tools/cohere_features_v2.py
   and tests/test-feature-parity.py for the parity proof).
2. Cross-attention mask respects encoder `feature_length` — padded encoder
   frames are masked with -inf rather than attended to.
3. Greedy decode gains an optional repetition penalty / no-repeat-ngram to
   break decoder repetition loops ("the the the", etc.).

Usage:
    python example_inference.py audio.wav
    python example_inference.py audio.wav --language ja
    python example_inference.py audio.wav --repetition-penalty 1.2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

# The corrected feature extractor lives next to this script as
# ``cohere_mel_spectrogram.py`` (see cohere-pytorch/processing_cohere_asr.py
# for the PyTorch reference).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cohere_mel_spectrogram import (  # noqa: E402
    CohereMelSpectrogram,
    pad_or_truncate_to_fixed,
)

# Language-specific prompts (first 10 tokens determine language).
# These IDs come from the original vocab.json; if future decoder exports
# change, re-verify against the PyTorch reference prompt.
LANGUAGE_PROMPTS = {
    "en": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13],
    "es": [13764, 7, 4, 16, 169, 169, 5, 9, 11, 13],
    "fr": [13764, 7, 4, 16, 69, 69, 5, 9, 11, 13],
    "de": [13764, 7, 4, 16, 76, 76, 5, 9, 11, 13],
    "it": [13764, 7, 4, 16, 97, 97, 5, 9, 11, 13],
    "pt": [13764, 7, 4, 16, 149, 149, 5, 9, 11, 13],
    "pl": [13764, 7, 4, 16, 148, 148, 5, 9, 11, 13],
    "nl": [13764, 7, 4, 16, 60, 60, 5, 9, 11, 13],
    "sv": [13764, 7, 4, 16, 173, 173, 5, 9, 11, 13],
    "tr": [13764, 7, 4, 16, 186, 186, 5, 9, 11, 13],
    "ru": [13764, 7, 4, 16, 155, 155, 5, 9, 11, 13],
    "zh": [13764, 7, 4, 16, 50, 50, 5, 9, 11, 13],
    "ja": [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13],
    "ko": [13764, 7, 4, 16, 110, 110, 5, 9, 11, 13],
}

EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0

# Encoder subsampling factor from mel frames → encoder frames.
# 3500 mel frames → 438 encoder frames (see exports/export-encoder.py).
MEL_FRAMES_FIXED = 3500
ENCODER_FRAMES_FIXED = 438


def load_models(model_dir: str = "."):
    model_dir = Path(model_dir)
    encoder_path = model_dir / "cohere_encoder.mlpackage"
    decoder_path = model_dir / "cohere_decoder_stateful.mlpackage"
    print(f"Loading encoder from {encoder_path}...")
    encoder = ct.models.MLModel(str(encoder_path))
    print(f"Loading decoder from {decoder_path}...")
    decoder = ct.models.MLModel(str(decoder_path))
    return encoder, decoder


def load_vocab(vocab_path: str) -> dict[int, str]:
    with open(vocab_path) as fh:
        vocab = json.load(fh)
    return {int(k): v for k, v in vocab.items()}


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(
            np.float32
        )
    return audio


def encode_audio(encoder, mel_processor: CohereMelSpectrogram, audio: np.ndarray):
    """Compute mel features and run the encoder.

    Returns (encoder_hidden, encoder_valid_frames) where encoder_valid_frames
    is the number of encoder output frames that correspond to real audio
    (everything beyond that is encoded padding and MUST be masked in
    cross-attention).
    """
    mel, valid_mel_len = mel_processor(audio)
    mel_fixed, feature_length = pad_or_truncate_to_fixed(
        mel, valid_mel_len, fixed_frames=MEL_FRAMES_FIXED
    )

    enc_out = encoder.predict({
        "input_features": mel_fixed.astype(np.float32),
        "feature_length": np.array([feature_length], dtype=np.int32),
    })
    encoder_hidden = enc_out["hidden_states"]

    # Encoder subsamples by MEL_FRAMES_FIXED / ENCODER_FRAMES_FIXED.
    # Map valid mel frames -> valid encoder frames (ceil to avoid truncating
    # the trailing partial frame).
    ratio = MEL_FRAMES_FIXED / ENCODER_FRAMES_FIXED  # ≈ 7.99
    encoder_valid = int(np.ceil(feature_length / ratio))
    encoder_valid = max(1, min(encoder_valid, encoder_hidden.shape[1]))
    return encoder_hidden, encoder_valid


def build_cross_attention_mask(encoder_seq_len: int, encoder_valid: int) -> np.ndarray:
    """Additive attention mask: 0.0 for valid positions, -inf for padded ones.

    Shape matches the decoder's expected cross_attention_mask: (1, 1, 1, E).
    """
    mask = np.zeros((1, 1, 1, encoder_seq_len), dtype=np.float16)
    if encoder_valid < encoder_seq_len:
        # Use a large negative number rather than -inf to avoid fp16 NaN
        # propagation in any matmul-then-softmax path that sums masks.
        mask[:, :, :, encoder_valid:] = -1.0e4
    return mask


def decode_with_stateful(
    decoder,
    encoder_hidden: np.ndarray,
    encoder_valid: int,
    prompt_ids: list[int],
    max_tokens: int = 108,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
) -> list[int]:
    """Greedy decode with optional repetition penalty / no-repeat-ngram.

    Args:
        repetition_penalty: 1.0 disables. Values >1 (e.g. 1.1 .. 1.3)
            divide the logit of already-emitted tokens, following the
            CTRL / HuggingFace convention.
        no_repeat_ngram: 0 disables. Otherwise, any n-gram already
            present in the output is prevented from being completed.
    """
    state = decoder.make_state()

    enc_seq_len = encoder_hidden.shape[1]
    cross_mask = build_cross_attention_mask(enc_seq_len, encoder_valid)

    all_tokens: list[int] = []  # every step, for rep penalty / n-gram
    output_tokens: list[int] = []  # tokens AFTER the language prompt
    last_token = None

    for step in range(max_tokens):
        current_token = prompt_ids[step] if step < len(prompt_ids) else last_token

        input_id = np.array([[current_token]], dtype=np.int32)
        attention_mask = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
        position_ids = np.array([[step]], dtype=np.int32)

        decoder_out = decoder.predict(
            {
                "input_id": input_id,
                "encoder_hidden_states": encoder_hidden.astype(np.float16),
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_mask,
                "position_ids": position_ids,
            },
            state=state,
        )

        logits = decoder_out["logits"][0].astype(np.float32).copy()

        # --- Repetition penalty ---
        if repetition_penalty != 1.0 and all_tokens:
            seen = np.array(sorted(set(all_tokens)))
            pos = logits[seen] >= 0
            logits[seen] = np.where(
                pos,
                logits[seen] / repetition_penalty,
                logits[seen] * repetition_penalty,
            )

        # --- No-repeat n-gram ---
        if no_repeat_ngram > 0 and len(all_tokens) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            prefix = tuple(all_tokens[-(n - 1):]) if n > 1 else ()
            forbidden: set[int] = set()
            # Scan history for any occurrence of `prefix` and forbid the
            # token that followed it.
            hist = all_tokens
            for i in range(len(hist) - (n - 1)):
                if tuple(hist[i : i + n - 1]) == prefix:
                    nxt_idx = i + n - 1
                    if nxt_idx < len(hist):
                        forbidden.add(hist[nxt_idx])
            for tok in forbidden:
                logits[tok] = -1e9

        next_token = int(np.argmax(logits))
        last_token = next_token
        all_tokens.append(next_token)

        if step >= len(prompt_ids) - 1:
            output_tokens.append(next_token)
            if next_token == EOS_TOKEN_ID:
                break

    return output_tokens


_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def tokens_to_text(tokens: list[int], vocab: dict[int, str]) -> str:
    """Detokenize Cohere/SentencePiece ids to text.

    Handles ``<0xHH>`` byte-fallback pieces so CJK output (which the tokenizer
    emits as a run of UTF-8 bytes) comes out as real characters. Runs of
    byte-fallback pieces are buffered and flushed as a single UTF-8 decode
    (errors="replace") whenever a non-byte piece or the end is reached.
    """
    out: list[str] = []
    byte_buf: list[int] = []

    def flush_bytes() -> None:
        if byte_buf:
            out.append(bytes(byte_buf).decode("utf-8", errors="replace"))
            byte_buf.clear()

    for tok in tokens:
        if tok <= 4 or tok == EOS_TOKEN_ID:
            continue
        s = vocab.get(tok, "")
        if s.startswith("<|"):
            continue
        m = _BYTE_FALLBACK_RE.match(s)
        if m is not None:
            byte_buf.append(int(m.group(1), 16))
            continue
        flush_bytes()
        out.append(s)

    flush_bytes()
    return "".join(out).replace("\u2581", " ").strip()


def transcribe(
    audio_path: str,
    model_dir: str = ".",
    language: str = "en",
    max_tokens: int = 108,
    repetition_penalty: float = 1.1,
    no_repeat_ngram: int = 3,
    verbose: bool = True,
) -> str:
    encoder, decoder = load_models(model_dir)
    vocab = load_vocab(str(Path(model_dir) / "vocab.json"))

    if verbose:
        print("[1/4] Loading audio...")
    audio = load_audio(audio_path)
    if verbose:
        print(f"   Duration: {len(audio) / 16000:.2f}s")

    if verbose:
        print("[2/4] Encoding audio...")
    mel_processor = CohereMelSpectrogram()
    encoder_hidden, encoder_valid = encode_audio(encoder, mel_processor, audio)
    if verbose:
        print(f"   Encoder output: {encoder_hidden.shape}  valid_frames={encoder_valid}")

    if verbose:
        print("[3/4] Decoding...")
    prompt_ids = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])
    tokens = decode_with_stateful(
        decoder,
        encoder_hidden,
        encoder_valid,
        prompt_ids,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram=no_repeat_ngram,
    )
    if verbose:
        print(f"   Generated {len(tokens)} tokens")

    if verbose:
        print("[4/4] Converting to text...")
    return tokens_to_text(tokens, vocab)


def main():
    ap = argparse.ArgumentParser(description="Cohere Transcribe CoreML inference")
    ap.add_argument("audio")
    ap.add_argument("--model-dir", default=".")
    ap.add_argument("--language", "-l", default="en", choices=list(LANGUAGE_PROMPTS))
    ap.add_argument("--max-tokens", type=int, default=108)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--quiet", "-q", action="store_true")
    args = ap.parse_args()

    text = transcribe(
        args.audio,
        model_dir=args.model_dir,
        language=args.language,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print()
        print("=" * 70)
        print("TRANSCRIPTION")
        print("=" * 70)
    print(text)


if __name__ == "__main__":
    main()
