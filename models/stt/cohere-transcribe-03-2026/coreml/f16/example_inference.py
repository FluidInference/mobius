#!/usr/bin/env python3
"""Complete inference example for Cohere Transcribe CoreML models.

This example demonstrates:
1. Loading CoreML models from HuggingFace
2. Audio preprocessing with mel spectrogram
3. Encoding audio to hidden states
4. Decoding with stateful decoder
5. Token-to-text conversion

Requirements:
    pip install coremltools numpy soundfile huggingface-hub

Usage:
    python example_inference.py audio.wav
    python example_inference.py audio.wav --language ja  # Japanese
    python example_inference.py audio.wav --max-tokens 256  # Longer output
"""

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("Error: soundfile not installed. Install with: pip install soundfile")
    sys.exit(1)

from cohere_mel_spectrogram import CohereMelSpectrogram

# Language-specific prompts (first 10 tokens determine language)
LANGUAGE_PROMPTS = {
    "en": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13],  # English
    "es": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 14],  # Spanish
    "fr": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 15],  # French
    "de": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 16],  # German
    "it": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 17],  # Italian
    "pt": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 18],  # Portuguese
    "pl": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 19],  # Polish
    "nl": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 20],  # Dutch
    "sv": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 21],  # Swedish
    "tr": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 22],  # Turkish
    "ru": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 23],  # Russian
    "zh": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 24],  # Chinese
    "ja": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 25],  # Japanese
    "ko": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 26],  # Korean
}

# Special tokens
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0


def load_models(model_dir="."):
    """Load CoreML models from directory.

    Args:
        model_dir: Directory containing the model files (.mlpackage format)

    Returns:
        (encoder, decoder) tuple
    """
    model_dir = Path(model_dir)

    print(f"Loading models from {model_dir}...")
    print("(First load takes ~20s for ANE compilation, then cached)")

    # ML Program models must use .mlpackage format
    encoder_path = model_dir / "cohere_encoder.mlpackage"
    decoder_path = model_dir / "cohere_decoder_stateful.mlpackage"

    encoder = ct.models.MLModel(str(encoder_path))
    decoder = ct.models.MLModel(str(decoder_path))
    print("✓ Models loaded")

    return encoder, decoder


def load_vocab(vocab_path="vocab.json"):
    """Load vocabulary mapping.

    Args:
        vocab_path: Path to vocab.json

    Returns:
        Dictionary mapping token IDs to strings
    """
    with open(vocab_path) as f:
        vocab = json.load(f)
    return {int(k): v for k, v in vocab.items()}


def load_audio(audio_path, target_sr=16000):
    """Load audio file and resample to 16kHz.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)

    Returns:
        Audio array (float32, mono, 16kHz)
    """
    audio, sr = sf.read(audio_path, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed (simple method, consider librosa for better quality)
    if sr != target_sr:
        # Simple resampling (use librosa.resample for production)
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * target_sr / sr)),
            np.arange(len(audio)),
            audio,
        )

    return audio


def encode_audio(encoder, mel_processor, audio):
    """Encode audio to hidden states.

    Args:
        encoder: CoreML encoder model
        mel_processor: CohereMelSpectrogram instance
        audio: Audio array (float32, mono, 16kHz)

    Returns:
        Encoder hidden states (1, 438, 1024)
    """
    # Compute mel spectrogram
    mel = mel_processor(audio)

    # Pad or truncate to 3500 frames (35 seconds)
    if mel.shape[2] > 3500:
        mel_padded = mel[:, :, :3500]
        actual_length = 3500
    else:
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])))
        actual_length = mel.shape[2]

    # Encode
    encoder_out = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([actual_length], dtype=np.int32),
    })

    # Extract hidden states
    encoder_hidden = encoder_out["hidden_states"]

    return encoder_hidden


def decode_with_stateful(decoder, encoder_hidden, prompt_ids, max_tokens=108):
    """Decode hidden states to tokens using stateful decoder.

    Args:
        decoder: CoreML stateful decoder model
        encoder_hidden: Encoder output (1, 438, 1024)
        prompt_ids: Language prompt (list of 10 token IDs)
        max_tokens: Maximum tokens to generate (default: 108)

    Returns:
        List of generated token IDs
    """
    # Initialize decoder state
    state = decoder.make_state()

    # Prepare cross-attention mask
    enc_seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, enc_seq_len), dtype=np.float16)

    # Generation loop
    tokens = []
    last_token = None

    for step in range(max_tokens):
        # Feed prompt tokens for first 10 steps
        if step < len(prompt_ids):
            current_token = prompt_ids[step]
        else:
            current_token = last_token

        # Prepare decoder inputs
        input_id = np.array([[current_token]], dtype=np.int32)
        attention_mask = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
        position_ids = np.array([[step]], dtype=np.int32)

        # Run decoder
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

        # Get next token
        logits = decoder_out["logits"]
        next_token = int(np.argmax(logits[0]))
        last_token = next_token

        # Collect tokens after prompt
        if step >= len(prompt_ids) - 1:
            tokens.append(next_token)

            # Stop on EOS
            if next_token == EOS_TOKEN_ID:
                break

    return tokens


def tokens_to_text(tokens, vocab):
    """Convert token IDs to text.

    Args:
        tokens: List of token IDs
        vocab: Vocabulary dictionary

    Returns:
        Decoded text string
    """
    text_tokens = []
    for token_id in tokens:
        # Skip special tokens
        if token_id <= 4 or token_id == EOS_TOKEN_ID:
            continue

        token_str = vocab.get(token_id, "")

        # Skip control tokens
        if token_str.startswith("<|"):
            continue

        text_tokens.append(token_str)

    # Join and clean up
    text = "".join(text_tokens)
    text = text.replace("▁", " ")  # SentencePiece space marker
    text = text.strip()

    return text


def transcribe(
    audio_path,
    model_dir=".",
    language="en",
    max_tokens=108,
    verbose=True,
):
    """Complete transcription pipeline.

    Args:
        audio_path: Path to audio file
        model_dir: Directory containing CoreML models
        language: Language code (en, es, fr, etc.)
        max_tokens: Maximum tokens to generate
        verbose: Print progress messages

    Returns:
        Transcribed text string
    """
    if verbose:
        print(f"Transcribing: {audio_path}")
        print(f"Language: {language}")
        print()

    # Load models
    encoder, decoder = load_models(model_dir)
    vocab = load_vocab(Path(model_dir) / "vocab.json")

    # Load audio
    if verbose:
        print("[1/4] Loading audio...")
    audio = load_audio(audio_path)
    duration = len(audio) / 16000
    if verbose:
        print(f"   Duration: {duration:.2f}s")

    # Encode
    if verbose:
        print("[2/4] Encoding audio...")
    mel_processor = CohereMelSpectrogram()
    encoder_hidden = encode_audio(encoder, mel_processor, audio)
    if verbose:
        print(f"   Encoder output: {encoder_hidden.shape}")

    # Decode
    if verbose:
        print("[3/4] Decoding...")
    prompt_ids = LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])
    tokens = decode_with_stateful(decoder, encoder_hidden, prompt_ids, max_tokens)
    if verbose:
        print(f"   Generated {len(tokens)} tokens")

    # Convert to text
    if verbose:
        print("[4/4] Converting to text...")
    text = tokens_to_text(tokens, vocab)

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Cohere Transcribe CoreML"
    )
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument(
        "--model-dir",
        default=".",
        help="Directory containing CoreML models (default: current directory)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="en",
        choices=list(LANGUAGE_PROMPTS.keys()),
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=108,
        help="Maximum tokens to generate (default: 108)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print transcription result",
    )

    args = parser.parse_args()

    try:
        text = transcribe(
            args.audio,
            model_dir=args.model_dir,
            language=args.language,
            max_tokens=args.max_tokens,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print()
            print("=" * 70)
            print("TRANSCRIPTION")
            print("=" * 70)
        print(text)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
