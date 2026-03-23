"""Generate speech using the native NeMo/PyTorch MagpieTTS pipeline.

Used to produce a reference WAV for comparison against CoreML output.
"""
import argparse
import time

import numpy as np
import soundfile as sf
import torch


def generate(
    text: str,
    speaker: int = 0,
    language: str = "en",
    output_path: str = "pytorch_output.wav",
    seed: int = 42,
    use_cfg: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    from nemo.collections.tts.models import MagpieTTSModel
    from nemo.collections.tts.parts.utils.tts_dataset_utils import (
        chunk_and_tokenize_text_by_sentence,
    )

    print("Loading MagpieTTS model...")
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    # Resolve tokenizer
    language_tokenizer_map = {
        "en": ["english_phoneme", "english"],
        "es": ["spanish_phoneme", "spanish"],
        "de": ["german_phoneme", "german"],
        "fr": ["french_chartokenizer", "french"],
        "zh": ["mandarin_phoneme", "mandarin"],
        "ja": ["japanese_phoneme", "japanese"],
        "hi": ["hindi_chartokenizer", "hindi"],
        "it": ["italian_phoneme", "italian"],
        "vi": ["vietnamese_phoneme", "vietnamese"],
    }

    available_tokenizers = list(model.tokenizer.tokenizers.keys())
    tokenizer_name = available_tokenizers[0]
    if language in language_tokenizer_map:
        for candidate in language_tokenizer_map[language]:
            if candidate in available_tokenizers:
                tokenizer_name = candidate
                break

    print(f"Text: '{text}'")
    print(f"Speaker: {speaker}, Language: {language}, Tokenizer: {tokenizer_name}")

    # Tokenize text — returns lists (one per chunk)
    token_chunks, token_lens, text_chunks = chunk_and_tokenize_text_by_sentence(
        text=text,
        tokenizer_name=tokenizer_name,
        text_tokenizer=model.tokenizer,
        eos_token_id=model.eos_id,
    )

    # Use first chunk (short text fits in one)
    tokens_tensor = token_chunks[0]  # already a tensor
    token_len = token_lens[0]
    print(f"Tokens: {token_len}, shape: {tokens_tensor.shape}")

    # Build batch dict matching prepare_context_tensors expectations
    batch = {
        "text": tokens_tensor.unsqueeze(0) if tokens_tensor.ndim == 1 else tokens_tensor,
        "text_lens": torch.tensor([token_len], dtype=torch.long),
        "speaker_indices": torch.tensor([speaker], dtype=torch.long),
    }

    print(f"\nGenerating (use_cfg={use_cfg})...")
    start = time.time()

    with torch.no_grad():
        output = model.infer_batch(
            batch,
            use_cfg=use_cfg,
            use_local_transformer_for_inference=True,
        )

    elapsed = time.time() - start

    # Extract audio from output
    print(f"Output type: {type(output).__name__}")

    # Inspect all fields
    if hasattr(output, '__dict__'):
        for k, v in vars(output).items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            elif isinstance(v, (int, float)):
                print(f"  {k}: {v}")
            elif v is not None:
                print(f"  {k}: type={type(v).__name__}")
    elif isinstance(output, dict):
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    # Find audio — try common attribute names
    audio = None
    for attr in ["predicted_audio", "audio", "audio_pred", "audio_output", "waveform"]:
        if hasattr(output, attr) and getattr(output, attr) is not None:
            audio = getattr(output, attr)
            print(f"\nUsing attribute '{attr}' for audio")
            break

    # Fall back to decoding codec tokens
    if audio is None:
        token_preds = getattr(output, "token_predictions", None)
        if isinstance(output, dict):
            token_preds = output.get("token_predictions", token_preds)
        if token_preds is not None:
            print("\nDecoding codec tokens to audio via NanoCodec...")
            print(f"  token_predictions shape: {tuple(token_preds.shape)}")
            audio = model.codec_model.decode(tokens=token_preds.long())
            if isinstance(audio, tuple):
                audio = audio[0]
            print(f"  Decoded audio shape: {tuple(audio.shape)}")

    if audio is None:
        raise ValueError(f"Could not find audio in output: {type(output)}")

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if hasattr(audio, 'squeeze'):
        audio = audio.squeeze()

    # Normalize same as CoreML pipeline
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    sample_rate = 22050
    sf.write(output_path, audio, sample_rate)
    duration = len(audio) / sample_rate
    print(f"\nSaved to {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Generation time: {elapsed:.2f}s")
    print(f"RTF: {elapsed / duration:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech with PyTorch MagpieTTS")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker index (0-4)")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--output", type=str, default="pytorch_output.wav")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cfg", action="store_true")
    args = parser.parse_args()

    generate(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        output_path=args.output,
        seed=args.seed,
        use_cfg=not args.no_cfg,
    )
