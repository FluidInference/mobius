#!/usr/bin/env python3
"""Test stateless decoder in PyTorch vs CoreML."""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import coremltools as ct
from transformers import AutoModelForSpeechSeq2Seq

sys.path.insert(0, str(Path(__file__).parent / "f16"))
from cohere_mel_spectrogram import CohereMelSpectrogram

ENGLISH_PROMPT = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
EOS_TOKEN_ID = 3


def test_pytorch_stateless(encoder_hidden_np):
    """Test stateless decoding in PyTorch."""
    print("="*70)
    print("PyTorch Stateless Decoder Test")
    print("="*70)

    # Load model
    print("\n[1/2] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Use CoreML encoder output (already has projection applied)
    encoder_hidden = torch.from_numpy(encoder_hidden_np).float()
    print(f"   Encoder hidden: {encoder_hidden.shape}")

    # Decode stateless (reprocess all tokens each step)
    print("[2/2] Decoding (stateless - no cache)...")
    tokens = []
    all_input_ids = [ENGLISH_PROMPT[0]]  # Start with BOS

    cross_mask = torch.ones(1, encoder_hidden.shape[1])

    for step in range(108):
        # Feed ALL tokens so far (no cache!)
        input_ids_tensor = torch.tensor([all_input_ids], dtype=torch.long)
        positions = torch.arange(len(all_input_ids)).unsqueeze(0)

        with torch.no_grad():
            decoder_outputs, _ = model.transf_decoder(
                input_ids=input_ids_tensor,
                positions=positions,
                encoder_hidden_states=encoder_hidden,
                self_attention_mask=None,  # Causal mask handled internally
                cross_attention_mask=cross_mask,
                past_key_values=None,  # No cache!
                cache_position=None,
                kv_seq_len=None,
            )

        # Get logits for LAST token
        last_hidden = decoder_outputs[:, -1:, :]
        logits = model.log_softmax(last_hidden).squeeze(1)
        next_token = int(torch.argmax(logits[0]))

        # Add to sequence
        if step < len(ENGLISH_PROMPT) - 1:
            # Still feeding prompt
            all_input_ids.append(ENGLISH_PROMPT[step + 1])
        else:
            # Generate
            all_input_ids.append(next_token)
            tokens.append(next_token)

            if next_token == EOS_TOKEN_ID:
                break

    print(f"   Generated {len(tokens)} tokens")

    return tokens


def test_coreml_stateless(encoder_hidden):
    """Test CoreML stateless decoder."""
    print("\n" + "="*70)
    print("CoreML Stateless Decoder Test")
    print("="*70)

    print("\n[1/2] Loading CoreML decoder...")
    decoder = ct.models.MLModel("build-stateless-nn/cohere_decoder_stateless.mlpackage")

    print("[2/2] Decoding...")
    tokens = []
    all_input_ids = [ENGLISH_PROMPT[0]]  # Start with BOS

    cross_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float32)

    for step in range(108):
        # Feed ALL tokens so far
        input_ids = np.array([all_input_ids], dtype=np.int32)

        decoder_out = decoder.predict({
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden.astype(np.float32),
            "cross_attention_mask": cross_mask,
        })

        next_token = int(np.argmax(decoder_out["logits"][0]))

        # Add to sequence
        if step < len(ENGLISH_PROMPT) - 1:
            # Still feeding prompt
            all_input_ids.append(ENGLISH_PROMPT[step + 1])
        else:
            # Generate
            all_input_ids.append(next_token)
            tokens.append(next_token)

            if next_token == EOS_TOKEN_ID:
                break

    print(f"   Generated {len(tokens)} tokens")

    return tokens


def tokens_to_text(tokens, vocab):
    """Convert tokens to text."""
    text_tokens = []
    for token_id in tokens:
        if token_id <= 4 or token_id == EOS_TOKEN_ID:
            continue
        token_str = vocab.get(token_id, "")
        if token_str.startswith("<|"):
            continue
        text_tokens.append(token_str)

    text = "".join(text_tokens).replace("▁", " ").strip()
    return text


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/kikow/brandon/voicelink/FluidAudio/test_sentence_int8.wav"

    # Load vocab
    vocab_path = Path("f16/vocab.json")
    with open(vocab_path) as f:
        vocab = {int(k): v for k, v in json.load(f).items()}

    # Encode audio with CoreML encoder (has projection built-in)
    print("="*70)
    print("Encoding Audio (CoreML)")
    print("="*70)

    encoder = ct.models.MLModel("f16/cohere_encoder.mlpackage")

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    mel_processor = CohereMelSpectrogram()
    mel = mel_processor(audio)

    if mel.shape[2] > 3500:
        mel_padded = mel[:, :, :3500]
        actual_length = 3500
    else:
        mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])))
        actual_length = mel.shape[2]

    encoder_out = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([actual_length], dtype=np.int32),
    })
    encoder_hidden = encoder_out["hidden_states"]

    print(f"Encoder output: {encoder_hidden.shape}")

    # Test PyTorch
    tokens_pytorch = test_pytorch_stateless(encoder_hidden)
    text_pytorch = tokens_to_text(tokens_pytorch, vocab)

    # Test CoreML
    tokens_coreml = test_coreml_stateless(encoder_hidden)
    text_coreml = tokens_to_text(tokens_coreml, vocab)

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nPyTorch tokens: {len(tokens_pytorch)}")
    print(f"PyTorch text:   {text_pytorch}")
    print(f"\nCoreML tokens:  {len(tokens_coreml)}")
    print(f"CoreML text:    {text_coreml}")

    match = "✅" if text_pytorch == text_coreml else "❌"
    print(f"\nMatch: {match}")

    if text_pytorch != text_coreml:
        print("\n❌ CoreML output differs from PyTorch!")
        print("This indicates a conversion issue, not a prompt feeding bug.")
    else:
        print("\n✅ CoreML matches PyTorch perfectly!")


if __name__ == "__main__":
    main()
