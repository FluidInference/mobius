#!/usr/bin/env python3
"""Convert CharsiuG2P ByT5 multilingual G2P model to CoreML.

Path: PyTorch -> ONNX (torch.onnx.export) -> CoreML (coremltools 7.x)

Source model: charsiu/g2p_multilingual_byT5_tiny_16_layers_100
Architecture: ByT5 encoder-decoder with byte-level tokenization
Languages: 100+ (9 used by Kokoro TTS)

Produces:
  - G2PEncoder.mlpackage: encoder (input_ids, attention_mask -> last_hidden_state)
  - G2PDecoder.mlpackage: decoder (decoder_input_ids, encoder_hidden_states,
                                    encoder_attention_mask -> logits)
  - g2p_config.json: token IDs and sequence length limits

Requires coremltools>=7.0,<8.0 for ONNX->CoreML conversion support.

Usage:
  uv sync
  uv run python convert-to-coreml.py
  uv run python convert-to-coreml.py --output-dir ./build/custom
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

import coremltools as ct

MODEL_NAME = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
MAX_INPUT_LEN = 64
MAX_OUTPUT_LEN = 128


class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


class DecoderWrapper(nn.Module):
    def __init__(self, decoder, lm_head):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask):
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self.lm_head(out.last_hidden_state)


def export_onnx(model, output_dir: Path):
    """Export encoder and decoder to ONNX."""
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    hidden_size = model.config.d_model

    # Encoder
    print("Exporting encoder to ONNX...")
    enc = EncoderWrapper(model.encoder)
    enc.eval()
    seq_len = 16
    torch.onnx.export(
        enc,
        (torch.ones(1, seq_len, dtype=torch.long), torch.ones(1, seq_len, dtype=torch.long)),
        str(onnx_dir / "encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "last_hidden_state": {1: "seq_len"},
        },
        opset_version=14,
    )
    print(f"  Saved: {onnx_dir / 'encoder.onnx'}")

    # Decoder — trace with dec_len > 1 so T5 uses full self-attention path
    print("Exporting decoder to ONNX...")
    dec = DecoderWrapper(model.decoder, model.lm_head)
    dec.eval()
    dec_len = 8
    torch.onnx.export(
        dec,
        (
            torch.ones(1, dec_len, dtype=torch.long),
            torch.randn(1, seq_len, hidden_size),
            torch.ones(1, seq_len, dtype=torch.long),
        ),
        str(onnx_dir / "decoder.onnx"),
        input_names=["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "decoder_input_ids": {1: "dec_len"},
            "encoder_hidden_states": {1: "enc_len"},
            "encoder_attention_mask": {1: "enc_len"},
            "logits": {1: "dec_len"},
        },
        opset_version=14,
    )
    print(f"  Saved: {onnx_dir / 'decoder.onnx'}")
    return onnx_dir


def convert_onnx_to_coreml(onnx_dir: Path, output_dir: Path):
    """Convert ONNX models to CoreML using coremltools 7.x ONNX support."""

    # Encoder
    print("Converting encoder ONNX -> CoreML...")
    enc_mlmodel = ct.converters.convert(
        str(onnx_dir / "encoder.onnx"),
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    enc_path = output_dir / "G2PEncoder.mlpackage"
    enc_mlmodel.save(str(enc_path))
    print(f"  Saved: {enc_path}")
    del enc_mlmodel

    # Decoder
    print("Converting decoder ONNX -> CoreML...")
    dec_mlmodel = ct.converters.convert(
        str(onnx_dir / "decoder.onnx"),
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    dec_path = output_dir / "G2PDecoder.mlpackage"
    dec_mlmodel.save(str(dec_path))
    print(f"  Saved: {dec_path}")
    del dec_mlmodel


def verify_coreml(output_dir: Path, model, tokenizer):
    """Verify CoreML output matches PyTorch for a test word."""
    encoder_path = output_dir / "G2PEncoder.mlpackage"
    decoder_path = output_dir / "G2PDecoder.mlpackage"

    if not encoder_path.exists() or not decoder_path.exists():
        print("Skipping verification — CoreML models not found.")
        return

    print("\nVerifying end-to-end (CoreML vs PyTorch)...")
    encoder = ct.models.MLModel(str(encoder_path))
    decoder = ct.models.MLModel(str(decoder_path))

    test_word = "hello"
    lang = "eng-us"
    prefixed = f"<{lang}>: {test_word}"

    # PyTorch reference
    tok = tokenizer(prefixed, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        pt_out = model.generate(**tok, num_beams=1, max_length=50)
    pt_phonemes = tokenizer.decode(pt_out[0].tolist(), skip_special_tokens=True)

    # CoreML greedy decode
    tok_np = tokenizer(prefixed, add_special_tokens=False, return_tensors="np")
    input_ids = tok_np["input_ids"].astype(np.int32)
    attention_mask = tok_np["attention_mask"].astype(np.int32)

    enc_out = encoder.predict({"input_ids": input_ids, "attention_mask": attention_mask})
    enc_hidden = enc_out["last_hidden_state"]

    decoder_ids = np.array([[0]], dtype=np.int32)  # decoder start token
    output_tokens = []

    for _ in range(MAX_OUTPUT_LEN):
        dec_out = decoder.predict({
            "decoder_input_ids": decoder_ids,
            "encoder_hidden_states": enc_hidden,
            "encoder_attention_mask": attention_mask,
        })
        logits = dec_out["logits"]
        next_token = int(np.argmax(logits[0, -1, :]))

        if next_token == tokenizer.eos_token_id:
            break

        output_tokens.append(next_token)
        decoder_ids = np.array([[0] + output_tokens], dtype=np.int32)

    coreml_phonemes = tokenizer.decode(output_tokens, skip_special_tokens=True)

    print(f"  PyTorch: '{test_word}' -> {pt_phonemes}")
    print(f"  CoreML:  '{test_word}' -> {coreml_phonemes}")
    match = pt_phonemes.strip() == coreml_phonemes.strip()
    print(f"  {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print("  WARNING: CoreML output differs from PyTorch reference.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CharsiuG2P ByT5 multilingual G2P to CoreML"
    )
    parser.add_argument("--model", default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--output-dir", default="./build", help="Output directory")
    parser.add_argument("--skip-verify", action="store_true", help="Skip end-to-end verification")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    # Save config
    config = {
        "model_name": args.model,
        "max_input_len": MAX_INPUT_LEN,
        "max_output_len": MAX_OUTPUT_LEN,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "decoder_start_token_id": model.config.decoder_start_token_id,
    }
    config_path = output_dir / "g2p_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {config_path}")

    # Convert
    onnx_dir = export_onnx(model, output_dir)
    convert_onnx_to_coreml(onnx_dir, output_dir)

    # Verify
    if not args.skip_verify:
        verify_coreml(output_dir, model, tokenizer)

    print("\nDone! Outputs:")
    print(f"  {output_dir / 'G2PEncoder.mlpackage'}")
    print(f"  {output_dir / 'G2PDecoder.mlpackage'}")
    print(f"  {config_path}")


if __name__ == "__main__":
    main()
