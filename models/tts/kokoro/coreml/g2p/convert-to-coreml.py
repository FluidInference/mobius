#!/usr/bin/env python3
"""Convert the grapheme-to-phoneme BART model to CoreML format.

Produces:
  - G2PEncoder.mlpackage: encoder (input_ids -> encoder_hidden_states)
  - G2PDecoder.mlpackage: decoder step (decoder_input_ids + encoder output -> logits)
  - g2p_vocab.json: character-to-index mappings for tokenization/detokenization

The decoder uses a manual implementation to avoid HuggingFace ops (new_ones,
dynamic mask creation) that coremltools can't convert.

Usage:
  uv sync
  python convert_to_coreml.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import coremltools as ct
from transformers import BartForConditionalGeneration

model_dir = "PeterReid/graphemes_to_phonemes_en_us"

# --- Load model ---
print("Loading model...")
model = BartForConditionalGeneration.from_pretrained(model_dir)
model.eval()
config = model.config

# --- Save g2p_vocab ---
grapheme_chars = list(config.grapheme_chars[4:])  # strip leading "____"
phoneme_chars = list(config.phoneme_chars[4:])
special_vocab = ["<pad>", "<s>", "</s>", "<unk>"]
grapheme_vocab = special_vocab + grapheme_chars
phoneme_vocab = special_vocab + phoneme_chars
grapheme_to_id = {ch: i for i, ch in enumerate(grapheme_vocab)}
id_to_phoneme = {str(i): ch for i, ch in enumerate(phoneme_vocab)}

with open("g2p_vocab.json", "w") as f:
    json.dump({
        "grapheme_to_id": grapheme_to_id,
        "id_to_phoneme": id_to_phoneme,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }, f, ensure_ascii=False, indent=2)
print(f"Saved g2p_vocab.json ({len(grapheme_vocab)} graphemes, {len(phoneme_vocab)} phonemes)")


# --- Encoder wrapper ---
class EncoderModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids):
        return self.encoder(input_ids=input_ids).last_hidden_state


# --- Manual decoder (avoids HF ops that coremltools can't convert) ---
class DecoderModel(nn.Module):
    """Manual BART decoder implementation (1 layer, 1 attention head).

    Uses only basic torch ops that CoreML supports. The causal attention mask
    and position IDs are passed as explicit inputs instead of being created
    dynamically inside the model.
    """
    def __init__(self, model):
        super().__init__()
        decoder = model.get_decoder()
        layer = decoder.layers[0]

        # Embeddings — use raw nn.Embedding for positions so that input
        # values are used directly (BartLearnedPositionalEmbedding ignores
        # input values and creates positions internally via torch.arange).
        self.embed_tokens = decoder.embed_tokens
        self.embed_positions = nn.Embedding.from_pretrained(
            decoder.embed_positions.weight, freeze=True,
        )
        self.layernorm_embedding = decoder.layernorm_embedding

        # Self-attention projections
        self.self_attn_q = layer.self_attn.q_proj
        self.self_attn_k = layer.self_attn.k_proj
        self.self_attn_v = layer.self_attn.v_proj
        self.self_attn_out = layer.self_attn.out_proj
        self.self_attn_norm = layer.self_attn_layer_norm

        # Cross-attention projections
        self.cross_attn_q = layer.encoder_attn.q_proj
        self.cross_attn_k = layer.encoder_attn.k_proj
        self.cross_attn_v = layer.encoder_attn.v_proj
        self.cross_attn_out = layer.encoder_attn.out_proj
        self.cross_attn_norm = layer.encoder_attn_layer_norm

        # FFN
        self.fc1 = layer.fc1
        self.fc2 = layer.fc2
        self.final_layer_norm = layer.final_layer_norm

        # LM head
        self.lm_head = model.lm_head
        self.register_buffer("final_logits_bias", model.final_logits_bias)

        self.scale = float(config.d_model // config.decoder_attention_heads) ** -0.5

    def forward(self, decoder_input_ids, encoder_hidden_states, position_ids, causal_mask):
        """
        Args:
            decoder_input_ids:    (1, dec_len) int32
            encoder_hidden_states:(1, enc_len, d_model) float32
            position_ids:         (1, dec_len) int32 — values [2, 3, 4, ...] (BART offset=2)
            causal_mask:          (1, dec_len, dec_len) float32 — 0 to attend, -1e4 to mask
        """
        # Token + position embeddings
        hidden = self.embed_tokens(decoder_input_ids) + self.embed_positions(position_ids)
        hidden = self.layernorm_embedding(hidden)

        # Self-attention with causal mask
        residual = hidden
        q = self.self_attn_q(hidden)
        k = self.self_attn_k(hidden)
        v = self.self_attn_v(hidden)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        hidden = torch.matmul(attn_weights, v)
        hidden = self.self_attn_out(hidden)
        hidden = residual + hidden
        hidden = self.self_attn_norm(hidden)

        # Cross-attention (no mask needed — attend to all encoder positions)
        residual = hidden
        q = self.cross_attn_q(hidden)
        k = self.cross_attn_k(encoder_hidden_states)
        v = self.cross_attn_v(encoder_hidden_states)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        hidden = torch.matmul(attn_weights, v)
        hidden = self.cross_attn_out(hidden)
        hidden = residual + hidden
        hidden = self.cross_attn_norm(hidden)

        # FFN
        residual = hidden
        hidden = F.gelu(self.fc1(hidden))
        hidden = self.fc2(hidden)
        hidden = residual + hidden
        hidden = self.final_layer_norm(hidden)

        # Project to vocab
        logits = self.lm_head(hidden) + self.final_logits_bias
        return logits


# --- Build models ---
encoder_model = EncoderModel(model.get_encoder())
encoder_model.eval()
decoder_model = DecoderModel(model)
decoder_model.eval()


# --- Verify manual decoder matches HuggingFace ---
print("Verifying manual decoder matches HuggingFace...")
test_word = "hello"
input_ids = [1] + [grapheme_to_id.get(c, 3) for c in test_word] + [2]
input_tensor = torch.tensor([input_ids])

with torch.no_grad():
    hf_encoder_out = model.get_encoder()(input_tensor).last_hidden_state
    hf_decoder_out = model.get_decoder()(
        input_ids=torch.tensor([[1]]),
        encoder_hidden_states=hf_encoder_out,
    ).last_hidden_state
    hf_logits = model.lm_head(hf_decoder_out) + model.final_logits_bias

    manual_encoder_out = encoder_model(input_tensor)
    manual_logits = decoder_model(
        torch.tensor([[1]]),
        manual_encoder_out,
        torch.tensor([[2]]),  # position_ids with BART offset
        torch.zeros(1, 1, 1),  # single token, no masking needed
    )

max_diff = (hf_logits - manual_logits).abs().max().item()
print(f"  Max diff from HuggingFace: {max_diff:.2e}")
assert max_diff < 1e-5, f"Manual decoder doesn't match HF! diff={max_diff}"


# --- Convert encoder ---
print("\nTracing encoder...")
with torch.no_grad():
    traced_encoder = torch.jit.trace(encoder_model, input_tensor)

print("Converting encoder to CoreML...")
encoder_coreml = ct.convert(
    traced_encoder,
    inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 64)), dtype=np.int32)],
    outputs=[ct.TensorType(name="encoder_hidden_states")],
    minimum_deployment_target=ct.target.iOS17,
)
encoder_coreml.save("G2PEncoder.mlpackage")
print("Saved G2PEncoder.mlpackage")


# --- Convert decoder ---
print("\nTracing decoder...")
dec_len = 3
example_args = (
    torch.tensor([[1, 5, 6]]),                                            # decoder_input_ids
    torch.randn(1, 7, config.d_model),                                    # encoder_hidden_states
    torch.tensor([[2, 3, 4]]),                                             # position_ids
    torch.triu(torch.full((1, dec_len, dec_len), -1e4), diagonal=1),       # causal_mask
)
with torch.no_grad():
    traced_decoder = torch.jit.trace(decoder_model, example_args)

print("Converting decoder to CoreML...")
decoder_coreml = ct.convert(
    traced_decoder,
    inputs=[
        ct.TensorType(name="decoder_input_ids",    shape=(1, ct.RangeDim(1, 64)),                        dtype=np.int32),
        ct.TensorType(name="encoder_hidden_states", shape=(1, ct.RangeDim(1, 64), config.d_model),       dtype=np.float32),
        ct.TensorType(name="position_ids",          shape=(1, ct.RangeDim(1, 64)),                        dtype=np.int32),
        ct.TensorType(name="causal_mask",           shape=(1, ct.RangeDim(1, 64), ct.RangeDim(1, 64)),   dtype=np.float32),
    ],
    outputs=[ct.TensorType(name="logits")],
    minimum_deployment_target=ct.target.iOS17,
)
decoder_coreml.save("G2PDecoder.mlpackage")
print("Saved G2PDecoder.mlpackage")


# --- End-to-end verification: CoreML vs PyTorch ---
print("\nVerifying end-to-end (CoreML vs PyTorch)...")

# PyTorch reference
with torch.no_grad():
    generated = model.generate(input_ids=input_tensor)
pt_phonemes = "".join(
    id_to_phoneme.get(str(t), "?") for t in generated[0].tolist() if t > 3
)

# CoreML greedy decode
enc_pred = encoder_coreml.predict({"input_ids": np.array([input_ids], dtype=np.int32)})
enc_hidden = enc_pred["encoder_hidden_states"]

decoder_ids = [1]  # BOS
for _ in range(64):
    n = len(decoder_ids)
    dec_pred = decoder_coreml.predict({
        "decoder_input_ids": np.array([decoder_ids], dtype=np.int32),
        "encoder_hidden_states": enc_hidden,
        "position_ids": np.array([[i + 2 for i in range(n)]], dtype=np.int32),
        "causal_mask": np.triu(np.full((1, n, n), -1e4, dtype=np.float32), k=1),
    })
    next_token = int(np.argmax(dec_pred["logits"][0, -1, :]))
    if next_token == 2:  # EOS
        break
    decoder_ids.append(next_token)

coreml_phonemes = "".join(
    id_to_phoneme.get(str(t), "?") for t in decoder_ids if t > 3
)

print(f"  PyTorch: '{test_word}' -> {pt_phonemes}")
print(f"  CoreML:  '{test_word}' -> {coreml_phonemes}")
print(f"  {'MATCH' if pt_phonemes == coreml_phonemes else 'MISMATCH'}")

print("\nDone! Created G2PEncoder.mlpackage, G2PDecoder.mlpackage, and g2p_vocab.json")
