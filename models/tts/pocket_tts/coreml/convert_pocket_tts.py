#!/usr/bin/env python3
"""Convert PocketTTS to CoreML format."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np
from pathlib import Path

print("Loading PocketTTS model...")
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states

# Load the model
model = TTSModel.load_model()
model.eval()

print(f"Model loaded. Sample rate: {model.sample_rate}")
print(f"FlowLM dim: {model.flow_lm.dim}, ldim: {model.flow_lm.ldim}")

# ============================================================================
# Component 1: Text Encoder (text tokens -> embeddings)
# ============================================================================
class TextEncoder(nn.Module):
    """Wraps the text conditioner for CoreML export."""
    def __init__(self, conditioner):
        super().__init__()
        self.conditioner = conditioner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [1, seq_len] int64 -> embeddings: [1, seq_len, dim]"""
        from pocket_tts.conditioners.base import TokenizedText
        return self.conditioner(TokenizedText(tokens))

# ============================================================================
# Component 2: Mimi Encoder (audio -> latents for voice cloning)
# ============================================================================
class MimiEncoder(nn.Module):
    """Wraps Mimi encoder for voice cloning."""
    def __init__(self, mimi, speaker_proj_weight):
        super().__init__()
        self.mimi = mimi
        self.speaker_proj_weight = speaker_proj_weight

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: [1, 1, samples] -> conditioning: [1, frames, 1024]"""
        encoded = self.mimi.encode_to_latent(audio)
        latents = encoded.transpose(-1, -2).to(torch.float32)
        conditioning = torch.nn.functional.linear(latents, self.speaker_proj_weight)
        return conditioning

# ============================================================================
# Component 3: FlowLM Backbone (for autoregressive generation)
# ============================================================================
class FlowLMBackbone(nn.Module):
    """FlowLM transformer backbone without KV cache (stateless)."""
    def __init__(self, flow_lm):
        super().__init__()
        self.flow_lm = flow_lm

    def forward(
        self,
        text_embeddings: torch.Tensor,  # [1, text_len, dim]
        audio_conditioning: torch.Tensor,  # [1, audio_len, dim]
        latent_sequence: torch.Tensor,  # [1, gen_len, ldim]
    ) -> torch.Tensor:
        """Returns transformer output for flow decoding."""
        # Replace NaN with BOS embedding
        latent_sequence = torch.where(
            torch.isnan(latent_sequence),
            self.flow_lm.bos_emb,
            latent_sequence
        )

        # Project latents to transformer dim
        input_ = self.flow_lm.input_linear(latent_sequence)

        # Concatenate conditioning
        combined_cond = torch.cat([text_embeddings, audio_conditioning], dim=1)
        full_input = torch.cat([combined_cond, input_], dim=1)

        # Run transformer (stateless - no KV cache)
        # This would need modification for streaming
        transformer_out = self.flow_lm.transformer.transformer(full_input)

        if self.flow_lm.out_norm:
            transformer_out = self.flow_lm.out_norm(transformer_out)

        # Return only the latent positions
        transformer_out = transformer_out[:, -latent_sequence.shape[1]:]
        return transformer_out

# ============================================================================
# Component 4: Flow Network (transformer_out + noise -> latent)
# ============================================================================
class FlowDecoder(nn.Module):
    """Flow network for single-step LSD decoding."""
    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net
        self.ldim = flow_lm.ldim

    def forward(
        self,
        transformer_out: torch.Tensor,  # [1, dim] - last position
        noise: torch.Tensor,  # [1, ldim]
    ) -> torch.Tensor:
        """Single-step flow decoding (LSD with 1 step)."""
        # s=0, t=1 for single step
        s = torch.zeros_like(noise[..., :1])
        t = torch.ones_like(noise[..., :1])
        flow_dir = self.flow_net(transformer_out, s, t, noise)
        return noise + flow_dir

# ============================================================================
# Component 5: EOS Detector
# ============================================================================
class EOSDetector(nn.Module):
    """Detects end of sequence from transformer output."""
    def __init__(self, flow_lm, threshold=-4.0):
        super().__init__()
        self.out_eos = flow_lm.out_eos
        self.threshold = threshold

    def forward(self, transformer_out: torch.Tensor) -> torch.Tensor:
        """Returns 1.0 if EOS, 0.0 otherwise."""
        logit = self.out_eos(transformer_out)
        return (logit > self.threshold).float()

# ============================================================================
# Component 6: Mimi Decoder (latents -> audio)
# ============================================================================
class MimiDecoder(nn.Module):
    """Wraps Mimi decoder for audio synthesis."""
    def __init__(self, mimi, emb_mean, emb_std):
        super().__init__()
        self.mimi = mimi
        self.emb_mean = emb_mean
        self.emb_std = emb_std

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [1, 1, ldim] -> audio: [1, 1, samples]"""
        # Denormalize
        denorm = latent * self.emb_std + self.emb_mean
        transposed = denorm.transpose(-1, -2)
        quantized = self.mimi.quantizer(transposed)
        # Note: This is stateless decoding - streaming would need state
        audio = self.mimi.decoder(self.mimi.decoder_transformer(quantized))
        return audio

# ============================================================================
# Export functions
# ============================================================================
def convert_text_encoder():
    """Convert text encoder to CoreML."""
    print("\n--- Converting Text Encoder ---")
    text_enc = TextEncoder(model.flow_lm.conditioner)
    text_enc.eval()

    # Trace with dummy input
    dummy_tokens = torch.zeros((1, 32), dtype=torch.int64)
    traced = torch.jit.trace(text_enc, dummy_tokens)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=(1, ct.RangeDim(1, 256)), dtype=np.int32)],
        outputs=[ct.TensorType(name="embeddings")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save("pocket_tts_text_encoder.mlpackage")
    print("Saved: pocket_tts_text_encoder.mlpackage")
    return mlmodel

def convert_mimi_decoder():
    """Convert Mimi decoder to CoreML."""
    print("\n--- Converting Mimi Decoder ---")
    decoder = MimiDecoder(
        model.mimi,
        model.flow_lm.emb_mean,
        model.flow_lm.emb_std
    )
    decoder.eval()

    ldim = model.flow_lm.ldim  # 32
    dummy_latent = torch.randn(1, 1, ldim)

    with torch.no_grad():
        traced = torch.jit.trace(decoder, dummy_latent)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=(1, 1, ldim), dtype=np.float32)],
        outputs=[ct.TensorType(name="audio")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save("pocket_tts_mimi_decoder.mlpackage")
    print("Saved: pocket_tts_mimi_decoder.mlpackage")
    return mlmodel

def convert_flow_decoder():
    """Convert flow network to CoreML."""
    print("\n--- Converting Flow Decoder ---")
    flow_dec = FlowDecoder(model.flow_lm)
    flow_dec.eval()

    dim = model.flow_lm.dim  # 1024
    ldim = model.flow_lm.ldim  # 32

    dummy_transformer_out = torch.randn(1, dim)
    dummy_noise = torch.randn(1, ldim)

    with torch.no_grad():
        traced = torch.jit.trace(flow_dec, (dummy_transformer_out, dummy_noise))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="transformer_out", shape=(1, dim), dtype=np.float32),
            ct.TensorType(name="noise", shape=(1, ldim), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="latent")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save("pocket_tts_flow_decoder.mlpackage")
    print("Saved: pocket_tts_flow_decoder.mlpackage")
    return mlmodel

def convert_eos_detector():
    """Convert EOS detector to CoreML."""
    print("\n--- Converting EOS Detector ---")
    eos = EOSDetector(model.flow_lm)
    eos.eval()

    dim = model.flow_lm.dim
    dummy = torch.randn(1, dim)

    with torch.no_grad():
        traced = torch.jit.trace(eos, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="transformer_out", shape=(1, dim), dtype=np.float32)],
        outputs=[ct.TensorType(name="is_eos")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save("pocket_tts_eos_detector.mlpackage")
    print("Saved: pocket_tts_eos_detector.mlpackage")
    return mlmodel

if __name__ == "__main__":
    os.makedirs("coreml", exist_ok=True)
    os.chdir("coreml")

    print("=" * 60)
    print("PocketTTS CoreML Conversion")
    print("=" * 60)

    # Start with simpler components
    try:
        convert_flow_decoder()
    except Exception as e:
        print(f"Flow decoder failed: {e}")

    try:
        convert_eos_detector()
    except Exception as e:
        print(f"EOS detector failed: {e}")

    try:
        convert_mimi_decoder()
    except Exception as e:
        print(f"Mimi decoder failed: {e}")

    try:
        convert_text_encoder()
    except Exception as e:
        print(f"Text encoder failed: {e}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
