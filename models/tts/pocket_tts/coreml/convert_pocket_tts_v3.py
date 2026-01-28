#!/usr/bin/env python3
"""Convert PocketTTS to CoreML format - v3 with proper state handling."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
import copy

print("Loading PocketTTS model...")
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states, increment_steps

# Load the model
model = TTSModel.load_model()
model.eval()

print(f"Model loaded. Sample rate: {model.sample_rate}")
print(f"FlowLM dim: {model.flow_lm.dim}, ldim: {model.flow_lm.ldim}")

# ============================================================================
# Simple wrapper that embeds the model state
# ============================================================================
class MimiDecoderWithState(nn.Module):
    """Mimi decoder that initializes state internally."""
    def __init__(self, mimi, emb_mean, emb_std):
        super().__init__()
        self.mimi = mimi
        self.emb_mean = nn.Parameter(emb_mean, requires_grad=False)
        self.emb_std = nn.Parameter(emb_std, requires_grad=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [1, seq_len, ldim] -> audio: [1, 1, samples]"""
        # Create fresh state for this forward pass
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=latent.shape[1] * 16)

        # Denormalize
        denorm = latent * self.emb_std + self.emb_mean
        transposed = denorm.transpose(-1, -2)

        # Quantize
        quantized = self.mimi.quantizer(transposed)

        # Decode using the model's method with state
        audio = self.mimi.decode_from_latent(quantized, mimi_state)
        return audio

class FlowLMBackboneWithState(nn.Module):
    """FlowLM backbone that initializes state internally."""
    def __init__(self, flow_lm):
        super().__init__()
        self.flow_lm = flow_lm

    def forward(self, conditioning: torch.Tensor, latent_sequence: torch.Tensor) -> torch.Tensor:
        """Full forward pass with fresh state."""
        # Create fresh state
        total_len = conditioning.shape[1] + latent_sequence.shape[1]
        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=total_len)

        # Replace NaN with BOS
        latent_sequence = torch.where(
            torch.isnan(latent_sequence),
            self.flow_lm.bos_emb,
            latent_sequence
        )

        # Project
        projected = self.flow_lm.input_linear(latent_sequence)

        # Combine
        full_input = torch.cat([conditioning, projected], dim=1)

        # Transformer with state
        transformer_out = self.flow_lm.transformer(full_input, model_state)

        if self.flow_lm.out_norm is not None:
            transformer_out = self.flow_lm.out_norm(transformer_out)

        return transformer_out[:, -latent_sequence.shape[1]:]

class VoiceEncoderWithState(nn.Module):
    """Voice encoder that handles state properly."""
    def __init__(self, mimi, speaker_proj_weight):
        super().__init__()
        self.mimi = mimi
        self.speaker_proj_weight = nn.Parameter(speaker_proj_weight, requires_grad=False)
        self.frame_size = mimi.frame_size

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: [1, 1, samples] -> conditioning: [1, frames, 1024]"""
        # Use the model's encode_to_latent which handles state
        encoded = self.mimi.encode_to_latent(audio)
        latents = encoded.transpose(-1, -2).to(torch.float32)
        conditioning = torch.nn.functional.linear(latents, self.speaker_proj_weight)
        return conditioning

# ============================================================================
# Simple components (no state needed)
# ============================================================================
class TextEncoder(nn.Module):
    def __init__(self, conditioner):
        super().__init__()
        self.conditioner = conditioner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        from pocket_tts.conditioners.base import TokenizedText
        return self.conditioner(TokenizedText(tokens))

class FlowDecoder(nn.Module):
    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net
        self.ldim = flow_lm.ldim

    def forward(self, transformer_out: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s = torch.zeros_like(noise[..., :1])
        t = torch.ones_like(noise[..., :1])
        flow_dir = self.flow_net(transformer_out, s, t, noise)
        return noise + flow_dir

class EOSDetector(nn.Module):
    def __init__(self, flow_lm, threshold=-4.0):
        super().__init__()
        self.out_eos = flow_lm.out_eos
        self.threshold = threshold

    def forward(self, transformer_out: torch.Tensor) -> torch.Tensor:
        logit = self.out_eos(transformer_out)
        return (logit > self.threshold).float()

# ============================================================================
# Conversion functions
# ============================================================================
def convert_text_encoder():
    print("\n--- Converting Text Encoder ---")
    text_enc = TextEncoder(model.flow_lm.conditioner)
    text_enc.eval()

    dummy_tokens = torch.zeros((1, 32), dtype=torch.int64)
    traced = torch.jit.trace(text_enc, dummy_tokens)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=(1, ct.RangeDim(1, 256)), dtype=np.int32)],
        outputs=[ct.TensorType(name="embeddings")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save("pocket_tts_text_encoder.mlpackage")
    print("Saved: pocket_tts_text_encoder.mlpackage")

def convert_flow_decoder():
    print("\n--- Converting Flow Decoder ---")
    flow_dec = FlowDecoder(model.flow_lm)
    flow_dec.eval()

    dim = model.flow_lm.dim
    ldim = model.flow_lm.ldim

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

def convert_eos_detector():
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

def convert_mimi_decoder():
    print("\n--- Converting Mimi Decoder ---")
    decoder = MimiDecoderWithState(
        model.mimi,
        model.flow_lm.emb_mean,
        model.flow_lm.emb_std
    )
    decoder.eval()

    ldim = model.flow_lm.ldim  # 32
    dummy_latent = torch.randn(1, 8, ldim)

    with torch.no_grad():
        print(f"Testing decoder with input: {dummy_latent.shape}")
        output = decoder(dummy_latent)
        print(f"Output shape: {output.shape}")
        traced = torch.jit.trace(decoder, dummy_latent)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latent", shape=(1, ct.RangeDim(1, 256), ldim), dtype=np.float32)],
        outputs=[ct.TensorType(name="audio")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save("pocket_tts_mimi_decoder.mlpackage")
    print("Saved: pocket_tts_mimi_decoder.mlpackage")

def convert_backbone():
    print("\n--- Converting FlowLM Backbone ---")
    backbone = FlowLMBackboneWithState(model.flow_lm)
    backbone.eval()

    dim = model.flow_lm.dim
    ldim = model.flow_lm.ldim

    dummy_conditioning = torch.randn(1, 64, dim)
    dummy_latents = torch.full((1, 8, ldim), float('nan'))

    with torch.no_grad():
        print(f"Testing backbone...")
        output = backbone(dummy_conditioning, dummy_latents)
        print(f"Output shape: {output.shape}")
        traced = torch.jit.trace(backbone, (dummy_conditioning, dummy_latents))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="conditioning", shape=(1, ct.RangeDim(1, 512), dim), dtype=np.float32),
            ct.TensorType(name="latent_sequence", shape=(1, ct.RangeDim(1, 256), ldim), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="transformer_out")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save("pocket_tts_backbone.mlpackage")
    print("Saved: pocket_tts_backbone.mlpackage")

def convert_voice_encoder():
    print("\n--- Converting Voice Encoder ---")
    voice_enc = VoiceEncoderWithState(model.mimi, model.flow_lm.speaker_proj_weight)
    voice_enc.eval()

    # Audio must be multiple of frame_size (1920 samples = 80ms at 24kHz)
    frame_size = model.mimi.frame_size
    num_frames = 12  # ~1 second
    audio_samples = frame_size * num_frames
    dummy_audio = torch.randn(1, 1, audio_samples)

    with torch.no_grad():
        print(f"Testing voice encoder with {audio_samples} samples ({audio_samples/24000:.2f}s)...")
        output = voice_enc(dummy_audio)
        print(f"Output shape: {output.shape}")
        traced = torch.jit.trace(voice_enc, dummy_audio)

    # Allow variable length audio (multiples of frame_size)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio", shape=(1, 1, ct.RangeDim(frame_size, frame_size * 375)), dtype=np.float32)],
        outputs=[ct.TensorType(name="voice_conditioning")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.save("pocket_tts_voice_encoder.mlpackage")
    print("Saved: pocket_tts_voice_encoder.mlpackage")

if __name__ == "__main__":
    os.makedirs("coreml", exist_ok=True)
    os.chdir("coreml")

    print("=" * 60)
    print("PocketTTS CoreML Conversion v3")
    print("=" * 60)

    components = [
        ("Text Encoder", convert_text_encoder),
        ("Flow Decoder", convert_flow_decoder),
        ("EOS Detector", convert_eos_detector),
        ("Mimi Decoder", convert_mimi_decoder),
        ("Backbone", convert_backbone),
        ("Voice Encoder", convert_voice_encoder),
    ]

    results = []
    for name, func in components:
        try:
            func()
            results.append((name, "✓"))
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            traceback.print_exc()
            results.append((name, f"✗ {str(e)[:50]}"))

    print("\n" + "=" * 60)
    print("Conversion Results:")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 60)

    # Print model sizes
    print("\nModel Sizes:")
    import glob
    for pkg in sorted(glob.glob("*.mlpackage")):
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(pkg) for f in fn)
        print(f"  {pkg}: {size / 1024 / 1024:.1f} MB")
