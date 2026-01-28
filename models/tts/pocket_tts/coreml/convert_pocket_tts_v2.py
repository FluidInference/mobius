#!/usr/bin/env python3
"""Convert PocketTTS to CoreML format - v2 with fixed Mimi decoder."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import coremltools as ct
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
# Component 1: Text Encoder
# ============================================================================
class TextEncoder(nn.Module):
    def __init__(self, conditioner):
        super().__init__()
        self.conditioner = conditioner

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        from pocket_tts.conditioners.base import TokenizedText
        return self.conditioner(TokenizedText(tokens))

# ============================================================================
# Component 2: Flow Decoder (unchanged)
# ============================================================================
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

# ============================================================================
# Component 3: EOS Detector (unchanged)
# ============================================================================
class EOSDetector(nn.Module):
    def __init__(self, flow_lm, threshold=-4.0):
        super().__init__()
        self.out_eos = flow_lm.out_eos
        self.threshold = threshold

    def forward(self, transformer_out: torch.Tensor) -> torch.Tensor:
        logit = self.out_eos(transformer_out)
        return (logit > self.threshold).float()

# ============================================================================
# Component 4: Mimi Decoder (STATELESS version)
# ============================================================================
class MimiDecoderStateless(nn.Module):
    """Stateless Mimi decoder - processes full latent sequence at once."""
    def __init__(self, mimi, emb_mean, emb_std):
        super().__init__()
        # Copy weights but make stateless versions
        self.quantizer = mimi.quantizer
        self.upsample = mimi.upsample if hasattr(mimi, 'upsample') else None
        self.decoder_transformer = mimi.decoder_transformer
        self.decoder = mimi.decoder
        self.emb_mean = emb_mean
        self.emb_std = emb_std

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [1, seq_len, ldim] -> audio: [1, 1, samples]"""
        # Denormalize
        denorm = latent * self.emb_std + self.emb_mean
        transposed = denorm.transpose(-1, -2)  # [1, ldim, seq_len]

        # Quantize (identity for DummyQuantizer)
        quantized = self.quantizer(transposed)

        # Upsample if needed (pass None for model_state)
        if self.upsample is not None:
            upsampled = self.upsample(quantized, model_state=None)
        else:
            upsampled = quantized

        # Decoder transformer (pass None for model_state)
        (transformed,) = self.decoder_transformer(upsampled, model_state=None)

        # SEANet decoder (pass None for model_state)
        audio = self.decoder(transformed, model_state=None)

        return audio

# ============================================================================
# Component 5: FlowLM Backbone Transformer (STATELESS)
# ============================================================================
class FlowLMBackboneStateless(nn.Module):
    """Stateless FlowLM backbone - no KV cache."""
    def __init__(self, flow_lm):
        super().__init__()
        self.bos_emb = flow_lm.bos_emb
        self.input_linear = flow_lm.input_linear
        self.transformer = flow_lm.transformer
        self.out_norm = flow_lm.out_norm

    def forward(
        self,
        conditioning: torch.Tensor,  # [1, cond_len, dim] (text + audio combined)
        latent_sequence: torch.Tensor,  # [1, gen_len, ldim]
    ) -> torch.Tensor:
        """Returns transformer output at all latent positions."""
        # Replace NaN with BOS embedding
        latent_sequence = torch.where(
            torch.isnan(latent_sequence),
            self.bos_emb,
            latent_sequence
        )

        # Project latents
        projected = self.input_linear(latent_sequence)

        # Concatenate
        full_input = torch.cat([conditioning, projected], dim=1)

        # Run transformer stateless (pass None for model_state)
        transformer_out = self.transformer(full_input, model_state=None)

        if self.out_norm is not None:
            transformer_out = self.out_norm(transformer_out)

        # Return only latent positions
        return transformer_out[:, -latent_sequence.shape[1]:]

# ============================================================================
# Component 6: Voice Encoder (for voice cloning)
# ============================================================================
class VoiceEncoder(nn.Module):
    """Encode reference audio to voice conditioning."""
    def __init__(self, mimi, speaker_proj_weight):
        super().__init__()
        self.encoder = mimi.encoder
        self.encoder_transformer = mimi.encoder_transformer
        self.downsample = mimi.downsample if hasattr(mimi, 'downsample') else None
        self.speaker_proj_weight = speaker_proj_weight
        self.frame_size = mimi.frame_size

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: [1, 1, samples] -> conditioning: [1, frames, 1024]"""
        # Encode
        emb = self.encoder(audio, model_state=None)

        # Encoder transformer
        (emb,) = self.encoder_transformer(emb, model_state=None)

        # Downsample if needed
        if self.downsample is not None:
            emb = self.downsample(emb, model_state=None)

        # Project to conditioning space
        latents = emb.transpose(-1, -2).to(torch.float32)
        conditioning = torch.nn.functional.linear(latents, self.speaker_proj_weight)
        return conditioning

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
    print("\n--- Converting Mimi Decoder (Stateless) ---")
    decoder = MimiDecoderStateless(
        model.mimi,
        model.flow_lm.emb_mean,
        model.flow_lm.emb_std
    )
    decoder.eval()

    ldim = model.flow_lm.ldim  # 32
    # Test with multiple frames
    dummy_latent = torch.randn(1, 8, ldim)  # 8 frames

    with torch.no_grad():
        # Test it first
        print(f"Testing decoder with input shape: {dummy_latent.shape}")
        try:
            output = decoder(dummy_latent)
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Direct call failed: {e}")
            raise

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
    print("\n--- Converting FlowLM Backbone (Stateless) ---")
    backbone = FlowLMBackboneStateless(model.flow_lm)
    backbone.eval()

    dim = model.flow_lm.dim  # 1024
    ldim = model.flow_lm.ldim  # 32

    # Dummy inputs
    dummy_conditioning = torch.randn(1, 64, dim)  # text + audio conditioning
    dummy_latents = torch.full((1, 8, ldim), float('nan'))  # BOS tokens

    with torch.no_grad():
        print(f"Testing backbone with conditioning: {dummy_conditioning.shape}, latents: {dummy_latents.shape}")
        try:
            output = backbone(dummy_conditioning, dummy_latents)
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Direct call failed: {e}")
            raise

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
    voice_enc = VoiceEncoder(model.mimi, model.flow_lm.speaker_proj_weight)
    voice_enc.eval()

    # 1 second of audio at 24kHz
    dummy_audio = torch.randn(1, 1, 24000)

    with torch.no_grad():
        print(f"Testing voice encoder with input: {dummy_audio.shape}")
        try:
            output = voice_enc(dummy_audio)
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Direct call failed: {e}")
            raise

        traced = torch.jit.trace(voice_enc, dummy_audio)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio", shape=(1, 1, ct.RangeDim(1920, 720000)), dtype=np.float32)],
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
    print("PocketTTS CoreML Conversion v2")
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
            print(f"Error: {e}")
            results.append((name, f"✗ {str(e)[:50]}"))

    print("\n" + "=" * 60)
    print("Conversion Results:")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 60)
