"""Convert NanoCodec decoder to CoreML.

The NanoCodec decoder converts discrete codec tokens back to audio waveform.
It's a causal HiFi-GAN-based vocoder.

The main conversion challenge is Snake activation, which uses a TorchScript
function internally. We replace it with a plain PyTorch equivalent before tracing.

Usage:
    python convert/convert_nanocodec.py [--nemo-path /path/to/model.nemo]
"""
import sys
import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct


def _snake_plain(x, alpha):
    """Plain PyTorch snake activation: x + (1/alpha) * sin^2(alpha * x)."""
    return x + (1.0 / (alpha + 1e-9)) * torch.sin(alpha * x).pow(2)


class TraceableSnake(nn.Module):
    """Snake activation without TorchScript — traceable by coremltools."""

    def __init__(self, original_snake):
        super().__init__()
        self.alpha = original_snake.alpha

    def forward(self, x):
        return _snake_plain(x, self.alpha)


class TraceableHalfSnake(nn.Module):
    """HalfSnake activation without TorchScript."""

    def __init__(self, original):
        super().__init__()
        self.snake_channels = original.snake_channels
        self.snake_act = TraceableSnake(original.snake_act)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        snake_out = self.snake_act(x[:, :self.snake_channels, :])
        lrelu_out = self.lrelu(x[:, self.snake_channels:, :])
        return torch.cat([snake_out, lrelu_out], dim=1)


def replace_snake_activations(module):
    """Recursively replace Snake/HalfSnake with traceable versions."""
    replaced = 0
    for name, child in module.named_children():
        cls_name = type(child).__name__
        if cls_name == 'HalfSnake':
            setattr(module, name, TraceableHalfSnake(child))
            replaced += 1
        elif cls_name == 'Snake':
            setattr(module, name, TraceableSnake(child))
            replaced += 1
        else:
            replaced += replace_snake_activations(child)
    return replaced


def _patch_causal_convs(module):
    """Make all CausalConv1d modules CoreML-traceable.

    NeMo's CausalConv1dNorm stores kernel_size, stride, padding_total as 0-dim
    tensor buffers and uses torch.ceil().to(int64) / .long() in
    _get_extra_padding_for_conv1d. These produce `aten::Int(tensor)` ops that
    coremltools cannot convert.

    Fix:
    1. Replace tensor buffers with plain Python ints
    2. Replace _get_extra_padding_for_conv1d with pure-Python-arithmetic version
    """
    import math
    import types

    def _safe_extra_padding(self, hidden_states):
        # For fixed-size inputs (max_frames=256), extra padding is always 0.
        # Returning a constant avoids symbolic shape arithmetic that produces
        # untraceable aten::Int ops in the traced graph.
        return 0

    for name, child in module.named_modules():
        # Convert tensor buffers to plain ints
        for attr in ('padding_total', 'kernel_size', 'stride'):
            val = getattr(child, attr, None)
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                int_val = int(val.item())
                if attr in dict(child.named_buffers()):
                    delattr(child, attr)
                child.__dict__[attr] = int_val

        # Replace the problematic method
        if hasattr(child, '_get_extra_padding_for_conv1d'):
            child._get_extra_padding_for_conv1d = types.MethodType(_safe_extra_padding, child)


def _patch_mask_noop():
    """Replace mask_sequence_tensor with a no-op in NeMo's audio codec module.

    Since we use fixed-size inputs (max_frames), all positions are valid and
    masking is unnecessary. This eliminates the einops rearrange and comparison
    ops inside mask_sequence_tensor that produce untraceable symbolic int ops.
    """
    import nemo.collections.tts.modules.audio_codec_modules as acm
    acm.mask_sequence_tensor = lambda tensor, lengths: tensor


def _disable_typecheck():
    """Disable NeMo's @typecheck decorator so we can pass None for input_len."""
    from nemo.core.classes.common import typecheck
    typecheck.set_typecheck_enabled(False)


def _patch_conv_transpose_unpad(module):
    """Patch CausalConvTranspose1dNorm to use negative indexing for unpadding.

    The original code does:
        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., self.padding_left : end]

    This produces aten::size → prim::NumToTensor → aten::Int in the traced graph,
    which coremltools cannot convert. Using negative indexing avoids querying shape:
        hidden_states = hidden_states[..., self.padding_left : -self.padding_right]
    """
    import types

    def _forward_neg_index(self, inputs, input_len):
        hidden_states = self.conv(inputs)
        # Use negative indexing to avoid aten::size → aten::Int
        if self.padding_right > 0:
            hidden_states = hidden_states[..., self.padding_left : -self.padding_right]
        elif self.padding_left > 0:
            hidden_states = hidden_states[..., self.padding_left :]
        hidden_states = self.activation(hidden_states)
        return hidden_states  # mask_sequence_tensor already patched to no-op

    import nemo.collections.tts.modules.audio_codec_modules as acm
    for name, child in module.named_modules():
        if isinstance(child, acm.CausalConvTranspose1dNorm):
            child.forward = types.MethodType(_forward_neg_index, child)


class TraceableNanoCodecDecoder(nn.Module):
    """Wrapper around NanoCodec's decode path for tracing.

    The NeMo codec uses symbolic int operations (.long(), .to(int64)) in:
    1. CausalConv1dNorm._get_extra_padding_for_conv1d — shape-derived padding
    2. CausalHiFiGANDecoder.forward — (audio_len * up_sample_rate).long()
    3. mask_sequence_tensor — length-based masking

    CoreML's converter cannot handle these symbolic int casts. Since we always
    use fixed-size inputs (max_frames), we:
    - Monkey-patch _get_extra_padding to use Python int arithmetic
    - Patch mask_sequence_tensor to be a no-op (no padding to mask)
    - Bypass the audio_decoder.forward to avoid symbolic length tracking
    - Pass None for input_len everywhere (typecheck disabled)
    """

    def __init__(self, codec_model, max_frames):
        super().__init__()
        self.codec_model = codec_model
        self.max_frames = max_frames

        # Disable NeMo typecheck so modules accept None for input_len
        _disable_typecheck()

        # Patch mask_sequence_tensor to no-op (fixed-size inputs need no masking)
        _patch_mask_noop()

        # Patch CausalConv1d modules for CoreML traceability
        _patch_causal_convs(codec_model.audio_decoder)

        # Patch ConvTranspose1d unpadding to use negative indexing
        _patch_conv_transpose_unpad(codec_model.audio_decoder)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, num_codebooks, max_frames) int32 - codec token IDs
        Returns:
            audio: (B, T_samples) float32 - decoded audio waveform
        """
        # --- dequantize (bypass tokens_len masking, not needed for fixed input) ---
        codec = self.codec_model
        tokens_cbt = tokens.permute(1, 0, 2)  # (C, B, T)
        indices_grouped = tokens_cbt.chunk(codec.vector_quantizer.num_groups, dim=0)
        dequantized_parts = []
        for indices_group, fsq_group in zip(indices_grouped, codec.vector_quantizer.fsqs):
            # FSQ decode: rearrange 'D B T -> B D T' without einops (avoids aten::Int)
            idx = indices_group.permute(1, 0, 2)
            codes_nonneg = (idx // fsq_group.dim_base_index) % fsq_group.num_levels
            dq = fsq_group.nonnegative_to_codes(codes_nonneg)
            dequantized_parts.append(dq)
        dequantized = torch.cat(dequantized_parts, dim=1)
        dequantized = dequantized.to(codec.dtype)

        # --- audio decoder (no input_len needed, masking is patched to no-op) ---
        ad = codec.audio_decoder
        out = ad.pre_conv(inputs=dequantized, input_len=None)
        for act, res_layer, up_conv, _rate in zip(
            ad.activations, ad.res_layers, ad.up_sample_conv_layers, ad.up_sample_rates
        ):
            out = act(out)
            out = up_conv(inputs=out, input_len=None)
            out = res_layer(inputs=out, input_len=None)

        out = ad.post_activation(out)
        out = ad.post_conv(inputs=out, input_len=None)
        audio = ad.out_activation(out)
        # Use squeeze instead of einops rearrange to avoid symbolic aten::Int ops
        audio = audio.squeeze(1)  # (B, 1, T) -> (B, T)
        return audio


def convert_nanocodec(nemo_path=None, max_frames=256, output_path="build/nanocodec_decoder.mlpackage"):
    # Load model
    print("Loading MagpieTTS model...")
    from nemo.collections.tts.models import MagpieTTSModel
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    codec = model._codec_model
    codec.eval()

    num_codebooks = codec.num_codebooks
    codebook_size = codec.codebook_size
    print(f"NanoCodec: {num_codebooks} codebooks, {codebook_size} codes, "
          f"{codec.sample_rate}Hz, {codec.samples_per_frame} samples/frame")

    # Replace Snake activations with traceable versions
    print("Replacing Snake activations...")
    replaced = replace_snake_activations(codec)
    print(f"  Replaced {replaced} Snake/HalfSnake modules")

    # Verify replacement
    with torch.no_grad():
        test_tokens = torch.randint(0, codebook_size, (1, num_codebooks, 10), dtype=torch.long)
        test_len = torch.tensor([10], dtype=torch.long)
        audio_test = codec.decode(tokens=test_tokens, tokens_len=test_len)
        print(f"  Post-replacement decode test: audio shape = {audio_test[0].shape}")

    # Create traceable wrapper
    print("Creating traceable NanoCodec decoder...")
    traceable_codec = TraceableNanoCodecDecoder(codec, max_frames)
    traceable_codec.eval()

    # Example inputs
    B = 1
    T = max_frames
    tokens = torch.randint(0, codebook_size, (B, num_codebooks, T), dtype=torch.long)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        audio = traceable_codec(tokens)
        print(f"Audio output shape: {audio.shape}")
        expected_samples = T * codec.samples_per_frame
        print(f"Expected ~{expected_samples} samples, got {audio.shape[-1]}")

    # Trace
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(traceable_codec, (tokens,))

    # Verify trace
    with torch.no_grad():
        ref_out = traceable_codec(tokens)
        traced_out = traced(tokens)
        diff = (ref_out - traced_out).abs().max().item()
        print(f"Trace verification - max diff: {diff:.6e}")

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="tokens", shape=(1, num_codebooks, T), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="audio", dtype=np.float32),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Test CoreML
    print("\nTesting CoreML model...")
    coreml_model = ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    test_tokens = np.random.randint(0, codebook_size, (1, num_codebooks, T)).astype(np.int32)

    out = coreml_model.predict({
        "tokens": test_tokens,
    })

    audio_out = out["audio"]
    print(f"Audio output shape: {audio_out.shape}")
    print(f"Audio range: [{audio_out.min():.4f}, {audio_out.max():.4f}]")
    print("Done!")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--output", type=str, default="build/nanocodec_decoder.mlpackage")
    args = parser.parse_args()
    convert_nanocodec(args.nemo_path, args.max_frames, args.output)
