"""Encoder wrapper with Int8 cache I/O + dynamic per-tensor scale.

C2 cache state INT8 compression. Cache_channel and cache_time are
passed as Int8 tensors with companion Float scales (per-tensor
symmetric quantization, scale = max(|x|)/127 computed dynamically
per chunk — no calibration data dependency).

Schema:
  Inputs:
    features                  Float32 [1, 128, T_mel]
    length                    Int32   [1]
    cache_channel_int8        Int8    [1, 24, 42, 1024]    (was Float32)
    cache_channel_scale       Float32 [1]                   NEW
    cache_time_int8           Int8    [1, 24, 1024, 8]     (was Float32)
    cache_time_scale          Float32 [1]                   NEW
    cache_last_channel_len    Int32   [1]
    prompt_id                 Int32   [1]
  Outputs:
    encoded                   Float32 [1, 1024, T_enc]
    encoded_length            Int32   [1]
    cache_channel_out_int8    Int8    [1, 24, 42, 1024]    (was Float32)
    cache_channel_out_scale   Float32 [1]                   NEW
    cache_time_out_int8       Int8    [1, 24, 1024, 8]     (was Float32)
    cache_time_out_scale      Float32 [1]                   NEW
    cache_len_out             Int32   [1]
    encoder_proj              Float32 [1, T_enc, 640]

Dequant inside the graph: cache_channel = cache_channel_int8.float() * cache_channel_scale
Quant on output: cache_channel_out_int8 = (cache_channel_out / scale_out).clamp(-127, 127) → Int8

The Int8 cast at the output boundary is naive rounding/truncation
(coremltools auto-inserted). Since we pre-divide by scale, values
are already in [-127, 127] range so the cast is lossless beyond the
0.5-unit rounding noise.
"""
from __future__ import annotations

from typing import Tuple

import torch

from multilingual_components import EncoderStreamingWithPostPrompt, NUM_PROMPTS


class EncoderStreamingWithInt8Cache(EncoderStreamingWithPostPrompt):
    """Extends EncoderStreamingWithPostPrompt with Int8 cache I/O.

    Internally still operates on Float; the Int8 representation is
    only at the model I/O boundary, saving transfer bandwidth between
    CPU memory and ANE/GPU SRAM.
    """

    def __init__(self, encoder, prompt_kernel, joint_enc, num_prompts=NUM_PROMPTS):
        super().__init__(encoder, prompt_kernel, num_prompts=num_prompts)
        self.joint_enc = joint_enc

    def forward(
        self,
        features: torch.Tensor,
        length: torch.Tensor,
        cache_channel_int8: torch.Tensor,  # Float at trace time; declared Int8 at conversion
        cache_channel_scale: torch.Tensor,  # Float [1]
        cache_time_int8: torch.Tensor,
        cache_time_scale: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
        prompt_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # Dequantize: cache_int8 (as Float after coremltools input cast) * scale
        cache_channel = cache_channel_int8 * cache_channel_scale
        cache_time = cache_time_int8 * cache_time_scale

        # Run encoder body
        conditioned, enc_len, cc_n, ct_n, cl_n = super().forward(
            features, length, cache_channel, cache_time,
            cache_last_channel_len, prompt_id,
        )

        # encoder_proj for B3 split (same as encprojsplit variant)
        encoder_proj = self.joint_enc(conditioned.transpose(1, 2))

        # Quantize cache outputs: per-tensor dynamic scale = absmax / 127
        cc_n_absmax = cc_n.abs().amax()
        ct_n_absmax = ct_n.abs().amax()
        # Avoid div-by-zero at chunk 0 (zero cache); use small floor
        cc_n_scale = (cc_n_absmax / 127.0).clamp_min(1e-6)
        ct_n_scale = (ct_n_absmax / 127.0).clamp_min(1e-6)

        # Divide + clamp; coremltools cast Float→Int8 at output boundary
        # will truncate to [-128, 127], so clamp([-127, 127]) gives safe margin
        cc_n_quant = (cc_n / cc_n_scale).clamp(-127.0, 127.0)
        ct_n_quant = (ct_n / ct_n_scale).clamp(-127.0, 127.0)

        # Return: (encoded, enc_len, cc_n_int8, cc_n_scale, ct_n_int8, ct_n_scale, cl_n, encoder_proj)
        return (
            conditioned, enc_len,
            cc_n_quant, cc_n_scale.unsqueeze(0),
            ct_n_quant, ct_n_scale.unsqueeze(0),
            cl_n,
            encoder_proj,
        )
