"""Encoder wrapper that also emits joint.enc(encoder_out) as a separate output.

PART OF B3 (enc_proj split). Pairs with joint_no_encproj.mlpackage so the
joint can skip the enc_proj matmul (which is constant across the per-token
inner loop). Expected RTFx win: +1-3%.

NOT YET INTEGRATED — this file documents the design. To complete B3:

1. Use this wrapper in place of EncoderStreamingWithPostPrompt in a new
   convert script (`convert_nemotron_multilingual_encprojsplit.py`).
2. Pass `model.joint.enc` as the `joint_enc` arg.
3. The traced encoder.mlpackage will emit `encoder_proj` as a 6th output
   (alongside encoded, enc_len, cache_ch_n, cache_t_n, cache_len_n).
4. Quantize encoder with mixed_layerpos.py — same boundaries should still work.
5. Swift: in StreamingNemotronMultilingualAsrManager+Pipeline.swift, after
   the encoder.prediction call, ALSO extract `encoder_proj` (multiArrayValue).
   Slice per-frame `encStepProj = encoder_proj[:, t:t+1, :]`. Feed encStepProj
   to joint_no_encproj.mlpackage as the `encoder_proj` input instead of the
   raw encoder_step.
6. Use joint_no_encproj.mlpackage (already traced at
   `build_encproj_split_test/`) in place of joint.mlpackage.

The wrapper itself is correct; the remaining work is converter + Swift
integration. Expected total: ~2 hours engineering for ~1-3% RTFx, with
real risk that overhead-bound ANE makes the win smaller than predicted.
"""
from __future__ import annotations

from typing import Tuple

import torch

from multilingual_components import EncoderStreamingWithPostPrompt, NUM_PROMPTS


class EncoderStreamingWithPostPromptAndEncProj(EncoderStreamingWithPostPrompt):
    """Extension of EncoderStreamingWithPostPrompt that also emits enc_proj.

    Inputs identical to parent.
    Outputs: (conditioned, enc_len, cache_ch_n, cache_t_n, cache_len_n, encoder_proj)
    where encoder_proj = joint.enc(conditioned.transpose(1, 2))  # [B, T_enc, 640]
    """

    def __init__(self, encoder, prompt_kernel, joint_enc, num_prompts=NUM_PROMPTS):
        super().__init__(encoder, prompt_kernel, num_prompts=num_prompts)
        self.joint_enc = joint_enc  # joint's enc Linear: [1024 -> 640]

    def forward(
        self,
        features: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
        prompt_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conditioned, enc_len, cc_n, ct_n, cl_n = super().forward(
            features, length, cache_last_channel, cache_last_time,
            cache_last_channel_len, prompt_id,
        )
        # Pre-project encoder output for joint reuse across inner-loop tokens
        # conditioned: [B, D=1024, T_enc]
        encoder_proj = self.joint_enc(conditioned.transpose(1, 2))  # [B, T_enc, 640]
        return conditioned, enc_len, cc_n, ct_n, cl_n, encoder_proj
